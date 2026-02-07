#!/usr/bin/env python3
"""
export_weights.py — PyTorch checkpoint → flat binary for C inference

Usage:
    python export_weights.py \
        --config weights/fastenhancer_s/config.yaml \
        --ckpt-dir weights/fastenhancer_s \
        --output weights/fe_small.bin

Weight order must match fe_load_weights() in fastenhancer.c exactly:
  1. enc_pre (StridedConv reshaped)
  2. enc[0..2] (Conv1d k=3)
  3. rf_pre_lin, rf_pre_conv
  4. rf[0..2] { gru, rnn_fc, attn.qkv, attn_fc, pe(block0 only) }
  5. rf_post_lin, rf_post_conv
  6. dec_1x1[0..2], dec_3x3[0..2]
  7. dec_post_1x1, dec_post_up
"""
import argparse
import struct
import sys
import os

import torch
import numpy as np

# Add fastenhancer repo to path — import selectively to avoid tensorboard dependency
FE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fastenhancer')
sys.path.insert(0, FE_REPO)

# Stub out utils package to avoid tensorboard import
import types
utils_pkg = types.ModuleType('utils')
utils_pkg.__path__ = [os.path.join(FE_REPO, 'utils')]
sys.modules['utils'] = utils_pkg

from utils.hparams import get_hparams
from models.fastenhancer.default.model import ONNXModel


def write_tensor(f, tensor, name=""):
    """Write tensor as flat float32 binary."""
    data = tensor.detach().cpu().float().contiguous().numpy()
    f.write(data.tobytes())
    numel = data.size
    print(f"  {name:40s} shape={str(list(tensor.shape)):20s} numel={numel:8d}")
    return numel


def export(args):
    # Load config
    hps = get_hparams(args.config, args.name)

    # Find latest checkpoint in ckpt_dir
    import re
    ckpt_dir = args.ckpt_dir
    pth_files = [f for f in os.listdir(ckpt_dir) if re.match(r'[0-9]{5,}\.pth', f)]
    if not pth_files:
        print(f"ERROR: No .pth files found in {ckpt_dir}")
        sys.exit(1)
    latest = sorted(pth_files)[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    raw_sd = checkpoint['model']

    # Create ONNX model and load weights
    model = ONNXModel(**dict(hps.model_kwargs.items()))
    model.load_state_dict(raw_sd, strict=True)
    model.eval()

    # Fuse BN + remove weight_norm
    model.remove_weight_reparameterizations()
    model.flatten_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters after fusion: {total_params}")

    sd = model.state_dict()

    # Read architecture params from config
    kernel_sizes = list(hps.model_kwargs.kernel_size)
    enc_blocks = len(kernel_sizes) - 1  # first is pre-net
    rf_blocks = hps.model_kwargs.rnnformer_kwargs.num_blocks
    dec_blocks = enc_blocks  # decoder mirrors encoder

    print(f"\nArchitecture: enc_blocks={enc_blocks}, rf_blocks={rf_blocks}, dec_blocks={dec_blocks}")

    total_floats = 0
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'wb') as f:

        print("\n=== Encoder PreNet (StridedConv reshaped) ===")
        total_floats += write_tensor(f, sd['enc_pre.0.weight'], 'enc_pre.weight')
        total_floats += write_tensor(f, sd['enc_pre.0.bias'], 'enc_pre.bias')

        print("\n=== Encoder Blocks ===")
        for i in range(enc_blocks):
            total_floats += write_tensor(f, sd[f'encoder.{i}.0.weight'], f'enc[{i}].weight')
            total_floats += write_tensor(f, sd[f'encoder.{i}.0.bias'], f'enc[{i}].bias')

        print("\n=== RNNFormer PreNet ===")
        # rf_pre.0 = Linear(F1→F2), rf_pre.1 = Conv1d(C1→C2, k=1)
        total_floats += write_tensor(f, sd['rf_pre.0.weight'], 'rf_pre_lin.weight')
        total_floats += write_tensor(f, sd['rf_pre.1.weight'], 'rf_pre_conv.weight')
        total_floats += write_tensor(f, sd['rf_pre.1.bias'], 'rf_pre_conv.bias')

        print("\n=== RNNFormer Blocks ===")
        for i in range(rf_blocks):
            prefix = f'rf_block.{i}'
            print(f"\n  --- Block {i} ---")

            # GRU
            total_floats += write_tensor(f, sd[f'{prefix}.rnn.weight_ih_l0'], f'rf[{i}].gru.W_ih')
            total_floats += write_tensor(f, sd[f'{prefix}.rnn.bias_ih_l0'], f'rf[{i}].gru.b_ih')
            total_floats += write_tensor(f, sd[f'{prefix}.rnn.weight_hh_l0'], f'rf[{i}].gru.W_hh')
            total_floats += write_tensor(f, sd[f'{prefix}.rnn.bias_hh_l0'], f'rf[{i}].gru.b_hh')

            # RNN FC (fused BN → has bias)
            total_floats += write_tensor(f, sd[f'{prefix}.rnn_fc.weight'], f'rf[{i}].rnn_fc.weight')
            total_floats += write_tensor(f, sd[f'{prefix}.rnn_fc.bias'], f'rf[{i}].rnn_fc.bias')

            # Attention QKV
            total_floats += write_tensor(f, sd[f'{prefix}.attn.qkv.weight'], f'rf[{i}].attn.qkv.weight')
            # QKV may or may not have bias depending on pre_norm fusing
            if f'{prefix}.attn.qkv.bias' in sd:
                total_floats += write_tensor(f, sd[f'{prefix}.attn.qkv.bias'], f'rf[{i}].attn.qkv.bias')

            # Attention FC (fused BN → has bias)
            total_floats += write_tensor(f, sd[f'{prefix}.attn_fc.weight'], f'rf[{i}].attn_fc.weight')
            total_floats += write_tensor(f, sd[f'{prefix}.attn_fc.bias'], f'rf[{i}].attn_fc.bias')

            # Positional embedding (first block only)
            if i == 0:
                total_floats += write_tensor(f, sd[f'{prefix}.pe'], f'rf[{i}].pe')

        print("\n=== RNNFormer PostNet ===")
        total_floats += write_tensor(f, sd['rf_post.0.weight'], 'rf_post_lin.weight')
        total_floats += write_tensor(f, sd['rf_post.1.weight'], 'rf_post_conv.weight')
        total_floats += write_tensor(f, sd['rf_post.1.bias'], 'rf_post_conv.bias')

        print("\n=== Decoder Blocks ===")
        for i in range(dec_blocks):
            dec_idx = i
            total_floats += write_tensor(f, sd[f'decoder.{dec_idx}.0.weight'], f'dec_1x1[{i}].weight')
            total_floats += write_tensor(f, sd[f'decoder.{dec_idx}.0.bias'], f'dec_1x1[{i}].bias')
            total_floats += write_tensor(f, sd[f'decoder.{dec_idx}.2.weight'], f'dec_3x3[{i}].weight')
            total_floats += write_tensor(f, sd[f'decoder.{dec_idx}.2.bias'], f'dec_3x3[{i}].bias')

        print("\n=== Decoder PostNet ===")
        total_floats += write_tensor(f, sd['dec_post.0.weight'], 'dec_post_1x1.weight')
        total_floats += write_tensor(f, sd['dec_post.0.bias'], 'dec_post_1x1.bias')
        total_floats += write_tensor(f, sd['dec_post.2.weight'], 'dec_post_up.weight')
        total_floats += write_tensor(f, sd['dec_post.2.bias'], 'dec_post_up.bias')

    file_size = os.path.getsize(args.output)
    print(f"\n{'='*60}")
    print(f"Total floats written: {total_floats}")
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"Expected:  {total_floats * 4:,} bytes")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to config.yaml")
    parser.add_argument('-d', '--ckpt-dir', type=str, required=True,
                        help="Directory containing .pth checkpoint files")
    parser.add_argument('-o', '--output', type=str, default='weights/fe_small.bin')
    args = parser.parse_args()

    # Use ckpt_dir as the base_dir for the wrapper (name parameter)
    args.name = args.ckpt_dir

    export(args)
