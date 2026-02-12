#!/usr/bin/env python3
"""
export_dereverb_weights.py — 2sderev PyTorch checkpoint → flat binary for C inference

Usage:
    python export_dereverb_weights.py \
        --repo-dir /path/to/2sderev \
        --config CI \
        --output weights/derev_wpe.bin

Binary layout (all float32, little-endian):

    WPE DNN:
      W_ih        [2048, 257]    = 526,336 floats
      b_ih        [2048]         = 2,048
      W_hh        [2048, 512]    = 1,048,576
      b_hh        [2048]         = 2,048
      clean_W     [257, 512]     = 131,584

    WPE Stats:
      mean        [257]
      std         [257]

    PF DNN:
      W_ih        [2048, 257]    = 526,336
      b_ih        [2048]         = 2,048
      W_hh        [2048, 512]    = 1,048,576
      b_hh        [2048]         = 2,048
      clean_W     [257, 512]     = 131,584
      interf_W    [257, 512]     = 131,584

    PF Stats:
      mean        [257]
      std         [257]

Total: ~14.2 MB
"""
import argparse
import os
import sys
import json

import torch
import numpy as np


def write_tensor(f, tensor, name=""):
    """Write tensor as flat float32 binary."""
    data = tensor.detach().cpu().float().contiguous().numpy()
    f.write(data.tobytes())
    numel = data.size
    print(f"  {name:45s} shape={str(list(tensor.shape)):20s} numel={numel:8d}")
    return numel


def export(args):
    repo_dir = args.repo_dir
    config = args.config  # "CI" or "HA"

    # Load params
    params_path = os.path.join(repo_dir, "derev_params.json")
    if not os.path.exists(params_path):
        print(f"ERROR: derev_params.json not found in {repo_dir}")
        sys.exit(1)

    with open(params_path) as f:
        params = json.load(f)

    # Determine model files from nested JSON structure
    # Config keys: "wpe+pf_ci", "wpe+pf_ha" (full pipeline with Wiener PF)
    config_key = f"wpe+pf_{config.lower()}"
    if config_key not in params:
        print(f"ERROR: Config key '{config_key}' not found in derev_params.json")
        print(f"       Available keys: {list(params.keys())}")
        sys.exit(1)

    cfg = params[config_key]
    wpe_model_path = os.path.join(repo_dir, cfg["dnn_wpe"]["ckpt"])
    pf_model_path = os.path.join(repo_dir, cfg["dnn_pf"]["ckpt"])

    # Stats paths use {} format pattern for mean/std
    wpe_stats_pattern = cfg["dnn_wpe"]["stats"]  # e.g. "stats/reverberant/tr_{}_noiseless_reverberant_abs.pt"
    pf_stats_pattern = cfg["dnn_pf"]["stats"]

    # Check files exist
    for path in [wpe_model_path, pf_model_path]:
        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path}")
            print(f"       Download models from the 2sderev repository.")
            sys.exit(1)

    print(f"Config: {config} ({config_key})")
    print(f"WPE model: {wpe_model_path}")
    print(f"PF model:  {pf_model_path}")

    # Load state dicts
    wpe_sd = torch.load(wpe_model_path, map_location='cpu', weights_only=True)
    pf_sd = torch.load(pf_model_path, map_location='cpu', weights_only=True)

    # Load normalization stats (pattern uses {} for mean/std)
    wpe_mean_path = os.path.join(repo_dir, wpe_stats_pattern.format("mean"))
    wpe_std_path = os.path.join(repo_dir, wpe_stats_pattern.format("std"))
    pf_mean_path = os.path.join(repo_dir, pf_stats_pattern.format("mean"))
    pf_std_path = os.path.join(repo_dir, pf_stats_pattern.format("std"))

    for path in [wpe_mean_path, wpe_std_path, pf_mean_path, pf_std_path]:
        if not os.path.exists(path):
            print(f"ERROR: Stats file not found: {path}")
            sys.exit(1)

    wpe_mean = torch.load(wpe_mean_path, map_location='cpu', weights_only=True)
    wpe_std = torch.load(wpe_std_path, map_location='cpu', weights_only=True)
    pf_mean = torch.load(pf_mean_path, map_location='cpu', weights_only=True)
    pf_std = torch.load(pf_std_path, map_location='cpu', weights_only=True)

    print(f"\nWPE DNN keys: {list(wpe_sd.keys())}")
    print(f"PF DNN keys:  {list(pf_sd.keys())}")
    print(f"WPE stats: mean={wpe_mean.shape}, std={wpe_std.shape}")
    print(f"PF stats:  mean={pf_mean.shape}, std={pf_std.shape}")

    total_floats = 0
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'wb') as f:
        print("\n=== WPE DNN Weights ===")
        total_floats += write_tensor(f, wpe_sd['lstm_layers.weight_ih_l0'], 'wpe.W_ih')
        total_floats += write_tensor(f, wpe_sd['lstm_layers.bias_ih_l0'],   'wpe.b_ih')
        total_floats += write_tensor(f, wpe_sd['lstm_layers.weight_hh_l0'], 'wpe.W_hh')
        total_floats += write_tensor(f, wpe_sd['lstm_layers.bias_hh_l0'],   'wpe.b_hh')
        total_floats += write_tensor(f, wpe_sd['clean_map.weight'],         'wpe.clean_W')

        print("\n=== WPE Normalization Stats ===")
        total_floats += write_tensor(f, wpe_mean.flatten(), 'wpe_stats.mean')
        total_floats += write_tensor(f, wpe_std.flatten(),  'wpe_stats.std')

        print("\n=== PF DNN Weights ===")
        total_floats += write_tensor(f, pf_sd['lstm_layers.weight_ih_l0'],  'pf.W_ih')
        total_floats += write_tensor(f, pf_sd['lstm_layers.bias_ih_l0'],    'pf.b_ih')
        total_floats += write_tensor(f, pf_sd['lstm_layers.weight_hh_l0'],  'pf.W_hh')
        total_floats += write_tensor(f, pf_sd['lstm_layers.bias_hh_l0'],    'pf.b_hh')
        total_floats += write_tensor(f, pf_sd['clean_map.weight'],          'pf.clean_W')
        total_floats += write_tensor(f, pf_sd['interference_map.weight'],   'pf.interf_W')

        print("\n=== PF Normalization Stats ===")
        total_floats += write_tensor(f, pf_mean.flatten(), 'pf_stats.mean')
        total_floats += write_tensor(f, pf_std.flatten(),  'pf_stats.std')

    file_size = os.path.getsize(args.output)
    print(f"\n{'='*60}")
    print(f"Total floats written: {total_floats:,}")
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    print(f"Expected:  {total_floats * 4:,} bytes")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export 2sderev LSTM weights to flat binary")
    parser.add_argument('-r', '--repo-dir', type=str, required=True,
                        help="Path to cloned 2sderev repository")
    parser.add_argument('-c', '--config', type=str, default='CI',
                        choices=['CI', 'HA'],
                        help="Model config: CI (hearing aid) or HA")
    parser.add_argument('-o', '--output', type=str,
                        default='weights/derev_wpe.bin',
                        help="Output binary file path")
    args = parser.parse_args()
    export(args)
