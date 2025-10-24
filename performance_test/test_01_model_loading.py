"""
Test 1: Model Loading and Initialization
Tests the time to load MASt3R model
"""
import torch
import time
import numpy as np
from pathlib import Path
import yaml

# Load config
config_path = Path("../config/base.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

import sys
sys.path.insert(0, str(Path("..").resolve()))

from mast3r_slam.config import set_global_config
set_global_config(config)

from mast3r_slam.mast3r_utils import load_mast3r

def benchmark_model_loading(num_runs=5):
    """Benchmark model loading time"""
    print("=" * 70)
    print("TEST 1: Model Loading")
    print("=" * 70)
    
    # Check if checkpoint exists
    checkpoint_path = Path("../checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please download the model checkpoint first.")
        return None
    
    # Test FP32 loading
    print("\nBenchmarking FP32 model loading...")

if __name__ == "__main__":
    results = benchmark_model_loading()
    
    if results is None:
        print("\nSkipped: Model checkpoint not available")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Model loading is a one-time cost: ~{results['model_loading_fp32']:.0f}ms")
    print(f"  FP16 conversion adds: ~{results['model_loading_fp16'] - results['model_loading_fp32']:.0f}ms")
