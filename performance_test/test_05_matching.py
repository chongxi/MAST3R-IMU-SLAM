"""
Test 5: Matching (Feature Matching with CUDA Backend)
Tests the matching functions using mast3r_slam_backends
"""
import torch
import time
import numpy as np
from pathlib import Path
import yaml
import lietorch

config_path = Path("../config/base.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

import sys
sys.path.insert(0, str(Path("..").resolve()))

from mast3r_slam.config import set_global_config
set_global_config(config)

from mast3r_slam.mast3r_utils import load_mast3r, mast3r_asymmetric_inference
from mast3r_slam.frame import create_frame
from mast3r_slam.matching import match_iterative_proj

def benchmark_matching(num_warmup=10, num_runs=50):
    """Benchmark matching operations"""
    print("=" * 70)
    print("TEST 5: Matching (Feature Matching + CUDA Backend)")
    print("=" * 70)
    
    print("\nSkipped: Test needs 4D tensor format (b,h,w,c), needs refactoring")
    print("The matching function expects image-shaped tensors, not flattened points")
    return None
    
    # Create realistic test data
    batch_size = 1
    num_points = 512 * 512  # Full image worth of points
    
    # Create dummy 3D points and descriptors
    X11 = torch.randn(batch_size, num_points, 3, device='cuda', dtype=torch.float32)
    X21 = torch.randn(batch_size, num_points, 3, device='cuda', dtype=torch.float32)
    D11 = torch.randn(batch_size, num_points, 24, device='cuda', dtype=torch.float32)
    D21 = torch.randn(batch_size, num_points, 24, device='cuda', dtype=torch.float32)
    
    print(f"\nTest data shape:")
    print(f"  Points: {X11.shape}")
    print(f"  Descriptors: {D11.shape}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(num_warmup):
        idx_1_to_2_init, valid_proj2 = match_iterative_proj(X11, X21, D11, D21)
        torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking matching...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        idx_1_to_2_init, valid_proj2 = match_iterative_proj(X11, X21, D11, D21)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Matching: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Matches found: {valid_proj2.sum().item()}/{num_points}")
    
    # Test with smaller point set (more realistic after downsampling)
    print("\nTesting with downsampled points (64x64)...")
    num_points_small = 64 * 64
    X11_small = X11[:, :num_points_small, :]
    X21_small = X21[:, :num_points_small, :]
    D11_small = D11[:, :num_points_small, :]
    D21_small = D21[:, :num_points_small, :]
    
    # Warmup
    for _ in range(num_warmup):
        idx_1_to_2_init, valid_proj2 = match_iterative_proj(X11_small, X21_small, D11_small, D21_small)
        torch.cuda.synchronize()
    
    # Benchmark
    times_small = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        idx_1_to_2_init, valid_proj2 = match_iterative_proj(X11_small, X21_small, D11_small, D21_small)
        torch.cuda.synchronize()
        times_small.append((time.perf_counter() - start) * 1000)
    
    mean_time_small = np.mean(times_small)
    std_time_small = np.std(times_small)
    
    print(f"  Matching (downsampled): {mean_time_small:.2f} ± {std_time_small:.2f} ms")
    print(f"  Speedup: {mean_time/mean_time_small:.2f}x")
    
    return {
        "matching_full": mean_time,
        "matching_downsampled": mean_time_small
    }

if __name__ == "__main__":
    results = benchmark_matching()
    
    if results is None:
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("SUMMARY:")
    print(f"  Matching runs for EVERY frame-keyframe pair")
    print(f"  Full resolution: {results['matching_full']:.2f}ms")
    print(f"  Downsampled (64x64): {results['matching_downsampled']:.2f}ms")
    print(f"\n  With 3 keyframes: matching cost = {results['matching_downsampled']*3:.1f}ms per frame")
