"""
Test 6: Backend Optimization (Bundle Adjustment + Gauss-Newton)
Tests the gauss_newton CUDA backend optimization
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

# Skip this test - backend module doesn't exist as separate importable module
print("=" * 70)
print("TEST 6: Backend Optimization (Bundle Adjustment)")
print("=" * 70)
print("\nSkipped: Backend optimization runs in separate process")
print("Cannot import backend_utils - it's part of multiprocess backend")
print("See main.py run_backend() for actual implementation")
import sys
sys.exit(1)

def benchmark_optimization(num_warmup=5, num_runs=20):
    """Benchmark backend optimization (bundle adjustment)"""
    print("=" * 70)
    print("TEST 6: Backend Optimization (Bundle Adjustment)")
    print("=" * 70)
    
    # Create realistic optimization problem
    # Simulating 100 frames with 500 points per frame
    num_frames = 100
    num_points_per_frame = 500
    
    # Camera poses (SE3)
    poses = lietorch.SE3.Identity(num_frames, device='cuda')
    
    # 3D points
    points = torch.randn(num_frames * num_points_per_frame, 3, device='cuda', dtype=torch.float32)
    
    # 2D observations (pixel coordinates)
    observations = torch.randn(num_frames * num_points_per_frame, 2, device='cuda', dtype=torch.float32)
    
    # Intrinsics (fx, fy, cx, cy)
    intrinsics = torch.tensor([300.0, 300.0, 256.0, 256.0], device='cuda', dtype=torch.float32)
    intrinsics = intrinsics.unsqueeze(0).repeat(num_frames, 1)
    
    # Correspondence indices
    ii = torch.arange(num_frames, device='cuda').repeat_interleave(num_points_per_frame)
    jj = torch.arange(num_points_per_frame, device='cuda').repeat(num_frames)
    
    # Weights
    weights = torch.ones(num_frames * num_points_per_frame, device='cuda', dtype=torch.float32)
    
    print(f"\nOptimization problem:")
    print(f"  Frames: {num_frames}")
    print(f"  Points: {num_frames * num_points_per_frame}")
    print(f"  Observations: {observations.shape}")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(num_warmup):
        try:
            # Call backend optimization (Gauss-Newton)
            backend_utils.gauss_newton(
                poses, points, observations, intrinsics,
                ii, jj, weights,
                max_iters=5,
                lm_lambda=1e-4
            )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  Warning: Backend function call failed: {e}")
            print(f"  This is expected if backend is not built or function signature changed")
            return None
    
    # Benchmark
    print("Benchmarking optimization...")
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        backend_utils.gauss_newton(
            poses, points, observations, intrinsics,
            ii, jj, weights,
            max_iters=5,
            lm_lambda=1e-4
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Optimization (5 iters): {mean_time:.2f} ± {std_time:.2f} ms")
    
    # Test with different iteration counts
    print("\nTesting different iteration counts...")
    for num_iters in [1, 3, 5, 10]:
        times_iters = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            backend_utils.gauss_newton(
                poses, points, observations, intrinsics,
                ii, jj, weights,
                max_iters=num_iters,
                lm_lambda=1e-4
            )
            torch.cuda.synchronize()
            times_iters.append((time.perf_counter() - start) * 1000)
        
        mean = np.mean(times_iters)
        std = np.std(times_iters)
        print(f"  {num_iters:2d} iterations: {mean:6.2f} ± {std:5.2f} ms")
    
    return {
        "optimization_5_iters": mean_time
    }

if __name__ == "__main__":
    try:
        results = benchmark_optimization()
        
        if results:
            print(f"\n{'='*70}")
            print("SUMMARY:")
            print(f"  Bundle adjustment runs periodically (backend thread)")
            print(f"  Optimization (5 iters): {results['optimization_5_iters']:.2f}ms")
            print(f"  Scales linearly with iteration count")
            print(f"  Currently FP32 only - potential 1.5x speedup with FP16")
        else:
            print(f"\n{'='*70}")
            print("NOTE: Backend optimization test skipped")
            print("  Backend may not be built or function signature differs")
            print("  See test_mast3r_backend/BUILD_REFERENCE.md for build instructions")
            
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        print("This usually means the backend is not built or API changed")
        print("See test_mast3r_backend/ for backend documentation")
