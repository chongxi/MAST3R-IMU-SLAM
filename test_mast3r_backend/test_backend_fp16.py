"""
Test script to check FP16 support in mast3r_slam_backends

This script tests:
1. Which backend functions support FP16
2. Measures FP16 vs FP32 performance
3. Quantifies conversion overhead
4. Validates numerical accuracy
"""

import torch
import time
import numpy as np
import mast3r_slam_backends

print("=" * 70)
print("MASt3R SLAM Backend FP16 Support Test")
print("=" * 70)
print(f"Device: {torch.cuda.get_device_name()}")
print(f"CUDA: {torch.version.cuda}")
print(f"PyTorch: {torch.__version__}")
print()

# Get all available functions
backend_funcs = [f for f in dir(mast3r_slam_backends) if not f.startswith('_')]
print(f"Available backend functions: {backend_funcs}")
print()


def test_iter_proj():
    """Test iter_proj function with FP16 and FP32"""
    print("=" * 70)
    print("TEST 1: iter_proj (Iterative Projection Matching)")
    print("=" * 70)
    
    batch_size = 4
    num_points = 1024
    h, w = 64, 64
    
    # Create test data in FP32
    rays_img_fp32 = torch.randn(batch_size, h, w, 9, device='cuda', dtype=torch.float32)
    pts_3d_norm_fp32 = torch.randn(batch_size, num_points, 3, device='cuda', dtype=torch.float32)
    pts_3d_norm_fp32 = pts_3d_norm_fp32 / pts_3d_norm_fp32.norm(dim=2, keepdim=True)  # Normalize
    p_init_fp32 = torch.rand(batch_size, num_points, 2, device='cuda', dtype=torch.float32)
    p_init_fp32[:, :, 0] *= w - 2
    p_init_fp32[:, :, 1] *= h - 2
    p_init_fp32 += 1  # Keep within [1, w-2] and [1, h-2]
    
    # Test FP32 (baseline)
    print("\n1a. Testing FP32 (baseline)...")
    try:
        # Warmup
        for _ in range(5):
            p_new_fp32, valid_fp32 = mast3r_slam_backends.iter_proj(
                rays_img_fp32, pts_3d_norm_fp32, p_init_fp32, 10, 1e-4, 0.1
            )
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            p_new_fp32, valid_fp32 = mast3r_slam_backends.iter_proj(
                rays_img_fp32, pts_3d_norm_fp32, p_init_fp32, 10, 1e-4, 0.1
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time_fp32 = np.mean(times)
        std_time_fp32 = np.std(times)
        print(f"   ✓ FP32 works: {mean_time_fp32:.3f} ± {std_time_fp32:.3f} ms")
        print(f"   Output dtype: {p_new_fp32.dtype}")
        print(f"   Output shape: {p_new_fp32.shape}")
    except Exception as e:
        print(f"   ✗ FP32 failed: {e}")
        mean_time_fp32 = None
    
    # Test FP16
    print("\n1b. Testing FP16...")
    rays_img_fp16 = rays_img_fp32.half()
    pts_3d_norm_fp16 = pts_3d_norm_fp32.half()
    p_init_fp16 = p_init_fp32.half()
    
    try:
        # Warmup
        for _ in range(5):
            p_new_fp16, valid_fp16 = mast3r_slam_backends.iter_proj(
                rays_img_fp16, pts_3d_norm_fp16, p_init_fp16, 10, 1e-4, 0.1
            )
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            p_new_fp16, valid_fp16 = mast3r_slam_backends.iter_proj(
                rays_img_fp16, pts_3d_norm_fp16, p_init_fp16, 10, 1e-4, 0.1
            )
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time_fp16 = np.mean(times)
        std_time_fp16 = np.std(times)
        print(f"   ✓ FP16 works: {mean_time_fp16:.3f} ± {std_time_fp16:.3f} ms")
        print(f"   Output dtype: {p_new_fp16.dtype}")
        print(f"   Speedup: {mean_time_fp32/mean_time_fp16:.2f}x")
        
        # Check accuracy
        diff = (p_new_fp32 - p_new_fp16.float()).abs()
        print(f"   Accuracy: mean error = {diff.mean():.6f}, max error = {diff.max():.6f}")
        
    except Exception as e:
        print(f"   ✗ FP16 failed: {e}")
        if "mat1 and mat2 must have the same dtype" in str(e) or "Float" in str(e) and "Half" in str(e):
            print(f"   → Backend does NOT support FP16 (dtype mismatch)")
        else:
            print(f"   → Unknown error")
    
    print()


def test_refine_matches():
    """Test refine_matches function with FP16 and FP32"""
    print("=" * 70)
    print("TEST 2: refine_matches (Descriptor Match Refinement)")
    print("=" * 70)
    
    batch_size = 4
    h, w = 64, 64
    num_matches = 512
    fdim = 24  # Feature dimension
    
    # Create test data in FP32
    D11_fp32 = torch.randn(batch_size, h, w, fdim, device='cuda', dtype=torch.float32)
    D21_fp32 = torch.randn(batch_size, num_matches, fdim, device='cuda', dtype=torch.float32)
    p1_fp32 = torch.randint(0, min(h, w), (batch_size, num_matches, 2), device='cuda', dtype=torch.long)
    
    # Test FP32 (baseline)
    print("\n2a. Testing FP32 (baseline)...")
    try:
        # Warmup
        for _ in range(5):
            p1_new_fp32 = mast3r_slam_backends.refine_matches(D11_fp32, D21_fp32, p1_fp32, 2, 4)
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            p1_new_fp32 = mast3r_slam_backends.refine_matches(D11_fp32, D21_fp32, p1_fp32, 2, 4)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time_fp32 = np.mean(times)
        std_time_fp32 = np.std(times)
        print(f"   ✓ FP32 works: {mean_time_fp32:.3f} ± {std_time_fp32:.3f} ms")
        print(f"   Output dtype: {p1_new_fp32[0].dtype}")
        print(f"   Output shape: {p1_new_fp32[0].shape}")
    except Exception as e:
        print(f"   ✗ FP32 failed: {e}")
        mean_time_fp32 = None
    
    # Test FP16
    print("\n2b. Testing FP16...")
    D11_fp16 = D11_fp32.half()
    D21_fp16 = D21_fp32.half()
    
    try:
        # Warmup
        for _ in range(5):
            p1_new_fp16 = mast3r_slam_backends.refine_matches(D11_fp16, D21_fp16, p1_fp32, 2, 4)
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            p1_new_fp16 = mast3r_slam_backends.refine_matches(D11_fp16, D21_fp16, p1_fp32, 2, 4)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time_fp16 = np.mean(times)
        std_time_fp16 = np.std(times)
        print(f"   ✓ FP16 works: {mean_time_fp16:.3f} ± {std_time_fp16:.3f} ms")
        print(f"   Output dtype: {p1_new_fp16[0].dtype}")
        print(f"   Speedup: {mean_time_fp32/mean_time_fp16:.2f}x")
        
        # Check if results match
        matches = (p1_new_fp32[0] == p1_new_fp16[0]).all()
        print(f"   Results match: {matches}")
        
    except Exception as e:
        print(f"   ✗ FP16 failed: {e}")
    
    print()


def test_conversion_overhead():
    """Measure overhead of FP16→FP32 conversion"""
    print("=" * 70)
    print("TEST 3: Conversion Overhead Analysis")
    print("=" * 70)
    
    # Realistic tensor sizes from SLAM pipeline
    batch_size = 4
    h, w = 512, 512
    
    X = torch.randn(batch_size, h, w, 3, device='cuda', dtype=torch.float16)
    C = torch.randn(batch_size, h, w, device='cuda', dtype=torch.float16)
    D = torch.randn(batch_size, h, w, 24, device='cuda', dtype=torch.float16)
    Q = torch.randn(batch_size, h, w, device='cuda', dtype=torch.float16)
    
    total_elements = X.numel() + C.numel() + D.numel() + Q.numel()
    total_mb_fp16 = total_elements * 2 / (1024**2)
    total_mb_fp32 = total_elements * 4 / (1024**2)
    
    print(f"\nTensor sizes:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  FP16 memory: {total_mb_fp16:.2f} MB")
    print(f"  FP32 memory: {total_mb_fp32:.2f} MB")
    
    # Test individual .float() calls
    print("\n3a. Individual .float() calls (current approach)...")
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        X_fp32 = X.float()
        C_fp32 = C.float()
        D_fp32 = D.float()
        Q_fp32 = Q.float()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time_individual = np.mean(times)
    print(f"   Time: {mean_time_individual:.3f} ± {np.std(times):.3f} ms")
    
    # Test async .to() calls
    print("\n3b. Async .to() conversions...")
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        X_fp32 = X.to(dtype=torch.float32, non_blocking=True)
        C_fp32 = C.to(dtype=torch.float32, non_blocking=True)
        D_fp32 = D.to(dtype=torch.float32, non_blocking=True)
        Q_fp32 = Q.to(dtype=torch.float32, non_blocking=True)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    mean_time_async = np.mean(times)
    print(f"   Time: {mean_time_async:.3f} ± {np.std(times):.3f} ms")
    print(f"   Speedup: {mean_time_individual/mean_time_async:.2f}x")
    
    # Impact at 30 FPS
    print(f"\n3c. Impact on frame rate:")
    print(f"   Overhead per frame: {mean_time_individual:.3f} ms")
    print(f"   At 30 FPS: {mean_time_individual * 30:.1f} ms/sec ({mean_time_individual * 30 / 10:.1f}% of time)")
    frame_time_without = 1000 / 30  # 33.33 ms
    frame_time_with = frame_time_without + mean_time_individual
    fps_loss = 30 - (1000 / frame_time_with)
    print(f"   FPS loss: {fps_loss:.1f} FPS")
    
    print()


def test_gauss_newton():
    """Test Gauss-Newton optimization functions"""
    print("=" * 70)
    print("TEST 4: Gauss-Newton Bundle Adjustment")
    print("=" * 70)
    
    num_poses = 10
    num_points = 256
    num_edges = 15
    
    # Create test data
    Twc = torch.randn(num_poses, 8, device='cuda', dtype=torch.float32)  # [t, q, s]
    Twc[:, 6] = Twc[:, 6].abs()  # Normalize quaternion w component
    Xs = torch.randn(num_poses, num_points, 3, device='cuda', dtype=torch.float32)
    Cs = torch.rand(num_poses, num_points, 1, device='cuda', dtype=torch.float32)
    
    ii = torch.randint(0, num_poses, (num_edges,), device='cuda', dtype=torch.long)
    jj = torch.randint(0, num_poses, (num_edges,), device='cuda', dtype=torch.long)
    idx_ii2jj = torch.randint(0, num_points, (num_edges, num_points), device='cuda', dtype=torch.long)
    valid_match = torch.rand(num_edges, num_points, 1, device='cuda') > 0.5
    Q = torch.rand(num_edges, num_points, 1, device='cuda', dtype=torch.float32)
    
    print("\n4a. Testing gauss_newton_points (FP32)...")
    try:
        start = time.perf_counter()
        dx = mast3r_slam_backends.gauss_newton_points(
            Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
            sigma_point=1.0, C_thresh=0.1, Q_thresh=0.1, max_iter=5, delta_thresh=1e-6
        )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"   ✓ FP32 works: {elapsed:.3f} ms")
        print(f"   Output shape: {dx[0].shape}")
    except Exception as e:
        print(f"   ✗ FP32 failed: {e}")
    
    print("\n4b. Testing with FP16 inputs...")
    Twc_fp16 = Twc.half()
    Xs_fp16 = Xs.half()
    Cs_fp16 = Cs.half()
    Q_fp16 = Q.half()
    
    try:
        dx_fp16 = mast3r_slam_backends.gauss_newton_points(
            Twc_fp16, Xs_fp16, Cs_fp16, ii, jj, idx_ii2jj, valid_match, Q_fp16,
            sigma_point=1.0, C_thresh=0.1, Q_thresh=0.1, max_iter=5, delta_thresh=1e-6
        )
        print(f"   ✓ FP16 works")
    except Exception as e:
        print(f"   ✗ FP16 failed: {e}")
        if "mat1 and mat2 must have the same dtype" in str(e) or "Float" in str(e) and "Half" in str(e):
            print(f"   → Backend does NOT support FP16")
    
    print()


def main():
    """Run all tests"""
    test_iter_proj()
    test_refine_matches()
    test_conversion_overhead()
    test_gauss_newton()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBased on the CUDA source code analysis:")
    print("✓ refine_matches: SUPPORTS FP16 (uses AT_DISPATCH_FLOATING_TYPES_AND_HALF)")
    print("✗ iter_proj: FP32 ONLY (hardcoded torch::PackedTensorAccessor32<float,...")
    print("✗ gauss_newton_*: FP32 ONLY (hardcoded float types)")
    print("\nTo add FP16 support, modify the .cu files to use templates.")
    print("See test_mast3r_backend/HOWTO_ADD_FP16.md for detailed instructions.")
    print()


if __name__ == "__main__":
    main()
