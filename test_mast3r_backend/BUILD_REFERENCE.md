# Quick Reference: Backend Source Files

## File Locations

```
mast3r_slam/backend/
├── include/
│   └── gn.h                     # Function declarations (102 lines)
└── src/
    ├── gn.cpp                   # Python bindings (133 lines)
    ├── gn_kernels.cu           # Bundle adjustment kernels (2847 lines)
    └── matching_kernels.cu     # Matching kernels (351 lines)

Compiled output:
mast3r_slam/mast3r_slam_backends.cpython-312-aarch64-linux-gnu.so
```

## Functions by File

### `matching_kernels.cu`
- `refine_matches_kernel<scalar_t>` - **✓ FP16 supported**
- `iter_proj_kernel` - **✗ FP32 only** (needs modification)

### `gn_kernels.cu`
All **✗ FP32 only**:
- `point_align_kernel` - Point-to-point bundle adjustment
- `ray_align_kernel` - Ray-to-ray bundle adjustment  
- `calib_proj_kernel` - Calibrated projection bundle adjustment
- `pose_retr_kernel` - Pose retraction on Sim(3) manifold
- Device functions: `quat_comp`, `actSim3`, `expSim3`, etc.

## Rebuild Commands

### Option 1: Using setup.py
```bash
cd /home/chongxi/Work/scene_reconstruction/MASt3R-SLAM
python setup.py build_ext --inplace
```

### Option 2: Using pip (editable install)
```bash
cd /home/chongxi/Work/scene_reconstruction/MASt3R-SLAM
pip install -e . --no-build-isolation
```

### Option 3: Clean rebuild
```bash
cd /home/chongxi/Work/scene_reconstruction/MASt3R-SLAM
rm -rf build/ mast3r_slam/*.so
python setup.py build_ext --inplace
```

## Expected Compile Time

On AGX Thor:
- Initial build: ~5-10 minutes
- Incremental rebuild: ~30-60 seconds

The longest file is `gn_kernels.cu` (2847 lines of complex CUDA code).

## Compile Flags

Check `setup.py` for compilation settings. Should include:

```python
extra_compile_args={
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-arch=sm_87',  # AGX Thor (Ampere)
        '--expt-relaxed-constexpr',
        '-std=c++17'
    ]
}
```

## Verifying Compilation

After rebuild:

```bash
# Check the .so file exists
ls -lh mast3r_slam/*.so

# Test import
python -c "import mast3r_slam_backends; print('✓ Import successful')"

# Run test suite
cd test_mast3r_backend
python test_backend_fp16.py
```

## Common Compilation Errors

### Error: "nvcc not found"
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: "torch/extension.h: No such file"
```bash
# Reinstall PyTorch
pip install --force-reinstall torch
```

### Error: "unsupported GPU architecture"
```bash
# Check your CUDA arch
python -c "import torch; print(torch.cuda.get_device_capability())"
# Update -arch=sm_XX in setup.py to match
```

### Error: "undefined symbol" at runtime
```bash
# Check dependencies
ldd mast3r_slam/mast3r_slam_backends.*.so

# Rebuild clean
rm -rf build/ mast3r_slam/*.so
python setup.py build_ext --inplace
```

## Testing Changes

After modifying CUDA files:

1. **Rebuild**
   ```bash
   python setup.py build_ext --inplace
   ```

2. **Test specific function**
   ```bash
   python -c "
   import torch
   import mast3r_slam_backends
   X = torch.randn(1, 100, 3, device='cuda', dtype=torch.float16)
   Y = torch.randn(1, 100, 3, device='cuda', dtype=torch.float16)
   p_init = torch.rand(1, 100, 2, device='cuda', dtype=torch.float16) * 50
   result = mast3r_slam_backends.iter_proj(X, Y, p_init, 10, 1e-4, 0.1)
   print(f'✓ FP16 works! Result dtype: {result[0].dtype}')
   "
   ```

3. **Run full test suite**
   ```bash
   cd test_mast3r_backend
   python test_backend_fp16.py
   ```

## Development Workflow

1. Edit CUDA file (e.g., `mast3r_slam/backend/src/matching_kernels.cu`)
2. Rebuild: `python setup.py build_ext --inplace`
3. Test: `python test_mast3r_backend/test_backend_fp16.py`
4. Iterate

For faster iteration during development:
```bash
# Watch for changes and auto-rebuild (requires inotify-tools)
while inotifywait -e modify mast3r_slam/backend/src/*.cu; do
    python setup.py build_ext --inplace && echo "✓ Rebuilt"
done
```

## Debugging Tips

### Enable verbose compilation
```bash
python setup.py build_ext --inplace --verbose
```

### Check CUDA errors
Add to kernel code:
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
```

### Print from kernel
```cpp
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Debug: u=%f, v=%f\n", u, v);
}
```

### Use cuda-gdb
```bash
cuda-gdb --args python main.py --fp8
```

## Performance Profiling

### Using nsys (Nsight Systems)
```bash
nsys profile --trace=cuda,nvtx python main.py --fp8
nsys-ui report.qdrep  # View in GUI
```

### Using nvprof
```bash
nvprof --print-gpu-trace python -c "
import torch
import mast3r_slam_backends
# ... test code ...
"
```

### Kernel timing
Add timing inside Python:
```python
import torch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
result = mast3r_slam_backends.iter_proj(...)
end.record()
torch.cuda.synchronize()

print(f"Kernel time: {start.elapsed_time(end):.3f} ms")
```

## Resources

- PyTorch C++ Extension Guide: https://pytorch.org/tutorials/advanced/cpp_extension.html
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch Accessor API: https://pytorch.org/cppdocs/api/structat_1_1_tensor_accessor.html
- Eigen Sparse Solvers: https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
