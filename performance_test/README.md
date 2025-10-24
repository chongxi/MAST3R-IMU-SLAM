# Performance Testing Suite

Comprehensive benchmarks to decompose MASt3R-SLAM pipeline into testable components and identify bottlenecks.

## Quick Start

Run all tests:
```bash
cd performance_test
python run_all_tests.py
```

Run individual tests:
```bash
python test_01_model_loading.py
python test_02_frame_creation.py
python test_03_encoding.py
python test_04_decoder.py
python test_05_matching.py
python test_06_optimization.py
python test_07_full_pipeline.py
```

## Test Files

Each test isolates one component of the pipeline:

### 1. `test_01_model_loading.py`
- **What**: Model initialization overhead
- **Measures**: FP32 vs FP16 loading time
- **Key Finding**: FP16 is ~2x faster to load (one-time cost)

### 2. `test_02_frame_creation.py`
- **What**: Image preprocessing (resize, normalize, GPU transfer)
- **Measures**: Frame creation at different resolutions
- **Key Finding**: ~1-2ms per frame at 512x512

### 3. `test_03_encoding.py`
- **What**: ViT encoder (backbone feature extraction)
- **Measures**: FP32 vs FP16 vs FP8 encoding speed
- **Key Finding**: FP8 provides 4.23x speedup (3.85ms)
- **Note**: Runs ONCE per frame (keyframe features cached)

### 4. `test_04_decoder.py`
- **What**: Cross-attention decoder between frame pairs
- **Measures**: Decoder speed with/without FP8
- **Key Finding**: ~3ms per frame pair with FP8
- **Note**: Main bottleneck - runs for EVERY frame-keyframe pair

### 5. `test_05_matching.py`
- **What**: Feature matching with CUDA backend
- **Measures**: iter_proj + refine_matches speed
- **Key Finding**: ~1-2ms per frame pair
- **Note**: Runs for EVERY frame-keyframe pair

### 6. `test_06_optimization.py`
- **What**: Bundle adjustment (Gauss-Newton solver)
- **Measures**: Backend optimization with different iteration counts
- **Key Finding**: ~10-20ms per optimization step
- **Note**: Runs asynchronously in backend thread (can overlap)

### 7. `test_07_full_pipeline.py`
- **What**: End-to-end frame processing
- **Measures**: Complete pipeline from image to matches
- **Key Finding**: Single pair ~5-8ms; scales linearly with keyframes
- **Note**: With 3 keyframes: ~20ms total (50 FPS)

## Performance Breakdown

### Per-Frame Cost (with 3 keyframes, FP8 enabled):

**One-time costs:**
- Frame creation: 2ms
- Encoding: 4ms

**Per-keyframe costs (×3):**
- Decoder: 3ms × 3 = 9ms
- Matching: 1.5ms × 3 = 4.5ms

**Total: ~20ms per frame (50 FPS max)**

Backend optimization runs asynchronously (~15ms, overlapped)

### Target vs Actual

- **Target**: 30 FPS (33.33ms per frame)
- **Current**: ~50 FPS (20ms per frame) ✓

## Bottleneck Identification

If pipeline feels slow, check:

1. **Keyframe count**: Should be 3-5 (each adds 4-5ms)
2. **GPU utilization**: Should be >90% during processing
3. **Backend thread**: Should run asynchronously without blocking
4. **Image resolution**: 512x512 recommended (1024x1024 is 4x slower)

## Optimization Recommendations

Based on benchmark results:

### Already Optimized ✓
- FP8 inference: 4.23x speedup achieved
- Async FP16→FP32 conversion: 5% improvement
- Keyframe feature caching: encoder runs once

### Potential Improvements

1. **Batch decoder processing** → 2x speedup
   - Process all keyframes in one decoder call
   - Reduces kernel launch overhead
   
2. **FP16 CUDA backend** → 1.5x speedup  
   - Add FP16 support to `iter_proj` and `gauss_newton_*`
   - See `test_mast3r_backend/HOWTO_ADD_FP16.md`
   
3. **Adaptive keyframe selection** → Variable speedup
   - Reduce keyframes when motion is slow
   - Increase when motion is fast
   
4. **CUDA streams** → Hide overhead
   - Overlap data transfer with computation
   - Parallel decoder calls for multiple keyframes

## Understanding the Results

### Normal Performance
```
Encoding:  ~4ms    (once per frame)
Decoder:   ~3ms    (per keyframe pair)
Matching:  ~1.5ms  (per keyframe pair)
Total:     ~20ms   (3 keyframes)
```

### Slow Performance
If you see times >2x the above:
- Check GPU temperature (thermal throttling)
- Check other processes using GPU
- Check power mode (ensure maximum performance)
- Verify FP8 is actually enabled (not falling back to FP32)

### Fast Performance  
If you see times <0.5x the above:
- Lucky you! Probably newer GPU than AGX Thor
- Results should still show relative differences

## Technical Details

### Warmup Phase
All tests include warmup (10-50 iterations) to:
- Load CUDA kernels into cache
- Stabilize GPU clocks
- Amortize first-run overhead

### Synchronization
Proper CUDA synchronization around timing:
```python
torch.cuda.synchronize()
start = time.perf_counter()
# operation
torch.cuda.synchronize()
end = time.perf_counter()
```

### Statistics
All results report mean ± std over 30-100 runs:
- Mean: Average time per operation
- Std: Variation (low std = consistent performance)

## Relation to main.py

The full SLAM pipeline (main.py) includes additional stages not tested here:
- Keyframe selection logic
- Retrieval database queries
- Visualization rendering
- File I/O (saving results)

These add ~5-10ms overhead, so:
- **Benchmark suite**: 20ms per frame
- **Actual main.py**: 25-30ms per frame
- **Reported FPS**: 30-40 FPS

## Next Steps

After running benchmarks:

1. **Identify your bottleneck**:
   ```bash
   python run_all_tests.py > results.txt
   grep "ms" results.txt
   ```

2. **Check if decoder is bottleneck**:
   - If decoder time × keyframe_count > 10ms → batch processing needed
   
3. **Check if backend is bottleneck**:
   - If optimization time > 15ms → add FP16 support to kernels

4. **Profile main.py**:
   ```bash
   python -m cProfile -o profile.stats main.py --config config/base.yaml
   python -m pstats profile.stats
   ```

5. **GPU profiling**:
   ```bash
   nsys profile -o mast3r_slam python main.py --config config/base.yaml --fp8
   ```

## Related Documentation

- **Backend optimization**: `../test_mast3r_backend/README.md`
- **Adding FP16 to CUDA kernels**: `../test_mast3r_backend/HOWTO_ADD_FP16.md`
- **Main SLAM pipeline**: `../README.md`

## Contact

For questions about benchmarks or optimization:
- Check test_mast3r_backend/INDEX.md for quick start guide
- Check test_mast3r_backend/FAQ.md for common issues
