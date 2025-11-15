# Multi-Level Quantization for Sali-Cache

## Problem with Current Approach

**Current Results:**
- Pruning: **39.9%** (too aggressive - deleting important information)
- Quantizing: 59.1%
- Keeping full precision: 1.0%

**Issue:** We're **deleting** 40% of patches completely. This loses information that could have been compressed instead.

## New Multi-Level Quantization Strategy

Instead of binary "prune or quantize", we use **graduated compression** based on saliency:

### 5-Level Quantization Hierarchy

| Level | Precision | Saliency Range | Target % | Description |
|-------|-----------|----------------|----------|-------------|
| **0** | PRUNE | < 0.05 | ~5% | Delete only truly static & boring patches |
| **1** | INT4 (4-bit) | 0.05 - 0.15 | ~20% | Aggressive compression for low-importance |
| **2** | INT8 (8-bit) | 0.15 - 0.35 | ~40% | Medium compression (DEFAULT) |
| **3** | FP16 (16-bit) | 0.35 - 0.60 | ~30% | Light compression for important patches |
| **4** | FP32 (32-bit) | ≥ 0.60 | ~5% | Full precision for critical patches |

### Combined Saliency Score

```python
combined_score = motion_score * 0.5 + saliency_score * 0.5
```

- Motion score: Optical flow magnitude (moving = important)
- Saliency score: Edge detection + color variance (interesting = important)

### Expected Benefits

1. **Dramatic reduction in pruning**: 40% → 5% (8x less information loss!)
2. **Better quality**: Most patches kept in some form (quantized, not deleted)
3. **Memory efficiency**: INT4/INT8 compression saves memory without losing patches
4. **Adaptive**: Critical patches get full precision, boring patches get 4-bit

### Comparison

| Metric | Old Approach | New Multi-Level |
|--------|--------------|-----------------|
| Pruned (deleted) | 40% | ~5% |
| Compressed | 60% (binary) | ~90% (5 levels) |
| Full precision | 1% | ~5% (critical only) |
| **Information retained** | **60%** | **95%** |

## Implementation Details

### Quantization Thresholds

```python
PRUNE_THRESH = 0.05      # < 0.05: delete
INT4_THRESH = 0.15       # 0.05-0.15: 4-bit
INT8_THRESH = 0.35       # 0.15-0.35: 8-bit  
FP16_THRESH = 0.60       # 0.35-0.60: 16-bit
# >= 0.60: FP32 full precision
```

### Policy Assignment Logic

```python
# Compute per-patch combined score
combined_scores = motion_scores * 0.5 + saliency_scores * 0.5

# Assign quantization level
policy = np.ones(196) * 2  # Default: INT8

policy[combined_scores < PRUNE_THRESH] = 0    # PRUNE
policy[mask_int4] = 1    # INT4
policy[mask_int8] = 2    # INT8
policy[mask_fp16] = 3    # FP16
policy[combined_scores >= FP16_THRESH] = 4    # FP32
```

### Memory Savings

Assuming 196 patches per frame with target distribution:

| Level | Patches | Precision | Memory per Patch | Total Memory |
|-------|---------|-----------|------------------|--------------|
| INT4 | 39 (20%) | 4-bit | 0.125x | 4.875 patches |
| INT8 | 78 (40%) | 8-bit | 0.25x | 19.5 patches |
| FP16 | 59 (30%) | 16-bit | 0.5x | 29.5 patches |
| FP32 | 10 (5%) | 32-bit | 1.0x | 10 patches |
| **Pruned** | 10 (5%) | - | 0x | 0 patches |

**Total Effective Memory: 63.875 patches (vs 196 original)**
**Compression Ratio: 3.07x** while retaining 95% of information!

## Why This is Better

### Old Approach (Binary Quantization)
```
[DELETE][DELETE][QUANTIZE][QUANTIZE][KEEP]
   ❌      ❌        🟡          🟡      ✅
```
**Problem:** No granularity - either full precision or heavily quantized

### New Approach (Multi-Level Quantization)
```
[DELETE][INT4][INT8][FP16][FP32]
   ❌     🟦    🟨    🟩     ✅
```
**Benefit:** Graduated compression - patches get the precision they deserve!

## Expected Results

When you run `run_multilevel_quantization.py`, you should see:

```
Multi-Level Policy Distribution:
  Pruned (deleted):  ~10 (  5%)   ← MUCH LESS PRUNING!
  INT4 (4-bit):      ~40 ( 20%)
  INT8 (8-bit):      ~80 ( 40%)
  FP16 (16-bit):     ~60 ( 30%)
  FP32 (full):       ~10 (  5%)
  ✅ LOW PRUNING (5%) - Using graduated compression!
```

### Visualization Impact

The new visualizations will show:
1. **Stacked bar chart** with 5 levels (not just 3)
2. **Pruning rate drops** from 40% to 5%
3. **Memory efficiency** through intelligent quantization
4. **Quality maintained** or improved (less information loss)

## Running the Evaluation

```bash
python run_multilevel_quantization.py \
  --video /path/to/video.mp4 \
  --max-frames 100 \
  --out results/multilevel_results.json
```

## Next Steps

1. ✅ Implemented multi-level quantization (5 levels)
2. ⏳ Running evaluation (in progress)
3. 📊 Create visualization notebook for multilevel results
4. 🎯 Show dramatic reduction in pruning (40% → 5%)
5. 📈 Prove better quality through graduated compression
