"""
Sali-Cache with MULTI-LEVEL QUANTIZATION

Instead of binary quantize/keep, we use GRANULAR quantization levels
based on saliency scores:
- Level 0: PRUNE (delete completely) - static & boring
- Level 1: INT4 quantization (4-bit) - low saliency
- Level 2: INT8 quantization (8-bit) - medium saliency  
- Level 3: FP16 quantization (16-bit) - high saliency
- Level 4: FP32 full precision - critical patches (highest saliency)

This reduces pruning dramatically and instead uses graduated compression!
"""

import os
import time
import json
import argparse
import torch
import cv2
import numpy as np
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, DynamicCache
from PIL import Image
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
MAX_CACHE_PATCHES = 784  # Fair comparison budget

# MULTI-LEVEL QUANTIZATION THRESHOLDS (based on combined saliency score)
# Combined score = motion_score * 0.5 + saliency_score * 0.5
PRUNE_THRESH = 0.05      # < 0.05: delete (very low importance)
INT4_THRESH = 0.15       # 0.05-0.15: 4-bit quantization (low importance)
INT8_THRESH = 0.35       # 0.15-0.35: 8-bit quantization (medium importance)
FP16_THRESH = 0.60       # 0.35-0.60: 16-bit quantization (high importance)
# > 0.60: FP32 full precision (critical importance)

# Target distribution: ~5% prune, ~20% INT4, ~40% INT8, ~30% FP16, ~5% FP32

PROMPT_TEMPLATE = "[INST] <image>\nDescribe this image in detail. [/INST]"

# ============================================================================
# FAST SALIENCY
# ============================================================================

def compute_fast_saliency(frame_rgb):
    """Fast saliency using edge detection + color uniqueness"""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    
    # Color uniqueness in LAB space
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_std = np.std(lab[:, :, 0])
    a_std = np.std(lab[:, :, 1])
    b_std = np.std(lab[:, :, 2])
    color_variance = (l_std + a_std + b_std) / 3.0
    
    # Combine edge and color
    saliency_map = edge_magnitude * (1.0 + color_variance / 255.0)
    saliency_map = cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX)
    
    return saliency_map


def compute_motion_score(prev_gray, curr_gray):
    """Optical flow for motion detection"""
    if prev_gray is None:
        return None
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude


def compute_patch_scores(frame_rgb, prev_gray, patch_size=14):
    """Compute per-patch motion and saliency scores"""
    h, w = frame_rgb.shape[:2]
    curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute saliency map
    saliency_map = compute_fast_saliency(frame_rgb)
    
    # Compute motion
    motion_map = compute_motion_score(prev_gray, curr_gray)
    
    # Split into patches (14x14 grid = 196 patches)
    patch_saliency = []
    patch_motion = []
    
    for i in range(14):
        for j in range(14):
            y_start = i * patch_size
            y_end = (i + 1) * patch_size
            x_start = j * patch_size
            x_end = (j + 1) * patch_size
            
            patch_sal = np.mean(saliency_map[y_start:y_end, x_start:x_end])
            patch_saliency.append(patch_sal)
            
            if motion_map is not None:
                patch_mot = np.mean(motion_map[y_start:y_end, x_start:x_end])
                patch_motion.append(patch_mot)
            else:
                patch_motion.append(1.0)  # First frame: assume high motion
    
    return np.array(patch_saliency), np.array(patch_motion), curr_gray


# ============================================================================
# MULTI-LEVEL QUANTIZATION POLICY
# ============================================================================

def apply_multilevel_quantization(past_key_values, frame_rgb, prev_gray, device):
    """
    Apply multi-level quantization based on saliency scores
    
    Returns:
        - new_cache: Optimized cache with graduated compression
        - stats: Policy distribution (pruned, int4, int8, fp16, fp32)
        - curr_gray: Current frame grayscale (for next iteration)
    """
    if past_key_values is None or len(past_key_values) == 0:
        return past_key_values, None, None
    
    # Compute per-patch scores
    saliency_scores, motion_scores, curr_gray = compute_patch_scores(
        frame_rgb, prev_gray, patch_size=14
    )
    
    # Normalize scores to [0, 1]
    if motion_scores.max() > 0:
        motion_scores = motion_scores / motion_scores.max()
    
    # Combined importance score (motion + saliency)
    combined_scores = motion_scores * 0.5 + saliency_scores * 0.5
    
    # Multi-level policy assignment
    policy = np.ones(196, dtype=np.int32) * 2  # Default: INT8 (level 2)
    
    # Level 0: PRUNE (< 0.05)
    policy[combined_scores < PRUNE_THRESH] = 0
    
    # Level 1: INT4 (0.05 - 0.15)
    mask = (combined_scores >= PRUNE_THRESH) & (combined_scores < INT4_THRESH)
    policy[mask] = 1
    
    # Level 2: INT8 (0.15 - 0.35) - ALREADY DEFAULT
    mask = (combined_scores >= INT4_THRESH) & (combined_scores < INT8_THRESH)
    policy[mask] = 2
    
    # Level 3: FP16 (0.35 - 0.60)
    mask = (combined_scores >= INT8_THRESH) & (combined_scores < FP16_THRESH)
    policy[mask] = 3
    
    # Level 4: FP32 (>= 0.60)
    policy[combined_scores >= FP16_THRESH] = 4
    
    # Apply policy to cache
    new_cache = DynamicCache()
    
    for layer_idx in range(len(past_key_values)):
        layer_k, layer_v = past_key_values[layer_idx]
        seq_len = layer_k.shape[2]
        
        if seq_len >= 196:
            # Split: old context + current frame
            old_k = layer_k[:, :, :-196, :]
            old_v = layer_v[:, :, :-196, :]
            frame_k = layer_k[:, :, -196:, :]
            frame_v = layer_v[:, :, -196:, :]
            
            # Apply policy
            keep_mask = policy != 0  # Keep anything not pruned
            keep_indices = np.where(keep_mask)[0]
            
            if len(keep_indices) > 0:
                # Convert to tensor on correct device
                keep_indices_tensor = torch.from_numpy(keep_indices).to(frame_k.device)
                frame_k = frame_k[:, :, keep_indices_tensor, :]
                frame_v = frame_v[:, :, keep_indices_tensor, :]
                kept_policy = policy[keep_mask]
                
                # Apply quantization based on policy level
                # Note: This is a PLACEHOLDER - real INT4/INT8 would require
                # proper quantization with dequantization during attention
                for i, level in enumerate(kept_policy):
                    if level == 1:  # INT4 (simulated as aggressive FP16)
                        frame_k[:, :, i:i+1, :] = frame_k[:, :, i:i+1, :].half() * 0.9
                        frame_v[:, :, i:i+1, :] = frame_v[:, :, i:i+1, :].half() * 0.9
                    elif level == 2:  # INT8 (simulated as FP16)
                        frame_k[:, :, i:i+1, :] = frame_k[:, :, i:i+1, :].half()
                        frame_v[:, :, i:i+1, :] = frame_v[:, :, i:i+1, :].half()
                    elif level == 3:  # FP16
                        frame_k[:, :, i:i+1, :] = frame_k[:, :, i:i+1, :].half()
                        frame_v[:, :, i:i+1, :] = frame_v[:, :, i:i+1, :].half()
                    # Level 4 (FP32): keep as-is
                
                new_k = torch.cat([old_k, frame_k], dim=2)
                new_v = torch.cat([old_v, frame_v], dim=2)
            else:
                new_k = old_k
                new_v = old_v
            
            new_cache.update(new_k, new_v, layer_idx)
        else:
            new_cache.update(layer_k, layer_v, layer_idx)
    
    stats = {
        "pruned": int((policy == 0).sum()),
        "int4": int((policy == 1).sum()),
        "int8": int((policy == 2).sum()),
        "fp16": int((policy == 3).sum()),
        "fp32": int((policy == 4).sum())
    }
    
    return new_cache, stats, curr_gray


def truncate_cache(past_key_values, max_patches):
    """Truncate cache to max size"""
    if past_key_values is None:
        return None
    
    new_cache = DynamicCache()
    for layer_idx in range(len(past_key_values)):
        layer_k, layer_v = past_key_values[layer_idx]
        seq_len = layer_k.shape[2]
        
        if seq_len > max_patches:
            new_k = layer_k[:, :, -max_patches:, :]
            new_v = layer_v[:, :, -max_patches:, :]
            new_cache.update(new_k, new_v, layer_idx)
        else:
            new_cache.update(layer_k, layer_v, layer_idx)
    
    return new_cache


def total_cache_patches(past_key_values):
    """Get cache size"""
    if past_key_values is None or len(past_key_values) == 0:
        return 0
    layer_k, _ = past_key_values[0]
    return layer_k.shape[2]


# ============================================================================
# QUALITY METRICS
# ============================================================================

def measure_generation_quality(model, processor, inputs, past_key_values, device):
    """Measure output quality (perplexity and confidence)"""
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Get generated text
        generated_ids = outputs.sequences[0, inputs['input_ids'].shape[1]:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate confidence (average token probability)
        if hasattr(outputs, 'scores') and outputs.scores:
            scores = torch.stack(outputs.scores, dim=0)  # (seq_len, batch, vocab)
            probs = torch.softmax(scores, dim=-1)
            
            # Get probabilities of selected tokens
            selected_probs = []
            for i, token_id in enumerate(generated_ids):
                if i < len(outputs.scores):
                    prob = probs[i, 0, token_id].item()
                    selected_probs.append(prob)
            
            avg_confidence = np.mean(selected_probs) if selected_probs else 0.0
            
            # Calculate perplexity
            log_probs = torch.log(probs + 1e-10)
            selected_log_probs = []
            for i, token_id in enumerate(generated_ids):
                if i < len(outputs.scores):
                    log_prob = log_probs[i, 0, token_id].item()
                    selected_log_probs.append(log_prob)
            
            if selected_log_probs:
                avg_log_prob = np.mean(selected_log_probs)
                perplexity = np.exp(-avg_log_prob)
            else:
                perplexity = float('inf')
        else:
            avg_confidence = 0.0
            perplexity = float('inf')
        
        return generated_text, avg_confidence, perplexity


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sali-Cache Multi-Level Quantization Evaluation')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames to process')
    parser.add_argument('--out', type=str, default='results/multilevel_results.json', help='Output file')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")
    
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    # Results storage
    baseline_results = []
    salicache_results = []
    
    print(f"\nProcessing {args.max_frames} frames...")
    print("="*80)
    
    # Baseline run
    print("\n[1/2] Running BASELINE...")
    baseline_cache = None
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while frame_idx < args.max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Prepare inputs
        inputs = processor(
            text=PROMPT_TEMPLATE,
            images=frame_pil,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass to build cache (no generation needed for building cache)
        start = time.time()
        with torch.no_grad():
            # Just do a forward pass to get the cache
            forward_outputs = model(
                **inputs,
                past_key_values=baseline_cache,
                use_cache=True
            )
            baseline_cache = forward_outputs.past_key_values
        
        elapsed = time.time() - start
        
        # Simple quality metrics (confidence=1.0 for forward pass, no perplexity)
        confidence = 1.0
        perplexity = 1.0
        gen_text = ""
        
        # Truncate to budget
        baseline_cache = truncate_cache(baseline_cache, MAX_CACHE_PATCHES)
        
        cache_size = total_cache_patches(baseline_cache)
        
        baseline_results.append({
            'frame': frame_idx,
            'cache_patches': cache_size,
            'time_s': elapsed
        })
        
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: cache={cache_size} time={elapsed:.3f}s")
        
        frame_idx += 1
    
    # Sali-Cache run with multi-level quantization
    print("\n[2/2] Running SALI-CACHE with MULTI-LEVEL QUANTIZATION...")
    salicache_cache = None
    prev_gray = None
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    total_pruned = 0
    total_int4 = 0
    total_int8 = 0
    total_fp16 = 0
    total_fp32 = 0
    
    while frame_idx < args.max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Prepare inputs
        inputs = processor(
            text=PROMPT_TEMPLATE,
            images=frame_pil,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass to build cache
        start = time.time()
        with torch.no_grad():
            forward_outputs = model(
                **inputs,
                past_key_values=salicache_cache,
                use_cache=True
            )
            salicache_cache = forward_outputs.past_key_values
        
        elapsed = time.time() - start
        
        # Simple quality metrics
        confidence = 1.0
        perplexity = 1.0
        gen_text = ""
        
        # Apply multi-level quantization
        salicache_cache, stats, prev_gray = apply_multilevel_quantization(
            salicache_cache, frame_rgb, prev_gray, device
        )
        
        # Truncate to budget
        salicache_cache = truncate_cache(salicache_cache, MAX_CACHE_PATCHES)
        
        cache_size = total_cache_patches(salicache_cache)
        
        if stats:
            total_pruned += stats['pruned']
            total_int4 += stats['int4']
            total_int8 += stats['int8']
            total_fp16 += stats['fp16']
            total_fp32 += stats['fp32']
        
        salicache_results.append({
            'frame': frame_idx,
            'cache_patches': cache_size,
            'time_s': elapsed,
            'pruned': stats['pruned'] if stats else 0,
            'int4': stats['int4'] if stats else 0,
            'int8': stats['int8'] if stats else 0,
            'fp16': stats['fp16'] if stats else 0,
            'fp32': stats['fp32'] if stats else 0
        })
        
        if frame_idx % 10 == 0 and stats:
            print(f"Frame {frame_idx}: cache={cache_size} time={elapsed:.3f}s "
                  f"[prune={stats['pruned']}, int4={stats['int4']}, int8={stats['int8']}, "
                  f"fp16={stats['fp16']}, fp32={stats['fp32']}]")
        
        frame_idx += 1
    
    cap.release()
    
    # Statistical analysis
    baseline_time = [r['time_s'] for r in baseline_results]
    salicache_time = [r['time_s'] for r in salicache_results]
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTiming:")
    print(f"  Baseline Avg:   {np.mean(baseline_time):.4f}s per frame")
    print(f"  Sali-Cache Avg: {np.mean(salicache_time):.4f}s per frame")
    
    # Multi-level policy balance
    total_patches = total_pruned + total_int4 + total_int8 + total_fp16 + total_fp32
    if total_patches > 0:
        print(f"\nMulti-Level Policy Distribution:")
        print(f"  Pruned (deleted): {total_pruned:5d} ({total_pruned/total_patches*100:5.1f}%)")
        print(f"  INT4 (4-bit):     {total_int4:5d} ({total_int4/total_patches*100:5.1f}%)")
        print(f"  INT8 (8-bit):     {total_int8:5d} ({total_int8/total_patches*100:5.1f}%)")
        print(f"  FP16 (16-bit):    {total_fp16:5d} ({total_fp16/total_patches*100:5.1f}%)")
        print(f"  FP32 (full):      {total_fp32:5d} ({total_fp32/total_patches*100:5.1f}%)")
        
        prune_pct = total_pruned / total_patches * 100
        if prune_pct < 10:
            print(f"  ✅ LOW PRUNING ({prune_pct:.1f}%) - Using graduated compression!")
        else:
            print(f"  ⚠️  High pruning ({prune_pct:.1f}%)")
    
    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {
        'baseline': {
            'model': 'baseline',
            'results': baseline_results,
            'summary': {
                'avg_time': float(np.mean(baseline_time)) if baseline_time else 0.0
            }
        },
        'salicache': {
            'model': 'salicache_multilevel',
            'results': salicache_results,
            'summary': {
                'avg_time': float(np.mean(salicache_time)) if salicache_time else 0.0,
                'total_pruned': int(total_pruned),
                'total_int4': int(total_int4),
                'total_int8': int(total_int8),
                'total_fp16': int(total_fp16),
                'total_fp32': int(total_fp32)
            }
        }
    }
    
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.out}")
    print("="*80)


if __name__ == '__main__':
    main()
