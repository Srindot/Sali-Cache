import os
import time
import json
import argparse
import torch
import cv2
import numpy as np
from transformers import AutoProcessor, AutoConfig, DynamicCache
from PIL import Image

# Import your custom model class
from models.sali_cache_llava import SaliCacheLlava

# --- 1. Configuration ---
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
U2NET_WEIGHTS_PATH = "models/saliency/u2netp.pth"
MAX_CACHE_PATCHES = 784  # Same as baseline for fair comparison

# Your VQA prompt. This will "guide" the summarization.
PROMPT_TEMPLATE = "[INST] <image>\nDescribe what is happening. [/INST]"

# Saliency-based cache parameters
PRUNE_RATIO = 0.3  # Keep 70% of patches based on importance
MOTION_WEIGHT = 0.6
SALIENCY_WEIGHT = 0.4


def compute_motion_score(prev_gray, curr_gray):
    """Compute motion between frames using optical flow magnitude"""
    if prev_gray is None:
        return None
    
    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Compute magnitude
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude


def compute_saliency_score(frame_rgb):
    """Simple saliency estimation using color and edge information"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Compute gradients
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize
    edge_magnitude = (edge_magnitude - edge_magnitude.min()) / (edge_magnitude.max() - edge_magnitude.min() + 1e-8)
    
    return edge_magnitude


def prune_cache_by_importance(past_key_values, motion_map, saliency_map, prune_ratio, patch_size=24):
    """
    Prune KV cache based on motion and saliency importance scores.
    Keeps most important patches, prunes least important ones.
    """
    if past_key_values is None or motion_map is None:
        return past_key_values
    
    from transformers import DynamicCache
    
    # Get dimensions
    H, W = motion_map.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Compute patch-level importance scores
    patch_scores = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y_start, y_end = i * patch_size, (i + 1) * patch_size
            x_start, x_end = j * patch_size, (j + 1) * patch_size
            
            motion_score = motion_map[y_start:y_end, x_start:x_end].mean()
            saliency_score = saliency_map[y_start:y_end, x_start:x_end].mean()
            
            # Combined importance score
            importance = MOTION_WEIGHT * motion_score + SALIENCY_WEIGHT * saliency_score
            patch_scores.append(importance)
    
    patch_scores = np.array(patch_scores)
    
    # Determine which patches to keep
    num_keep = int(num_patches * (1 - prune_ratio))
    keep_indices = np.argsort(patch_scores)[-num_keep:]  # Keep top patches
    keep_indices = sorted(keep_indices)
    
    # Get current sequence length
    layer_k, _ = past_key_values[0]
    seq_len = layer_k.shape[2]
    
    # The last 'num_patches' tokens in the cache are the new image patches
    # We need to selectively prune them
    if seq_len >= num_patches:
        new_cache = DynamicCache()
        
        for layer_idx in range(len(past_key_values)):
            layer_k, layer_v = past_key_values[layer_idx]
            
            # Split into old cache and new patches
            old_seq_len = seq_len - num_patches
            old_k = layer_k[:, :, :old_seq_len, :]
            old_v = layer_v[:, :, :old_seq_len, :]
            
            new_k = layer_k[:, :, old_seq_len:, :]
            new_v = layer_v[:, :, old_seq_len:, :]
            
            # Keep only important patches from new_k and new_v
            pruned_k = new_k[:, :, keep_indices, :]
            pruned_v = new_v[:, :, keep_indices, :]
            
            # Concatenate old cache with pruned new patches
            combined_k = torch.cat([old_k, pruned_k], dim=2)
            combined_v = torch.cat([old_v, pruned_v], dim=2)
            
            new_cache.update(combined_k, combined_v, layer_idx)
        
        return new_cache
    else:
        return past_key_values


def truncate_cache(past_key_values, max_patches):
    """Truncate cache to max_patches (same as baseline)"""
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
    if past_key_values is None:
        return 0
    if len(past_key_values) > 0:
        layer_k, _ = past_key_values[0]
        return int(layer_k.shape[2])
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.environ.get("VIDEO_PATH", "data/sample_video.mp4"), help="Path to input video")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--out", type=str, default="results/salicache.json", help="Output results JSON path")
    parser.add_argument("--u2net-weights", type=str, default=U2NET_WEIGHTS_PATH, help="Path to U2Net weights")
    args = parser.parse_args()

    video_path = args.video
    max_frames = args.max_frames
    out_path = args.out
    u2net_weights = args.u2net_weights

    # Ensure results dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Basic environment checks
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a machine with CUDA GPUs.")

    # --- 2. Load Models and Processor ---
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

    print("Loading Sali-Cache LLaVA model (NOTE: Using placeholder saliency model)...")
    # For now, use the base model since SaliCacheLlava requires real U2Net weights
    # The optimization logic is TODO in the custom model
    from transformers import LlavaNextForConditionalGeneration
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()
    
    print("Model loaded successfully (NOTE: No cache optimization applied yet - U2Net and pruning logic are placeholders)")

    # --- 3. Video Processing Loop ---
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    past_key_values = None
    prev_gray = None
    
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image and grayscale for motion detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for processing (LLaVA 1.6 uses 336x336)
        frame_rgb_resized = cv2.resize(frame_rgb, (336, 336))
        curr_gray_resized = cv2.resize(curr_gray, (336, 336))
        
        image = Image.fromarray(frame_rgb_resized)

        # Process inputs (image + text)
        inputs = processor(text=PROMPT_TEMPLATE, images=image, return_tensors="pt").to(model.device, torch.float16)

        start = time.time()
        with torch.no_grad():
            # Call forward pass
            outputs = model(
                **inputs,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Get the full cache
            full_cache = outputs.past_key_values
            
            # Apply saliency-based pruning
            motion_map = compute_motion_score(prev_gray, curr_gray_resized)
            saliency_map = compute_saliency_score(frame_rgb_resized)
            
            if motion_map is not None:
                # Prune cache based on motion and saliency
                full_cache = prune_cache_by_importance(
                    full_cache, motion_map, saliency_map, PRUNE_RATIO
                )
            
            # Then apply sliding window truncation
            past_key_values = truncate_cache(full_cache, MAX_CACHE_PATCHES)
            
            # Update previous frame
            prev_gray = curr_gray_resized

        elapsed = time.time() - start
        cache_patches = total_cache_patches(past_key_values)

        results.append({
            "frame": frame_num,
            "cache_patches": cache_patches,
            "time_s": elapsed
        })

        print(f"Sali-Cache Frame {frame_num} processed. cache_patches={cache_patches} time={elapsed:.3f}s")
        
        frame_num += 1
        if frame_num >= max_frames:
            break

    cap.release()

    # Save results
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Sali-Cache experiment complete. Results saved to {out_path}")


if __name__ == '__main__':
    main()