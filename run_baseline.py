import os
import time
import json
import argparse
import torch
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, DynamicCache
from PIL import Image
import cv2

# --- Baseline Configuration ---
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
PROMPT_TEMPLATE = "[INST] <image>\nDescribe what is happening. [/INST]"

# THIS IS YOUR BASELINE'S "MEMORY"
# We'll allow 4 * 196 = 784 patches in the cache (a ~4-frame window)
MAX_CACHE_PATCHES = 784

def truncate_cache(past_key_values, max_patches):
    """
    This is the "dumb" sliding window logic.
    It chops the cache to keep only the 'max_patches' most recent.
    """
    if past_key_values is None:
        return None
    
    # Create a new cache with truncated values
    new_cache = DynamicCache()
    
    for layer_idx in range(len(past_key_values)):
        layer_k, layer_v = past_key_values[layer_idx]
        
        seq_len = layer_k.shape[2]
        
        if seq_len > max_patches:
            # Keep only the last 'max_patches' items
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
        # Get the sequence length from the first layer's key tensor
        layer_k, _ = past_key_values[0]
        return int(layer_k.shape[2])
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.environ.get("VIDEO_PATH", "data/sample_video.mp4"), help="Path to input video")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process")
    parser.add_argument("--max-cache-patches", type=int, default=MAX_CACHE_PATCHES, help="Max cache patches for sliding window")
    parser.add_argument("--out", type=str, default="results/baseline.json", help="Output results JSON path")
    args = parser.parse_args()

    video_path = args.video
    max_frames = args.max_frames
    max_cache_patches = args.max_cache_patches
    out_path = args.out

    # Ensure results dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Basic environment checks
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a machine with CUDA GPUs.")

    # Load model and processor
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    # Video Loop
    cap = cv2.VideoCapture(video_path)
    past_key_values = None
    frame_num = 0

    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=PROMPT_TEMPLATE, images=image, return_tensors="pt").to(model.device, torch.float16)

        start = time.time()
        with torch.no_grad():
            outputs = model(
                **inputs,
                past_key_values=past_key_values,
                use_cache=True
            )

            full_cache = outputs.past_key_values
            past_key_values = truncate_cache(full_cache, max_cache_patches)

        elapsed = time.time() - start
        cache_patches = total_cache_patches(past_key_values)

        results.append({
            "frame": frame_num,
            "cache_patches": cache_patches,
            "time_s": elapsed
        })

        print(f"Baseline Frame {frame_num} processed. cache_patches={cache_patches} time={elapsed:.3f}s")
        frame_num += 1
        if frame_num >= max_frames:
            break

    cap.release()

    # Save results
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Baseline test complete. Results saved to {out_path}")


if __name__ == '__main__':
    main()