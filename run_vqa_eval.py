"""
MSRVTT-QA Evaluation Script

This script evaluates both baseline and Sali-Cache models on the MSRVTT-QA dataset.
The goal is to measure **accuracy** with the same memory budget (MAX_CACHE_PATCHES).

Key Hypothesis:
Given the SAME cache size limit, Sali-Cache should achieve HIGHER accuracy
because it intelligently curates which patches to keep/quantize/prune.
"""

import os
import time
import json
import argparse
import csv
import torch
import cv2
import numpy as np
from pathlib import Path
from transformers import LlavaNextForConditionalGeneration, AutoProcessor, DynamicCache
from PIL import Image
from difflib import SequenceMatcher

# Import Sali-Cache components
from models.sali_cache_llava import SaliCacheLlava
from models.saliency.u2net import U2NetpWrapper
from utils.patch_utils import get_patch_scores

# --- Configuration ---
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DATASET_ROOT = "/ssd_scratch/vishnu/datasets/valerytamrazov/msrvttqa/versions/2"
VIDEO_DIR = os.path.join(DATASET_ROOT, "archive/TrainValVideo")
VAL_CSV = os.path.join(DATASET_ROOT, "val.csv")

# THIS IS THE KEY: Both models use the SAME memory budget
MAX_CACHE_PATCHES = 784  # ~4 frames worth of patches (196 patches per frame)

# Sampling parameters
MAX_FRAMES_PER_VIDEO = 8  # Sample 8 frames from each video
FRAME_SAMPLE_RATE = 5     # Sample every 5th frame

# Sali-Cache parameters
MOTION_THRESHOLD = 0.3
SALIENCY_THRESHOLD = 0.6


def truncate_cache(past_key_values, max_patches):
    """Sliding window cache truncation (used by both models)"""
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
    """Get total cache size"""
    if past_key_values is None:
        return 0
    if len(past_key_values) > 0:
        layer_k, _ = past_key_values[0]
        return layer_k.shape[2]
    return 0


def similarity_ratio(pred, gt):
    """Calculate similarity between predicted and ground truth answers"""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    return SequenceMatcher(None, pred, gt).ratio()


def extract_answer(response_text):
    """Extract the answer from the model's response"""
    # The model outputs: "[INST] <image>\n{question} [/INST] {answer}"
    # We want to extract the answer part
    if "[/INST]" in response_text:
        answer = response_text.split("[/INST]")[-1].strip()
    else:
        answer = response_text.strip()
    
    # Take only the first sentence or first few words
    answer = answer.split('.')[0].split('\n')[0].strip()
    
    return answer


def compute_motion_score(prev_gray, curr_gray):
    """Compute motion score using optical flow"""
    if prev_gray is None:
        return None
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return magnitude


def compute_saliency_score(frame_rgb, saliency_model):
    """Compute saliency score using U²-Net"""
    # Convert numpy array to PIL Image
    pil_img = Image.fromarray(frame_rgb)
    
    # Get saliency map from model
    saliency_map = saliency_model(pil_img)
    
    return saliency_map


def apply_sali_cache_optimization(model, past_key_values, raw_image, prev_gray):
    """
    Apply Sali-Cache optimization: motion-based pruning + saliency-based quantization
    """
    if past_key_values is None or total_cache_patches(past_key_values) == 0:
        return past_key_values, None
    
    # Convert image to grayscale for motion detection
    if isinstance(raw_image, Image.Image):
        frame_rgb = np.array(raw_image)
    else:
        frame_rgb = raw_image
    
    curr_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.resize(curr_gray, (336, 336))
    
    # Compute motion scores
    motion_map = compute_motion_score(prev_gray, curr_gray)
    if motion_map is None:
        return past_key_values, curr_gray  # First frame, no pruning
    
    motion_map = cv2.resize(motion_map, (336, 336))
    motion_scores = get_patch_scores(motion_map, patch_size=24, aggregation_mode='max')
    
    # Compute saliency scores
    saliency_map = compute_saliency_score(frame_rgb, model.saliency_model)
    saliency_scores = get_patch_scores(saliency_map, patch_size=24, aggregation_mode='mean')
    
    # Create policy: 0=prune, 1=quantize, 2=keep
    policy = np.ones(196, dtype=np.int32)  # Default: quantize
    
    # Low motion → prune (these patches are redundant)
    policy[motion_scores < MOTION_THRESHOLD] = 0
    
    # High motion + high saliency → keep at full precision
    policy[(motion_scores >= MOTION_THRESHOLD) & (saliency_scores > SALIENCY_THRESHOLD)] = 2
    
    # Apply pruning/quantization based on policy
    new_cache = DynamicCache()
    
    for layer_idx in range(len(past_key_values)):
        layer_k, layer_v = past_key_values[layer_idx]
        batch_size, num_heads, seq_len, head_dim = layer_k.shape
        
        # The last 196 tokens correspond to the current frame's patches
        if seq_len >= 196:
            # Split into [old context] + [current frame patches]
            old_k = layer_k[:, :, :-196, :]
            old_v = layer_v[:, :, :-196, :]
            frame_k = layer_k[:, :, -196:, :]
            frame_v = layer_v[:, :, -196:, :]
            
            # Apply policy to current frame patches
            keep_mask = policy != 0  # Keep anything that's not pruned
            keep_indices = np.where(keep_mask)[0]
            
            if len(keep_indices) > 0:
                frame_k = frame_k[:, :, keep_indices, :]
                frame_v = frame_v[:, :, keep_indices, :]
                
                # Quantize patches where policy=1 (optional: for now just keep them)
                # TODO: Implement real quantization here
                
                # Concatenate old context with optimized frame
                new_k = torch.cat([old_k, frame_k], dim=2)
                new_v = torch.cat([old_v, frame_v], dim=2)
            else:
                # All pruned, keep only old context
                new_k = old_k
                new_v = old_v
            
            new_cache.update(new_k, new_v, layer_idx)
        else:
            # Not enough tokens yet, keep as is
            new_cache.update(layer_k, layer_v, layer_idx)
    
    return new_cache, curr_gray


def load_validation_data(max_samples=None):
    """Load MSRVTT-QA validation data"""
    qa_pairs = []
    
    with open(VAL_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qa_pairs.append({
                'question': row['question'],
                'answer': row['answer'],
                'video_id': row['video_id'],
                'id': row['id']
            })
    
    if max_samples:
        qa_pairs = qa_pairs[:max_samples]
    
    return qa_pairs


def sample_frames(video_path, max_frames=8, sample_rate=5):
    """Sample frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (336, 336))
            frames.append(frame_rgb)
        
        frame_idx += 1
    
    cap.release()
    return frames


def evaluate_baseline(qa_pairs, device):
    """Evaluate baseline sliding-window cache model with multi-frame encoding"""
    print("\n" + "="*80)
    print("EVALUATING BASELINE MODEL")
    print("="*80)
    
    # Load model
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    
    results = []
    correct = 0
    total = 0
    
    for idx, qa in enumerate(qa_pairs):
        video_path = os.path.join(VIDEO_DIR, f"video{qa['video_id']}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Skipping {qa['video_id']}: video not found")
            continue
        
        # Sample frames
        frames = sample_frames(video_path, MAX_FRAMES_PER_VIDEO, FRAME_SAMPLE_RATE)
        if len(frames) == 0:
            print(f"Skipping {qa['video_id']}: no frames extracted")
            continue
        
        start_time = time.time()
        past_key_values = None
        
        # STEP 1: Encode all video frames to build up cache
        for frame_idx, frame in enumerate(frames):
            pil_img = Image.fromarray(frame)
            
            # Encode frame without generating text
            inputs = processor(
                text="[INST] <image>\nWhat is shown? [/INST]",
                images=pil_img,
                return_tensors="pt"
            ).to(device, torch.float16)
            
            with torch.no_grad():
                # Forward pass to build cache
                outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Apply sliding window truncation (baseline strategy)
                past_key_values = truncate_cache(past_key_values, MAX_CACHE_PATCHES)
        
        # STEP 2: Answer question with last frame + accumulated cache
        last_frame = Image.fromarray(frames[-1])
        question_prompt = f"[INST] <image>\n{qa['question']} Answer briefly. [/INST]"
        inputs = processor(
            text=question_prompt,
            images=last_frame,
            return_tensors="pt"
        ).to(device, torch.float16)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True
            )
        
        # Get answer
        response = processor.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        
        elapsed = time.time() - start_time
        
        # Check if answer is correct
        is_correct = similarity_ratio(predicted_answer, qa['answer']) > 0.5
        if is_correct:
            correct += 1
        total += 1
        
        final_cache_size = total_cache_patches(past_key_values)
        
        results.append({
            'video_id': qa['video_id'],
            'question': qa['question'],
            'ground_truth': qa['answer'],
            'prediction': predicted_answer,
            'correct': is_correct,
            'time_s': elapsed,
            'cache_size': final_cache_size
        })
        
        print(f"[{idx+1}/{len(qa_pairs)}] Video {qa['video_id']} | "
              f"GT: '{qa['answer']}' | Pred: '{predicted_answer}' | "
              f"Correct: {is_correct} | Cache: {final_cache_size} | Time: {elapsed:.2f}s")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\nBASELINE RESULTS:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Cache Size: {MAX_CACHE_PATCHES} patches")
    
    return {
        'model': 'baseline',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'cache_size': MAX_CACHE_PATCHES,
        'results': results
    }


def evaluate_salicache(qa_pairs, device):
    """Evaluate Sali-Cache optimized model with smart cache curation"""
    print("\n" + "="*80)
    print("EVALUATING SALI-CACHE MODEL")
    print("="*80)
    
    # Load base model
    model = LlavaNextForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    
    # Attach Sali-Cache components
    model.saliency_model = U2NetpWrapper(weights_path="models/saliency/u2netp.pth")
    model.saliency_model.to(device)
    
    processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    
    results = []
    correct = 0
    total = 0
    total_pruned = 0
    
    for idx, qa in enumerate(qa_pairs):
        video_path = os.path.join(VIDEO_DIR, f"video{qa['video_id']}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Skipping {qa['video_id']}: video not found")
            continue
        
        # Sample frames
        frames = sample_frames(video_path, MAX_FRAMES_PER_VIDEO, FRAME_SAMPLE_RATE)
        if len(frames) == 0:
            print(f"Skipping {qa['video_id']}: no frames extracted")
            continue
        
        start_time = time.time()
        past_key_values = None
        prev_gray = None
        video_pruned = 0
        
        # STEP 1: Encode all video frames with Sali-Cache optimization
        for frame_idx, frame in enumerate(frames):
            pil_img = Image.fromarray(frame)
            
            # Encode frame
            inputs = processor(
                text="[INST] <image>\nWhat is shown? [/INST]",
                images=pil_img,
                return_tensors="pt"
            ).to(device, torch.float16)
            
            with torch.no_grad():
                # Forward pass to build cache
                outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Apply Sali-Cache intelligent optimization
                cache_before = total_cache_patches(past_key_values)
                past_key_values, prev_gray = apply_sali_cache_optimization(
                    model, past_key_values, frame, prev_gray
                )
                cache_after = total_cache_patches(past_key_values)
                
                pruned = cache_before - cache_after
                video_pruned += pruned
                
                # Then apply same sliding window as baseline for fair comparison
                past_key_values = truncate_cache(past_key_values, MAX_CACHE_PATCHES)
        
        # STEP 2: Answer question with last frame + optimized cache
        last_frame = Image.fromarray(frames[-1])
        question_prompt = f"[INST] <image>\n{qa['question']} Answer briefly. [/INST]"
        inputs = processor(
            text=question_prompt,
            images=last_frame,
            return_tensors="pt"
        ).to(device, torch.float16)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=10,
                do_sample=False,
                use_cache=True
            )
        
        # Get answer
        response = processor.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        
        elapsed = time.time() - start_time
        
        # Check if answer is correct
        is_correct = similarity_ratio(predicted_answer, qa['answer']) > 0.5
        if is_correct:
            correct += 1
        total += 1
        total_pruned += video_pruned
        
        final_cache_size = total_cache_patches(past_key_values)
        
        results.append({
            'video_id': qa['video_id'],
            'question': qa['question'],
            'ground_truth': qa['answer'],
            'prediction': predicted_answer,
            'correct': is_correct,
            'time_s': elapsed,
            'cache_size': final_cache_size,
            'pruned_patches': video_pruned
        })
        
        print(f"[{idx+1}/{len(qa_pairs)}] Video {qa['video_id']} | "
              f"GT: '{qa['answer']}' | Pred: '{predicted_answer}' | "
              f"Correct: {is_correct} | Cache: {final_cache_size} | Pruned: {video_pruned} | Time: {elapsed:.2f}s")
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\nSALI-CACHE RESULTS:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Cache Size: {MAX_CACHE_PATCHES} patches")
    
    return {
        'model': 'salicache',
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'cache_size': MAX_CACHE_PATCHES,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA on MSRVTT-QA")
    parser.add_argument('--model', type=str, choices=['baseline', 'salicache', 'both'],
                        default='both', help='Which model to evaluate')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of validation samples to use')
    parser.add_argument('--out', type=str, default='results/vqa_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load validation data
    qa_pairs = load_validation_data(max_samples=args.samples)
    print(f"Loaded {len(qa_pairs)} QA pairs from validation set")
    
    # Evaluate models
    all_results = {}
    
    if args.model in ['baseline', 'both']:
        baseline_results = evaluate_baseline(qa_pairs, device)
        all_results['baseline'] = baseline_results
    
    if args.model in ['salicache', 'both']:
        salicache_results = evaluate_salicache(qa_pairs, device)
        all_results['salicache'] = salicache_results
    
    # Save results
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {args.out}")
    print(f"{'='*80}")
    
    # Print comparison if both models were evaluated
    if 'baseline' in all_results and 'salicache' in all_results:
        print("\nCOMPARISON:")
        print(f"  Baseline Accuracy:   {all_results['baseline']['accuracy']:.2f}%")
        print(f"  Sali-Cache Accuracy: {all_results['salicache']['accuracy']:.2f}%")
        print(f"  Improvement:         +{all_results['salicache']['accuracy'] - all_results['baseline']['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
