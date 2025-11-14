"""
Quick test to verify the cache pruning logic works correctly.
This doesn't require a real video or model - just tests the pruning function.
"""
import torch
import numpy as np
from transformers import DynamicCache

# Import the pruning function from run_experiment
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Copy the relevant functions here for testing
MOTION_WEIGHT = 0.6
SALIENCY_WEIGHT = 0.4

def prune_cache_by_importance(past_key_values, motion_map, saliency_map, prune_ratio, patch_size=24):
    """
    Prune KV cache based on motion and saliency importance scores.
    Keeps most important patches, prunes least important ones.
    """
    if past_key_values is None or motion_map is None:
        return past_key_values
    
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


def test_pruning():
    """Test that pruning actually reduces cache size"""
    print("Testing cache pruning logic...")
    
    # Create fake cache (2 layers, batch=1, heads=8, seq=196+196=392, dim=64)
    # First 196 are old tokens, last 196 are new image patches (14x14 grid)
    fake_cache = DynamicCache()
    num_layers = 2
    batch = 1
    heads = 8
    old_seq = 196
    new_seq = 196
    total_seq = old_seq + new_seq
    dim = 64
    
    for layer_idx in range(num_layers):
        k = torch.randn(batch, heads, total_seq, dim)
        v = torch.randn(batch, heads, total_seq, dim)
        fake_cache.update(k, v, layer_idx)
    
    # Create fake motion and saliency maps (336x336 to match 14x14 patches with 24x24 patch size)
    motion_map = np.random.rand(336, 336)
    saliency_map = np.random.rand(336, 336)
    
    # Make some patches more important (top-left corner has high importance)
    motion_map[:100, :100] = 0.9
    saliency_map[:100, :100] = 0.9
    
    print(f"Original cache size: {total_seq} tokens")
    
    # Apply pruning (30% pruning = keep 70%)
    prune_ratio = 0.3
    pruned_cache = prune_cache_by_importance(fake_cache, motion_map, saliency_map, prune_ratio)
    
    # Check new size
    k_pruned, _ = pruned_cache[0]
    new_size = k_pruned.shape[2]
    
    expected_new_patches = int(new_seq * (1 - prune_ratio))
    expected_total = old_seq + expected_new_patches
    
    print(f"Pruned cache size: {new_size} tokens")
    print(f"Expected size: {expected_total} tokens")
    print(f"Reduction: {total_seq - new_size} tokens ({100*(total_seq - new_size)/total_seq:.1f}%)")
    
    if new_size == expected_total:
        print("✅ Pruning works correctly!")
        return True
    else:
        print(f"❌ Pruning failed! Expected {expected_total}, got {new_size}")
        return False


if __name__ == "__main__":
    success = test_pruning()
    sys.exit(0 if success else 1)
