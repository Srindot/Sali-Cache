import numpy as np

def get_patch_scores(
    mask: np.ndarray, 
    patch_size: int = 24, 
    aggregation_mode: str = "max"
) -> np.ndarray:
    """
    Efficiently aggregates pixel-level scores into patch-level scores.
    
    Args:
        mask (np.ndarray): A 2D (e.g., 336x336) map of scores.
        patch_size (int): The size of one patch (e.g., 24).
        aggregation_mode (str): 'max' or 'mean'
        
    Returns:
        np.ndarray: A 1D array of patch scores (e.g., 196 scores).
    """
    # Get dimensions
    H, W = mask.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Reshape the mask into (num_patches_h, patch_size, num_patches_w, patch_size)
    # This groups pixels by their patch
    reshaped_mask = mask.reshape(
        num_patches_h, 
        patch_size, 
        num_patches_w, 
        patch_size
    )
    
    # Swap axes to (num_patches_h, num_patches_w, patch_size, patch_size)
    # This puts all pixels for a patch [i, j] together
    patch_grouped_mask = reshaped_mask.transpose(0, 2, 1, 3)
    
    # Now, aggregate over the last two dimensions (the patch_size x patch_size)
    if aggregation_mode == "max":
        patch_scores = patch_grouped_mask.max(axis=(2, 3))
    elif aggregation_mode == "mean":
        patch_scores = patch_grouped_mask.mean(axis=(2, 3))
    else:
        raise ValueError("mode must be 'max' or 'mean'")
        
    # Finally, flatten from a 2D grid of scores (14x14) to a 1D list (196)
    return patch_scores.flatten()