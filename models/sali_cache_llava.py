import torch
import torch.nn as nn
from transformers import LlavaNextForConditionalGeneration, DynamicCache
from PIL import Image
import numpy as np
import cv2

# Import our helper files
from .saliency.u2net import U2NetpWrapper
from utils.video_processing import get_motion_map
from utils.patch_utils import get_patch_scores
from utils.quantization import quantize_tensor, dequantize_tensor

# --- Configuration ---
# These are "hyperparameters" you will tune
MOTION_THRESHOLD = 0.3   # Lower = prune more static patches (was 0.5)
SALIENCY_THRESHOLD = 0.6 # Higher = keep fewer as full precision (was 0.5)
PATCH_SIZE = 24          # For LLaVA 1.6
NUM_PATCHES_PER_DIM = 14 # 336 / 24 = 14

class SaliCacheLlava(LlavaNextForConditionalGeneration):
    """
    Our custom LLaVA model with a Saliency-Aware KV Cache.
    """
    def __init__(self, config, u2net_weights_path: str):
        super().__init__(config)
        
        print(f"Initializing Sali-Cache LLaVA. Loading U2Net from {u2net_weights_path}...")
        # 1. Load the Saliency Model
        self.saliency_model = U2NetpWrapper(u2net_weights_path)
        self.saliency_model.eval()

        # 2. Initialize Optical Flow state
        self.prev_frame_gray = None
        self.current_raw_image = None
        
        # 3. Track optimization stats
        self.last_policy = None
        self.last_pruned_count = 0
        self.last_quantized_count = 0
        self.last_kept_count = 0

    @torch.no_grad()
    def get_cache_policy(self, pixel_values):
        """
        The core logic to decide what to do with each patch.
        This runs our "helper" models.
        """
        # Convert tensor to PIL Image / NumPy array for processing
        # Note: pixel_values are normalized. We need to de-normalize for CV.
        # This is a complex step, for a real implementation.
        # For this skeleton, we assume 'pixel_values' is a processable image.
        
        # Prefer using the provided raw image if available
        if self.current_raw_image is not None:
            pil_img = self.current_raw_image
            img_np = np.array(pil_img.resize((336, 336)))
            current_frame_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            # Fallback random image (shouldn't happen in proper run)
            dummy_image_np = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
            current_frame_gray = cv2.cvtColor(dummy_image_np, cv2.COLOR_RGB2GRAY)

        # --- 1. Temporal Pruning (Optical Flow) ---
        if self.prev_frame_gray is None:
            motion_scores = torch.ones(NUM_PATCHES_PER_DIM**2, device=self.device)
        else:
            motion_map = get_motion_map(self.prev_frame_gray, current_frame_gray)
            motion_scores = get_patch_scores(motion_map, PATCH_SIZE, "mean")
            motion_scores = torch.tensor(motion_scores, device=self.device)

        # Update for next frame
        self.prev_frame_gray = current_frame_gray

        # --- 2. Spatial Saliency (U²-Net) ---
        # The saliency model expects a PIL Image
        if self.current_raw_image is not None:
            saliency_mask = self.saliency_model(self.current_raw_image)
        else:
            saliency_mask = np.zeros((336, 336))

        saliency_scores = get_patch_scores(saliency_mask, PATCH_SIZE, "max")
        saliency_scores = torch.tensor(saliency_scores, device=self.device)

        # --- 3. Create the Final Policy ---
        # Policy: 0=Prune, 1=Quantize, 2=Keep Full
        
        # Start by assuming we quantize everything
        policy = torch.ones(NUM_PATCHES_PER_DIM**2, device=self.device, dtype=torch.int)
        
        # If motion is low, prune it (set policy to 0)
        policy[motion_scores < MOTION_THRESHOLD] = 0
        
        # If motion is high AND saliency is high, keep at full precision (set to 2)
        is_salient = saliency_scores > SALIENCY_THRESHOLD
        is_moving = motion_scores >= MOTION_THRESHOLD
        policy[is_moving & is_salient] = 2
        
        # Store stats
        self.last_policy = policy
        self.last_pruned_count = int((policy == 0).sum())
        self.last_quantized_count = int((policy == 1).sum())
        self.last_kept_count = int((policy == 2).sum())

        return policy


    def apply_cache_optimization(self, past_key_values, policy):
        """
        Modify past_key_values according to policy per-patch.
        policy: tensor of length N_patches with values {0=prune,1=quantize,2=keep}
        """
        if past_key_values is None:
            return None

        num_patches = NUM_PATCHES_PER_DIM ** 2
        new_cache = DynamicCache()

        for layer_idx in range(len(past_key_values)):
            layer_k, layer_v = past_key_values[layer_idx]
            seq_len = layer_k.shape[2]

            if seq_len < num_patches:
                # nothing to do; copy as-is
                new_cache.update(layer_k, layer_v, layer_idx)
                continue

            old_seq_len = seq_len - num_patches
            old_k = layer_k[:, :, :old_seq_len, :]
            old_v = layer_v[:, :, :old_seq_len, :]

            new_k = layer_k[:, :, old_seq_len:, :]
            new_v = layer_v[:, :, old_seq_len:, :]

            # Build list of kept indices
            keep_mask = (policy != 0).cpu().numpy()
            keep_indices = [i for i, k in enumerate(keep_mask) if k]

            if len(keep_indices) == 0:
                combined_k = old_k
                combined_v = old_v
            else:
                kept_k = new_k[:, :, keep_indices, :]
                kept_v = new_v[:, :, keep_indices, :]

                # Quantize patches that have policy == 1
                for qi, patch_pos in enumerate(keep_indices):
                    if policy[patch_pos] == 1:
                        # quantize the patch across heads/batch/dim
                        kept_k[:, :, qi:qi+1, :] = quantize_tensor(kept_k[:, :, qi:qi+1, :])
                        kept_v[:, :, qi:qi+1, :] = quantize_tensor(kept_v[:, :, qi:qi+1, :])

                combined_k = torch.cat([old_k, kept_k], dim=2)
                combined_v = torch.cat([old_v, kept_v], dim=2)

            new_cache.update(combined_k, combined_v, layer_idx)

        return new_cache

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        past_key_values: tuple = None,
        **kwargs,
    ):
        """
        This is the main function we override.
        """
        
        # --- 1. Get the Optimization Policy ---
        # We only run this logic if it's a vision frame
        # (i.e., pixel_values are provided)
        cache_policy = None
        # Accept raw PIL Image for better saliency/motion calculation
        self.current_raw_image = kwargs.pop("raw_image", None)
        if pixel_values is not None:
            cache_policy = self.get_cache_policy(pixel_values)

        # --- 2. Run the Original LLaVA 'forward' Pass ---
        # This will generate the *full, un-optimized* new KV cache
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            **kwargs,
        )

        # --- 3. Apply Our Custom Cache Logic (Post-Processing) ---
        # Now we modify the 'outputs.past_key_values'
        # This is where the magic happens
        
        if cache_policy is not None and outputs.past_key_values is not None:
            optimized_cache = self.apply_cache_optimization(outputs.past_key_values, cache_policy)
            outputs.past_key_values = optimized_cache

        # Clear raw image pointer
        self.current_raw_image = None

        return outputs