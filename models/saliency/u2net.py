import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image

# TODO:
# You must DOWNLOAD the 'u2net.py' file from the official U²-Net repo
# and place it in this folder.
# from .u2net_full import U2NETP # <-- Import the actual model class

# --- Placeholder class until you download the real one ---
class U2NETP_Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.DUMMY_LAYER = nn.Linear(1, 1)
        print("WARNING: Using U2NETP_Placeholder. Download the real model!")
        
    def forward(self, x):
        # Return a dummy 320x320 mask
        return torch.rand(1, 1, 320, 320)
# --------------------------------------------------------


class U2NetpWrapper(nn.Module):
    """
    A simple wrapper for U²-Netp to load weights and handle
    the pre/post-processing needed for our project.
    """
    def __init__(self, weights_path):
        super().__init__()
        
        # TODO: Replace 'U2NETP_Placeholder' with the real 'U2NETP' class
        self.model = U2NETP_Placeholder()
        
        # TODO: Load the real weights
        # self.model.load_state_dict(torch.load(weights_path))
        print(f"TODO: Would load U2Net weights from {weights_path}")
        
        # U²-Net expects a 320x320 input
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @torch.no_grad()
    def forward(self, pil_image: Image.Image) -> np.ndarray:
        """
        Takes a PIL Image, returns a 336x336 numpy saliency mask.
        """
        image_tensor = self.transform(pil_image).unsqueeze(0)
        image_tensor = image_tensor.to(next(self.model.parameters()).device)
        
        # The model's output is a "d0" tensor
        d0 = self.model(image_tensor)
        
        # --- Post-processing ---
        mask = d0[:, 0, :, :]
        # Normalize to 0-1 range
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        
        # Convert to numpy array and resize to 336x336
        mask_np = mask.cpu().numpy()[0] # -> (320, 320)
        mask_resized = cv2.resize(mask_np, (336, 336), interpolation=cv2.INTER_LINEAR)
        
        return mask_resized # Returns a (336, 336) np.array