import cv2
import numpy as np

def get_motion_map(prev_frame_gray: np.ndarray, current_frame_gray: np.ndarray) -> np.ndarray:
    """
    Calculates the dense optical flow (motion) between two frames.
    Returns a 2D map of motion magnitudes.
    """
    
    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_frame_gray,
        next=current_frame_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    # Calculate the magnitude of the 2D motion vectors
    # flow is (height, width, 2) -> (x_flow, y_flow)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize the magnitude map to 0-1 for easier thresholding
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
        
    return magnitude # (height, width)