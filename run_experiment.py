import torch
import cv2
from transformers import AutoProcessor
from PIL import Image

# Import your custom model class
from models.sali_cache_llava import SaliCacheLlava

# --- 1. Configuration ---
VIDEO_PATH = "data/sample_video.mp4"
LLAVA_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
U2NET_WEIGHTS_PATH = "models/saliency/u2netp.pth"

# Your VQA prompt. This will "guide" the summarization.
PROMPT_TEMPLATE = "[INST] <image>\nThis is frame {frame_num}. Describe what is happening. [/INST]"

# --- 2. Load Models and Processor ---
print("Loading processor...")
processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)

print("Loading Sali-Cache LLaVA model...")
# Initialize your custom model
model = SaliCacheLlava.from_pretrained(
    LLAVA_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    u2net_weights_path=U2NET_WEIGHTS_PATH  # Pass the weights path
)

print("Model loaded successfully.")

# --- 3. Video Processing Loop ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0
past_key_values = None  # This is the cache, starts empty

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image (expected by processor)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Create the prompt for this specific frame
    prompt = PROMPT_TEMPLATE.format(frame_num=frame_num)

    # Process inputs (image + text)
    inputs = processor(prompt, image, return_tensors="pt").to(model.device, torch.float16)

    # --- THIS IS THE CORE OF YOUR PROJECT ---
    # We call our custom model, passing in the cache from the last step.
    # Our overridden 'forward' pass will automatically apply all
    # the pruning and quantization logic to the 'past_key_values'.
    
    print(f"--- Processing Frame {frame_num} ---")
    
    with torch.no_grad():
        # 'generate' is a high-level loop. For deep customization,
        # we'd call 'model.forward()' directly in a loop.
        # But for this project, we can let 'generate' manage the loop
        # and our custom 'forward' pass will intercept the cache.
        
        # Here, we just ask for a 1-token reply to keep it fast
        # and let our custom 'forward' hook do its cache magic.
        output = model.generate(
            **inputs,
            past_key_values=past_key_values,
            max_new_tokens=5, # We don't care about the output, just the cache
            use_cache=True
        )
        
        # 'generate' doesn't return the cache in a simple way.
        # A more advanced loop would call model.forward() directly.
        
        # For this skeleton, we will run the 'forward' pass MANUALLY
        # to show how the cache is built.
        
        # --- A BETTER LOOP (Manual Forward Pass) ---
        
        # 1. Manually call our custom 'forward' pass
        # This will run our pruning/quantization logic
        outputs = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True
        )
        
        # 2. Get the optimized cache from the outputs
        # Our custom 'forward' pass has already pruned/quantized this!
        past_key_values = outputs.past_key_values

        # 3. (Optional) Get the generated text for summarization
        # We would "sample" the logits to get the next token
        # For now, we're just building the cache.

    print(f"Frame {frame_num} processed. Cache size: {len(past_key_values) if past_key_values else 0} layers.")
    
    frame_num += 1

    # For testing, let's just run 10 frames
    if frame_num > 10:
        print("Finished processing 10 frames.")
        break

cap.release()
print("Experiment complete.")