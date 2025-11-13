import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import requests
from PIL import Image
import warnings

# Suppress a specific warning from transformers
warnings.filterwarnings(
    "ignore", 
    message=".*_attn_implementation_internal.*", 
    category=UserWarning
)

print("--- Starting Model Setup ---")

# 1. Define Model ID and Data Type
# This is the official Hugging Face ID for LLaVA 1.6 with Mistral-7B
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

# We use float16 (half-precision) to fit in 24GB VRAM
torch_dtype = torch.float16 

# 2. Load the Processor
# The "processor" bundles the text tokenizer and the image processor
print(f"Loading processor for {model_id}...")
processor = LlavaNextProcessor.from_pretrained(model_id)

# 3. Load the Model
print(f"Loading model {model_id} onto GPU...")
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,  # Use float16
    device_map="auto",        # Automatically uses your GPU
    low_cpu_mem_usage=True    # Helps with large models
)

print("\n✅ Model and Processor Loaded Successfully!")
print(f"   Model is using device: {model.device}")

# --- 4. Run a Test Inference ---
print("\n--- Running a Test Inference ---")

try:
    # Load a sample image
    url = "https://www.ilankelman.org/stopsigns/stop-sign.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # Define a prompt
    # The prompt MUST follow the model's chat template
    prompt = "[INST] <image>\nWhat is the text on the sign? [/INST]"

    # Process the inputs (image + text)
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)

    # Generate a response
    print("Generating response...")
    output = model.generate(**inputs, max_new_tokens=100)

    # Decode and print the response
    response_text = processor.decode(output[0], skip_special_tokens=True)
    
    print("\n--- Test Result ---")
    print(response_text)
    print("-------------------")

except Exception as e:
    print(f"\n❌ An error occurred during the test inference: {e}")
    print("   Please ensure you have a stable internet connection and all libraries are installed.")