# Sali Cache
A Two-Stage Spatio-Temporal KV Cache for Efficient Video VLM Inference


## Problem Statement
Processing long videos with Vision-Language Models (VLMs) is computationally prohibitive. The Key-Value (KV) cache, which stores context from past frames, grows linearly with video length, quickly exhausting GPU memory (even 24GB). This is due to high temporal redundancy (static backgrounds) and spatial redundancy (unimportant pixels).


## Proposed Solution 

A novel, two-stage cache management pipeline that runs before tokens are cached.

Stage 1 (Temporal): "Deduplicates" tokens that are redundant from the previous frame.

Stage 2 (Spatial): "Adaptively quantizes" new, unique tokens based on their visual importance, as determined by a cheap, external saliency model.


## Model used for this framework

### 1. Vision "Summarization" Model (VLM)

Model: LLaVA 1.6 (Mistral-7B)

Hugging Face ID: llava-hf/llava-v1.6-mistral-7b-hf

Why:
- built on the very strong Mistral 7B base model. 
- Loaded in float16 (half-precision), will consoume ~15-16GB of VRAM transformers library from Hugging Face 


### 2. Salient Segmentation Model

Model: U²-Netp (U-Squared Net, lite/pruned version)

Why: 
- famous and highly effective saliency model
- lightweight U-2-Netp version is tiny (only ~4.7 MB) and very fast (runs at ~40 FPS)
- designed to produce clean segmentation masks of the "most important" object in a frame 
- uses a negligible amount of VRAM and can easily run on the GPU alongside LLaVA.

Library: open-source PyTorch implementations and pre-trained models 

### 3. Optical Flow Model

Model: Dense Optical Flow (Farneback's Algorithm)

Why: 
- classic computer vision algorithm that is perfect for this task
- extremely fast 
- runs on the CPU, meaning it uses zero VRAM

Library: OpenCV library.

Function: cv2.calcOpticalFlowFarneback()