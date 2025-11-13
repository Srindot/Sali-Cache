# Sali Cache
A Two-Stage Spatio-Temporal KV Cache for Efficient Video VLM Inference

## Problem Statement
Processing long videos with Vision-Language Models (VLMs) is computationally prohibitive. The Key-Value (KV) cache, which stores context from past frames, grows linearly with video length, quickly exhausting GPU memory (even 24GB). This is due to high temporal redundancy (static backgrounds) and spatial redundancy (unimportant pixels).

## Proposed Solution 

A novel, two-stage cache management pipeline that runs before tokens are cached.

Stage 1 (Temporal): "Deduplicates" tokens that are redundant from the previous frame.

Stage 2 (Spatial): "Adaptively quantizes" new, unique tokens based on their visual importance, as determined by a cheap, external saliency model.

