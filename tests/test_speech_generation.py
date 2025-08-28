#!/usr/bin/env python
# coding=utf-8
"""
Simple test script to verify that speech generation works correctly after our fixes.
"""

import torch
import numpy as np
import pytest
from vibevoice.model import load_vibevoice_model

def test_speech_generation():
    """Test that speech generation works with our fixes."""
    print("Loading model...")
    model, processor = load_vibevoice_model(
        "microsoft/VibeVoice-1.5B",
        device="cpu",
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    print("Model loaded successfully.")
    
    # Create a test input with voice sample
    text_input = "Speaker 1: This is a test of the speech generation system."
    dummy_voice_sample = np.zeros(24000, dtype=np.float32)
    
    print("Processing input...")
    inputs = processor(
        text=[text_input],
        voice_samples=[[dummy_voice_sample]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    print("Generating speech...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            tokenizer=processor.tokenizer,
            do_sample=False,
        )
    
    print("Speech generation completed successfully!")
    print(f"Output sequences shape: {outputs.sequences.shape}")
    print(f"Speech outputs: {outputs.speech_outputs is not None}")

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), reason="No GPU available for compile test")
def test_speech_generation_compiled():
    """Test that speech generation works with torch.compile enabled."""
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Loading compiled model on {device}...")
    model, processor = load_vibevoice_model(
        "microsoft/VibeVoice-1.5B",
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32, # float16 for CUDA, float32 for MPS
        attn_implementation="eager", # Use eager for this test to avoid flash_attention issues on some setups
        use_compile=True
    )
    print("Model loaded successfully.")
    
    # Create a test input with voice sample
    text_input = "Speaker 1: This is a test of the speech generation system."
    dummy_voice_sample = np.zeros(24000, dtype=np.float32)
    
    print("Processing input...")
    inputs = processor(
        text=[text_input],
        voice_samples=[[dummy_voice_sample]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Move inputs to the correct device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("Generating speech...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            tokenizer=processor.tokenizer,
            do_sample=False,
        )
    
    print("Speech generation completed successfully!")
    print(f"Output sequences shape: {outputs.sequences.shape}")
    print(f"Speech outputs: {outputs.speech_outputs is not None}")

if __name__ == "__main__":
    test_speech_generation()
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        test_speech_generation_compiled()
