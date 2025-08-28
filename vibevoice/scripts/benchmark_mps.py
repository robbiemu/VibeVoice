#!/usr/bin/env python
# coding=utf-8
"""
Benchmark script for measuring VibeVoice inference performance across different devices (MPS, CUDA, CPU).

This script measures end-to-end generation latency with proper synchronization and cache handling
for accurate timing on each backend.
"""

import time
import torch
import os
from typing import Dict, List
import numpy as np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print(
        "Warning: psutil not available. Memory usage will not be reported. To enable, run: pip install psutil"
    )

from vibevoice.utils.device_config import get_optimal_config
from vibevoice.model import load_vibevoice_model


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information."""
    if not PSUTIL_AVAILABLE:
        return {}

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
    }


def benchmark_inference(model, processor, test_inputs: List[str], device: str) -> Dict:
    """
    Benchmark inference performance with backend-appropriate synchronization.

    Args:
        model: The loaded VibeVoice model
        processor: The VibeVoice processor
        test_inputs: List of text inputs to benchmark
        device: The device string ("mps", "cuda", or "cpu")

    Returns:
        Dictionary with timing results
    """
    model.eval()
    times = []

    # Get initial memory usage
    initial_memory = get_memory_usage()

    with torch.no_grad():
        for text_input in test_inputs:
            # Create a dummy voice sample (1 second of silence) for the processor
            # This is required by the VibeVoice model API
            dummy_voice_sample = np.zeros(24000, dtype=np.float32)

            # Tokenize input and move to device
            inputs = processor(
                text=[text_input],  # Wrap in list for batch processing
                voice_samples=[[dummy_voice_sample]], # Provide the required voice prompt
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move all tensor inputs to the target device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Clear cache before timing
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

            start_time = time.time()
            # Run full generation for accurate benchmarking
            with torch.no_grad():
                _ = model.generate(
                    **inputs, # Pass all processor outputs to generate
                    max_new_tokens=16,  # Generate 16 new tokens for benchmarking
                    tokenizer=processor.tokenizer,
                    do_sample=False,    # Deterministic generation for consistent timing
                )

            # For CUDA, synchronize to ensure the operation is complete before stopping the timer
            if device == "cuda":
                torch.cuda.synchronize()
            # For MPS, synchronize if available (newer PyTorch versions)
            elif device == "mps" and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    # Get final memory usage
    final_memory = get_memory_usage()

    # Calculate memory difference if psutil is available
    memory_diff = {}
    if initial_memory and final_memory:
        memory_diff = {
            "rss_diff_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "vms_diff_mb": final_memory["vms_mb"] - initial_memory["vms_mb"],
        }

    return {
        "avg_time": sum(times) / len(times),
        "times": times,
        "min_time": min(times),
        "max_time": max(times),
        "memory_usage": final_memory,
        "memory_diff": memory_diff,
    }


def main():
    """Main benchmark function."""
    print("=== VibeVoice Performance Benchmark ===")

    # Get optimal configuration
    config = get_optimal_config()
    device = config["device"]
    dtype = config["dtype"]
    attn_impl = config["attn_implementation"]

    print(f"Selected device: {device}")
    print(f"Data type: {dtype}")
    print(f"Attention implementation: {attn_impl}")

    if device == "mps":
        print(
            "Note: MPS uses unified memory. Timings can have variance. Recommend median of multiple runs."
        )
    elif device == "cuda":
        print("Note: CUDA timings include synchronization for accuracy.")
    else:
        print("Note: CPU timings do not require synchronization.")

    # Define test inputs
    test_inputs = [
        "Speaker 1: Short test sentence.",
        "Speaker 1: This is a medium length sentence for testing performance.",
        "Speaker 1: This is a much longer sentence that will test the model's ability to handle extended input sequences. " \
        * 3,
        "Speaker 1: A conversation between two people discussing the future of artificial intelligence.\nSpeaker 2: Yes, it's fascinating how quickly the field is evolving.\nSpeaker 1: Indeed, we're seeing breakthroughs almost daily.",
    ]
    print(f"\nTest inputs: {len(test_inputs)} prompts of varying lengths")

    # --- Eager Mode Benchmark ---
    print("\n--- Benchmarking Eager Mode ---")
    model_eager, processor = load_vibevoice_model(
        "microsoft/VibeVoice-1.5B",
        device=device,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        use_compile=False,
    )
    print("Eager model loaded. Performing warmup run...")
    warmup_result_eager = benchmark_inference(model_eager, processor, [test_inputs[0]], device)
    print(f"Warmup completed in {warmup_result_eager['avg_time']:.3f}s")

    print("\nStarting eager benchmark...")
    results_eager = benchmark_inference(model_eager, processor, test_inputs, device)
    print("\n--- Eager Mode Results ---")
    print(f"Average inference time: {results_eager['avg_time']:.3f}s")
    print(f"Min inference time: {results_eager['min_time']:.3f}s")
    print(f"Max inference time: {results_eager['max_time']:.3f}s")
    if results_eager["memory_usage"]:
        print(f"  Memory RSS: {results_eager['memory_usage']['rss_mb']:.1f} MB")

    # --- Compiled Mode Benchmark ---
    # Only run if the device is not CPU, as compile is not supported
    if device != "cpu":
        print("\n--- Benchmarking Compiled Mode ---")
        model_compiled, processor = load_vibevoice_model(
            "microsoft/VibeVoice-1.5B",
            device=device,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            use_compile=True,
        )
        print("Compiled model loaded. Performing warmup run...")
        warmup_result_compiled = benchmark_inference(model_compiled, processor, [test_inputs[0]], device)
        print(f"Warmup completed in {warmup_result_compiled['avg_time']:.3f}s")

        print("\nStarting compiled benchmark...")
        results_compiled = benchmark_inference(model_compiled, processor, test_inputs, device)
        print("\n--- Compiled Mode Results ---")
        print(f"Average inference time: {results_compiled['avg_time']:.3f}s")
        print(f"Min inference time: {results_compiled['min_time']:.3f}s")
        print(f"Max inference time: {results_compiled['max_time']:.3f}s")
        if results_compiled["memory_usage"]:
            print(f"  Memory RSS: {results_compiled['memory_usage']['rss_mb']:.1f} MB")

    print(
        "\nNote: For MPS, it's recommended to take the median of several full benchmark runs for stable results."
    )



if __name__ == "__main__":
    main()
