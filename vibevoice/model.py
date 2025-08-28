import torch
import warnings
from vibevoice.utils.device_config import get_optimal_config
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def load_vibevoice_model(model_name, device="auto", torch_dtype=None, attn_implementation="auto", use_compile=True):
    """
    Loads the VibeVoice model and processor with explicit device placement.

    Args:
        model_name (str): The name of the model to load from the Hugging Face Hub.
        device (str, optional): The device to load the model on. Defaults to "auto".
        torch_dtype (torch.dtype, optional): The dtype to use for the model. Defaults to None.
        attn_implementation (str, optional): The attention implementation to use. Defaults to "auto".
        use_compile (bool, optional): Whether to compile the model with torch.compile. Defaults to True.

    Returns:
        tuple: A tuple containing the model and the processor.
    """
    optimal_device, optimal_dtype, optimal_attn_impl = get_optimal_config()

    if device == "auto":
        device = optimal_device
    if torch_dtype is None:
        torch_dtype = optimal_dtype
    if attn_implementation == "auto":
        attn_implementation = optimal_attn_impl

    print("\n" + "="*50)
    print("Loading VibeVoice Model with Configuration:")
    print(f"  Model Name: {model_name}")
    print(f"  Device: {device}")
    print(f"  Data Type: {torch_dtype}")
    print(f"  Attention Implementation: {attn_implementation}")
    print("="*50 + "\n")

    model_kwargs = {"torch_dtype": torch_dtype}
    # Only add the argument if it's a valid, known implementation
    if attn_implementation in ("flash_attention_2", "sdpa"):
        model_kwargs["attn_implementation"] = attn_implementation

    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Explicitly move the model to the device. This is more reliable for MPS.
    model.to(torch.device(device))
    model.eval()

    if use_compile:
        if device == "cuda":
            try:
                print("Attempting to compile model with backend: inductor...")
                model = torch.compile(model, backend="inductor", mode="reduce-overhead")
                print("Model compiled successfully.")
            except Exception as e:
                warnings.warn(
                    f"torch.compile failed with backend 'inductor' ({e}). "
                    "Falling back to eager mode. For details, run with TORCH_LOGS=+dynamo.",
                    UserWarning,
                    stacklevel=2,
                )
        elif device == "mps":
            try:
                print("Attempting to compile model for MPS...")
                model = torch.compile(model, mode="reduce-overhead") # No backend specified
                print("Model compiled successfully.")
            except Exception as e:
                warnings.warn(
                    f"torch.compile failed for MPS ({e}). "
                    "Falling back to eager mode. For details, run with TORCH_LOGS=+dynamo.",
                    UserWarning,
                    stacklevel=2,
                )

    processor = VibeVoiceProcessor.from_pretrained(model_name)

    return model, processor

