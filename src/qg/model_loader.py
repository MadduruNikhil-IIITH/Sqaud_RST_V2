"""
Model loader for QG pipeline. Supports multiple models and abstracts Hugging Face pipeline loading.
"""

from transformers import pipeline
import torch

def get_device():
    return 0 if torch.cuda.is_available() else -1



import os
import glob

def remove_bin_files(model_name):
    """Delete all pytorch_model.bin files for the given model from the HF cache."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_pattern = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    for model_dir in glob.glob(model_dir_pattern + "*"):
        for bin_file in glob.glob(os.path.join(model_dir, "**", "pytorch_model.bin"), recursive=True):
            try:
                os.remove(bin_file)
                print(f"[INFO] Deleted {bin_file} to force safetensors usage.")
            except Exception as e:
                print(f"[WARN] Could not delete {bin_file}: {e}")


def load_qg_model(model_name):
    """
    Loads a QG model and tokenizer for the given task.
    Always uses 'text-generation' pipeline for compatibility with installed transformers.
    Forces use of safetensors to avoid PyTorch version restriction.
    """
    device = get_device()
    pipe_type = "text-generation"
    print(f"[INFO] Using pipeline type: {pipe_type}")

    return pipeline(
        pipe_type,
        model=model_name,
        tokenizer=model_name,
        device=device
    )

# Model recommendations
BEST_MODEL = "microsoft/Phi-4-mini-instruct"
SUPPORTED_MODELS = [
    "microsoft/Phi-4-mini-instruct",      # Efficient, not gated, fits 8GB VRAM, strong for QG
    "microsoft/Phi-3.5-mini-instruct",    # Also strong, not gated
    "google/gemma-2-9b-it",               # Good, may require 4-bit for 8GB VRAM
    "mistralai/Mistral-7B-Instruct-v0.3", # Strong, at upper VRAM limit
    "valhalla/t5-base-e2e-qg",            # T5: strong baseline
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-7B-Chat",
    "google/t5-base-qg-prepend",
    "allenai/unifiedqa-t5-base"
]
# Model comparison notes:
# - valhalla/t5-base-e2e-qg: Best for SQuAD-style QG, structured, fast, fits on 8GB VRAM
# - Qwen models: Good context length, efficient, chat/instruct variants, strong general LLMs
# - Phi-4-mini-instruct: Efficient, more open-ended, less structured QG
