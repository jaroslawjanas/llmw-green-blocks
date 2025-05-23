#!/usr/bin/env python3
"""
Utility functions for LLM Watermarking.
"""

import os
import random
from typing import Dict
from datasets import load_dataset
import src.paths as paths


def load_hf_token():
    """Load HuggingFace token from hf_token file."""
    token_path = "hf_token"
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    return token

def get_random_essay(seed=None, ) -> str:
    """
    Get a random essay from the dataset.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Random essay text
    """
    if seed is not None:
        random.seed(seed)
    
    # Get HuggingFace token if available
    token = load_hf_token()
    
    # Configure dataset loading options
    dataset_kwargs = {
        "cache_dir": paths.DATASETS_CACHE_DIR  # Use datasets subdirectory
    }
    if token:
        dataset_kwargs["token"] = token
    
    # Load the dataset
    dataset = load_dataset("ChristophSchuhmann/essays-with-instructions", **dataset_kwargs)
    
    # Select a random essay
    essay_idx = random.randint(0, len(dataset["train"]) - 1)
    essay_data = dataset["train"][essay_idx]
    
    # Get essay text
    essay_text = essay_data.get("essay", "")
    if not essay_text:
        # Fallback if "essay" field doesn't exist
        text_fields = [v for k, v in essay_data.items() if isinstance(v, str) and len(v) > 100]
        if text_fields:
            essay_text = random.choice(text_fields)
    
    # Trim if too long
    if len(essay_text) > 1000:
        start_idx = random.randint(0, len(essay_text) - 1000)
        essay_text = essay_text[start_idx:start_idx + 1000]
    
    return essay_text

def save_to_file(prompt: str, generated_text: str, stats: Dict, output_file: str, 
                 seed: int, model_name: str, context_window: int, bias: float, 
                 green_fraction: float, temperature: float, hash_window: int = 1):
    """
    Save the prompt, generated text and stats to a file.
    
    Args:
        prompt: The input prompt
        generated_text: The generated text
        stats: Statistics about the watermarking
        output_file: Path to the output file
        seed: Random seed used for generation
        model_name: Name of the model used for generation
        context_window: Maximum context window size
        bias: Bias value added to green tokens
        green_fraction: Fraction of tokens in green list
        temperature: Sampling temperature used for generation
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== INPUT PROMPT ===\n")
        f.write(prompt)
        f.write("\n\n=== GENERATED TEXT ===\n")
        f.write(generated_text)
        f.write("\n\n=== WATERMARK STATISTICS ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Green tokens: {stats['green_tokens']}\n")
        f.write(f"Red tokens: {stats['red_tokens']}\n")
        f.write(f"Total tokens: {stats['total_tokens']}\n")
        f.write(f"Green ratio: {stats['green_ratio']:.4f}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Context window: {context_window}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Green fraction: {green_fraction}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Hash window: {hash_window}\n")