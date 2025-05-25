#!/usr/bin/env python3
"""
Utility functions for LLM Watermarking.
"""

import os
import random
from typing import Dict, List
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
    essay_text = essay_data.get("instructions", "")
    if not essay_text:
        raise ValueError("No text found in dataset entry")
    
    return essay_text

def save_to_file(
        prompt: str,
        generated_text: str,
        stats: Dict,
        green_red_mask:List[int],
        block_count: int,
        output_file: str, 
        seed: int,
        model_name: str,
        context_window: int,
        bias: float,
        green_fraction: float,
        temperature: float,
        block_size: int,
        hash_window: int = 1
    ):

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
        f.write("\n\n=== GREEN/RED MASK ===\n")
        f.write(str(green_red_mask))
        f.write("\n\n=== WATERMARK STATISTICS ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"\n")
        f.write(f"Green tokens: {stats['green_tokens']}\n")
        f.write(f"Red tokens: {stats['red_tokens']}\n")
        f.write(f"Total tokens: {stats['total_tokens']}\n")
        f.write(f"Green ratio: {stats['green_ratio']:.4f}\n")
        f.write(f"Block count: {block_count}\n")
        f.write(f"\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Context window: {context_window}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Green fraction: {green_fraction}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Block size: {block_size}\n")
        f.write(f"Hash window: {hash_window}\n")

def count_green_blocks(mask: List[int], block_size: int) -> int:
    """
    Counts the number of intact green blocks (1s) in a mask (of 1s and 0s)

    Args:
        mask: A list of integers (0s and 1s) representing the green/red mask.
        block_size: The required size of an intact block of 1s.

    Returns:
        The number of intact blocks of 1s found in the mask.
    """
    if not mask or block_size <= 0 or block_size > len(mask):
        return 0

    block_count = 0
    green_in_row = 0

    for color in mask:
        if color == 1: # Green
            green_in_row += 1
        elif color == 0: # Red
            green_in_row = 0
        else:
            raise ValueError("Mask must contain only 0s and 1s")

        if green_in_row == block_size:
            block_count += 1
            green_in_row = 0 # Reset count after finding a block

    return block_count