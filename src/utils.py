#!/usr/bin/env python3
"""
Utility functions for LLM Watermarking.
"""

import os
import sys
import random
from typing import Dict, List, Tuple
from datasets import load_dataset
import src.paths as paths


class Tee:
    """
    A class that writes to multiple file-like objects simultaneously.
    Used to capture console output to both stdout and a log file.
    """
    def __init__(self, *files):
        self.files = files

    def write(self, text):
        for file in self.files:
            file.write(text)
            file.flush()  # Ensure immediate writing

    def flush(self):
        for file in self.files:
            file.flush()


def load_hf_token():
    """Load HuggingFace token from hf_token file."""
    token_path = "hf_token"
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    return token

def get_shuffled_essays(
    dataset_name: str,
    dataset_subset: str,
    dataset_split: str,
    dataset_column: str,
    seed: int,
    n_prompts: int
) -> List[str]:
    """
    Get a shuffled list of prompts from a specified Hugging Face dataset, subset, split, and column.
    
    Args:
        dataset_name: The name of the Hugging Face dataset (e.g., "ChristophSchuhmann/essays-with-instructions").
        dataset_subset: The name of the dataset subset (e.g., "default").
        dataset_split: The name of the dataset split (e.g., "train", "validation", "test").
        dataset_column: The name of the column in the dataset to extract prompts from (e.g., "text" or "instructions").
        seed: Random seed for reproducibility.
        n_prompts: Number of prompts to return.
        
    Returns:
        List of prompt texts (shuffled deterministically based on seed).
    """
    # Set random seed for reproducible shuffling
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
    try:
        dataset = load_dataset(dataset_name, dataset_subset, **dataset_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}' with subset '{dataset_subset}': {e}")
    
    if dataset_split not in dataset:
        raise ValueError(f"Dataset '{dataset_name}' with subset '{dataset_subset}' does not have a '{dataset_split}' split. "
                         f"Available splits: {list(dataset.keys())}")
        
    total_prompts_in_dataset = len(dataset[dataset_split])
    
    # Check if we have enough prompts
    if n_prompts > total_prompts_in_dataset:
        raise ValueError(f"Requested {n_prompts} prompts but dataset '{dataset_name}' "
                         f"({dataset_subset} subset, {dataset_split} split) only contains {total_prompts_in_dataset} entries.")
    
    # Create a list of indices and shuffle them
    prompt_indices = list(range(total_prompts_in_dataset))
    random.shuffle(prompt_indices)
    
    # Get the first n_prompts indices
    selected_indices = prompt_indices[:n_prompts]
    
    # Extract the prompts
    prompts = []
    for idx in selected_indices:
        prompt_data = dataset[dataset_split][idx]
        try:
            prompt_text = prompt_data[dataset_column]
        except KeyError:
            raise ValueError(f"Column '{dataset_column}' not found in dataset entry at index {idx} of split '{dataset_split}'. "
                             f"Available columns: {list(prompt_data.keys())}")
        if not prompt_text:
            raise ValueError(f"No text found in dataset entry at index {idx} for column '{dataset_column}' in split '{dataset_split}'.")
        prompts.append(prompt_text)
    
    return prompts

def save_generation_details(
        prompt: str,
        generated_text: str,
        file_path: str,
        stats: Dict,
        green_red_mask:List[int],
        block_counts: List[Tuple[int, int]],
        seed: int,
        model_name: str,
        dataset_name: str,
        context_window: int,
        bias: float,
        green_fraction: float,
        temperature: float,
        hash_window: int = 1
    ):

    """
    Save the prompt, generated text and stats to a file.
    
    Args:
        prompt: The input prompt
        generated_text: The generated text
        stats: Statistics about the watermarking
        block_counts: A list of tuples, where each tuple contains (block_size, counted_blocks)
                              for each block size considered.
        seed: Random seed used for generation
        model_name: Name of the model used for generation
        dataset_name: Name of the dataset used for prompts
        context_window: Maximum context window size
        bias: Bias value added to green tokens
        green_fraction: Fraction of tokens in green list
        temperature: Sampling temperature used for generation
    """
    # Generate output filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== INPUT PROMPT ===\n")
        f.write(prompt)

        f.write("\n\n=== GENERATED TEXT ===\n")
        f.write(generated_text)

        f.write("\n\n=== GREEN/RED MASK ===\n")
        f.write(str(green_red_mask))

        f.write("\n\n=== WATERMARK STATISTICS ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"\n")
        f.write(f"Green tokens: {stats['green_tokens']}\n")
        f.write(f"Red tokens: {stats['red_tokens']}\n")
        f.write(f"Total tokens: {stats['total_tokens']}\n")
        f.write(f"Green ratio: {stats['green_ratio']:.4f}\n")
        f.write(f"\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Context window: {context_window}\n")
        f.write(f"Bias: {bias}\n")
        f.write(f"Green fraction: {green_fraction}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Hash window: {hash_window}\n")
        f.write(f"\n")
        for b_size, b_count in block_counts: # Iterate and write
            f.write(f"Block count (size {b_size}): {b_count}\n")

def count_green_blocks(mask: List[int], block_sizes: List[int]) -> List[Tuple[int, int]]:
    """
    Counts the number of intact green blocks (1s) in a mask (of 1s and 0s) for multiple block sizes.

    Args:
        mask: A list of integers (0s and 1s) representing the green/red mask.
        block_sizes: A list of required sizes for intact blocks of 1s.

    Returns:
        A list of tuples, where each tuple contains (block_size, counted_blocks) for each
        block size provided in block_sizes.
    """
    results = []
    
    for b_size in block_sizes:
        if not mask or b_size <= 0 or b_size > len(mask):
            results.append((b_size, 0))
            continue

        block_count = 0
        green_in_row = 0

        for color in mask:
            if color == 1: # Green
                green_in_row += 1
            elif color == 0: # Red
                green_in_row = 0
            else:
                raise ValueError("Mask must contain only 0s and 1s")

            if green_in_row == b_size:
                block_count += 1
                green_in_row = 0 # Reset count after finding a block
        results.append((b_size, block_count))

    return results

def save_average_block_counts(
    average_block_counts: Dict[int, float],
    batch_output_dir: str,
    model_name: str,
    dataset_name: str,
    total_prompts: int,
    block_sizes_analyzed: List[int]
):
    """
    Save the calculated average block counts for the entire batch to a separate file.

    Args:
        average_block_counts: A dictionary where keys are block_size and values are
                              the average b_count for that size across the batch.
        batch_output_dir: The directory where the batch output files are saved.
        model_name: Name of the model used for generation.
        dataset_name: Name of the dataset used for prompts.
        total_prompts: Total number of prompts processed in the batch.
        block_sizes_analyzed: List of block sizes that were analyzed.
    """
    average_output_filepath = os.path.join(batch_output_dir, "average_block_counts.txt")
    with open(average_output_filepath, "w", encoding="utf-8") as f:
        f.write("=== AVERAGE BLOCK COUNTS PER BLOCK SIZE (BATCH SUMMARY) ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n\n")
        f.write(f"Total Prompts Processed: {total_prompts}\n")
        f.write(f"Block Sizes Analyzed: {block_sizes_analyzed}\n\n")
        for b_size in sorted(average_block_counts.keys()):
            f.write(f"Average Block Count (size {b_size}): {average_block_counts[b_size]:.2f}\n")
    print(f"\nAverage block counts saved to: {average_output_filepath}")
