#!/usr/bin/env python3
"""
LLM Watermarking Implementation based on Kirchenbauer et al., 2023
https://arxiv.org/abs/2301.10226

This script implements the red-green token watermarking technique with greedy sampling.
"""

import argparse
import hashlib
import random
import os
import torch
import numpy as np
import datetime
from typing import List, Tuple, Dict, Optional, Union
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
from model_formatters import format_prompt_for_model
from utils import load_hf_token

# Set the default cache and output directories
CACHE_DIR = "./cache"
MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")
DATASETS_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")
OUTPUT_DIR = "./output"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)
os.makedirs(DATASETS_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LLMWatermarker:
    def __init__(
        self,
        model_name: str,
        green_list_fraction: float = 0.5,
        bias: float = 6.0,
        seed: int = 4242,
        cache_dir: str = CACHE_DIR,
        device: Optional[str] = None,
        context_window: int = 1024,
        temperature: float = 0.0,
        hash_window: int = 1,
    ):
        """
        Initialize the watermarker with the specified model and parameters.
        
        Args:
            model_name: HuggingFace model identifier
            green_list_fraction: Fraction of tokens to include in green list (default: 0.5)
            bias: Logit bias to apply to green tokens (default: 6.0)
            seed: Random seed for reproducibility
            cache_dir: Directory to cache models
            device: Device to run model on ('cuda', 'cpu', or None for auto-detection)
            context_window: Maximum number of tokens to use as context for generation (default: 1024)
            temperature: Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
            hash_window: Number of previous tokens to hash together (default: 1)
        """
        self.model_name = model_name
        self.green_list_fraction = green_list_fraction
        self.bias = bias
        self.seed = seed
        self.cache_dir = cache_dir
        self.context_window = context_window
        self.temperature = temperature
        self.hash_window = hash_window
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Stats
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0
        
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        # Get HuggingFace token if available
        token = load_hf_token()
        
        # Configure tokenizer options
        tokenizer_kwargs = {
            "cache_dir": MODELS_CACHE_DIR,  # Use models subdirectory
        }
        if token:
            # Use token instead of deprecated use_auth_token
            tokenizer_kwargs["token"] = token
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            **tokenizer_kwargs
        )
        
        # Configure model loading options
        model_kwargs = {
            "cache_dir": MODELS_CACHE_DIR,  # Use models subdirectory
        }
        if token:
            model_kwargs["token"] = token
        
        # Load model with appropriate settings for the device
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                device_map="auto",
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            self.model.to(self.device)
            
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _get_red_green_tokens(self, token_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Generate red and green token lists based on the hash of previous tokens.
        
        Args:
            token_ids: List of token IDs to hash
            
        Returns:
            Tuple of (green_tokens, red_tokens)
        """
        # Create a hash from the specified window of tokens
        hash_input = "-".join([str(tid) for tid in token_ids])
        hash_object = hashlib.sha256(hash_input.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert hash to a numeric seed
        hash_seed = int(hash_hex, 16) % (2**32)
        
        # Create a local random generator with this seed
        rng = random.Random(hash_seed)
        
        # Get vocabulary size
        vocab_size = len(self.tokenizer)
        
        # Generate a random permutation of token indices
        all_tokens = list(range(vocab_size))
        rng.shuffle(all_tokens)
        
        # Split into green and red lists
        split_point = int(vocab_size * self.green_list_fraction)
        green_tokens = all_tokens[:split_point]
        red_tokens = all_tokens[split_point:]
        
        return green_tokens, red_tokens
        
    def _modify_logits(self, logits: torch.Tensor, token_window: List[int]) -> torch.Tensor:
        """
        Modify logits by adding bias to green tokens.
        
        Args:
            logits: Original logits from the model
            token_window: List of previous token IDs to use for hashing
            
        Returns:
            Modified logits tensor
        """
        # Ensure logits are in the right shape (batch_size, seq_len, vocab_size)
        # The tensor might be (1, 1, vocab_size) or just (1, vocab_size)
        if len(logits.shape) == 3:
            # Get the last token's logits and flatten to 1D
            logits = logits[0, -1, :]
        elif len(logits.shape) == 2:
            # Already (batch_size, vocab_size), get first batch
            logits = logits[0, :]
        
        # Get vocabulary size from logits
        vocab_size = logits.shape[-1]
        
        # Get green and red tokens
        green_tokens, _ = self._get_red_green_tokens(token_window)
        
        # Clone logits for modification
        modified_logits = logits.clone()
        
        # Add bias to each green token individually, checking bounds
        for token_id in green_tokens:
            if token_id < vocab_size:
                modified_logits[token_id] += self.bias
        
        # Reshape back to original format
        return modified_logits.unsqueeze(0).unsqueeze(0)
        
    def generate_text(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        verbose: bool = True
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate text with watermarking.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (generated_text, statistics)
        """
        # Reset counters
        self.green_tokens_selected = 0
        self.red_tokens_selected = 0
        
        # Format the prompt for the specific model
        formatted_prompt = format_prompt_for_model(prompt, self.model_name, self.tokenizer)
        
        # Tokenize the formatted prompt
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Store generated ids
        generated_ids = input_ids.clone()[0].tolist()
        
        # Setup progress tracking
        progress_bar = tqdm(range(max_new_tokens), disable=not verbose)
        
        # Generate tokens one by one
        for _ in progress_bar:
            # Only use the last context_window tokens if needed (to save memory and time)
            if len(generated_ids) > self.context_window:
                input_ids = torch.tensor([generated_ids[-self.context_window:]], device=self.device)
            else:
                input_ids = torch.tensor([generated_ids], device=self.device)
            
            # Get logits from the model
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1:, :]  # Get logits of last token
            
            # Create a window of previous tokens for hashing
            if len(generated_ids) >= self.hash_window:
                token_window = generated_ids[-self.hash_window:]
            else:
                token_window = generated_ids
            
            # Modify logits with watermark using the token window
            modified_logits = self._modify_logits(logits, token_window)
            
            # Apply temperature if set
            if self.temperature > 0:
                # Scale logits by temperature
                modified_logits = modified_logits / self.temperature
                
            # Get probabilities through softmax
            probs = torch.nn.functional.softmax(modified_logits, dim=-1)
            
            # Token sampling based on temperature
            if self.temperature == 0 or self.temperature < 1e-6:
                # Greedy sampling (select token with highest probability)
                next_token_id = torch.argmax(probs, dim=-1).item()
            else:
                # Sample from the distribution
                next_token_id = torch.multinomial(probs.squeeze(), 1).item()
            
            # Track green/red selection
            green_tokens, red_tokens = self._get_red_green_tokens(token_window)
            
            # Check vocabulary bounds for safety
            vocab_size = len(self.tokenizer)
            if next_token_id < vocab_size:  # Make sure token is in vocabulary range
                if next_token_id in green_tokens:
                    self.green_tokens_selected += 1
                else:
                    self.red_tokens_selected += 1
            
            # Update progress bar with stats
            green_ratio = self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10)
            progress_bar.set_description(f"Green: {self.green_tokens_selected}, Red: {self.red_tokens_selected}, Ratio: {green_ratio:.2f}")
            
            # Add the new token to generated ids
            generated_ids.append(next_token_id)
            
            # Check if we've reached an EOS token
            if next_token_id == self.tokenizer.eos_token_id:
                break
                
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Compile statistics
        stats = {
            "green_tokens": self.green_tokens_selected,
            "red_tokens": self.red_tokens_selected,
            "total_tokens": self.green_tokens_selected + self.red_tokens_selected,
            "green_ratio": self.green_tokens_selected / (self.green_tokens_selected + self.red_tokens_selected + 1e-10)
        }
        
        return generated_text, stats


def get_random_essay(seed=None) -> str:
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
        "cache_dir": DATASETS_CACHE_DIR  # Use datasets subdirectory
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

def main():
    parser = argparse.ArgumentParser(description="LLM Watermarking Implementation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green tokens")
    parser.add_argument("--seed", type=int, default=4242, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay if not provided)")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Cache directory for models")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--output", type=str, help="Custom filename for output in the output/ directory (if not specified, a filename will be auto-generated)")
    parser.add_argument("--context-window", type=int, default=1024, help="Maximum number of tokens to use as context for generation (default: 1024)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)")
    
    args = parser.parse_args()
    
    # Determine device
    device = "cpu" if args.no_cuda else None
    
    # Initialize the watermarker
    watermarker = LLMWatermarker(
        model_name=args.model,
        green_list_fraction=args.green_fraction,
        bias=args.bias,
        seed=args.seed,
        cache_dir=args.cache_dir,
        device=device,
        context_window=args.context_window,
        temperature=args.temperature,
        hash_window=args.hash_window
    )
    
    # Get prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = get_random_essay(seed=args.seed)
        print("\n--- Random Essay Prompt ---")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("---------------------------\n")
    
    # Generate text
    print(f"Generating {args.max_tokens} tokens with watermarking...")
    generated_text, stats = watermarker.generate_text(
        prompt=prompt,
        max_new_tokens=args.max_tokens
    )
    
    # Print results
    print("\n--- Generated Text ---")
    print(generated_text)
    print("---------------------\n")
    
    print("--- Watermark Statistics ---")
    print(f"Model: {args.model}")
    print(f"Green tokens: {stats['green_tokens']}")
    print(f"Red tokens: {stats['red_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Green ratio: {stats['green_ratio']:.4f}")
    print(f"Seed: {args.seed}")
    print(f"Context window: {args.context_window}")
    print(f"Bias: {args.bias}")
    print(f"Green fraction: {args.green_fraction}")
    print(f"Temperature: {args.temperature}")
    print(f"Hash window: {args.hash_window}")
    print("---------------------------")
    
    # Save output to file if requested
    if args.output:
        output_path = os.path.join(OUTPUT_DIR, args.output)
        # Save to file with all parameters
        save_to_file(prompt, generated_text, stats, output_path, args.seed, args.model,
                    args.context_window, args.bias, args.green_fraction, args.temperature, 
                    args.hash_window)
        print(f"\nOutput saved to: {output_path}")
    else:
        # Generate a default filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split("/")[-1]
        output_file = f"{model_name}_gen_{timestamp}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        # Stats for file output with all parameters
        save_to_file(prompt, generated_text, stats, output_path, args.seed, args.model,
                    args.context_window, args.bias, args.green_fraction, args.temperature,
                    args.hash_window)
        print(f"\nOutput saved to: {output_path}")
    

if __name__ == "__main__":
    main()
