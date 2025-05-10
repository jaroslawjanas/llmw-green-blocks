#!/usr/bin/env python3
"""
Wrapper script for running the LLM watermarking implementation.
This provides a simple way to run experiments with different models and parameters.
"""

import argparse
import json
import os
import subprocess
import sys

# Default configuration file
CONFIG_FILE = "model_config.json"

def load_config():
    """Load configuration from file if it exists."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"model": "facebook/opt-125m"}  # Default model

def main():
    # Load default configuration
    config = load_config()
    default_model = config.get("model", "facebook/opt-125m")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run watermarking experiments with different models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main options
    parser.add_argument("--select-model", action="store_true", 
                        help="Run the model selector")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"Model to use (current default: {default_model})")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--bias", type=float, default=6.0,
                        help="Bias to add to green tokens")
    
    # Additional parameters
    parser.add_argument("--green-fraction", type=float, default=0.5,
                        help="Fraction of tokens in green list")
    parser.add_argument("--seed", type=int, default=4242,
                        help="Random seed")
    parser.add_argument("--prompt", type=str,
                        help="Custom prompt (uses random essay if not provided)")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                        help="Cache directory for models")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Run model selector if requested
    if args.select_model:
        print("Running model selector...")
        subprocess.run([sys.executable, "model_selector.py"])
        return
    
    # Build command for running the watermarker
    cmd = [
        sys.executable, "llm_watermark.py",
        "--model", args.model,
        "--max-tokens", str(args.max_tokens),
        "--green-fraction", str(args.green_fraction),
        "--bias", str(args.bias),
        "--seed", str(args.seed),
        "--cache-dir", args.cache_dir
    ]
    
    # Add optional arguments if provided
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.no_cuda:
        cmd.append("--no-cuda")
    
    # Remove temperature parameter - no longer used
    # In the original Kirchenbauer et al. implementation, only bias is used
    
    # Run the watermarker
    print(f"Running watermarker with model: {args.model}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
