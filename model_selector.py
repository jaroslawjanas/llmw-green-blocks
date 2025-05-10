#!/usr/bin/env python3
"""
Model Selector for LLM Watermarking implementation

This script helps users select appropriate models for watermarking,
considering memory constraints (especially for 8GB VRAM).
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import List, Dict, Any, Optional

def load_hf_token():
    """Load HuggingFace token from hf_token file."""
    token_path = "hf_token"
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    return token

# Define the cache directory
CACHE_DIR = "./cache"
MODELS_CACHE_DIR = os.path.join(CACHE_DIR, "models")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_CACHE_DIR, exist_ok=True)

# Models recommended for 8GB VRAM
RECOMMENDED_MODELS = [
    {
        "name": "facebook/opt-125m",
        "description": "Small OPT model (125M parameters), very fast, works on CPU or GPU",
        "min_vram": 0,  # Works on CPU
        "tags": ["fast", "small"]
    },
    {
        "name": "facebook/opt-350m",
        "description": "Small OPT model (350M parameters), good balance of performance and speed",
        "min_vram": 1,  # 1GB minimum
        "tags": ["fast", "small"]
    },
    {
        "name": "facebook/opt-1.3b",
        "description": "Medium OPT model (1.3B parameters), better quality output",
        "min_vram": 2,  # 2GB minimum
        "tags": ["medium"]
    },
    {
        "name": "facebook/opt-2.7b",
        "description": "Large OPT model (2.7B parameters), good performance",
        "min_vram": 5,  # 5GB minimum
        "tags": ["large"]
    },
    {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Small but capable chat-tuned Llama model",
        "min_vram": 2,  # 2GB minimum
        "tags": ["medium", "chat-tuned"]
    },
    {
        "name": "microsoft/phi-1_5",
        "description": "Microsoft Phi-1.5 (1.3B parameters), excellent for its size",
        "min_vram": 3,  # 3GB minimum
        "tags": ["medium"]
    },
    {
        "name": "microsoft/phi-2",
        "description": "Microsoft Phi-2 (2.7B parameters), very strong performance",
        "min_vram": 5,  # 5GB minimum
        "tags": ["large", "chat-tuned"]
    },
    {
        "name": "stabilityai/stablelm-2-1_6b",
        "description": "StableLM 2 (1.6B parameters), balanced performance",
        "min_vram": 3,  # 3GB minimum
        "tags": ["medium"]
    },
    {
        "name": "stabilityai/stablelm-3b-4e1t",
        "description": "StableLM (3B parameters), good performance",
        "min_vram": 6,  # 6GB minimum
        "tags": ["large"]
    },
    {
        "name": "google/gemma-2b",
        "description": "Google Gemma (2B parameters), high-quality small model",
        "min_vram": 4,  # 4GB minimum
        "tags": ["medium", "chat-tuned"]
    },
    {
        "name": "google/gemma-2b-it",
        "description": "Instruction-tuned Google Gemma (2B parameters)",
        "min_vram": 4,  # 4GB minimum
        "tags": ["medium", "instruction-tuned"]
    },
    {
        "name": "google/gemma-3-1b-it",
        "description": "Instruction-tuned Google Gemma 3 (1B parameters), latest generation",
        "min_vram": 3,  # 3GB minimum
        "tags": ["medium", "instruction-tuned"]
    },
    {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Mistral 7B, can work with 8GB VRAM using quantization",
        "min_vram": 7,  # 7GB minimum
        "tags": ["very-large"]
    }
]

def get_available_vram() -> float:
    """
    Get available VRAM in GB.
    
    Returns:
        Available VRAM in GB, or 0 if no GPU is available
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        # Get total VRAM
        device = torch.cuda.current_device()
        vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
        
        # Get cached/allocated VRAM
        vram_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
        vram_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # Convert to GB
        
        # Calculate available VRAM (approximation)
        vram_available = vram_total - max(vram_allocated, vram_reserved)
        
        return vram_available
    except Exception as e:
        print(f"Error getting VRAM: {e}")
        return 0

def filter_models_by_vram(models: List[Dict[str, Any]], available_vram: float) -> List[Dict[str, Any]]:
    """
    Filter models based on available VRAM.
    
    Args:
        models: List of model info dictionaries
        available_vram: Available VRAM in GB
        
    Returns:
        Filtered list of models
    """
    return [model for model in models if model["min_vram"] <= available_vram]

def filter_models_by_tag(models: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    """
    Filter models based on tag.
    
    Args:
        models: List of model info dictionaries
        tag: Tag to filter on
        
    Returns:
        Filtered list of models
    """
    return [model for model in models if tag in model["tags"]]

def download_model(model_name: str, cache_dir: str = CACHE_DIR):
    """
    Download model and tokenizer to cache.
    
    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache the model
    """
    print(f"Downloading model: {model_name}")
    
    # Get HuggingFace token if available
    token = load_hf_token()
    
    # Configure tokenizer options
    tokenizer_kwargs = {
        "cache_dir": MODELS_CACHE_DIR,
    }
    if token:
        tokenizer_kwargs["token"] = token
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Check if we have enough VRAM for full precision
    available_vram = get_available_vram()
    use_fp16 = available_vram > 0  # Use fp16 if we have a GPU
    
    # Configure model loading options
    model_kwargs = {
        "cache_dir": MODELS_CACHE_DIR,
    }
    if token:
        model_kwargs["token"] = token
    
    # Download model with appropriate precision
    if use_fp16:
        print(f"Using float16 precision (GPU mode)")
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs
        )
    else:
        print(f"Using CPU mode")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    
    print(f"Model and tokenizer successfully downloaded to {MODELS_CACHE_DIR}")

def list_downloaded_models(cache_dir: str = MODELS_CACHE_DIR) -> List[str]:
    """
    List models that have already been downloaded.
    
    Args:
        cache_dir: Cache directory for models
        
    Returns:
        List of downloaded model names
    """
    if not os.path.exists(cache_dir):
        return []
    
    # Look for model directories with the pattern "models--org--name"
    models = []
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if os.path.isdir(item_path) and item.startswith("models--"):
            # Extract org/model from directory name (e.g., "models--facebook--opt-125m" -> "facebook/opt-125m")
            parts = item.split("--")
            if len(parts) >= 3:
                # Skip the "models" prefix and join the rest with "/"
                model_name = "/".join(parts[1:])
                models.append(model_name)
    
    return models

# Configuration functionality removed as requested

def main():
    parser = argparse.ArgumentParser(description="LLM Model Selector")
    parser.add_argument("--list", action="store_true", help="List all recommended models")
    parser.add_argument("--filter", type=str, help="Filter models by tag (e.g., small, medium, large)")
    parser.add_argument("--download", type=str, help="Download a specific model")
    # Removed set-default argument as requested
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Cache directory for models")
    
    args = parser.parse_args()
    
    # Check available VRAM
    available_vram = get_available_vram()
    print(f"Available VRAM: {available_vram:.2f}GB")
    
    # Filter models based on available VRAM
    compatible_models = filter_models_by_vram(RECOMMENDED_MODELS, available_vram)
    
    if args.list:
        # If tag filter is provided, apply it
        if args.filter:
            compatible_models = filter_models_by_tag(compatible_models, args.filter)
            print(f"Models compatible with your GPU (filtered by tag '{args.filter}'):")
        else:
            print("Models compatible with your GPU:")
            
        if not compatible_models:
            print("No compatible models found for your GPU and filter criteria.")
            return
            
        for i, model in enumerate(compatible_models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Description: {model['description']}")
            print(f"   Min VRAM: {model['min_vram']}GB")
            print(f"   Tags: {', '.join(model['tags'])}")
            print()
    
    elif args.download:
        download_model(args.download, args.cache_dir)
    
    # set_default functionality removed
    
    else:
        # Interactive mode
        print("\nRecommended models for your system:")
        for i, model in enumerate(compatible_models, 1):
            print(f"{i}. {model['name']} - {model['description']}")
        
        print("\nAlready downloaded models:")
        downloaded = list_downloaded_models(MODELS_CACHE_DIR)
        if downloaded:
            for i, model in enumerate(downloaded, 1):
                print(f"{i}. {model}")
        else:
            print("No models downloaded yet.")
        
        print("\nOptions:")
        print("1. List all compatible models")
        print("2. Download a model")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            print("\nCompatible models:")
            for i, model in enumerate(compatible_models, 1):
                print(f"{i}. {model['name']}")
                print(f"   Description: {model['description']}")
                print(f"   Min VRAM: {model['min_vram']}GB")
                print(f"   Tags: {', '.join(model['tags'])}")
                print()
                
        elif choice == "2":
            print("\nSelect a model to download:")
            for i, model in enumerate(compatible_models, 1):
                print(f"{i}. {model['name']} - {model['description']}")
                
            model_idx = int(input("\nEnter model number: ")) - 1
            if 0 <= model_idx < len(compatible_models):
                download_model(compatible_models[model_idx]["name"], args.cache_dir)
            else:
                print("Invalid model number.")
                
        elif choice == "3":
            # Exit the program
            print("Exiting model selector.")

if __name__ == "__main__":
    main()
