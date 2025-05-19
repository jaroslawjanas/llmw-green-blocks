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

# Recommended models with VRAM requirements
RECOMMENDED_MODELS = [
    # Microsoft Phi-4 models
    {
        "name": "microsoft/Phi-4-mini-reasoning",
        "description": "Microsoft Phi-4 Mini Reasoning (3.8B parameters), compact but powerful reasoning capabilities",
        "min_vram": 6,  # 6GB minimum
        "tags": ["medium", "reasoning"]
    },
    {
        "name": "microsoft/phi-4",
        "description": "Microsoft Phi-4 base model (14.7B parameters), versatile foundation model with strong general capabilities",
        "min_vram": 20,  # 20GB minimum, similar to other 14B models
        "tags": ["very-large", "base-model"]
    },
    {
        "name": "microsoft/Phi-4-reasoning",
        "description": "Microsoft Phi-4 Reasoning (14.7B parameters), powerful reasoning and logic capabilities",
        "min_vram": 12,  # 12GB minimum
        "tags": ["large", "reasoning"]
    },
    {
        "name": "microsoft/Phi-4-reasoning-plus",
        "description": "Microsoft Phi-4 Reasoning Plus (14.7B parameters), enhanced reasoning with expanded capabilities",
        "min_vram": 20,  # 20GB minimum
        "tags": ["very-large", "reasoning"]
    },
    
    # Qwen3 models
    {
        "name": "Qwen/Qwen3-4B",
        "description": "Qwen 3 (4B parameters), efficient general-purpose model with strong multilingual capabilities",
        "min_vram": 8,  # 8GB minimum
        "tags": ["medium", "multilingual"]
    },
    {
        "name": "Qwen/Qwen3-8B",
        "description": "Qwen 3 (8B parameters), balanced performance with excellent multilingual support",
        "min_vram": 14,  # 14GB minimum
        "tags": ["large", "multilingual"]
    },
    {
        "name": "Qwen/Qwen3-14B",
        "description": "Qwen 3 (14B parameters), high-quality generation with advanced capabilities",
        "min_vram": 20,  # 20GB minimum
        "tags": ["very-large", "multilingual"]
    },
    
    # Google Gemma 3 models
    {
        "name": "google/gemma-3-1b-it",
        "description": "Instruction-tuned Google Gemma 3 (1B parameters), compact and efficient",
        "min_vram": 3,  # 3GB minimum
        "tags": ["small", "instruction-tuned"]
    },
    {
        "name": "google/gemma-3-4b-it",
        "description": "Instruction-tuned Google Gemma 3 (4B parameters), good balance of size and capability",
        "min_vram": 8,  # 8GB minimum
        "tags": ["medium", "instruction-tuned"]
    },
    {
        "name": "google/gemma-3-12b-it",
        "description": "Instruction-tuned Google Gemma 3 (12B parameters), high-quality responses with strong reasoning",
        "min_vram": 18,  # 18GB minimum
        "tags": ["very-large", "instruction-tuned"]
    },
    {
        "name": "google/gemma-3-27b-it",
        "description": "Instruction-tuned Google Gemma 3 (27B parameters), powerful capabilities with extensive knowledge",
        "min_vram": 40,  # 40GB minimum
        "tags": ["enormous", "instruction-tuned"]
    },
    
    # Meta Llama models
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta Llama 3.1 (8B parameters), instruction-tuned for chat and reasoning",
        "min_vram": 14,  # 14GB minimum
        "tags": ["large", "instruction-tuned"]
    },
    {
        "name": "meta-llama/Llama-3.2-1B",
        "description": "Meta Llama 3.2 (1B parameters), compact base model with impressive capabilities",
        "min_vram": 2,  # 2GB minimum
        "tags": ["small", "base-model"]
    },
    {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "description": "Meta Llama 3.2 (1B parameters), instruction-tuned for direct responses",
        "min_vram": 2,  # 2GB minimum
        "tags": ["small", "instruction-tuned"]
    },
    {
        "name": "meta-llama/Llama-3.2-3B",
        "description": "Meta Llama 3.2 (3B parameters), efficient base model with strong general capabilities",
        "min_vram": 6,  # 6GB minimum
        "tags": ["medium", "base-model"]
    },
    {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Meta Llama 3.2 (3B parameters), instruction-tuned for chat and assistance",
        "min_vram": 6,  # 6GB minimum
        "tags": ["medium", "instruction-tuned"]
    },
    {
        "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "description": "Meta Llama 3.2 (11B parameters), vision-language model for multimodal chat and image understanding",
        "min_vram": 16,  # 16GB minimum
        "tags": ["large", "instruction-tuned", "vision", "multimodal"]
    },
    {
        "name": "meta-llama/Llama-3.3-70B-Instruct",
        "description": "Meta Llama 3.3 (70B parameters), large-scale instruction-tuned model with remarkable capabilities",
        "min_vram": 80,  # 80GB minimum
        "tags": ["enormous", "instruction-tuned"]
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E",
        "description": "Meta Llama 4 Scout (17B parameters), high-performance base model with cutting-edge capabilities",
        "min_vram": 24,  # 24GB minimum
        "tags": ["very-large", "base-model"]
    },
    {
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "description": "Meta Llama 4 Scout (17B parameters), instruction-tuned for advanced reasoning and assistance",
        "min_vram": 24,  # 24GB minimum
        "tags": ["very-large", "instruction-tuned"]
    }
]

def get_gpu_info() -> tuple:
    """
    Get information about available GPUs.
    
    Returns:
        Tuple of (num_gpus, min_gpu_vram, total_vram)
        where min_gpu_vram is the VRAM of the GPU with least memory in GB
        and total_vram is the sum of available VRAM across all GPUs
    """
    if not torch.cuda.is_available():
        return 0, 0, 0
    
    try:
        # Get number of GPUs
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            return 0, 0, 0
            
        # Calculate available VRAM for each GPU
        gpu_vram_available = []
        
        for device in range(num_gpus):
            # Get total VRAM
            vram_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # Convert to GB
            
            # Get cached/allocated VRAM
            vram_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # Convert to GB
            vram_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # Convert to GB
            
            # Calculate available VRAM (approximation)
            vram_available = vram_total - max(vram_allocated, vram_reserved)
            gpu_vram_available.append(vram_available)
        
        # Find minimum VRAM among all GPUs and calculate total VRAM
        min_gpu_vram = min(gpu_vram_available) if gpu_vram_available else 0
        total_vram = sum(gpu_vram_available)
        
        return num_gpus, min_gpu_vram, total_vram
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return 0, 0, 0

def get_available_vram() -> float:
    """
    Get total available VRAM in GB across all GPUs.
    
    Returns:
        Total available VRAM in GB, or 0 if no GPU is available
    """
    _, _, total_vram = get_gpu_info()
    return total_vram

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
    parser.add_argument("--download", type=str, help="Download a model from the recommended list")
    parser.add_argument("--download-custom", type=str, help="Download any model by specifying its full name (e.g., 'organization/model-name')")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR, help="Cache directory for models")
    
    args = parser.parse_args()
    
    # Check available GPU resources
    num_gpus, min_gpu_vram, total_vram = get_gpu_info()
    
    if num_gpus > 0:
        print(f"GPUs: {num_gpus} x {min_gpu_vram:.2f}GB = {total_vram:.2f}GB")
    else:
        print("No GPU detected. Running in CPU mode.")
    
    # Filter models based on total available VRAM across all GPUs
    compatible_models = filter_models_by_vram(RECOMMENDED_MODELS, total_vram)
    
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
    
    elif args.download_custom:
        print(f"Attempting to download custom model: {args.download_custom}")
        try:
            download_model(args.download_custom, args.cache_dir)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Please check the model name and ensure you have the correct permissions.")
    
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
        print("2. Download a model from the list")
        print("3. Download a custom model by name")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
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
            # Download a custom model by name
            custom_model = input("\nEnter the model name (e.g., 'organization/model-name'): ")
            if custom_model:
                print(f"\nAttempting to download model: {custom_model}")
                try:
                    download_model(custom_model, args.cache_dir)
                except Exception as e:
                    print(f"Error downloading model: {e}")
                    print("Please check the model name and ensure you have the correct permissions.")
            else:
                print("No model name provided.")
                
        elif choice == "4":
            # Exit the program
            print("Exiting model selector.")

if __name__ == "__main__":
    main()
