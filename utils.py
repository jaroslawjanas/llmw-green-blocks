#!/usr/bin/env python3
"""
Utility functions for LLM Watermarking.
"""

import os

def load_hf_token():
    """Load HuggingFace token from hf_token file."""
    token_path = "hf_token"
    token = None
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
    return token
