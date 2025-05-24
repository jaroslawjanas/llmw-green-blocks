# LLM Watermarking Implementation

This project implements the red-green token watermarking technique proposed by [Kirchenbauer et al. (2023)](https://arxiv.org/abs/2301.10226). The implementation focuses on tracking how many red and green tokens are selected during text generation with a special interest in analyzing the red token selection patterns.

## Overview

The watermarking technique works as follows:

1. Generate logits from the model
2. Randomly split the vocabulary into "green" and "red" token lists based on the hash of the previous token(s)
3. Add a bias value (typically 6.0) to the logits of green tokens
4. Apply softmax to convert logits to probabilities
5. Select the token with highest probability (or sample based on temperature)
6. Track how many green vs. red tokens are selected during generation

Also supports:
- Variable number of previous tokens to use for hashing (--hash-window)
- Temperature-based sampling (--temperature)

This technique increases the likelihood of selecting tokens from the green list, creating a statistical pattern that can be analyzed.

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU with 8GB VRAM (for larger models)
- Windows 10 (other platforms should work but are untested)
- Hugging Face account (for accessing models)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/jaroslawjanas/llmw-red-freq.git
   cd llmw-red-freq
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate llm-watermark
   ```

3. Set up HuggingFace authentication:
   ```bash
   # Rename the template file
   cp hf_token.template hf_token
   # Then edit the file and replace with your token
   ```
   You can get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens)

## Usage

### Selecting a Model

Use the model selector to choose an appropriate model for your hardware. Running it without arguments launches an interactive tool:

```bash
python main.py --model-selector
```

This interactive tool will:
- Check your available VRAM
- Recommend compatible models
- Allow you to download models
- Provide options to list all compatible models, download a model from the list, or download a custom model by name.

Alternatively, use command-line options:

```bash
# List all compatible models
python main.py --model-selector --list

# Filter by tag (small, medium, large)
python main.py --model-selector --list --filter medium

# Download a specific model
python main.py --model-selector --download facebook/opt-1.3b
```

### Running the Watermarker

Basic usage:

```bash
python llm_watermark.py
```

This will:
1. Use the default model (or facebook/opt-125m if none set)
2. Choose a random essay from the [essays-with-instructions](https://huggingface.co/datasets/ChristophSchuhmann/essags-with-instructions) dataset
3. Generate 100 tokens with watermarking
4. Show statistics about green vs. red token selection, including a detailed timing summary

Advanced options:

```bash
python llm_watermark.py --model facebook/opt-1.3b --max-tokens 200 --bias 8.0 --prompt "Your custom prompt here"
```

All options:

```
--model MODEL           Model to use
--max-tokens MAX_TOKENS Maximum tokens to generate (default: 100)
--green-fraction GREEN_FRACTION Fraction of tokens in green list (default: 0.5)
--bias BIAS             Bias to add to green tokens (default: 6.0)
--seed SEED             Random seed for reproducibility (default: 4242)
--prompt PROMPT         Custom prompt (uses random essay if not provided)
--cache-dir CACHE_DIR   Cache directory for models (optional)
--no-cuda               Disable CUDA even if available
--output OUTPUT         Custom filename for output in the output/ directory (if not specified, a filename will be auto-generated)
--context-window CONTEXT_WINDOW   Maximum number of tokens to use as context for generation (default: 1500)
--temperature, --temp TEMPERATURE Sampling temperature (default: 0.0 = greedy sampling, higher = more random)
--hash-window HASH_WINDOW         Number of previous tokens to hash together (default: 1)
--block-size BLOCK_SIZE           Size of a green block to consider as intact (default: 25)
```

Note: All watermarking results are automatically saved to the `output/` directory. If you don't specify a filename with `--output`, a filename will be automatically generated based on the model name and a timestamp (format: `modelname_gen_YYYYMMDD_HHMMSS.txt`). This ensures that multiple runs don't overwrite each other.

## Examples

### Basic Watermarking Example

```bash
python llm_watermark.py --model facebook/opt-125m --max-tokens 50
```

Sample output:
```
--- Watermark Statistics ---
Model: facebook/opt-125m
Green tokens: 42
Red tokens: 8
Total tokens: 50
Green ratio: 0.8400
Block count: 1
Seed: 4242
Context window: 1500
Bias: 6.0
Green fraction: 0.5
Temperature: 0.0
Block size: 25
Hash window: 1
---------------------------
```

### Varying the Bias Parameter

Lower bias results in more red tokens being selected:
```bash
python llm_watermark.py --bias 2.0
```

Higher bias suppresses red token selection more aggressively:
```bash
python llm_watermark.py --bias 10.0
```

## Model Compatibility

The model selector now includes a curated selection of the latest models with their VRAM requirements:

| Model Family | Examples | Min VRAM | Notes |
|--------------|----------|----------|-------|
| Small (1-3B) | meta-llama/Llama-3.2-1B, google/gemma-3-1b-it | 2-3GB | Works on low-end GPUs |
| Medium (3-4B) | microsoft/Phi-4-mini-reasoning, meta-llama/Llama-3.2-3B, google/gemma-3-4b-it | 6-8GB | Good for 8GB GPUs |
| Large (7-8B) | Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct | 12-14GB | Requires mid-range GPU |
| Very Large (12-14B) | microsoft/Phi-4-reasoning-plus, Qwen/Qwen3-14B, google/gemma-3-12b-it | 18-20GB | Needs high-end GPU |
| Enormous (27-70B) | google/gemma-3-27b-it, meta-llama/Llama-3.3-70B-Instruct | 40-80GB | Requires professional GPU |

### Important Note on Model Licensing

Some models (such as Gemma, Llama, and certain Mistral models) require you to accept a usage license before downloading. When attempting to use these models:

1. You may need to visit the model's page on Hugging Face first (e.g., [google/gemma-2b](https://huggingface.co/google/gemma-2b))
2. Read and accept the license terms on the Hugging Face website
3. Ensure you're logged in with the same account whose token you've configured in the `hf_token` file

If you encounter errors like "403 Client Error: Forbidden" when downloading a model, it typically means you need to accept the model's license agreement first.

## How It Works

### Prompt Formatting

The system automatically formats prompts to match the specific requirements of different models, ensuring compatibility and optimal performance.

The watermarking algorithm:

1. For each generated token, a window of previous token IDs is used to seed a random number generator
   - The window size is controlled by the `--hash-window` parameter (default: 1 token)
   - Using more tokens creates a more context-dependent watermark
2. The vocabulary is randomly partitioned into two sets: green (50% by default) and red (50%)
3. A bias value (default: 6.0) is added to the logits of all green tokens
4. With temperature=0, the highest probability token is selected; with temperature>0, tokens are sampled from the adjusted distribution
5. This increases the probability of selecting green tokens and decreases the likelihood of red token selection

A higher green ratio indicates stronger watermarking effect.

### About the Seed Parameter

The `--seed` parameter (default: 4242) controls randomness throughout the entire watermarking process:

1. **Model Initialization**: Sets random seeds for PyTorch, NumPy, and Python's random module to ensure deterministic behavior
2. **Prompt Selection**: When no custom prompt is provided, the seed determines which random essay is selected
3. **Model Behavior**: Affects any stochastic elements in the model (such as dropout)
4. **Reproducibility**: Using the same seed, model, and parameters will produce identical output

Note that the seed does NOT directly influence the token partitioning into green/red lists. That process uses a deterministic hash of the previous token ID, which is independent of the global random seed.

The seed value is recorded in both console output and saved output files to enable exact reproduction of results.

## References

- [Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Goldstein, T., & Miers, I. (2023). A Watermark for Large Language Models.](https://arxiv.org/abs/2301.10226)

## Disclaimer

This project was developed with assistance fro m Claude 3.7 Sonnet via the Cline VSCode extension. All code and documentation were reviewed and approved by a human developer, with proper oversight maintained throughout the development process.
