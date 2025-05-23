import os
import datetime
import argparse
from src.llm_watermark import LLMWatermarker
from src.utils import get_random_essay
from src.utils import save_to_file
import src.paths as paths


def main():
    parser = argparse.ArgumentParser(description="LLM Watermarking Implementation")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green tokens")
    parser.add_argument("--seed", type=int, default=4242, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay if not provided)")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--output", type=str, help="Custom filename for output in the output/ directory (if not specified, a filename will be auto-generated)")
    parser.add_argument("--context-window", type=int, default=1024, help="Maximum number of tokens to use as context for generation (default: 1024)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)")
    
    args = parser.parse_args()

    # Set global cache
    paths.set_cache_dir(args.cache_dir)
    paths.ensure_directories()
    
    # Determine device
    device = "cpu" if args.no_cuda else None
    
    # Initialize the watermarker
    watermarker = LLMWatermarker(
        model_name=args.model,
        green_list_fraction=args.green_fraction,
        bias=args.bias,
        seed=args.seed,
        cache_dir=paths.CACHE_DIR,
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
        output_path = os.path.join(paths.OUTPUT_DIR, args.output)
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
        output_path = os.path.join(paths.OUTPUT_DIR, output_file)
        # Stats for file output with all parameters
        save_to_file(prompt, generated_text, stats, output_path, args.seed, args.model,
                    args.context_window, args.bias, args.green_fraction, args.temperature,
                    args.hash_window)
        print(f"\nOutput saved to: {output_path}")
    

if __name__ == "__main__":
    main()
