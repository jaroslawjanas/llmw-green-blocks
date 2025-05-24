import os
import datetime
import argparse
from src.llm_watermark import LLMWatermarker
from src.utils import get_random_essay
from src.utils import save_to_file, count_green_blocks
import src.paths as paths
import src.model_selector


def main():
    parser = argparse.ArgumentParser(description="LLM Watermarking Implementation")
    parser.add_argument("--model-selector", action="store_true", help="Launch the model selector UI")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green tokens")
    parser.add_argument("--seed", type=int, default=4242, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay if not provided)")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--output", type=str, help="Custom filename for output in the output/ directory (if not specified, a filename will be auto-generated)")
    parser.add_argument("--context-window", type=int, default=1500, help="Maximum number of tokens to use as context for generation (default: 1500)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)")
    parser.add_argument("--block-size", type=int, default=25, help="Size of a green block to consider as intact (default: 25)")

    args, remaining_argv = parser.parse_known_args()

    if args.model_selector:
        # Pass the remaining arguments and the cache-dir to the model_selector's main function
        # This allows model_selector to parse its own arguments (e.g., --list, --download)
        model_selector_args = remaining_argv
        if args.cache_dir:
            model_selector_args.extend(["--cache-dir", args.cache_dir])
        src.model_selector.main(model_selector_args)
        return

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
    generated_text, stats, green_red_mask = watermarker.generate_text(
        prompt=prompt,
        max_new_tokens=args.max_tokens
    )

    # Find the number of intact blocks
    block_count = count_green_blocks(green_red_mask, args.block_size)
    
    # Print results
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n--- Green/Red Mask ---")
    print(green_red_mask)
    print("---------------------\n")
    
    print("--- Watermark Statistics ---")
    print(f"Model: {args.model}\n")
    print(f"Green tokens: {stats['green_tokens']}")
    print(f"Red tokens: {stats['red_tokens']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Green ratio: {stats['green_ratio']:.4f}")
    print(f"Block count: {block_count}\n")
    print(f"Seed: {args.seed}")
    print(f"Context window: {args.context_window}")
    print(f"Bias: {args.bias}")
    print(f"Green fraction: {args.green_fraction}")
    print(f"Temperature: {args.temperature}")
    print(f"Block size: {args.block_size}")
    print(f"Hash window: {args.hash_window}")
    print("---------------------------")
    
    # Save output under specific filename
    if args.output:
        output_path = os.path.join(paths.OUTPUT_DIR, args.output)
    else:
        # Generate a default filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split("/")[-1]
        output_file = f"{model_name}_gen_{timestamp}.txt"
        output_path = os.path.join(paths.OUTPUT_DIR, output_file)

    save_to_file(
        prompt          = prompt,
        generated_text  = generated_text,
        stats           = stats,
        green_red_mask  = green_red_mask,
        block_count     = block_count,
        output_file     = output_path,
        seed            = args.seed,
        model_name      = args.model,
        context_window  = args.context_window,
        bias            = args.bias,
        green_fraction  = args.green_fraction,
        temperature     = args.temperature,
        block_size      = args.block_size,
        hash_window     = args.hash_window
    )
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
