import os
import sys
import datetime
import argparse
from collections import defaultdict
from src.llm_watermark import LLMWatermarker
from src.utils import get_shuffled_essays
from src.utils import save_generation_details, count_green_blocks, save_average_block_counts, Tee
import src.paths as paths
import src.model_selector


def main():
    parser = argparse.ArgumentParser(description="LLM Watermarking Implementation")
    parser.add_argument("--model-selector", action="store_true", help="Launch the model selector UI")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to use")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--green-fraction", type=float, default=0.5, help="Fraction of tokens in green list")
    parser.add_argument("--bias", type=float, default=6.0, help="Bias to add to green tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--prompt", type=str, help="Custom prompt (uses random essay if not provided)")
    parser.add_argument("--cache-dir", type=str, default=paths.CACHE_DIR, help="Cache directory for models and datasets")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--context-window", type=int, default=1500, help="Maximum number of tokens to use as context for generation (default: 1500)")
    parser.add_argument("--temperature", "--temp", type=float, default=0.0, help="Sampling temperature (default: 0.0 = greedy sampling, higher = more random)")
    parser.add_argument("--hash-window", type=int, default=1, help="Number of previous tokens to hash together (default: 1)")
    parser.add_argument("--block-size", type=int, nargs='+', default=[25], help="Size(s) of a green block to consider as intact (default: 25)")
    parser.add_argument("--n-prompts", type=int, default=1, help="Number of prompts to process (default: 1)")

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
    print(f"Loading model: {args.model}")
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
    
    model_name = args.model.split("/")[-1]
    
    # Create a timestamp for the entire batch
    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir_name = f"{model_name}-{batch_timestamp}"
    batch_output_dir = os.path.join(paths.OUTPUT_DIR, batch_output_dir_name)
    
    # Ensure the batch output directory exists
    os.makedirs(batch_output_dir)

    # --- Begin Tee logging setup ---
    log_file_path = os.path.join(batch_output_dir, f"{batch_output_dir_name}.log")
    original_stdout = sys.stdout
    try:
        log_file = open(log_file_path, "w", encoding="utf-8")
        sys.stdout = Tee(original_stdout, log_file)
    except Exception as e:
        print(f"Failed to set up logging to file: {e}")
    # --- End Tee logging setup ---

    # Get prompts
    if args.prompt:
        # Single custom prompt
        prompts = [args.prompt]
        print(f"\n--- Using Custom Prompt ---")
        print(args.prompt[:200] + "..." if len(args.prompt) > 200 else args.prompt)
        print("---------------------------\n")
    else:
        # Get shuffled essays from dataset
        prompts = get_shuffled_essays(seed=args.seed, n_prompts=args.n_prompts)
        if args.n_prompts == 1:
            print("\n--- Random Essay Prompt ---")
            print(prompts[0][:200] + "..." if len(prompts[0]) > 200 else prompts[0])
            print("---------------------------\n")
        else:
            print(f"\n--- Processing {args.n_prompts} Shuffled Essays ---")
            print(f"First prompt preview: {prompts[0][:100] + '...' if len(prompts[0]) > 100 else prompts[0]}")
            print("---------------------------\n")
    
    # Process each prompt
    total_prompts = len(prompts)

    print(f"Generating {args.max_tokens} tokens per prompt with watermarking...")
    print(f"Processing {total_prompts} prompt(s) with model: {args.model}\n")
    print(f"Saving outputs to: {batch_output_dir}\n")
    
    output_paths = []
    
    # Initialize defaultdict to aggregate block counts for averaging
    # Keys will be block_size, values will be lists of b_count for that size
    aggregated_block_counts = defaultdict(list)

    for prompt_idx, prompt in enumerate(prompts, 1):
        if total_prompts > 1:
            print(f"\n{'='*60}")
            print(f"Processing Prompt {prompt_idx}/{total_prompts}")
            print(f"{'='*60}")
            print(f"Prompt preview: {prompt[:100] + '...' if len(prompt) > 100 else prompt}")
            print()
        
        # Generate text for this prompt
        generated_text, stats, green_red_mask = watermarker.generate_text(
            prompt=prompt,
            max_new_tokens=args.max_tokens
        )

        # Find the number of intact blocks for each block size
        block_counts = count_green_blocks(green_red_mask, args.block_size)
        
        # Aggregate block counts for batch averaging
        for b_size, b_count in block_counts:
            aggregated_block_counts[b_size].append(b_count)

        # Print results for this prompt
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
        for b_size, b_count in block_counts:
            print(f"Block count (size {b_size}): {b_count}")
        print(f"\nSeed: {args.seed}")
        print(f"Context window: {args.context_window}")
        print(f"Bias: {args.bias}")
        print(f"Green fraction: {args.green_fraction}")
        print(f"Temperature: {args.temperature}")
        print(f"Hash window: {args.hash_window}")
        print("---------------------------")
        
        # Generate output filename within the batch directory
        output_file = f"prompt{prompt_idx:05d}.txt"
        output_filepath = os.path.join(batch_output_dir, output_file)

        # Save output for this prompt
        save_generation_details(
            prompt          = prompt,
            generated_text  = generated_text,
            stats           = stats,
            green_red_mask  = green_red_mask,
            block_counts    = block_counts,
            file_path       = output_filepath,
            seed            = args.seed,
            model_name      = args.model,
            context_window  = args.context_window,
            bias            = args.bias,
            green_fraction  = args.green_fraction,
            temperature     = args.temperature,
            hash_window     = args.hash_window
        )
        
        output_paths.append(output_filepath)
        print(f"\nOutput saved to: {output_filepath}")
    
    # Calculate average block counts for the entire batch
    average_block_counts = {}
    for b_size, counts in aggregated_block_counts.items():
        average_block_counts[b_size] = sum(counts) / len(counts)

    # Print average block counts to console
    print("\n--- AVERAGE BLOCK COUNTS PER BLOCK SIZE (BATCH SUMMARY) ---")
    print(f"Model: {args.model}")
    print(f"Total Prompts Processed: {total_prompts}")
    print(f"Block Sizes Analyzed: {args.block_size}\n")
    for b_size in sorted(average_block_counts.keys()):
        print(f"Average Block Count (size {b_size}): {average_block_counts[b_size]:.4f}")
    print("-----------------------------------------------------------\n")

    # Save average block counts to a separate file using the utility function
    save_average_block_counts(
        average_block_counts=average_block_counts,
        batch_output_dir=batch_output_dir,
        model_name=args.model,
        total_prompts=total_prompts,
        block_sizes_analyzed=args.block_size
    )


    # Final summary
    if total_prompts > 0:
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Processed {total_prompts} prompt(s) successfully")
        print(f"Model: {args.model}")
        print(f"Output directory: {batch_output_dir}")
        print(f"Output files:")
        for i, output_path in enumerate(output_paths, 1):
            print(f"  {i}. {os.path.basename(output_path)}")
        print(f"  Batch average block counts: average_block_counts.txt") # Hardcode filename as it's fixed in the utility
        print(f"{'='*60}")

    # --- Restore original stdout and close log file ---
    try:
        sys.stdout = original_stdout
        if 'log_file' in locals():
            log_file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
