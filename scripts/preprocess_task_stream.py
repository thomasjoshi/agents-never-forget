#!/usr/bin/env python
"""
Preprocess SWE-Bench-CL Task Stream for faster loading.

This script converts the JSON task stream into a more efficient binary format
and performs preprocessing steps to accelerate training.
"""

import os
import sys
import json
import pickle
import time
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def process_task_chunk(task_chunk, verbose=False):
    """
    Process a chunk of tasks in parallel.
    
    Args:
        task_chunk: Dictionary of tasks to process
        verbose: Whether to print verbose output
        
    Returns:
        Processed task chunk
    """
    processed_chunk = {}
    
    for task_name, examples in task_chunk.items():
        # Skip if examples is not a list
        if not isinstance(examples, list):
            if verbose:
                print(f"Warning: Task '{task_name}' does not contain a list of examples, skipping...")
            continue
            
        # Process each example
        processed_examples = []
        for example in examples:
            # Validate example
            if not isinstance(example, dict):
                continue
                
            # Perform any preprocessing on the example here
            # For example, this could include:
            # 1. Removing unnecessary fields to reduce memory
            # 2. Preprocessing code snippets
            # 3. Formatting data consistently
            
            processed_examples.append(example)
            
        if processed_examples:
            processed_chunk[task_name] = processed_examples
            
    return processed_chunk

def preprocess_task_stream(input_file, output_file, num_processes=None, verbose=True):
    """
    Preprocess the task stream for faster loading.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the preprocessed binary file
        num_processes: Number of processes to use for parallel processing
        verbose: Whether to print verbose output
        
    Returns:
        None
    """
    start_time = time.time()
    
    if verbose:
        print(f"Preprocessing task stream: {input_file} -> {output_file}")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Load the task stream
    if verbose:
        print(f"Loading JSON data from {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            task_stream = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file {input_file}: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read file {input_file}: {str(e)}")
        sys.exit(1)
    
    # Extract metadata if present
    metadata = task_stream.pop('metadata', {})
    metadata['preprocessed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    metadata['original_file'] = os.path.basename(input_file)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), 8)  # Use up to 8 cores
    
    if verbose:
        print(f"Using {num_processes} processes for parallel processing")
    
    # Split tasks into chunks for parallel processing
    tasks = list(task_stream.items())
    chunk_size = max(1, len(tasks) // num_processes)
    chunks = [dict(tasks[i:i+chunk_size]) for i in range(0, len(tasks), chunk_size)]
    
    # Process chunks in parallel
    if verbose:
        print(f"Processing {len(task_stream)} tasks in parallel...")
    
    with mp.Pool(num_processes) as pool:
        process_func = partial(process_task_chunk, verbose=verbose)
        results = pool.map(process_func, chunks)
    
    # Combine results
    processed_tasks = {}
    for result in results:
        processed_tasks.update(result)
    
    # Get task ordering (from metadata or create sorted list)
    task_ordering = metadata.get('task_ordering', sorted(processed_tasks.keys()))
    
    # Update metadata
    metadata['task_ordering'] = task_ordering
    metadata['num_tasks'] = len(task_ordering)
    metadata['total_examples'] = sum(len(examples) for examples in processed_tasks.values())
    
    # Create final processed data
    processed_data = {
        **processed_tasks,
        'metadata': metadata
    }
    
    # Save as pickle
    if verbose:
        print(f"Saving preprocessed data to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f, protocol=4)  # Protocol 4 is compatible with most Python 3 versions
    
    # Compute statistics
    elapsed_time = time.time() - start_time
    file_size_original = os.path.getsize(input_file) / (1024 * 1024)  # MB
    file_size_processed = os.path.getsize(output_file) / (1024 * 1024)  # MB
    
    if verbose:
        print(f"\nPreprocessing complete in {elapsed_time:.2f} seconds")
        print(f"Original size: {file_size_original:.2f} MB")
        print(f"Processed size: {file_size_processed:.2f} MB")
        print(f"Size reduction: {100 * (1 - file_size_processed/file_size_original):.2f}%")
        print(f"Tasks: {len(task_ordering)}")
        print(f"Total examples: {metadata['total_examples']}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess SWE-Bench-CL Task Stream for faster loading")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", help="Path to save the preprocessed binary file (default: input file with .pkl extension)")
    parser.add_argument("--processes", type=int, help="Number of processes to use for parallel processing")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + ".pkl"
    
    preprocess_task_stream(args.input, args.output, args.processes, verbose=not args.quiet)

if __name__ == "__main__":
    main()
