#!/usr/bin/env python
"""
Generate continual learning task splits from SWE-Bench-Verified dataset.

This script creates sequential task splits for continual learning evaluation,
either by repository or chronological order.
"""

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# For reproducibility
np.random.seed(42)

def load_swe_bench_verified(input_file):
    """
    Load the SWE-Bench-Verified dataset from a file or HuggingFace.
    
    Args:
        input_file: Path to dataset file or HuggingFace dataset identifier
    
    Returns:
        DataFrame containing the SWE-Bench-Verified data
    """
    try:
        print(f"Loading data from {input_file}...")
        
        # Check if it's a local file first
        if os.path.exists(input_file):
            if input_file.endswith('.json'):
                print("Loading local JSON file...")
                with open(input_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif input_file.endswith('.parquet'):
                print("Loading local Parquet file...")
                df = pd.read_parquet(input_file)
            else:
                raise ValueError("Local file must be .json or .parquet format")
        else:
            # Not a local file, try to load as a HuggingFace dataset
            try:
                from datasets import load_dataset
                print("Loading HuggingFace dataset...")
                
                # Clean up the dataset path if needed
                if input_file.startswith("hf://"):
                    dataset_path = input_file[5:]
                else:
                    dataset_path = input_file
                
                # For SWE-bench_Verified, try to use the correct splits
                try:
                    # First try direct loading (works with princeton-nlp/SWE-bench_Verified)
                    dataset = load_dataset(dataset_path)
                    print(f"Available splits: {list(dataset.keys())}")
                    
                    # Choose the right split (test for SWE-bench_Verified)
                    if 'test' in dataset:
                        split_name = 'test'
                    else:
                        split_name = list(dataset.keys())[0]
                    
                    print(f"Using split: {split_name}")
                    df = dataset[split_name].to_pandas()
                except Exception as e:
                    print(f"Error with direct dataset loading: {e}")
                    print("Trying to load specific data file...")
                    
                    # Try with explicit data file pattern (works with explicit file paths)
                    if 'SWE-bench_Verified' in dataset_path:
                        # Use the test split for SWE-bench_Verified
                        dataset = load_dataset(dataset_path, data_files={'test': 'test-00000-of-00001.parquet'})
                        df = dataset['test'].to_pandas()
                    else:
                        # Generic approach for other datasets
                        dataset = load_dataset(dataset_path)
                        split_name = list(dataset.keys())[0]
                        df = dataset[split_name].to_pandas()
            
            except ImportError:
                print("Error: To use HuggingFace datasets, install required packages with:")
                print("pip install datasets huggingface_hub")
                sys.exit(1)
            except Exception as e:
                print(f"Error accessing dataset: {e}")
                print("Troubleshooting tips:")
                print("1. If this is an authentication issue, run 'huggingface-cli login' first")
                print("2. Check that the dataset name is correct")
                print("3. Try downloading the dataset locally first")
                sys.exit(1)
        
        # Display basic information
        print(f"Total issues: {len(df)}")
        print(f"Repositories: {df['repo'].nunique()}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Data Preprocessing
        # Convert created_at to datetime
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Clean up any potential issues with the data
        required_columns = ['repo', 'instance_id', 'base_commit', 'problem_statement']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Required column '{col}' not found in dataset")
        
        # Filter for rows that have all required columns
        df = df.dropna(subset=[col for col in required_columns if col in df.columns])
        
        # Ensure FAIL_TO_PASS and PASS_TO_PASS are properly loaded as lists
        if 'FAIL_TO_PASS' in df.columns:
            df['FAIL_TO_PASS_list'] = df['FAIL_TO_PASS'].apply(parse_test_list)
        if 'PASS_TO_PASS' in df.columns:
            df['PASS_TO_PASS_list'] = df['PASS_TO_PASS'].apply(parse_test_list)
        
        # Add difficulty score as an ordinal categorical variable
        difficulty_map = {
            '<15 min fix': 1,
            '15 min - 1 hour': 2,
            '1-4 hours': 3,
            '>4 hours': 4
        }
        
        if 'difficulty' in df.columns:
            print("Difficulty values in dataset:", df['difficulty'].unique())
            df['difficulty_score'] = df['difficulty'].map(difficulty_map).fillna(2)  # Default to medium if unknown
        else:
            print("Warning: 'difficulty' column not found in dataset")
        
        return df
    
    except Exception as e:
        print(f"Error loading SWE-Bench-Verified data: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def parse_test_list(test_str):
    """
    Parse test lists from the SWE-Bench-Verified dataset.
    
    Args:
        test_str: String representation of test list
    
    Returns:
        List of tests
    """
    if pd.isna(test_str):
        return []
    try:
        # Try to parse as JSON
        return json.loads(test_str)
    except:
        # If it fails, try to extract using regex
        if isinstance(test_str, str):
            import re
            matches = re.findall(r'"([^"]+)"', test_str)
            return matches
        return []

def create_task_splits_by_repo(df, num_tasks):
    """
    Create task splits by repository.
    
    Args:
        df: DataFrame with SWE-Bench-Verified data
        num_tasks: Number of tasks to create
    
    Returns:
        Dictionary with task splits
    """
    # Get unique repositories
    repos = df['repo'].unique().tolist()
    np.random.shuffle(repos)
    
    # Split repositories into num_tasks groups
    repo_splits = np.array_split(repos, num_tasks)
    
    # Create task dictionary
    tasks = {}
    for i, repo_group in enumerate(repo_splits):
        task_name = f"T_{i+1}"
        task_df = df[df['repo'].isin(repo_group)]
        tasks[task_name] = task_df.to_dict('records')
        print(f"Task {task_name}: {len(task_df)} issues from {len(repo_group)} repositories")
    
    return tasks

def create_task_splits_by_time(df, num_tasks):
    """
    Create task splits by chronological order.
    
    Args:
        df: DataFrame with SWE-Bench-Verified data
        num_tasks: Number of tasks to create
    
    Returns:
        Dictionary with task splits
    """
    # Sort by created_at
    if 'created_at' not in df.columns:
        print("Warning: 'created_at' column not found. Using random ordering instead.", file=sys.stderr)
        return create_random_task_splits(df, num_tasks)
    
    df_sorted = df.sort_values('created_at')
    
    # Split into num_tasks groups of approximately equal size
    split_indices = np.array_split(range(len(df_sorted)), num_tasks)
    
    # Create task dictionary
    tasks = {}
    for i, indices in enumerate(split_indices):
        task_name = f"T_{i+1}"
        task_df = df_sorted.iloc[indices]
        tasks[task_name] = task_df.to_dict('records')
        
        # Print time range for this task
        if len(task_df) > 0 and 'created_at' in task_df.columns:
            start_date = task_df['created_at'].min().strftime('%Y-%m-%d')
            end_date = task_df['created_at'].max().strftime('%Y-%m-%d')
            print(f"Task {task_name}: {len(task_df)} issues from {start_date} to {end_date}")
        else:
            print(f"Task {task_name}: {len(task_df)} issues")
    
    return tasks

def create_random_task_splits(df, num_tasks):
    """
    Create random task splits.
    
    Args:
        df: DataFrame with SWE-Bench-Verified data
        num_tasks: Number of tasks to create
    
    Returns:
        Dictionary with task splits
    """
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1, random_state=42)
    
    # Split into num_tasks groups of approximately equal size
    split_indices = np.array_split(range(len(df_shuffled)), num_tasks)
    
    # Create task dictionary
    tasks = {}
    for i, indices in enumerate(split_indices):
        task_name = f"T_{i+1}"
        task_df = df_shuffled.iloc[indices]
        tasks[task_name] = task_df.to_dict('records')
        print(f"Task {task_name}: {len(task_df)} issues (random split)")
    
    return tasks

def main():
    parser = argparse.ArgumentParser(description="Generate continual learning task splits from SWE-Bench-Verified")
    parser.add_argument("--input_file", required=True, help="Path to SWE-Bench-Verified JSON or parquet file")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of task splits to create")
    parser.add_argument("--split_by", choices=["repo", "time", "random"], default="repo", 
                        help="Method to split tasks: by repository, chronological order, or random")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        # Load SWE-Bench-Verified data
        df = load_swe_bench_verified(args.input_file)
        
        # Create task splits
        if args.split_by == "repo":
            tasks = create_task_splits_by_repo(df, args.num_tasks)
            stream_type = "repo"
        elif args.split_by == "time":
            tasks = create_task_splits_by_time(df, args.num_tasks)
            stream_type = "time"
        else:  # random
            tasks = create_random_task_splits(df, args.num_tasks)
            stream_type = "random"
        
        # Add metadata
        task_ordering = list(tasks.keys())
        metadata = {
            "stream_type": stream_type,
            "task_ordering": task_ordering,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_tasks": args.num_tasks,
            "total_issues": sum(len(tasks[t]) for t in task_ordering)
        }
        
        output_data = {**tasks, "metadata": metadata}
        
        # Save to output file
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Preprocess tasks to convert non-serializable objects
        for task_name, task_examples in tasks.items():
            for i, example in enumerate(task_examples):
                # Convert all values to properly serializable types
                for key, value in list(example.items()):
                    if isinstance(value, pd.Timestamp):
                        example[key] = value.isoformat()
                    elif isinstance(value, np.ndarray):
                        # Convert numpy arrays to lists
                        example[key] = value.tolist()
                    elif isinstance(value, (pd.Series, list)):
                        # Process each element in a list/series
                        if hasattr(value, 'tolist'):
                            example[key] = value.tolist()
                        else:
                            example[key] = list(value)
                    elif pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                        example[key] = None
                    elif isinstance(value, np.integer):
                        example[key] = int(value)
                    elif isinstance(value, np.floating):
                        example[key] = float(value)
        
        # Save using standard JSON dump (no need for custom encoder)
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSuccessfully created {args.num_tasks} task splits:")
        for task_name in task_ordering:
            print(f"  - {task_name}: {len(tasks[task_name])} issues")
        
        print(f"\nOutput saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
