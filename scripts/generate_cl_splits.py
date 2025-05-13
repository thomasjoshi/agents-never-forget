#!/usr/bin/env python3
"""
Generate continual learning splits from SWE-Bench-Verified dataset.

This script creates a continual learning task stream by splitting SWE-Bench-Verified
into a sequence of tasks (T1, T2, ..., Tn), optionally by repository or chronological order.
"""

import argparse
import json
import sys
import os
import random
from datetime import datetime
from collections import defaultdict

def load_data(input_file):
    """
    Load data from the SWE-Bench-Verified JSON file.
    
    Args:
        input_file (str): Path to the SWE-Bench-Verified JSON file
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {input_file}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

def partition_by_repo(data, num_tasks):
    """
    Partition tasks by repository.
    
    Args:
        data (dict): SWE-Bench-Verified data
        num_tasks (int): Number of task groups to create
        
    Returns:
        dict: Tasks partitioned by group
    """
    # Collect all tasks by repository
    repo_tasks = defaultdict(list)
    
    # Extract tasks from the data structure
    for repo_name, repo_data in data.items():
        for issue_id, issue_data in repo_data.get("issues", {}).items():
            task = {
                "repo": repo_name,
                "issue_id": issue_id,
                "title": issue_data.get("title", ""),
                "body": issue_data.get("body", ""),
                "timestamp": issue_data.get("created_at", ""),
                "difficulty": issue_data.get("difficulty", ""),
                "files": issue_data.get("files", []),
                "patch": issue_data.get("pr", {}).get("patch", "")
            }
            repo_tasks[repo_name].append(task)
    
    # Get list of repos
    repos = list(repo_tasks.keys())
    random.shuffle(repos)
    
    # Distribute repos across task groups
    task_groups = {f"T_{i+1}": [] for i in range(num_tasks)}
    for i, repo in enumerate(repos):
        group_idx = i % num_tasks
        group_name = f"T_{group_idx+1}"
        task_groups[group_name].extend(repo_tasks[repo])
    
    return task_groups, "repo"

def partition_by_time(data, num_tasks):
    """
    Partition tasks chronologically.
    
    Args:
        data (dict): SWE-Bench-Verified data
        num_tasks (int): Number of task groups to create
        
    Returns:
        dict: Tasks partitioned by group
    """
    # Extract all tasks with timestamps
    all_tasks = []
    
    for repo_name, repo_data in data.items():
        for issue_id, issue_data in repo_data.get("issues", {}).items():
            task = {
                "repo": repo_name,
                "issue_id": issue_id,
                "title": issue_data.get("title", ""),
                "body": issue_data.get("body", ""),
                "timestamp": issue_data.get("created_at", ""),
                "difficulty": issue_data.get("difficulty", ""),
                "files": issue_data.get("files", []),
                "patch": issue_data.get("pr", {}).get("patch", "")
            }
            all_tasks.append(task)
    
    # Sort tasks by timestamp
    def parse_timestamp(timestamp):
        try:
            return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
        except:
            # Default to epoch start if timestamp is invalid
            return datetime(1970, 1, 1)
    
    all_tasks.sort(key=lambda x: parse_timestamp(x.get("timestamp", "")))
    
    # Divide tasks into groups
    task_groups = {f"T_{i+1}": [] for i in range(num_tasks)}
    tasks_per_group = len(all_tasks) // num_tasks
    remainder = len(all_tasks) % num_tasks
    
    start_idx = 0
    for i in range(num_tasks):
        group_size = tasks_per_group + (1 if i < remainder else 0)
        end_idx = start_idx + group_size
        task_groups[f"T_{i+1}"] = all_tasks[start_idx:end_idx]
        start_idx = end_idx
    
    return task_groups, "time"

def main():
    parser = argparse.ArgumentParser(description="Generate continual learning task splits")
    parser.add_argument("--input_file", required=True, help="Path to SWE-Bench-Verified JSON file")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of task groups to create")
    parser.add_argument("--split_by", choices=["repo", "time"], default="repo", 
                       help="Split by repository or chronologically")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load data
    data = load_data(args.input_file)
    
    # Partition data
    if args.split_by == "repo":
        task_groups, stream_type = partition_by_repo(data, args.num_tasks)
    else:
        task_groups, stream_type = partition_by_time(data, args.num_tasks)
    
    # Create output structure
    output = task_groups.copy()
    output["metadata"] = {
        "stream_type": stream_type,
        "task_ordering": list(task_groups.keys()),
        "created_at": datetime.now().isoformat(),
        "num_tasks": args.num_tasks,
        "split_by": args.split_by,
        "seed": args.seed
    }
    
    # Save to file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Successfully wrote task splits to {args.output_file}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print statistics
    print("\nTask Stream Statistics:")
    print(f"Split by: {args.split_by}")
    print(f"Number of task groups: {args.num_tasks}")
    
    for group_name, tasks in task_groups.items():
        print(f"  {group_name}: {len(tasks)} tasks")
        
        # Print example task names (up to 3)
        if tasks:
            print(f"    Example tasks:")
            for i, task in enumerate(tasks[:3]):
                print(f"      - {task['repo']}/{task['issue_id']}: {task['title'][:50]}...")
    
    print(f"\nOutput saved to: {os.path.abspath(args.output_file)}")

if __name__ == "__main__":
    main()
