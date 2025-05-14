#!/usr/bin/env python
"""
Script to warm up Docker cache for SWE-Bench-CL evaluation.

This script pre-builds Docker images for repositories in the SWE-Bench-CL task stream
to avoid 90-second cold builds during evaluation.
"""

import os
import json
import argparse
import subprocess
import tempfile
from tqdm import tqdm


def load_task_stream(task_stream_file, filter_repo=None):
    """
    Load the task stream from a JSON file and optionally filter for specific repo.
    
    Args:
        task_stream_file: Path to task stream JSON file
        filter_repo: Optional repository name to filter tasks (e.g., 'sympy/sympy')
    
    Returns:
        Dictionary containing task stream data and task ordering
    """
    print(f"Loading task stream from {task_stream_file}...")
    
    try:
        with open(task_stream_file, 'r') as f:
            task_stream = json.load(f)
    except Exception as e:
        print(f"Error loading task stream: {e}")
        return None
    
    # If task_stream is not a dictionary with tasks, it might be a direct list
    if not isinstance(task_stream, dict):
        if isinstance(task_stream, list):
            # Convert list to dictionary for consistency
            tasks = {}
            for task in task_stream:
                repo = task.get('repo')
                if not repo:
                    continue
                if repo not in tasks:
                    tasks[repo] = []
                tasks[repo].append(task)
            task_stream = {"tasks": tasks, "task_ordering": list(tasks.keys())}
        else:
            print(f"Error: Unexpected task stream format: {type(task_stream)}")
            return None
    
    # If task_stream is already in expected format (dict with 'tasks' and 'task_ordering')
    if 'tasks' in task_stream and 'task_ordering' in task_stream:
        tasks = task_stream['tasks']
        task_ordering = task_stream['task_ordering']
    else:
        # Flatten the structure into a list of all tasks
        all_tasks = []
        task_ordering = []
        for task_name, task_examples in task_stream.items():
            task_ordering.append(task_name)
            for example in task_examples:
                # Add task name to example for reference
                example['task_name'] = task_name
                all_tasks.append(example)
        
        # Group tasks by repository
        tasks = {}
        for task in all_tasks:
            repo = task.get('repo')
            if not repo:
                continue
            if repo not in tasks:
                tasks[repo] = []
            tasks[repo].append(task)
    
    # Filter for specific repository if requested
    if filter_repo:
        if filter_repo in tasks:
            filtered_tasks = {filter_repo: tasks[filter_repo]}
            filtered_ordering = [to for to in task_ordering if to == filter_repo]
            return {"tasks": filtered_tasks, "task_ordering": filtered_ordering}
        else:
            print(f"Warning: Repository {filter_repo} not found in task stream.")
            return {"tasks": {}, "task_ordering": []}
    
    return {"tasks": tasks, "task_ordering": task_ordering}


def warm_docker_cache(task_stream, temp_dir):
    """
    Pre-build Docker images for repositories in the task stream.
    
    Args:
        task_stream: Dictionary containing task stream data
        temp_dir: Directory to use for temporary clones
    
    Returns:
        Number of repositories successfully cached
    """
    unique_repos = set()
    
    # Collect all unique repositories
    for task_name, task_examples in task_stream['tasks'].items():
        for example in task_examples:
            repo = example.get('repo')
            base_commit = example.get('base_commit')
            if repo and base_commit:
                unique_repos.add((repo, base_commit))
    
    print(f"Found {len(unique_repos)} unique repository/commit combinations to cache")
    
    # Clone and build Docker images for each repo
    success_count = 0
    for repo, base_commit in tqdm(unique_repos, desc="Building Docker images"):
        repo_path = os.path.join(temp_dir, repo.replace('/', '_'))
        os.makedirs(repo_path, exist_ok=True)
        
        try:
            # Clone the repository
            print(f"\nCloning {repo} at commit {base_commit[:8]}...")
            clone_cmd = [
                "git", "clone", 
                f"https://github.com/{repo}.git",
                repo_path
            ]
            clone_result = subprocess.run(clone_cmd, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, 
                                         check=False)
            
            if clone_result.returncode != 0:
                print(f"Failed to clone {repo}: {clone_result.stderr.decode()}")
                continue
            
            # Checkout the base commit
            checkout_cmd = ["git", "checkout", base_commit]
            checkout_result = subprocess.run(checkout_cmd, 
                                           cwd=repo_path, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, 
                                           check=False)
            
            if checkout_result.returncode != 0:
                print(f"Failed to checkout {base_commit} for {repo}: {checkout_result.stderr.decode()}")
                continue
            
            # Build the Docker image
            print(f"Building Docker image for {repo}:{base_commit[:8]}...")
            image_name = f"swebench-{repo.replace('/', '_')}-{base_commit[:8]}"
            build_cmd = [
                "docker", "build", 
                "-t", image_name, 
                "."
            ]
            build_result = subprocess.run(build_cmd, 
                                        cwd=repo_path,
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        check=False)
            
            if build_result.returncode != 0:
                print(f"Failed to build Docker image for {repo}: {build_result.stderr.decode()}")
                continue
                
            print(f"Successfully built Docker image: {image_name}")
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {repo}: {str(e)}")
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Warm Docker cache for SWE-Bench-CL tasks")
    parser.add_argument("--task_stream_file", required=True, 
                        help="Path to task stream JSON file")
    parser.add_argument("--filter_repo", default=None, 
                        help="Optional repository to filter tasks for")
    parser.add_argument("--temp_dir", default=None, 
                        help="Directory to use for temporary clones (default: system temp dir)")
    
    args = parser.parse_args()
    
    # Load task stream
    task_data = load_task_stream(args.task_stream_file, args.filter_repo)
    if not task_data:
        print("Failed to load task stream. Exiting.")
        return
    
    # Create temp directory if not specified
    temp_dir = args.temp_dir
    if not temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="swebench_cache_")
        print(f"Using temporary directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Warm the Docker cache
    success_count = warm_docker_cache(task_data, temp_dir)
    
    print(f"\nDocker cache warming completed. Successfully built {success_count} images.")
    
    # Clean up if we created a temp directory
    if not args.temp_dir:
        print(f"Cleaning up temporary directory: {temp_dir}")
        # Uncomment to enable auto-cleanup
        # import shutil
        # shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
