#!/usr/bin/env python
"""
Script to warm up Docker cache for SWE-Bench-CL evaluation.

This script pre-builds Docker images for repositories in the SWE-Bench-CL task stream
to avoid 90-second cold builds during evaluation.
"""

import os
import json
import pickle
import argparse
import subprocess
import tempfile
import shutil
import time
from tqdm import tqdm

# Global cache for Docker images
DOCKER_IMAGE_CACHE = {}

def docker_image_exists(image_name):
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--type=image", image_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return result.returncode == 0
    except Exception:
        return False

def cleanup_docker_resources():
    """Clean up Docker containers and images to free up disk space."""
    try:
        # Stop and remove all running containers
        subprocess.run(
            ["docker", "ps", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Remove all stopped containers
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            check=False
        )
        
        # Remove dangling images
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            check=False
        )
        
        # Remove unused volumes
        subprocess.run(
            ["docker", "volume", "prune", "-f"],
            check=False
        )
        
    except Exception as e:
        print(f"Warning: Error during Docker cleanup: {e}")


def load_task_stream(task_stream_file, filter_repo=None):
    """
    Load the task stream from a file (JSON or pickle) and optionally filter for specific repo.
    
    Args:
        task_stream_file: Path to task stream file (either JSON or pickle)
        filter_repo: Optional repository name to filter tasks (e.g., 'sympy/sympy')
    
    Returns:
        Dictionary containing task stream data and task ordering
    """
    print(f"Loading task stream from {task_stream_file}...")
    
    try:
        # Determine file type based on extension
        use_pickle = task_stream_file.endswith('.pkl')
        
        if use_pickle:
            with open(task_stream_file, 'rb') as f:
                task_stream = pickle.load(f)
        else:
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
    cached_count = 0
    
    for repo, base_commit in tqdm(unique_repos, desc="Processing repositories"):
        repo_path = os.path.join(temp_dir, repo.replace('/', '_'))
        image_name = f"swebench-{repo.replace('/', '_')}-{base_commit[:8]}"
        
        # Check if image already exists in cache
        if docker_image_exists(image_name):
            print(f"\nUsing cached Docker image for {repo}@{base_commit[:8]}")
            DOCKER_IMAGE_CACHE[(repo, base_commit)] = image_name
            cached_count += 1
            success_count += 1
            continue
            
        os.makedirs(repo_path, exist_ok=True)
        
        try:
            # Clone the repository if it doesn't exist or is empty
            if not os.path.exists(os.path.join(repo_path, '.git')):
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
                    # Clean up failed clone
                    shutil.rmtree(repo_path, ignore_errors=True)
                    continue
            
            # Checkout the base commit
            checkout_cmd = ["git", "fetch", "origin", base_commit]
            subprocess.run(checkout_cmd, cwd=repo_path, check=False)
            
            checkout_cmd = ["git", "checkout", base_commit]
            checkout_result = subprocess.run(checkout_cmd, 
                                          cwd=repo_path, 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          check=False)
            
            if checkout_result.returncode != 0:
                print(f"Failed to checkout {base_commit} for {repo}: {checkout_result.stderr.decode()}")
                continue
            
            # Check for Dockerfile
            dockerfile_path = os.path.join(repo_path, "Dockerfile")
            if not os.path.exists(dockerfile_path):
                print(f"No Dockerfile found in {repo}@{base_commit[:8]}, using default Python image")
                with open(dockerfile_path, 'w') as f:
                    f.write("FROM python:3.9\n")
                    f.write("WORKDIR /app\n")
                    f.write("COPY . .\n")
                    f.write("RUN pip install -r requirements.txt\n" if os.path.exists(os.path.join(repo_path, 'requirements.txt')) else "")
            
            # Build the Docker image with cache
            print(f"Building Docker image for {repo}:{base_commit[:8]}...")
            build_cmd = [
                "docker", "build", 
                "--no-cache=false",  # Use cache layers
                "-t", image_name, 
                "."
            ]
            
            start_time = time.time()
            build_result = subprocess.run(build_cmd, 
                                       cwd=repo_path,
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       check=False)
            build_time = time.time() - start_time
            
            if build_result.returncode != 0:
                print(f"Failed to build Docker image for {repo}: {build_result.stderr.decode()}")
                continue
                
            print(f"Successfully built Docker image in {build_time:.1f}s: {image_name}")
            DOCKER_IMAGE_CACHE[(repo, base_commit)] = image_name
            success_count += 1
            
            # Clean up after each build to save disk space
            cleanup_docker_resources()
            
        except Exception as e:
            print(f"Error processing {repo}: {str(e)}")
            # Clean up on error
            shutil.rmtree(repo_path, ignore_errors=True)
    
    print(f"\nDocker image caching complete. Success: {success_count}, Cached: {cached_count}, Total: {len(unique_repos)}")
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
