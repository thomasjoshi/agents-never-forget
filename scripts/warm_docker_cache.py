#!/usr/bin/env python
"""
Script to warm up Docker cache by pre-building images for SWE-Bench repositories.
This significantly speeds up test execution by avoiding the cold-build delay.
"""

import os
import json
import argparse
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import warnings

def check_docker_available():
    """Check if Docker is available and running."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        warnings.warn(f"Docker not available: {e}")
        return False

def warm_docker_cache(task_stream_file, output_dir=None):
    """
    Pre-build Docker images for all repositories in the task stream.
    
    Args:
        task_stream_file: Path to the task stream JSON file
        output_dir: Directory where to save the log (optional)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_docker_available():
        print("Docker not available. Cache warming skipped.")
        return False
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load task stream
    try:
        with open(task_stream_file, 'r') as f:
            task_stream = json.load(f)
    except Exception as e:
        print(f"Error loading task stream file: {e}")
        return False
    
    # Get unique repositories
    repos = set()
    for task_name, task_examples in task_stream.items():
        if task_name == "metadata":
            continue
        for example in task_examples:
            if "repo" in example:
                repos.add(example["repo"])
    
    print(f"Found {len(repos)} unique repositories to warm up")
    
    # Warm up cache for each repository
    success_count = 0
    for repo in tqdm(repos, desc="Warming Docker cache"):
        try:
            # Create a dummy patch file
            with open("/tmp/dummy.diff", "w") as f:
                f.write("# Empty patch for cache warming\n")
            
            # Build Docker image
            cmd = [
                "docker", "run", "--rm",
                "-v", f"/tmp/dummy.diff:/patch.diff",
                "swebench/tester:latest",
                "--repo", repo,
                "--build-only"
            ]
            
            # Run with timeout (3 minutes should be enough for build)
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if process.returncode == 0:
                success_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"Timeout building image for {repo}")
        except Exception as e:
            print(f"Error building image for {repo}: {e}")
    
    # Clean up
    try:
        os.unlink("/tmp/dummy.diff")
    except:
        pass
    
    print(f"Successfully pre-built {success_count}/{len(repos)} Docker images")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="Warm up Docker cache for SWE-Bench repositories")
    parser.add_argument("--task_stream_file", required=True, 
                        help="Path to task stream JSON file")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save output (optional)")
    
    args = parser.parse_args()
    
    start_time = time.time()
    success = warm_docker_cache(args.task_stream_file, args.output_dir)
    elapsed_time = time.time() - start_time
    
    print(f"Cache warming completed in {elapsed_time:.1f} seconds")
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
