#!/usr/bin/env python3
"""
This script analyzes the structural overlap between different patches in the SWE-Bench-CL dataset.
It samples unrelated patch pairs and computes similarity metrics between them.
"""

import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import csv

# === STEP 1: Load the dataset ===
def load_data(json_path):
    """Load the SWE-Bench-CL dataset from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_tasks(data):
    """Extract tasks from the dataset and organize them by repository."""
    repo_tasks = defaultdict(list)
    
    for sequence in data['sequences']:
        repo = sequence['repo']
        
        for task in sequence['tasks']:
            # Extract necessary information
            task_id = task['metadata']['instance_id']
            sequence_index = task['continual_learning'].get('sequence_position', None)
            if sequence_index is None:
                sequence_index = len(repo_tasks[repo])  # Use list index if missing
                
            patch = task['evaluation']['patch']
            files = task['continual_learning'].get('modified_files', [])
            
            # Some datasets might use 'task_id' instead of 'instance_id'
            if not task_id:
                task_id = task['metadata'].get('task_id', f"{repo}-{sequence_index}")
                
            # Store the task with relevant information
            repo_tasks[repo].append({
                'task_id': task_id,
                'sequence_index': sequence_index,
                'patch': patch,
                'files': files
            })
    
    # Sort tasks by sequence_index for each repo
    for repo in repo_tasks:
        repo_tasks[repo].sort(key=lambda x: x['sequence_index'])
    
    return repo_tasks

# === STEP 2: Sample unrelated patch pairs ===
def sample_unrelated_pairs(repo_tasks, max_pairs=20):
    """
    Sample unrelated task pairs (A → B) for each repo where:
    - A appears before B in the sequence
    - set(files_A) is disjoint from set(files_B)
    """
    pairs = []
    
    for repo, tasks in repo_tasks.items():
        repo_pairs = []
        
        for i, task_a in enumerate(tasks):
            for task_b in tasks[i+1:]:  # Only consider tasks that come after task_a
                files_a = set(task_a['files'])
                files_b = set(task_b['files'])
                
                # Check if the file sets are disjoint
                if not files_a.intersection(files_b):
                    repo_pairs.append((task_a, task_b))
                    
        # Sample up to max_pairs from the available pairs
        sampled_pairs = random.sample(repo_pairs, min(max_pairs, len(repo_pairs)))
        pairs.extend([(repo, pair[0], pair[1]) for pair in sampled_pairs])
    
    return pairs

# === STEP 3: Compute patch similarity ===
def compute_jaccard_similarity(patch_a, patch_b):
    """Compute Jaccard similarity between two patches."""
    tokens_a = set(patch_a.split())
    tokens_b = set(patch_b.split())
    
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def compute_cosine_similarity(patches):
    """Compute cosine similarity using TfidfVectorizer."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(patches)
    
    # Convert to dense arrays for easier manipulation
    dense_matrix = tfidf_matrix.toarray()
    
    # Compute cosine similarity (normalize dot product)
    norms = np.linalg.norm(dense_matrix, axis=1)
    normalized = dense_matrix / norms[:, np.newaxis]
    
    # Cosine similarity between patch_a and patch_b
    return normalized[0].dot(normalized[1])

# === STEP 4 & 5: Process pairs and save results ===
def process_pairs(pairs):
    """Process pairs, compute similarity, and collect results."""
    results = []
    jaccard_scores = []
    cosine_scores = []
    
    for repo, task_a, task_b in pairs:
        jaccard_sim = compute_jaccard_similarity(task_a['patch'], task_b['patch'])
        cosine_sim = compute_cosine_similarity([task_a['patch'], task_b['patch']])
        
        jaccard_scores.append(jaccard_sim)
        cosine_scores.append(cosine_sim)
        
        # Store the result
        result = {
            'repo': repo,
            'task_a_id': task_a['task_id'],
            'task_b_id': task_b['task_id'],
            'jaccard': jaccard_sim,
            'cosine': cosine_sim,
            'files_a': task_a['files'],
            'files_b': task_b['files']
        }
        results.append(result)
        
        # Print the result
        print(f"Repo: {repo}")
        print(f"Task A ID → Task B ID: {task_a['task_id']} → {task_b['task_id']}")
        print(f"Files A: {', '.join(task_a['files'])}")
        print(f"Files B: {', '.join(task_b['files'])}")
        print(f"Jaccard similarity: {jaccard_sim:.4f}")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print("-" * 50)
    
    # Print summary statistics
    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0
    avg_cosine = np.mean(cosine_scores) if cosine_scores else 0
    high_jaccard_count = sum(1 for score in jaccard_scores if score > 0.25)
    high_cosine_count = sum(1 for score in cosine_scores if score > 0.4)
    
    print("\nSummary:")
    print(f"Average Jaccard similarity: {avg_jaccard:.4f}")
    print(f"Average cosine similarity: {avg_cosine:.4f}")
    print(f"Count of Jaccard > 0.25: {high_jaccard_count}")
    print(f"Count of Cosine > 0.4: {high_cosine_count}")
    
    return results, jaccard_scores, cosine_scores

def save_results(results, output_json, output_csv):
    """Save results to JSON and CSV files."""
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['repo', 'task_a_id', 'task_b_id', 'jaccard', 'cosine', 'files_a', 'files_b'])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result['repo'],
                result['task_a_id'],
                result['task_b_id'],
                result['jaccard'],
                result['cosine'],
                ','.join(result['files_a']),
                ','.join(result['files_b'])
            ])

# === STEP 6: Plot figure ===
def plot_histograms(jaccard_scores, cosine_scores, output_file):
    """Create and save histograms of similarity scores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot Jaccard similarity histogram
    ax1.hist(jaccard_scores, bins=10, alpha=0.7, color='blue')
    ax1.set_title('Jaccard Similarity Distribution')
    ax1.set_xlabel('Jaccard Similarity')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cosine similarity histogram
    ax2.hist(cosine_scores, bins=10, alpha=0.7, color='green')
    ax2.set_title('Cosine Similarity Distribution')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    json_path = os.path.join(os.path.dirname(__file__), 'SWE-Bench-CL.json')
    output_json = os.path.join(os.path.dirname(__file__), 'overlap_results.json')
    output_csv = os.path.join(os.path.dirname(__file__), 'overlap_table.csv')
    output_figure = os.path.join(os.path.dirname(__file__), 'overlap_histogram.png')
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Load the dataset
    print("Loading SWE-Bench-CL dataset...")
    data = load_data(json_path)
    repo_tasks = extract_tasks(data)
    
    # Step 2: Sample unrelated patch pairs
    print("Sampling unrelated patch pairs...")
    pairs = sample_unrelated_pairs(repo_tasks)
    
    # Steps 3 & 4: Compute similarity and print results
    print("Computing patch similarity...")
    results, jaccard_scores, cosine_scores = process_pairs(pairs)
    
    # Step 5: Save results
    print("Saving results...")
    save_results(results, output_json, output_csv)
    
    # Step 6: Plot figure
    print("Generating histogram...")
    plot_histograms(jaccard_scores, cosine_scores, output_figure)
    
    print(f"Analysis complete. Results saved to {output_json}, {output_csv}, and {output_figure}")

if __name__ == "__main__":
    main()
