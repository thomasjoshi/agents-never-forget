#!/usr/bin/env python3
"""
This script analyzes the structural overlap between patches in the SWE-Bench-CL dataset
across different difficulty levels.
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

def extract_tasks_by_difficulty(data):
    """Extract tasks from the dataset and group them by difficulty level."""
    # Initialize difficulty buckets
    easy_tasks = []  # difficulty_score ≤ 2
    medium_tasks = []  # 2 < difficulty_score ≤ 3
    hard_tasks = []  # difficulty_score > 3
    
    for sequence in data['sequences']:
        repo = sequence['repo']
        
        for task in sequence['tasks']:
            # Extract necessary information
            task_id = task['metadata']['instance_id']
            if not task_id:
                task_id = task['metadata'].get('task_id', f"{repo}-{len(easy_tasks) + len(medium_tasks) + len(hard_tasks)}")
                
            sequence_index = task['continual_learning'].get('sequence_position', None)
            patch = task['evaluation']['patch']
            files = task['continual_learning'].get('modified_files', [])
            
            # Try to extract difficulty score
            difficulty_score = None
            
            # Check different possible locations for difficulty information
            if 'difficulty_score' in task.get('continual_learning', {}):
                difficulty_score = task['continual_learning']['difficulty_score']
            elif 'difficulty_score' in task.get('metadata', {}):
                difficulty_score = task['metadata']['difficulty_score']
            elif 'difficulty' in task.get('metadata', {}):
                # Try to convert text difficulty to numeric score
                difficulty_text = task['metadata']['difficulty']
                if '<15 min fix' in difficulty_text:
                    difficulty_score = 1
                elif '15-60 min fix' in difficulty_text:
                    difficulty_score = 2
                elif '1-3 hour fix' in difficulty_text:
                    difficulty_score = 3
                elif '>3 hour fix' in difficulty_text:
                    difficulty_score = 4
            
            # Skip tasks with no difficulty information
            if difficulty_score is None:
                continue
                
            # Store task with relevant information
            task_info = {
                'task_id': task_id,
                'repo': repo,
                'sequence_index': sequence_index,
                'patch': patch,
                'files': files,
                'difficulty_score': difficulty_score
            }
            
            # Assign to appropriate bucket
            if difficulty_score <= 2:
                easy_tasks.append(task_info)
            elif difficulty_score <= 3:
                medium_tasks.append(task_info)
            else:
                hard_tasks.append(task_info)
    
    return {
        'easy': easy_tasks,
        'medium': medium_tasks,
        'hard': hard_tasks
    }

# === STEP 2: Sample task pairs within and across difficulty levels ===
def sample_task_pairs(tasks_by_difficulty, pairs_per_group=10):
    """Sample task pairs within and across difficulty levels."""
    pairs = []
    difficulty_levels = ['easy', 'medium', 'hard']
    
    # Sample pairs within each difficulty level
    for level in difficulty_levels:
        tasks = tasks_by_difficulty[level]
        level_pairs = []
        
        # Try to find pairs with disjoint modified files
        for i, task_a in enumerate(tasks):
            for j in range(i+1, len(tasks)):
                task_b = tasks[j]
                files_a = set(task_a['files'])
                files_b = set(task_b['files'])
                
                # Check if the file sets are disjoint
                if not files_a.intersection(files_b):
                    level_pairs.append((task_a, task_b))
                    
                # Break early if we have enough pairs
                if len(level_pairs) >= pairs_per_group * 2:  # Sample more to allow for random selection
                    break
            if len(level_pairs) >= pairs_per_group * 2:
                break
        
        # Randomly sample the required number of pairs
        sampled_pairs = random.sample(level_pairs, min(pairs_per_group, len(level_pairs)))
        pairs.extend([('within', level, pair[0], pair[1]) for pair in sampled_pairs])
    
    # Sample pairs across difficulty levels
    for i, level_a in enumerate(difficulty_levels):
        for level_b in difficulty_levels[i+1:]:
            tasks_a = tasks_by_difficulty[level_a]
            tasks_b = tasks_by_difficulty[level_b]
            cross_pairs = []
            
            # Try to find pairs with disjoint modified files
            for task_a in tasks_a:
                for task_b in tasks_b:
                    files_a = set(task_a['files'])
                    files_b = set(task_b['files'])
                    
                    # Check if the file sets are disjoint
                    if not files_a.intersection(files_b):
                        cross_pairs.append((task_a, task_b))
                        
                    # Break early if we have enough pairs
                    if len(cross_pairs) >= pairs_per_group * 2:
                        break
                if len(cross_pairs) >= pairs_per_group * 2:
                    break
            
            # Randomly sample the required number of pairs
            sampled_pairs = random.sample(cross_pairs, min(pairs_per_group, len(cross_pairs)))
            pairs.extend([('across', f"{level_a}-{level_b}", pair[0], pair[1]) for pair in sampled_pairs])
    
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

# === STEP 4: Process pairs and compute similarity metrics ===
def process_difficulty_pairs(pairs):
    """Process pairs and compute similarity metrics by difficulty levels."""
    results = []
    
    # Dictionaries to store scores by group
    jaccard_by_group = defaultdict(list)
    cosine_by_group = defaultdict(list)
    
    for pair_type, group, task_a, task_b in pairs:
        jaccard_sim = compute_jaccard_similarity(task_a['patch'], task_b['patch'])
        cosine_sim = compute_cosine_similarity([task_a['patch'], task_b['patch']])
        
        # Store the result
        result = {
            'pair_type': pair_type,
            'group': group,
            'task_a_id': task_a['task_id'],
            'task_b_id': task_b['task_id'],
            'task_a_repo': task_a['repo'],
            'task_b_repo': task_b['repo'],
            'task_a_difficulty': task_a['difficulty_score'],
            'task_b_difficulty': task_b['difficulty_score'],
            'jaccard': jaccard_sim,
            'cosine': cosine_sim,
            'files_a': task_a['files'],
            'files_b': task_b['files']
        }
        results.append(result)
        
        # Add to group scores
        jaccard_by_group[group].append(jaccard_sim)
        cosine_by_group[group].append(cosine_sim)
        
        # Print the result
        print(f"Group: {group}")
        print(f"Task A ID → Task B ID: {task_a['task_id']} → {task_b['task_id']}")
        print(f"Task A Difficulty: {task_a['difficulty_score']}, Task B Difficulty: {task_b['difficulty_score']}")
        print(f"Files A: {', '.join(task_a['files'])}")
        print(f"Files B: {', '.join(task_b['files'])}")
        print(f"Jaccard similarity: {jaccard_sim:.4f}")
        print(f"Cosine similarity: {cosine_sim:.4f}")
        print("-" * 50)
    
    # Compute and print summary statistics by group
    print("\nSummary Statistics:")
    print("-" * 50)
    
    group_stats = {}
    for group, scores in jaccard_by_group.items():
        avg_jaccard = np.mean(scores) if scores else 0
        avg_cosine = np.mean(cosine_by_group[group]) if cosine_by_group[group] else 0
        
        group_stats[group] = {
            'avg_jaccard': avg_jaccard,
            'avg_cosine': avg_cosine,
            'count': len(scores)
        }
        
        print(f"Group: {group}")
        print(f"  Number of pairs: {len(scores)}")
        print(f"  Average Jaccard similarity: {avg_jaccard:.4f}")
        print(f"  Average cosine similarity: {avg_cosine:.4f}")
        print("-" * 30)
    
    return results, group_stats

# === STEP 5: Save results ===
def save_difficulty_results(results, group_stats, output_json, output_csv):
    """Save results to JSON and CSV files."""
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump({
            'results': results,
            'group_stats': group_stats
        }, f, indent=2)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'pair_type', 'group', 'task_a_id', 'task_b_id', 
            'task_a_repo', 'task_b_repo', 'task_a_difficulty', 'task_b_difficulty',
            'jaccard', 'cosine', 'files_a', 'files_b'
        ])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result['pair_type'],
                result['group'],
                result['task_a_id'],
                result['task_b_id'],
                result['task_a_repo'],
                result['task_b_repo'],
                result['task_a_difficulty'],
                result['task_b_difficulty'],
                result['jaccard'],
                result['cosine'],
                ','.join(result['files_a']),
                ','.join(result['files_b'])
            ])

# === STEP 6: Plot bar chart ===
def plot_difficulty_barplot(group_stats, output_file):
    """Create and save a bar chart comparing average similarities across groups."""
    # Separate within and across groups
    within_groups = {k: v for k, v in group_stats.items() if k in ['easy', 'medium', 'hard']}
    across_groups = {k: v for k, v in group_stats.items() if '-' in k}
    
    # Sort groups for consistent display
    within_order = ['easy', 'medium', 'hard']
    across_order = sorted(across_groups.keys())
    
    # Prepare data for within groups
    within_labels = [k for k in within_order if k in within_groups]
    within_jaccard = [group_stats[k]['avg_jaccard'] for k in within_labels]
    within_cosine = [group_stats[k]['avg_cosine'] for k in within_labels]
    
    # Prepare data for across groups
    across_labels = across_order
    across_jaccard = [group_stats[k]['avg_jaccard'] for k in across_labels]
    across_cosine = [group_stats[k]['avg_cosine'] for k in across_labels]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot Jaccard similarities
    ax1.set_title('Average Jaccard Similarity by Difficulty Level', fontsize=14)
    
    # Plot within groups
    bar_width = 0.35
    x_within = np.arange(len(within_labels))
    ax1.bar(x_within, within_jaccard, bar_width, label='Within Difficulty Level', color='blue', alpha=0.7)
    
    # Plot across groups
    x_across = np.arange(len(within_labels) + 1, len(within_labels) + 1 + len(across_labels))
    ax1.bar(x_across, across_jaccard, bar_width, label='Across Difficulty Levels', color='green', alpha=0.7)
    
    # Set labels and ticks
    all_x = np.concatenate([x_within, x_across])
    all_labels = within_labels + across_labels
    ax1.set_xticks(all_x)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.set_ylabel('Jaccard Similarity', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot Cosine similarities
    ax2.set_title('Average Cosine Similarity by Difficulty Level', fontsize=14)
    
    # Plot within groups
    ax2.bar(x_within, within_cosine, bar_width, label='Within Difficulty Level', color='blue', alpha=0.7)
    
    # Plot across groups
    ax2.bar(x_across, across_cosine, bar_width, label='Across Difficulty Levels', color='green', alpha=0.7)
    
    # Set labels and ticks
    ax2.set_xticks(all_x)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right')
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure in high resolution
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    json_path = os.path.join(os.path.dirname(__file__), 'SWE-Bench-CL.json')
    output_json = os.path.join(os.path.dirname(__file__), 'difficulty_overlap_results.json')
    output_csv = os.path.join(os.path.dirname(__file__), 'difficulty_overlap_table.csv')
    output_figure = os.path.join(os.path.dirname(__file__), 'difficulty_overlap_barplot.png')
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Load the dataset
    print("Loading SWE-Bench-CL dataset...")
    data = load_data(json_path)
    
    # Step 2: Extract tasks by difficulty
    print("Grouping tasks by difficulty level...")
    tasks_by_difficulty = extract_tasks_by_difficulty(data)
    print(f"Found {len(tasks_by_difficulty['easy'])} easy tasks, "
          f"{len(tasks_by_difficulty['medium'])} medium tasks, and "
          f"{len(tasks_by_difficulty['hard'])} hard tasks.")
    
    # Step 3: Sample task pairs
    print("Sampling task pairs within and across difficulty levels...")
    pairs = sample_task_pairs(tasks_by_difficulty)
    
    # Step 4: Process pairs and compute similarity metrics
    print("Computing patch similarity...")
    results, group_stats = process_difficulty_pairs(pairs)
    
    # Step 5: Save results
    print("Saving results...")
    save_difficulty_results(results, group_stats, output_json, output_csv)
    
    # Step 6: Plot bar chart
    print("Generating bar chart...")
    plot_difficulty_barplot(group_stats, output_figure)
    
    print(f"Analysis complete. Results saved to {output_json}, {output_csv}, and {output_figure}")

if __name__ == "__main__":
    main()