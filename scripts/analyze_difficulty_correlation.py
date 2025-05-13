#!/usr/bin/env python3
"""
This script analyzes the correlation between task difficulty and structural overlap
in the SWE-Bench-CL dataset, using continuous difficulty scores rather than
arbitrary categorization.
"""

import json
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import csv
import scipy.stats as stats
import seaborn as sns

# === STEP 1: Load the dataset ===
def load_data(json_path):
    """Load the SWE-Bench-CL dataset from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_tasks_with_difficulty(data):
    """Extract tasks from the dataset with their difficulty scores."""
    tasks = []
    
    for sequence in data['sequences']:
        repo = sequence['repo']
        
        for task in sequence['tasks']:
            # Extract necessary information
            task_id = task['metadata']['instance_id']
            if not task_id:
                task_id = task['metadata'].get('task_id', f"{repo}-{len(tasks)}")
                
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
            
            tasks.append(task_info)
    
    print(f"Extracted {len(tasks)} tasks with difficulty scores.")
    return tasks

# === STEP 2: Sample task pairs with disjoint modified files ===
def sample_task_pairs(tasks, max_pairs=200):
    """Sample task pairs with disjoint modified files."""
    # Group tasks by repository
    tasks_by_repo = defaultdict(list)
    for task in tasks:
        tasks_by_repo[task['repo']].append(task)
    
    all_pairs = []
    
    # Sample pairs from each repository
    for repo, repo_tasks in tasks_by_repo.items():
        repo_pairs = []
        
        # Sort tasks by sequence index to respect temporal ordering
        repo_tasks.sort(key=lambda x: x['sequence_index'] if x['sequence_index'] is not None else 0)
        
        # Find pairs with disjoint modified files
        for i, task_a in enumerate(repo_tasks):
            for j in range(i+1, len(repo_tasks)):
                task_b = repo_tasks[j]
                files_a = set(task_a['files'])
                files_b = set(task_b['files'])
                
                # Check if the file sets are disjoint
                if not files_a.intersection(files_b):
                    repo_pairs.append((task_a, task_b))
                    
                # Break early if we have enough pairs
                if len(repo_pairs) >= max_pairs:
                    break
            if len(repo_pairs) >= max_pairs:
                break
        
        # Add repository pairs to the total
        all_pairs.extend(repo_pairs)
    
    # Randomly shuffle and limit total pairs if needed
    random.shuffle(all_pairs)
    return all_pairs[:max_pairs]

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

# === STEP 4: Process pairs and compute correlation metrics ===
def process_pairs_continuous(pairs):
    """Process pairs and compute correlation between difficulty and similarity."""
    results = []
    
    # Lists to store data for correlation analysis
    difficulty_a = []
    difficulty_b = []
    difficulty_diff = []
    difficulty_avg = []
    jaccard_scores = []
    cosine_scores = []
    
    for task_a, task_b in pairs:
        diff_a = task_a['difficulty_score']
        diff_b = task_b['difficulty_score']
        
        # Compute similarity measures
        jaccard_sim = compute_jaccard_similarity(task_a['patch'], task_b['patch'])
        cosine_sim = compute_cosine_similarity([task_a['patch'], task_b['patch']])
        
        # Store the result
        result = {
            'task_a_id': task_a['task_id'],
            'task_b_id': task_b['task_id'],
            'repo': task_a['repo'],
            'difficulty_a': diff_a,
            'difficulty_b': diff_b,
            'difficulty_diff': abs(diff_a - diff_b),
            'difficulty_avg': (diff_a + diff_b) / 2,
            'jaccard': jaccard_sim,
            'cosine': cosine_sim,
            'files_a': task_a['files'],
            'files_b': task_b['files']
        }
        results.append(result)
        
        # Add to data for correlation analysis
        difficulty_a.append(diff_a)
        difficulty_b.append(diff_b)
        difficulty_diff.append(abs(diff_a - diff_b))
        difficulty_avg.append((diff_a + diff_b) / 2)
        jaccard_scores.append(jaccard_sim)
        cosine_scores.append(cosine_sim)
        
        # Print progress every 20 pairs
        if len(results) % 20 == 0:
            print(f"Processed {len(results)} pairs...")
    
    # Compute correlation coefficients
    pearson_jaccard_diff = stats.pearsonr(difficulty_diff, jaccard_scores)
    pearson_cosine_diff = stats.pearsonr(difficulty_diff, cosine_scores)
    spearman_jaccard_diff = stats.spearmanr(difficulty_diff, jaccard_scores)
    spearman_cosine_diff = stats.spearmanr(difficulty_diff, cosine_scores)
    
    pearson_jaccard_avg = stats.pearsonr(difficulty_avg, jaccard_scores)
    pearson_cosine_avg = stats.pearsonr(difficulty_avg, cosine_scores)
    spearman_jaccard_avg = stats.spearmanr(difficulty_avg, jaccard_scores)
    spearman_cosine_avg = stats.spearmanr(difficulty_avg, cosine_scores)
    
    # Calculate average scores by difficulty pairs
    difficulty_pair_jaccard = defaultdict(list)
    difficulty_pair_cosine = defaultdict(list)
    
    for result in results:
        diff_a = result['difficulty_a']
        diff_b = result['difficulty_b']
        # Ensure consistent ordering (lower difficulty first)
        diff_pair = (min(diff_a, diff_b), max(diff_a, diff_b))
        
        difficulty_pair_jaccard[diff_pair].append(result['jaccard'])
        difficulty_pair_cosine[diff_pair].append(result['cosine'])
    
    difficulty_pair_stats = {}
    for diff_pair, jaccard_vals in difficulty_pair_jaccard.items():
        cosine_vals = difficulty_pair_cosine[diff_pair]
        difficulty_pair_stats[diff_pair] = {
            'count': len(jaccard_vals),
            'avg_jaccard': np.mean(jaccard_vals),
            'avg_cosine': np.mean(cosine_vals),
            'std_jaccard': np.std(jaccard_vals),
            'std_cosine': np.std(cosine_vals)
        }
    
    # Print correlation results
    print("\nCorrelation Analysis Results:")
    print("-" * 50)
    print("Correlation with Difficulty Difference:")
    print(f"Pearson (Jaccard): r={pearson_jaccard_diff[0]:.4f}, p={pearson_jaccard_diff[1]:.4f}")
    print(f"Pearson (Cosine):  r={pearson_cosine_diff[0]:.4f}, p={pearson_cosine_diff[1]:.4f}")
    print(f"Spearman (Jaccard): rho={spearman_jaccard_diff[0]:.4f}, p={spearman_jaccard_diff[1]:.4f}")
    print(f"Spearman (Cosine):  rho={spearman_cosine_diff[0]:.4f}, p={spearman_cosine_diff[1]:.4f}")
    
    print("\nCorrelation with Average Difficulty:")
    print(f"Pearson (Jaccard): r={pearson_jaccard_avg[0]:.4f}, p={pearson_jaccard_avg[1]:.4f}")
    print(f"Pearson (Cosine):  r={pearson_cosine_avg[0]:.4f}, p={pearson_cosine_avg[1]:.4f}")
    print(f"Spearman (Jaccard): rho={spearman_jaccard_avg[0]:.4f}, p={spearman_jaccard_avg[1]:.4f}")
    print(f"Spearman (Cosine):  rho={spearman_cosine_avg[0]:.4f}, p={spearman_cosine_avg[1]:.4f}")
    
    print("\nDifficulty Pair Statistics:")
    for diff_pair, stats_dict in sorted(difficulty_pair_stats.items()):
        print(f"Difficulty {diff_pair[0]}-{diff_pair[1]} (n={stats_dict['count']}):")
        print(f"  Jaccard: {stats_dict['avg_jaccard']:.4f} ± {stats_dict['std_jaccard']:.4f}")
        print(f"  Cosine:  {stats_dict['avg_cosine']:.4f} ± {stats_dict['std_cosine']:.4f}")
    
    # Store all correlation results
    correlation_results = {
        'pearson_jaccard_diff': {
            'r': pearson_jaccard_diff[0],
            'p': pearson_jaccard_diff[1]
        },
        'pearson_cosine_diff': {
            'r': pearson_cosine_diff[0],
            'p': pearson_cosine_diff[1]
        },
        'spearman_jaccard_diff': {
            'r': spearman_jaccard_diff[0],
            'p': spearman_jaccard_diff[1]
        },
        'spearman_cosine_diff': {
            'r': spearman_cosine_diff[0],
            'p': spearman_cosine_diff[1]
        },
        'pearson_jaccard_avg': {
            'r': pearson_jaccard_avg[0],
            'p': pearson_jaccard_avg[1]
        },
        'pearson_cosine_avg': {
            'r': pearson_cosine_avg[0],
            'p': pearson_cosine_avg[1]
        },
        'spearman_jaccard_avg': {
            'r': spearman_jaccard_avg[0],
            'p': spearman_jaccard_avg[1]
        },
        'spearman_cosine_avg': {
            'r': spearman_cosine_avg[0],
            'p': spearman_cosine_avg[1]
        },
        'difficulty_pair_stats': {
            f"{pair[0]}-{pair[1]}": stats_dict 
            for pair, stats_dict in difficulty_pair_stats.items()
        }
    }
    
    # Data for plotting
    plot_data = {
        'difficulty_a': difficulty_a,
        'difficulty_b': difficulty_b,
        'difficulty_diff': difficulty_diff,
        'difficulty_avg': difficulty_avg,
        'jaccard_scores': jaccard_scores,
        'cosine_scores': cosine_scores
    }
    
    return results, correlation_results, plot_data

# === STEP 5: Save results ===
def save_correlation_results(results, correlation_results, output_json, output_csv):
    """Save results to JSON and CSV files."""
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump({
            'pairs': results,
            'correlation': correlation_results
        }, f, indent=2)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'task_a_id', 'task_b_id', 'repo',
            'difficulty_a', 'difficulty_b', 'difficulty_diff', 'difficulty_avg',
            'jaccard', 'cosine', 'files_a', 'files_b'
        ])
        
        # Write data rows
        for result in results:
            writer.writerow([
                result['task_a_id'],
                result['task_b_id'],
                result['repo'],
                result['difficulty_a'],
                result['difficulty_b'],
                result['difficulty_diff'],
                result['difficulty_avg'],
                result['jaccard'],
                result['cosine'],
                ','.join(result['files_a']),
                ','.join(result['files_b'])
            ])

# === STEP 6: Plot correlation visualizations ===
def plot_correlation_visualizations(plot_data, correlation_results, output_dir):
    """Create and save visualizations of the correlation between difficulty and similarity."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    # Extract data
    difficulty_a = np.array(plot_data['difficulty_a'])
    difficulty_b = np.array(plot_data['difficulty_b'])
    difficulty_diff = np.array(plot_data['difficulty_diff'])
    difficulty_avg = np.array(plot_data['difficulty_avg'])
    jaccard_scores = np.array(plot_data['jaccard_scores'])
    cosine_scores = np.array(plot_data['cosine_scores'])
    
    # Plot 1: Similarity vs. Difficulty Difference
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Jaccard vs Difficulty Difference
    ax1.scatter(difficulty_diff, jaccard_scores, alpha=0.6, s=50)
    # Add regression line
    m, b = np.polyfit(difficulty_diff, jaccard_scores, 1)
    ax1.plot(np.sort(difficulty_diff), m*np.sort(difficulty_diff) + b, color='red', linewidth=2)
    
    r_value = correlation_results['pearson_jaccard_diff']['r']
    p_value = correlation_results['pearson_jaccard_diff']['p']
    
    ax1.set_title(f'Jaccard Similarity vs. Difficulty Difference\nr = {r_value:.3f}, p = {p_value:.3f}', fontsize=14)
    ax1.set_xlabel('Absolute Difficulty Difference', fontsize=12)
    ax1.set_ylabel('Jaccard Similarity', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Cosine vs Difficulty Difference
    ax2.scatter(difficulty_diff, cosine_scores, alpha=0.6, s=50)
    # Add regression line
    m, b = np.polyfit(difficulty_diff, cosine_scores, 1)
    ax2.plot(np.sort(difficulty_diff), m*np.sort(difficulty_diff) + b, color='red', linewidth=2)
    
    r_value = correlation_results['pearson_cosine_diff']['r']
    p_value = correlation_results['pearson_cosine_diff']['p']
    
    ax2.set_title(f'Cosine Similarity vs. Difficulty Difference\nr = {r_value:.3f}, p = {p_value:.3f}', fontsize=14)
    ax2.set_xlabel('Absolute Difficulty Difference', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_vs_difficulty_diff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Similarity vs. Average Difficulty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Jaccard vs Average Difficulty
    ax1.scatter(difficulty_avg, jaccard_scores, alpha=0.6, s=50)
    # Add regression line
    m, b = np.polyfit(difficulty_avg, jaccard_scores, 1)
    ax1.plot(np.sort(difficulty_avg), m*np.sort(difficulty_avg) + b, color='red', linewidth=2)
    
    r_value = correlation_results['pearson_jaccard_avg']['r']
    p_value = correlation_results['pearson_jaccard_avg']['p']
    
    ax1.set_title(f'Jaccard Similarity vs. Average Difficulty\nr = {r_value:.3f}, p = {p_value:.3f}', fontsize=14)
    ax1.set_xlabel('Average Difficulty', fontsize=12)
    ax1.set_ylabel('Jaccard Similarity', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Cosine vs Average Difficulty
    ax2.scatter(difficulty_avg, cosine_scores, alpha=0.6, s=50)
    # Add regression line
    m, b = np.polyfit(difficulty_avg, cosine_scores, 1)
    ax2.plot(np.sort(difficulty_avg), m*np.sort(difficulty_avg) + b, color='red', linewidth=2)
    
    r_value = correlation_results['pearson_cosine_avg']['r']
    p_value = correlation_results['pearson_cosine_avg']['p']
    
    ax2.set_title(f'Cosine Similarity vs. Average Difficulty\nr = {r_value:.3f}, p = {p_value:.3f}', fontsize=14)
    ax2.set_xlabel('Average Difficulty', fontsize=12)
    ax2.set_ylabel('Cosine Similarity', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_vs_avg_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Heatmap of similarity by difficulty pairs
    difficulty_pairs = list(sorted([
        tuple(map(int, pair.split('-'))) 
        for pair in correlation_results['difficulty_pair_stats'].keys()
    ]))
    
    # Create matrices for heatmaps
    unique_difficulties = sorted(list(set([d for pair in difficulty_pairs for d in pair])))
    n_difficulties = len(unique_difficulties)
    
    jaccard_matrix = np.zeros((n_difficulties, n_difficulties))
    cosine_matrix = np.zeros((n_difficulties, n_difficulties))
    count_matrix = np.zeros((n_difficulties, n_difficulties))
    
    for i, d1 in enumerate(unique_difficulties):
        for j, d2 in enumerate(unique_difficulties):
            if i <= j:  # Only upper triangle (including diagonal)
                pair_key = f"{d1}-{d2}"
                if pair_key in correlation_results['difficulty_pair_stats']:
                    stats = correlation_results['difficulty_pair_stats'][pair_key]
                    jaccard_matrix[i, j] = stats['avg_jaccard']
                    cosine_matrix[i, j] = stats['avg_cosine']
                    count_matrix[i, j] = stats['count']
    
    # Make symmetric by mirroring upper triangle to lower triangle
    for i in range(n_difficulties):
        for j in range(i+1, n_difficulties):
            jaccard_matrix[j, i] = jaccard_matrix[i, j]
            cosine_matrix[j, i] = cosine_matrix[i, j]
            count_matrix[j, i] = count_matrix[i, j]
    
    # Jaccard heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(jaccard_matrix, dtype=bool)
    # mask[np.triu_indices_from(mask, k=1)] = True  # Hide upper triangle
    
    ax = sns.heatmap(
        jaccard_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        mask=mask,
        xticklabels=unique_difficulties,
        yticklabels=unique_difficulties,
        cbar_kws={'label': 'Average Jaccard Similarity'}
    )
    plt.title('Average Jaccard Similarity by Difficulty Pair', fontsize=16)
    plt.xlabel('Difficulty', fontsize=14)
    plt.ylabel('Difficulty', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jaccard_by_difficulty_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cosine heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(cosine_matrix, dtype=bool)
    # mask[np.triu_indices_from(mask, k=1)] = True  # Hide upper triangle
    
    ax = sns.heatmap(
        cosine_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        mask=mask,
        xticklabels=unique_difficulties,
        yticklabels=unique_difficulties,
        cbar_kws={'label': 'Average Cosine Similarity'}
    )
    plt.title('Average Cosine Similarity by Difficulty Pair', fontsize=16)
    plt.xlabel('Difficulty', fontsize=14)
    plt.ylabel('Difficulty', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_by_difficulty_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Count heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(count_matrix, dtype=bool)
    # mask[np.triu_indices_from(mask, k=1)] = True  # Hide upper triangle
    
    ax = sns.heatmap(
        count_matrix,
        annot=True,
        fmt=".0f",  # Use float format that displays as integer
        cmap="Greens",
        mask=mask,
        xticklabels=unique_difficulties,
        yticklabels=unique_difficulties,
        cbar_kws={'label': 'Number of Pairs'}
    )
    plt.title('Number of Task Pairs by Difficulty', fontsize=16)
    plt.xlabel('Difficulty', fontsize=14)
    plt.ylabel('Difficulty', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'count_by_difficulty_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths
    json_path = os.path.join(os.path.dirname(__file__), 'SWE-Bench-CL.json')
    output_json = os.path.join(os.path.dirname(__file__), 'difficulty_correlation_results.json')
    output_csv = os.path.join(os.path.dirname(__file__), 'difficulty_correlation_table.csv')
    output_plots_dir = os.path.join(os.path.dirname(__file__), 'difficulty_correlation_plots')
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Load the dataset
    print("Loading SWE-Bench-CL dataset...")
    data = load_data(json_path)
    
    # Step 2: Extract tasks with difficulty scores
    print("Extracting tasks with difficulty information...")
    tasks = extract_tasks_with_difficulty(data)
    
    # Step 3: Sample task pairs
    print("Sampling task pairs with disjoint modified files...")
    pairs = sample_task_pairs(tasks)
    print(f"Sampled {len(pairs)} task pairs.")
    
    # Step 4: Process pairs and compute correlation metrics
    print("Computing similarity metrics and correlation...")
    results, correlation_results, plot_data = process_pairs_continuous(pairs)
    
    # Step 5: Save results
    print("Saving results...")
    save_correlation_results(results, correlation_results, output_json, output_csv)
    
    # Step 6: Plot correlation visualizations
    print("Generating correlation visualizations...")
    plot_correlation_visualizations(plot_data, correlation_results, output_plots_dir)
    
    print(f"Analysis complete. Results saved to {output_json}, {output_csv}, and visualizations in {output_plots_dir}/")

if __name__ == "__main__":
    main()
