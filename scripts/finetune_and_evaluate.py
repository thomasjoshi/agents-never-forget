#!/usr/bin/env python3
"""
Sequential fine-tuning and evaluation for continual learning on SWE-Bench-CL.

This script loads a model, fine-tunes it sequentially on tasks from a continual learning
stream, and evaluates forgetting and forward transfer metrics across all tasks.
"""

import argparse
import json
import os
import sys
import csv
import logging
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Hugging Face Transformers
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer, 
        TrainingArguments, 
        Trainer, 
        default_data_collator,
        get_scheduler
    )
except ImportError:
    print("Error: This script requires the transformers library. Install with: pip install transformers", 
          file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SWEBenchDataset(Dataset):
    """Dataset for SWE-Bench code tasks."""
    
    def __init__(self, tasks, tokenizer, max_length=1024):
        """
        Initialize dataset with tasks.
        
        Args:
            tasks (list): List of task dictionaries
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length for tokenization
        """
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        # Format input (customize this based on your model and task format)
        task_input = f"Task: {task['title']}\n\n"
        task_input += f"Problem: {task['body']}\n\n"
        
        # Format output - this could be the patch or other task solution
        task_output = task.get('patch', '')
        
        # Prepare for causal language modeling (text generation)
        full_text = f"{task_input}\nSolution: {task_output}"
        
        # Tokenize inputs with attention mask
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare for causal LM training (shift labels)
        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]
        labels = input_ids.clone()
        
        # For causal LM, we use -100 to mask the input part from loss computation
        # Find the "Solution:" marker to separate input from output
        solution_tokens = self.tokenizer.encode("Solution:", add_special_tokens=False)
        solution_start = None
        
        for i in range(len(input_ids) - len(solution_tokens)):
            if input_ids[i:i+len(solution_tokens)].tolist() == solution_tokens:
                solution_start = i + len(solution_tokens)
                break
        
        if solution_start:
            # Mask input part from loss
            labels[:solution_start] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "task_id": f"{task['repo']}/{task['issue_id']}"
        }

def prepare_datasets(task_stream, tokenizer, max_length=1024):
    """
    Prepare datasets for all tasks.
    
    Args:
        task_stream (dict): Task stream dictionary
        tokenizer: HuggingFace tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Dictionary mapping task names to datasets
    """
    datasets = {}
    metadata = task_stream.get("metadata", {})
    task_ordering = metadata.get("task_ordering", [])
    
    for task_name in task_ordering:
        tasks = task_stream.get(task_name, [])
        if tasks:
            datasets[task_name] = SWEBenchDataset(tasks, tokenizer, max_length)
            logger.info(f"Prepared dataset for {task_name} with {len(tasks)} examples")
        else:
            logger.warning(f"No tasks found for {task_name}")
    
    return datasets

def evaluate_model(model, dataset, tokenizer, device, batch_size=4):
    """
    Evaluate model performance on a dataset.
    
    Args:
        model: HuggingFace model
        dataset: Dataset instance
        tokenizer: HuggingFace tokenizer
        device: PyTorch device
        batch_size (int): Batch size for evaluation
        
    Returns:
        float: Accuracy score (or other relevant metric)
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(**{k: v for k, v in batch.items() 
                             if k in ["input_ids", "attention_mask", "labels"]})
            
            total_loss += outputs.loss.item()
    
    # Convert loss to "accuracy" metric (1 - normalized loss)
    # This is just one approach - you might want a different metric
    avg_loss = total_loss / len(dataloader)
    
    # Higher is better (1 = perfect, 0 = worst)
    accuracy = max(0, 1 - avg_loss / 10)  # Normalize loss to 0-1 range
    
    return accuracy

def finetune_model(model, dataset, tokenizer, device, learning_rate=5e-5, 
                  epochs=3, batch_size=4, output_dir=None):
    """
    Fine-tune model on a dataset.
    
    Args:
        model: HuggingFace model
        dataset: Dataset instance
        tokenizer: HuggingFace tokenizer
        device: PyTorch device
        learning_rate (float): Learning rate
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        output_dir (str): Directory to save checkpoints (optional)
        
    Returns:
        model: Fine-tuned model
    """
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir if output_dir else "./temp_trainer",
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs") if output_dir else None,
        logging_steps=10,
        save_total_limit=1,
        save_steps=500,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    
    # Train the model
    trainer.train()
    
    return model

def prompt_based_update(model, dataset, tokenizer, device):
    """
    Update model using prompt-based learning (no parameter updates).
    This is just a placeholder - implement based on your specific approach.
    
    Args:
        model: HuggingFace model
        dataset: Dataset instance
        tokenizer: HuggingFace tokenizer
        device: PyTorch device
        
    Returns:
        model: Model (unchanged in this case)
    """
    # For prompt-based learning, we might not update the model at all,
    # but instead prepare prompt templates, examples, etc.
    logger.info("Using prompt-based learning (no parameter updates)")
    
    # This is where you would implement your prompt engineering approach
    # For this skeleton code, we just return the unchanged model
    return model

def track_metrics(task_accuracies, current_task, seen_tasks):
    """
    Calculate forgetting and other metrics.
    
    Args:
        task_accuracies (dict): Dictionary mapping task names to list of accuracies
        current_task (str): Name of current task
        seen_tasks (list): List of previously seen tasks
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "current_task": current_task,
        "current_accuracy": task_accuracies[current_task][-1],
        "avg_accuracy": np.mean([task_accuracies[t][-1] for t in seen_tasks]),
        "task_accuracies": {t: task_accuracies[t][-1] for t in seen_tasks},
        "forgetting": {},
        "forward_transfer": {}
    }
    
    # Calculate forgetting for each previously seen task
    for task in seen_tasks:
        if task == current_task:
            # No forgetting for current task
            metrics["forgetting"][task] = 0
            continue
            
        if len(task_accuracies[task]) >= 2:
            # Forgetting = max previous accuracy - current accuracy
            max_prev_acc = max(task_accuracies[task][:-1])
            curr_acc = task_accuracies[task][-1]
            forgetting = max(0, max_prev_acc - curr_acc)
            metrics["forgetting"][task] = forgetting
    
    # Calculate average forgetting
    if len(metrics["forgetting"]) > 0:
        metrics["avg_forgetting"] = np.mean(list(metrics["forgetting"].values()))
    else:
        metrics["avg_forgetting"] = 0
        
    return metrics

def plot_results(metrics_history, output_dir):
    """
    Create plots for accuracy and forgetting.
    
    Args:
        metrics_history (list): List of metrics dictionaries
        output_dir (str): Directory to save plots
    """
    # Extract data for plotting
    tasks = []
    avg_accuracies = []
    avg_forgetting = []
    
    for metrics in metrics_history:
        tasks.append(metrics["current_task"])
        avg_accuracies.append(metrics["avg_accuracy"])
        avg_forgetting.append(metrics.get("avg_forgetting", 0))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot average accuracy
    ax1.plot(tasks, avg_accuracies, 'o-', label='Average Accuracy')
    ax1.set_title('Average Accuracy Across Tasks')
    ax1.set_xlabel('Current Task')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # Plot average forgetting
    ax2.plot(tasks, avg_forgetting, 'o-', color='red', label='Average Forgetting')
    ax2.set_title('Average Forgetting Across Tasks')
    ax2.set_xlabel('Current Task')
    ax2.set_ylabel('Forgetting')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'cl_metrics.png')
    plt.savefig(plot_path)
    logger.info(f"Saved plots to {plot_path}")
    
    # Close figure to free memory
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate models on task streams")
    parser.add_argument("--model_name_or_path", required=True, 
                        help="Hugging Face model name or path")
    parser.add_argument("--task_stream_file", required=True, 
                        help="Path to task stream JSON file")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate for fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs per task")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--output_dir", default="./cl_results", 
                        help="Directory to save results")
    parser.add_argument("--update_method", choices=["finetune", "prompt"], default="finetune",
                        help="Method for updating model (fine-tuning or prompt-based)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load task stream
    try:
        with open(args.task_stream_file, 'r') as f:
            task_stream = json.load(f)
        logger.info(f"Loaded task stream from {args.task_stream_file}")
    except Exception as e:
        print(f"Error loading task stream: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get task ordering
    metadata = task_stream.get("metadata", {})
    task_ordering = metadata.get("task_ordering", [])
    
    if not task_ordering:
        print("Error: Task stream has no task ordering in metadata", file=sys.stderr)
        sys.exit(1)
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = "</s>"
        
        model.to(device)
        logger.info(f"Loaded model: {args.model_name_or_path}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Prepare datasets
    datasets = prepare_datasets(task_stream, tokenizer, args.max_length)
    
    # Track metrics for each task
    task_accuracies = {task_name: [] for task_name in task_ordering}
    metrics_history = []
    
    # CSV for saving results
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['current_task', 'task', 'accuracy', 'forgetting', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sequential training and evaluation
        seen_tasks = []
        
        for task_idx, current_task in enumerate(task_ordering):
            logger.info(f"==== Task {task_idx+1}/{len(task_ordering)}: {current_task} ====")
            
            # Get current task dataset
            if current_task not in datasets:
                logger.warning(f"Skipping task {current_task} - no dataset available")
                continue
                
            current_dataset = datasets[current_task]
            seen_tasks.append(current_task)
            
            # Update model based on specified method
            if args.update_method == "finetune":
                # Fine-tune on current task
                task_output_dir = os.path.join(args.output_dir, f"checkpoint_{current_task}")
                os.makedirs(task_output_dir, exist_ok=True)
                
                model = finetune_model(
                    model, 
                    current_dataset, 
                    tokenizer, 
                    device,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    output_dir=task_output_dir
                )
            else:
                # Prompt-based learning
                model = prompt_based_update(model, current_dataset, tokenizer, device)
            
            # Evaluate on all seen tasks
            logger.info(f"Evaluating on all {len(seen_tasks)} seen tasks")
            
            for task_name in seen_tasks:
                task_dataset = datasets[task_name]
                accuracy = evaluate_model(model, task_dataset, tokenizer, device, args.batch_size)
                task_accuracies[task_name].append(accuracy)
                
                # Save to CSV
                writer.writerow({
                    'current_task': current_task,
                    'task': task_name,
                    'accuracy': accuracy,
                    'forgetting': (max(task_accuracies[task_name]) - accuracy) 
                                 if len(task_accuracies[task_name]) > 1 else 0,
                    'timestamp': datetime.now().isoformat()
                })
                csvfile.flush()  # Make sure data is written to disk
                
                if task_name == current_task:
                    logger.info(f"Accuracy on current task ({task_name}): {accuracy:.4f}")
                else:
                    prev_max = max(task_accuracies[task_name][:-1]) if len(task_accuracies[task_name]) > 1 else 0
                    forgetting = max(0, prev_max - accuracy)
                    logger.info(f"Accuracy on previous task {task_name}: {accuracy:.4f} (forgetting: {forgetting:.4f})")
            
            # Calculate and display metrics
            metrics = track_metrics(task_accuracies, current_task, seen_tasks)
            metrics_history.append(metrics)
            
            logger.info(f"Average accuracy: {metrics['avg_accuracy']:.4f}")
            if task_idx > 0:
                logger.info(f"Average forgetting: {metrics['avg_forgetting']:.4f}")
            logger.info("=" * 50)
        
        # Plot final results
        plot_results(metrics_history, args.output_dir)
        
        # Save final summary
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"SWE-Bench-CL Continual Learning Experiment\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"Model: {args.model_name_or_path}\n")
            f.write(f"Update method: {args.update_method}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"Average accuracy across all tasks: {metrics_history[-1]['avg_accuracy']:.4f}\n")
            
            if len(metrics_history) > 1:
                f.write(f"Average forgetting: {metrics_history[-1]['avg_forgetting']:.4f}\n\n")
            
            f.write("Task-specific accuracies:\n")
            for task in task_ordering:
                if task in task_accuracies and task_accuracies[task]:
                    f.write(f"  {task}: {task_accuracies[task][-1]:.4f}\n")
            
        logger.info(f"Experiment complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)
