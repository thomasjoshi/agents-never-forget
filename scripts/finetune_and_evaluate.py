#!/usr/bin/env python
"""
Finetune and evaluate a model on a sequence of continual learning tasks.

This script loads a model, finetunes it sequentially on a stream of tasks,
and evaluates catastrophic forgetting and forward transfer.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_scheduler
)

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SWEBenchDataset(Dataset):
    """Dataset for SWE-Bench tasks."""
    
    def __init__(self, examples, tokenizer, max_length=1024):
        """
        Initialize dataset.
        
        Args:
            examples: List of examples from the task
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum length of encoded sequences
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Construct prompt from problem statement
        prompt = f"Fix the following issue in the code:\n\n{example['problem_statement']}"
        
        # Get ground truth solution (if available)
        solution = example.get('patch', '')
        if not solution:
            # Fallback to changes if patch not available
            solution = example.get('changes', '')
        
        # Combine for training
        combined = f"{prompt}\n\nSolution:\n{solution}"
        
        # Tokenize
        encoded = self.tokenizer(
            combined, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the input_ids
        item = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": encoded["input_ids"][0].clone()
        }
        
        return item


def load_task_stream(task_stream_file):
    """
    Load the task stream from a JSON file.
    
    Args:
        task_stream_file: Path to task stream JSON file
    
    Returns:
        Dictionary containing task stream data
    """
    try:
        with open(task_stream_file, 'r') as f:
            task_stream = json.load(f)
        
        # Verify that the file contains the expected structure
        if "metadata" not in task_stream:
            print("Warning: Task stream file does not contain metadata.", file=sys.stderr)
        
        # Get task ordering from metadata if available
        if "metadata" in task_stream and "task_ordering" in task_stream["metadata"]:
            task_ordering = task_stream["metadata"]["task_ordering"]
        else:
            # Default: sort task keys (excluding metadata)
            task_ordering = [k for k in task_stream.keys() if k != "metadata"]
            task_ordering.sort()
        
        print(f"Loaded task stream with {len(task_ordering)} tasks")
        print(f"Task ordering: {task_ordering}")
        
        return task_stream, task_ordering
    
    except Exception as e:
        print(f"Error loading task stream: {e}", file=sys.stderr)
        sys.exit(1)


def evaluate_model(model, tokenizer, task_examples, device, batch_size=4):
    """
    Evaluate model on a task.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
    
    Returns:
        Accuracy score for the task
    """
    model.eval()
    dataset = SWEBenchDataset(task_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
    
    # Convert loss to accuracy-like metric (lower is better for loss, higher is better for accuracy)
    # This is a simplified approach - in a real scenario, you'd evaluate if the model can generate correct code
    avg_loss = total_loss / len(dataloader)
    pseudo_accuracy = 1.0 - min(avg_loss / 10.0, 0.99)  # Normalize loss to a 0-1 scale
    
    return pseudo_accuracy


def finetune_model(model, tokenizer, task_examples, learning_rate, device, num_epochs=3, batch_size=4):
    """
    Finetune model on a task.
    
    Args:
        model: Model to finetune
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        learning_rate: Learning rate for finetuning
        device: Device to run finetuning on
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
    
    Returns:
        Finetuned model
    """
    model.train()
    dataset = SWEBenchDataset(task_examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(dataloader) * num_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item()})
    
    return model


def prompt_based_update(model, tokenizer, task_examples, device):
    """
    Add task examples to prompt set (no parameter updates).
    
    Args:
        model: Model to use
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        device: Device
    
    Returns:
        Model (unchanged) - this is a placeholder for prompt-based methods
    """
    # In a real prompt-based approach, you might store examples in a retrieval database
    # For simplicity, we're just demonstrating the API here
    print("Prompt-based update: storing examples without parameter updates")
    return model


def main():
    parser = argparse.ArgumentParser(description="Finetune and evaluate a model on a continual learning task stream")
    parser.add_argument("--model_name_or_path", required=True, help="HuggingFace model name or path")
    parser.add_argument("--task_stream_file", required=True, help="Path to task stream JSON file")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for finetuning")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--update_method", choices=["finetune", "prompt"], default="finetune",
                        help="Method to update model: finetune (full parameter update) or prompt (no parameter update)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs per task")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load model and tokenizer
        print(f"Loading model: {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        
        # If the tokenizer doesn't have a pad token, set it to eos token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device(args.device)
        model.to(device)
        
        # Load task stream
        task_stream, task_ordering = load_task_stream(args.task_stream_file)
        
        # Initialize metrics tracking
        metrics = {
            "task": [],
            "task_accuracy": [],
            "forgetting": [],
            "mean_accuracy": []
        }
        
        # Track per-task accuracy over time
        task_accuracies = {}
        
        # Iterate through tasks in order
        for task_idx, task_name in enumerate(task_ordering):
            print(f"\n{'='*50}")
            print(f"Task {task_idx+1}/{len(task_ordering)}: {task_name}")
            
            task_examples = task_stream[task_name]
            print(f"Examples in task: {len(task_examples)}")
            
            # Evaluate model on current task (before update)
            print("Evaluating on current task (before update)...")
            current_task_acc_before = evaluate_model(model, tokenizer, task_examples, device, args.batch_size)
            
            # Update model
            if args.update_method == "finetune":
                print(f"Finetuning model on {task_name}...")
                model = finetune_model(
                    model, tokenizer, task_examples, args.learning_rate, device, args.num_epochs, args.batch_size
                )
            else:  # prompt
                print(f"Updating prompt store with {task_name}...")
                model = prompt_based_update(model, tokenizer, task_examples, device)
            
            # Evaluate model on all seen tasks
            print("Evaluating on all seen tasks...")
            all_task_accuracies = {}
            
            for prev_task_name in task_ordering[:task_idx+1]:
                prev_task_examples = task_stream[prev_task_name]
                task_acc = evaluate_model(model, tokenizer, prev_task_examples, device, args.batch_size)
                all_task_accuracies[prev_task_name] = task_acc
                
                # Store accuracy for this task at this timestep
                if prev_task_name not in task_accuracies:
                    task_accuracies[prev_task_name] = [task_acc]
                else:
                    task_accuracies[prev_task_name].append(task_acc)
            
            # Compute mean accuracy across all seen tasks
            mean_accuracy = np.mean(list(all_task_accuracies.values()))
            
            # Compute forgetting for previous tasks
            forgetting = 0.0
            if task_idx > 0:
                forgetting_values = []
                for prev_task_name in task_ordering[:task_idx]:
                    # Forgetting = maximum previous accuracy - current accuracy
                    max_prev_acc = max(task_accuracies[prev_task_name][:-1])  # max accuracy before current update
                    current_acc = task_accuracies[prev_task_name][-1]         # accuracy after current update
                    task_forgetting = max(0, max_prev_acc - current_acc)      # forgetting cannot be negative
                    forgetting_values.append(task_forgetting)
                
                forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
            
            # Save metrics
            metrics["task"].append(task_name)
            metrics["task_accuracy"].append(all_task_accuracies[task_name])
            metrics["forgetting"].append(forgetting)
            metrics["mean_accuracy"].append(mean_accuracy)
            
            # Print current metrics
            print(f"\nMetrics after task {task_name}:")
            print(f"  Current task accuracy: {all_task_accuracies[task_name]:.4f}")
            print(f"  Mean accuracy across seen tasks: {mean_accuracy:.4f}")
            print(f"  Forgetting: {forgetting:.4f}")
            
            # Save current model if needed
            model_output_dir = os.path.join(args.output_dir, f"model_after_{task_name}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Optionally save model weights - comment out if not needed
            # model.save_pretrained(model_output_dir)
            # tokenizer.save_pretrained(model_output_dir)
            
            # Save task accuracies after each task
            task_acc_df = pd.DataFrame({
                task_name: accs + [None] * (len(task_ordering) - len(accs))
                for task_name, accs in task_accuracies.items()
            })
            task_acc_df.index = [f"After_{task}" for task in task_ordering[:len(task_acc_df)]]
            task_acc_df.to_csv(os.path.join(args.output_dir, "task_accuracies.csv"))
        
        # Convert metrics to DataFrame and save
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
        print(f"\nSaved metrics to {os.path.join(args.output_dir, 'metrics.csv')}")
        
        # Plot accuracy and forgetting
        plt.figure(figsize=(12, 8))
        
        # Accuracy plot
        plt.subplot(2, 1, 1)
        plt.plot(metrics["task"], metrics["task_accuracy"], marker='o', label="Current task accuracy")
        plt.plot(metrics["task"], metrics["mean_accuracy"], marker='s', label="Mean accuracy (seen tasks)")
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over tasks")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Forgetting plot
        plt.subplot(2, 1, 2)
        plt.plot(metrics["task"][1:], metrics["forgetting"][1:], marker='d', color='red')
        plt.xlabel("Task")
        plt.ylabel("Forgetting")
        plt.title("Forgetting over tasks")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "learning_curves.png"))
        print(f"Saved learning curves to {os.path.join(args.output_dir, 'learning_curves.png')}")
        
        print("\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
