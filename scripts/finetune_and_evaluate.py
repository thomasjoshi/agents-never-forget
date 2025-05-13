#!/usr/bin/env python
"""Finetune and evaluate a model on a sequence of continual learning tasks.

This script loads CodeLlama-3B-Instruct with quantization, finetunes it sequentially on SWE-Bench-CL tasks,
and evaluates catastrophic forgetting and forward transfer.
"""

import argparse
import json
import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil

# Required for loading and quantizing the model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig
)

# Required for parameter-efficient fine-tuning
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftConfig,
    PeftModel
)

# For accelerator and mixed precision training
from accelerate import Accelerator

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Constants
DEFAULT_MODEL = "codellama/CodeLlama-3B-Instruct-hf"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# Check for GPU availability
def check_gpu_availability():
    """Check if CUDA is available and provide helpful guidance if not."""
    if not torch.cuda.is_available():
        print("\n" + "="*80)
        print("ERROR: No CUDA device detected. This script requires a GPU to run efficiently.")
        print("\nTo set up an A100 VM on Google Cloud Platform (GCP):")
        print("  1. Visit https://console.cloud.google.com/")
        print("  2. Create a new VM instance with:")
        print("     - Machine type: n1-standard-16 (16 vCPUs, 60GB memory)")
        print("     - GPU: 1x NVIDIA A100 40GB or 2x NVIDIA T4")
        print("     - Boot disk: Ubuntu 20.04 LTS with at least 100GB disk space")
        print("  3. Install required packages:")
        print("     - CUDA drivers (https://developer.nvidia.com/cuda-downloads)")
        print("     - PyTorch with CUDA support")
        print("     - Required libraries: transformers, peft, bitsandbytes, accelerate")
        print("\nAlternatively, consider using Google Colab Pro with a GPU runtime.")
        print("="*80)
        sys.exit(1)
    else:
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Get memory usage information
def get_memory_usage():
    """Get current memory usage stats."""
    # GPU memory
    gpu_memory = {
        "allocated": torch.cuda.memory_allocated() / 1e9,  # Convert to GB
        "cached": torch.cuda.memory_reserved() / 1e9     # Convert to GB
    }
    
    # CPU memory
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1e9  # Convert to GB
    
    return {
        "gpu": gpu_memory,
        "cpu": cpu_memory
    }


class SWEBenchDataset(Dataset):
    """Dataset for SWE-Bench tasks."""
    
    def __init__(self, examples, tokenizer, max_length=1024, use_memory=False, memory_examples=None):
        """
        Initialize dataset.
        
        Args:
            examples: List of examples from the task
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum length of encoded sequences
            use_memory: Whether to include memory in the training examples
            memory_examples: List of memory examples to include (from previous tasks)
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_memory = use_memory
        self.memory_examples = memory_examples if memory_examples else []
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the instruction for CodeLlama-Instruct model
        repo = example.get('repo', 'unknown')
        instance_id = example.get('instance_id', 'unknown')
        
        # Construct prompt from problem statement with CodeLlama-Instruct format
        instruction = f"You are a helpful coding assistant. Fix the following issue in the {repo} repository (issue {instance_id}):"
        problem = example.get('problem_statement', '')
        
        # Get code context (e.g., patch, base_commit, etc.)
        context = ""
        if 'patch' in example and example['patch']:
            context += f"\nHere is the current code:\n```\n{example['patch']}\n```"
        
        # Include memory examples if enabled
        memory_context = ""
        if self.use_memory and self.memory_examples:
            # Include up to 2 relevant memory examples
            memory_examples = self.memory_examples
            if not isinstance(memory_examples, list):
                memory_examples = [memory_examples]
                
            for i, mem_ex in enumerate(memory_examples[:2]):
                problem = mem_ex.get('problem_statement', mem_ex.get('description', ''))
                solution = mem_ex.get('patch', mem_ex.get('solution', ''))
                memory_context += f"\n\nRELEVANT_EXAMPLE {i+1}:\nProblem: {str(problem)[:300]}\nSolution: {str(solution)[:300]}"
        
        # Get ground truth solution
        solution = example.get('patch', '')
        if not solution:
            # Fallback to other fields if patch not available
            solution = example.get('changes', example.get('solution', ''))
        
        # Combine components in CodeLlama-Instruct format
        if self.use_memory and memory_context:
            combined = f"<s>[INST] {instruction}\n\n{problem}\n\nPrevious related examples:{memory_context} [/INST]\n\n{solution}</s>"
        else:
            combined = f"<s>[INST] {instruction}\n\n{problem}{context} [/INST]\n\n{solution}</s>"
        
        # Tokenize
        encoded = self.tokenizer(
            combined, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the input_ids
        # But set them to -100 for the prompt part to avoid computing loss there
        labels = encoded["input_ids"][0].clone()
        
        # Find the [/INST] token position to only compute loss on the completion
        inst_tokens = self.tokenizer.encode(" [/INST]")
        
        if len(inst_tokens) > 0:
            # Find position of [/INST] token
            inst_token_pos = -1
            for i in range(len(labels) - len(inst_tokens) + 1):
                if labels[i:i+len(inst_tokens)].tolist() == inst_tokens:
                    inst_token_pos = i + len(inst_tokens) - 1
                    break
            
            # Set labels to -100 for the instruction part
            if inst_token_pos >= 0:
                labels[:inst_token_pos+1] = -100
        
        item = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels
        }
        
        return item


def load_model_and_tokenizer(model_name=DEFAULT_MODEL, use_8bit=True, use_lora=True, device="cuda"):
    """
    Load and prepare CodeLlama model with quantization.
    
    Args:
        model_name: HF model name or path
        use_8bit: Whether to use 8-bit quantization
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        device: Device to load the model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading model: {model_name}")
    print(f"Quantization: {'8-bit' if use_8bit else 'None'}")
    print(f"Using LoRA: {use_lora}")
    
    # Configure quantization
    if use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization config
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible solutions:")
        print("1. Check that you have enough GPU memory (try smaller model or increase quantization)")
        print("2. Ensure you have the latest versions of transformers and bitsandbytes installed")
        print("3. Try a different model checkpoint")
        sys.exit(1)
    
    # Apply LoRA for parameter-efficient fine-tuning
    if use_lora:
        lora_config = LoraConfig(
            r=16,               # Rank
            lora_alpha=32,      # Alpha parameter for LoRA scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Create PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Report memory usage after model loading
    memory_usage = get_memory_usage()
    print(f"Memory after model loading - GPU: {memory_usage['gpu']['allocated']:.2f} GB, "
          f"CPU: {memory_usage['cpu']:.2f} GB")
    
    return model, tokenizer


def load_task_stream(task_stream_file, filter_repo=None):
    """
    Load the task stream from a JSON file and optionally filter for specific repo.
    
    Args:
        task_stream_file: Path to task stream JSON file
        filter_repo: Optional repository name to filter tasks (e.g., 'sympy/sympy')
    
    Returns:
        Dictionary containing task stream data and task ordering
    """
    try:
        print(f"Loading task stream from {task_stream_file}...")
        with open(task_stream_file, 'r') as f:
            task_stream = json.load(f)
        
        # Verify that the file contains the expected structure
        if not task_stream or (not isinstance(task_stream, dict) or "tasks" not in task_stream):
            raise ValueError("Invalid task stream format. Expected a dictionary with a 'tasks' key.")
        
        # Get the list of tasks
        tasks = task_stream.get("tasks", [])
        
        # Filter tasks by repository if requested
        if filter_repo:
            print(f"Filtering tasks for repository: {filter_repo}")
            tasks = [task for task in tasks if task.get("repo", "").lower() == filter_repo.lower()]
        
        if not tasks:
            raise ValueError(f"No tasks found{f' for repository {filter_repo}' if filter_repo else ''}")
        
        # Group tasks by their task name (e.g., 'task1', 'task2', etc.)
        task_groups = {}
        for task in tasks:
            task_name = task.get("task_name", "unknown")
            if task_name not in task_groups:
                task_groups[task_name] = []
            task_groups[task_name].append(task)
        
        # Create task ordering based on task names
        task_ordering = sorted(task_groups.keys())
        
        # Create the task stream structure expected by the rest of the code
        task_stream = {
            "metadata": {
                "task_ordering": task_ordering,
                "total_tasks": len(task_ordering),
                "total_examples": sum(len(examples) for examples in task_groups.values()),
                "filtered_repo": filter_repo,
                "source_file": task_stream_file
            }
        }
        
        # Add task groups to the task stream
        task_stream.update(task_groups)
        
        # Print task distribution
        print(f"\nTask distribution:")
        for task_name in task_ordering:
            print(f"  - {task_name}: {len(task_groups[task_name])} examples")
        print(f"\nTotal tasks: {len(task_ordering)}")
        print(f"Total examples: {sum(len(examples) for examples in task_groups.values())}")
        
        return task_stream, task_ordering
    
    except Exception as e:
        print(f"Error loading task stream: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


def evaluate_model(model, tokenizer, task_examples, device, batch_size=4, use_memory=False, memory_examples=None):
    """
    Evaluate model on a task.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        use_memory: Whether to include memory in evaluation examples
        memory_examples: Examples from previous tasks to use as memory
    
    Returns:
        Accuracy score for the task
    """
    model.eval()
    dataset = SWEBenchDataset(task_examples, tokenizer, use_memory=use_memory, memory_examples=memory_examples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=default_data_collator,
        shuffle=False  # No need to shuffle for evaluation
    )
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nWARNING: GPU OOM during evaluation. Trying with smaller batch...")
                    # Clear cache
                    torch.cuda.empty_cache()
                    
                    # Try with batch size of 1
                    try:
                        # Process single items
                        for i in range(len(batch['input_ids'])):
                            single_batch = {
                                k: v[i:i+1].to(device) for k, v in batch.items()
                            }
                            outputs = model(**single_batch)
                            total_loss += outputs.loss.item()
                            num_batches += 1
                    except Exception as e2:
                        print(f"Single-item processing also failed: {e2}")
                        continue
                else:
                    print(f"Error during evaluation: {e}")
                    continue
    
    # Report memory usage
    memory_usage = get_memory_usage()
    print(f"Memory during evaluation - GPU: {memory_usage['gpu']['allocated']:.2f} GB")
    
    # Calculate pseudo-accuracy from loss
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        # Convert loss to accuracy-like metric (0-1 scale)
        # Lower loss → higher accuracy
        pseudo_accuracy = 1.0 - min(avg_loss / 10.0, 0.99)
        return pseudo_accuracy
    else:
        print("WARNING: No valid batches during evaluation")
        return 0.0  # Default value if evaluation failed


def finetune_model(model, tokenizer, task_examples, learning_rate, device, num_epochs=3, batch_size=4, 
                gradient_accumulation_steps=1, use_memory=False, memory_examples=None):
    """
    Finetune model on a task using LoRA with quantization.
    
    Args:
        model: Model to finetune
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        learning_rate: Learning rate for finetuning
        device: Device to run finetuning on
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients for
        use_memory: Whether to include memory in training examples
        memory_examples: Examples from previous tasks to use as memory
    
    Returns:
        Finetuned model
    """
    model.train()
    
    # Create dataset with memory if enabled
    dataset = SWEBenchDataset(task_examples, tokenizer, use_memory=use_memory, memory_examples=memory_examples)
    
    # Create dataloader with appropriate batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Batch size: {batch_size} (effective: {effective_batch_size} with {gradient_accumulation_steps} gradient accumulation steps)")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=default_data_collator
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    num_training_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    num_warmup_steps = min(100, int(num_training_steps * 0.1))  # 10% warmup
    
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop with gradient accumulation and error handling
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    accumulated_loss = 0.0
    step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps  # Scale loss for accumulation
                accumulated_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Track epoch stats
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1
                
                # Check if we should update weights
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(dataloader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update progress bar
                    step += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": accumulated_loss})
                    accumulated_loss = 0.0
                    
                    # Print memory usage every 10 steps
                    if step % 10 == 0:
                        memory_usage = get_memory_usage()
                        print(f"\nStep {step}/{num_training_steps}:")
                        print(f"  GPU memory: {memory_usage['gpu']['allocated']:.2f} GB allocated, {memory_usage['gpu']['cached']:.2f} GB cached")
                        print(f"  Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nWARNING: GPU OOM in batch {batch_idx+1}. Skipping to next batch and clearing cache...")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    # Try reducing batch size for next epoch
                    if batch_size > 1 and epoch < num_epochs - 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"Reducing batch size to {batch_size} for next epoch")
                else:
                    print(f"Error during training: {e}")
        
        # Print epoch stats
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch+1}/{num_epochs} completed:")
            print(f"  Average loss: {avg_epoch_loss:.4f}")
    
    # Report final memory usage
    memory_usage = get_memory_usage()
    print(f"Memory after training - GPU: {memory_usage['gpu']['allocated']:.2f} GB")
    
    # Return the fine-tuned model
    return model


def save_model_checkpoint(model, tokenizer, output_dir, task_name):
    """
    Save model checkpoint after training on a task.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Base directory for checkpoints
        task_name: Name of the current task
    """
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, f"checkpoint_after_{task_name}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Saving model checkpoint to {checkpoint_dir}...")
    
    # For LoRA models, save adapter weights
    try:
        model.save_pretrained(checkpoint_dir)
        # Also save tokenizer
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Successfully saved checkpoint for {task_name}")
        return checkpoint_dir
    except Exception as e:
        print(f"Error saving model checkpoint: {e}")
        return None


def plot_metrics(metrics_df, output_dir):
    """
    Plot accuracy and forgetting metrics with enhanced visualization.
    
    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Colors for with/without memory
    colors = {True: '#1f77b4', False: '#ff7f0e'}
    
    # Plot accuracy over time
    for memory_mode, subset in metrics_df.groupby('memory_enabled'):
        color = colors[memory_mode]
        label = 'With Memory' if memory_mode else 'Without Memory'
        
        # Plot accuracy
        ax1.plot(
            subset['task_index'], 
            subset['task_accuracy'], 
            marker='o',
            linestyle='-',
            color=color,
            label=label,
            alpha=0.8,
            linewidth=2
        )
        
        # Plot mean accuracy
        ax1.axhline(
            y=subset['mean_accuracy'].iloc[-1],
            color=color,
            linestyle='--',
            alpha=0.5,
            linewidth=1.5,
            label=f'{label} (Avg: {subset["mean_accuracy"].iloc[-1]:.2f})'
        )
    
    # Configure accuracy plot
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Task Accuracy Over Time', fontsize=14, pad=15)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(0, 1.05)  # 0-1 scale for accuracy
    
    # Plot forgetting over time
    for memory_mode, subset in metrics_df.groupby('memory_enabled'):
        color = colors[memory_mode]
        label = 'With Memory' if memory_mode else 'Without Memory'
        
        # Plot forgetting
        ax2.plot(
            subset['task_index'], 
            subset['mean_forgetting'],
            marker='s',
            linestyle='-',
            color=color,
            label=label,
            alpha=0.8,
            linewidth=2
        )
    
    # Configure forgetting plot
    ax2.set_xlabel('Task Index', fontsize=12)
    ax2.set_ylabel('Forgetting', fontsize=12)
    ax2.set_title('Catastrophic Forgetting Over Time', fontsize=14, pad=15)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(-0.05, 1.05)  # 0-1 scale for forgetting
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save high-resolution version
    plot_path = os.path.join(output_dir, 'continual_learning_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Also save individual plots
    plt.figure(figsize=(12, 5))
    
    # Save accuracy plot
    for memory_mode, subset in metrics_df.groupby('memory_enabled'):
        plt.plot(
            subset['task_index'], 
            subset['task_accuracy'], 
            marker='o',
            linestyle='-',
            color=colors[memory_mode],
            label='With Memory' if memory_mode else 'Without Memory',
            alpha=0.8,
            linewidth=2
        )
    
    plt.xlabel('Task Index', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Task Accuracy Over Time', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save forgetting plot
    plt.figure(figsize=(12, 5))
    
    for memory_mode, subset in metrics_df.groupby('memory_enabled'):
        plt.plot(
            subset['task_index'], 
            subset['mean_forgetting'], 
            marker='s',
            linestyle='-',
            color=colors[memory_mode],
            label='With Memory' if memory_mode else 'Without Memory',
            alpha=0.8,
            linewidth=2
        )
    
    plt.xlabel('Task Index', fontsize=12)
    plt.ylabel('Forgetting', fontsize=12)
    plt.title('Catastrophic Forgetting Over Time', fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forgetting_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plots to {output_dir}:")
    print(f"- {os.path.join(output_dir, 'continual_learning_metrics.png')}")
    print(f"- {os.path.join(output_dir, 'accuracy_over_time.png')}")
    print(f"- {os.path.join(output_dir, 'forgetting_over_time.png')}")


def main():
    parser = argparse.ArgumentParser(description="Finetune and evaluate a model on a continual learning task stream")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})")
    parser.add_argument("--task_stream_file", required=True, help="Path to task stream JSON file")
    parser.add_argument("--filter_repo", default="sympy/sympy", help="Repository to filter tasks for (default: sympy/sympy)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for finetuning")
    parser.add_argument("--output_dir", default="results", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients for")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs per task")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for model")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output and checkpoint directories
    output_dir = os.path.abspath(args.output_dir)
    checkpoint_dir = os.path.join(output_dir, CHECKPOINT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for GPU
    check_gpu_availability()
    
    try:
        # Prepare dataset with the filter on sympy repository
        task_stream, task_ordering = load_task_stream(
            args.task_stream_file, 
            filter_repo=args.filter_repo
        )
        
        # Create metrics tracking for both memory conditions
        all_metrics = []
        
        # Track task accuracies for with and without memory
        for memory_enabled in [False, True]:  # First without memory, then with memory
            print(f"\n{'='*80}")
            print(f"Starting experiment with memory {'enabled' if memory_enabled else 'disabled'}")
            print(f"{'='*80}")
            
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                model_name=args.model_name,
                use_8bit=args.use_8bit, 
                use_lora=args.use_lora,
                device=args.device
            )
            
            # Initialize metrics tracking
            task_metrics = []
            stored_memories = []  # Memory examples from previous tasks
            
            # Verify task ordering and stream structure
            if not task_ordering:
                raise ValueError("No tasks found in the task stream")
                
            print(f"\nStarting evaluation with {len(task_ordering)} tasks")
            
            # Iterate through tasks in order
            for task_idx, task_name in enumerate(task_ordering):
                print(f"\n{'='*80}")
                print(f"TASK {task_idx+1}/{len(task_ordering)}: {task_name}")
                print(f"{'='*80}")
                
                # Get task examples
                task_examples = task_stream.get(task_name, [])
                if not task_examples:
                    print(f"Warning: No examples found for task {task_name}, skipping...")
                    continue
                    
                print(f"Examples in task: {len(task_examples)}")
                
                # Log memory usage
                memory_usage = get_memory_usage()
                print(f"\nMemory usage before task {task_idx+1}:")
                print(f"GPU Allocated: {memory_usage['gpu']['allocated']:.2f} GB")
                print(f"CPU Usage: {memory_usage['cpu']:.2f} GB")
                
                # Evaluate model on current task (before update)
                print("Evaluating on current task (before update)...")
                current_task_acc_before = evaluate_model(
                    model, tokenizer, task_examples, args.device, args.batch_size,
                    use_memory=memory_enabled, memory_examples=stored_memories
                )
                
                try:
                    # Finetune model on current task
                    print(f"\nFinetuning model on {task_name}...")
                    model = finetune_model(
                        model, tokenizer, task_examples, args.learning_rate, args.device,
                        args.num_epochs, args.batch_size, args.gradient_accumulation_steps,
                        use_memory=memory_enabled, memory_examples=stored_memories
                    )
                    
                    # Save model checkpoint
                    save_model_checkpoint(model, tokenizer, checkpoint_dir, f"task_{task_idx}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\nCUDA out of memory error during training. Trying with reduced batch size...")
                        try:
                            # Try with half the batch size
                            reduced_batch_size = max(1, args.batch_size // 2)
                            print(f"Retrying with batch size {reduced_batch_size}...")
                            model = finetune_model(
                                model, tokenizer, task_examples, args.learning_rate, args.device,
                                args.num_epochs, reduced_batch_size, args.gradient_accumulation_steps * 2,
                                use_memory=memory_enabled, memory_examples=stored_memories
                            )
                            save_model_checkpoint(model, tokenizer, checkpoint_dir, f"task_{task_idx}_reduced_bs")
                        except Exception as e2:
                            print(f"\nFailed to train model on task {task_name}: {e2}")
                            print("Skipping to next task...")
                            continue
                    else:
                        print(f"\nError during training: {e}")
                        print("Skipping to next task...")
                        continue
                
                # If memory is enabled, store examples from this task for future tasks
                if memory_enabled and task_examples:
                    try:
                        # Store a subset of examples from this task as memory
                        memory_samples = min(5, len(task_examples))  # Store up to 5 examples
                        memory_to_add = task_examples[:memory_samples]
                        
                        # Ensure we're not storing too much in memory
                        max_memory_examples = 20  # Maximum number of examples to keep in memory
                        if len(stored_memories) + len(memory_to_add) > max_memory_examples:
                            # Remove oldest examples to make room
                            num_to_remove = (len(stored_memories) + len(memory_to_add)) - max_memory_examples
                            stored_memories = stored_memories[num_to_remove:]
                        
                        stored_memories.extend(memory_to_add)
                        print(f"\nAdded {len(memory_to_add)} examples to memory (total: {len(stored_memories)}/{max_memory_examples})")
                        
                        # Log memory usage after update
                        memory_usage = get_memory_usage()
                        print(f"Memory usage after memory update:")
                        print(f"GPU Allocated: {memory_usage['gpu']['allocated']:.2f} GB")
                        print(f"CPU Usage: {memory_usage['cpu']:.2f} GB")
                        
                    except Exception as e:
                        print(f"\nError updating memory: {e}")
                        print("Continuing without updating memory...")
                
                # Evaluate model on all seen tasks after update
                all_task_accuracies = {}
                task_forgetting_values = []
                
                print("Evaluating on all seen tasks after update...")
                for prev_task_idx, prev_task_name in enumerate(task_ordering[:task_idx+1]):
                    prev_task_examples = task_stream[prev_task_name]
                    prev_task_acc = evaluate_model(
                        model, tokenizer, prev_task_examples, args.device, args.batch_size,
                        use_memory=memory_enabled, memory_examples=stored_memories
                    )
                    all_task_accuracies[prev_task_name] = prev_task_acc
                    
                # Calculate forgetting metrics for all previous tasks
                task_forgetting_values = []
                for prev_task_idx in range(task_idx):
                    prev_task_name = task_ordering[prev_task_idx]
                    prev_task_acc = all_task_accuracies.get(prev_task_name, 0)
                    
                    # Find previous best accuracy for this task with the same memory setting
                    prev_best_acc = 0
                    for metric in task_metrics:
                        if (metric['task_name'] == prev_task_name and 
                            metric['memory_enabled'] == memory_enabled):
                            if metric['task_accuracy'] > prev_best_acc:
                                prev_best_acc = metric['task_accuracy']
                    
                    # Forgetting = max(0, prev_best_acc - current_accuracy)
                    if prev_best_acc > 0:  # Only calculate if we have a previous best
                        forgetting = max(0, prev_best_acc - prev_task_acc)
                        task_forgetting_values.append(forgetting)
                
                # Compute mean forgetting across all previous tasks
                mean_forgetting = np.mean(task_forgetting_values) if task_forgetting_values else 0.0
                
                # Compute mean accuracy across all seen tasks
                mean_accuracy = np.mean(list(all_task_accuracies.values()))
                
                # Store metrics for this task
                task_metrics.append({
                    "memory_enabled": memory_enabled,
                    "task_index": task_idx,
                    "task_name": task_name,
                    "task_accuracy": all_task_accuracies[task_name],
                    "mean_accuracy": mean_accuracy,
                    "mean_forgetting": mean_forgetting,
                    "num_memory_examples": len(stored_memories) if memory_enabled else 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Print detailed metrics
                print(f"\n{'='*80}")
                print(f"TASK COMPLETED: {task_name}")
                print(f"{'='*80}")
                print(f"Current task accuracy: {all_task_accuracies[task_name]:.4f}")
                print(f"Mean accuracy (all tasks): {mean_accuracy:.4f}")
                print(f"Mean forgetting: {mean_forgetting:.4f}")
                if memory_enabled:
                    print(f"Memory bank size: {len(stored_memories)}")
                
                # Print per-task accuracies
                print("\nTask Accuracies:")
                for t, acc in all_task_accuracies.items():
                    print(f"  - {t}: {acc:.4f}")
                
                if task_forgetting_values:
                    print("\nForgetting per task:")
                    for i, forgetting in enumerate(task_forgetting_values, 1):
                        print(f"  - Task {i}: {forgetting:.4f}")
                
                print(f"{'='*80}\n")
                
            # Add all task metrics to combined results
            all_metrics.extend(task_metrics)
            
            # Clean up resources before next round
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Convert all metrics to DataFrame and save
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nSaved metrics to {metrics_csv_path}")
        
        # Generate plots
        plot_metrics(metrics_df, output_dir)
        
        print("\nExperiment completed successfully!")
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def prompt_based_update(model, tokenizer, task_examples, device):
    """Add task examples to prompt set (no parameter updates).
    
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


if __name__ == "__main__":
    main()


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
