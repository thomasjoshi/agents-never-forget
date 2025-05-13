#!/usr/bin/env python
"""Finetune and evaluate a model on a sequence of continual learning tasks.

This script loads TinyLlama-1.1B-Chat with 4-bit quantization, finetunes it sequentially on SWE-Bench-CL tasks,
and evaluates catastrophic forgetting and forward transfer.
"""

import os
import json
import random
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import psutil
import GPUtil

# Set default model and training parameters
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_BATCH_SIZE = 16  # Increased from 4 for faster training
DEFAULT_GRAD_ACCUM_STEPS = 2  # Reduced from 4 for faster training
DEFAULT_LEARNING_RATE = 2e-4  # Increased from 5e-5
DEFAULT_NUM_EPOCHS = 2  # Reduced from 3 for faster training
DEFAULT_USE_4BIT = True  # Use 4-bit quantization for faster training

def check_gpu_availability():
    """Check if CUDA is available and provide helpful guidance if not."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This script is optimized for GPU usage.")
        print("Training will be very slow on CPU. Consider using a GPU runtime.")
        return False
    
    # Get GPU info
    gpu_info = GPUtil.getGPUs()
    if not gpu_info:
        print("WARNING: No GPUs found. This script is optimized for GPU usage.")
        return False
    
    print(f"Found {len(gpu_info)} GPU(s):")
    for i, gpu in enumerate(gpu_info):
        print(f"  GPU {i}: {gpu.name} with {gpu.memoryTotal}MB memory")
    
    return True

def get_memory_usage():
    """Get current memory usage stats."""
    gpu_mem = {"allocated": 0, "cached": 0}
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # Convert to GB
            "cached": torch.cuda.memory_reserved() / 1024**3,  # Convert to GB
        }
    
    return {
        "gpu": gpu_mem,
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
    
    # Format input text with instruction
    input_text = f"Fix the following code:\n\n{example['patch']}"
    
    # Add memory if enabled
    if self.use_memory and self.memory_examples:
        memory_context = "\n\n".join(
            [f"Remember:\n{mem['patch']}\n\nTest Case:\n{mem['test_patch']}" 
             for mem in self.memory_examples]
        )
        input_text = f"{memory_context}\n\n{input_text}"
    
    # Tokenize input and labels
    encoding = self.tokenizer(
        input_text,
        max_length=self.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Use test_patch as the target for language modeling
    labels = self.tokenizer(
        example.get("test_patch", ""),  # Fallback to empty string if not present
        max_length=self.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    
    return {
        "input_ids": encoding["input_ids"].squeeze(),
        "attention_mask": encoding["attention_mask"].squeeze(),
        "labels": labels.squeeze()
    }

def load_model_and_tokenizer(model_name=DEFAULT_MODEL, use_4bit=DEFAULT_USE_4BIT, use_lora=True, device="cuda"):
    """
    Load and prepare TinyLlama model with optimization for faster training.
    
    Args:
        model_name: HF model name or path
        use_4bit: Whether to use 4-bit quantization (faster than 8-bit)
        use_lora: Whether to use LoRA for parameter-efficient fine-tuning
        device: Device to load the model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_name} with 4-bit: {use_4bit}, LoRA: {use_lora}")
    
    # Configure quantization if enabled
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,  # Disable cache for faster training
    )
    
    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA if enabled
    if use_lora:
        lora_config = LoraConfig(
            r=16,  # Reduced rank for faster training
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,  # Use fast tokenizer for better performance
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
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
    print(f"Loading task stream from {task_stream_file}...")
    
    with open(task_stream_file, 'r') as f:
        task_stream = json.load(f)
    
    # Filter tasks by repository if specified
    if filter_repo:
        print(f"Filtering tasks for repository: {filter_repo}")
        filtered_tasks = {}
        for task_name, examples in task_stream.items():
            filtered = [ex for ex in examples if ex.get('repo') == filter_repo]
            if filtered:
                filtered_tasks[task_name] = filtered
        task_stream = filtered_tasks
    
    # Get task ordering (alphabetical by default)
    task_ordering = sorted(task_stream.keys())
    
    # Print task statistics
    print(f"\nLoaded {len(task_stream)} tasks with the following distribution:")
    for task in task_ordering:
        print(f"  {task}: {len(task_stream[task])} examples")
    
    return {
        "tasks": task_stream,
        "task_ordering": task_ordering
    }

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
    if not task_examples:
        return 0.0
    
    model.eval()
    dataset = SWEBenchDataset(
        task_examples, 
        tokenizer, 
        use_memory=use_memory,
        memory_examples=memory_examples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            # Get predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy
            labels = batch["labels"]
            mask = labels != -100  # Ignore padding tokens
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"  Evaluation accuracy: {accuracy:.4f}")
    return accuracy

def finetune_model(model, tokenizer, task_examples, learning_rate, device, 
                  num_epochs=DEFAULT_NUM_EPOCHS, 
                  batch_size=DEFAULT_BATCH_SIZE, 
                  gradient_accumulation_steps=DEFAULT_GRAD_ACCUM_STEPS, 
                  use_memory=False, memory_examples=None,
                  max_steps=100):
    """
    Finetune model on a task with optimizations for faster training.
    
    Args:
        model: Model to finetune
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        learning_rate: Learning rate for training
        device: Device to run training on
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_memory: Whether to include memory in training examples
        memory_examples: Examples from previous tasks to use as memory
        max_steps: Maximum number of training steps (safety limit)
    
    Returns:
        Finetuned model
    """
    print(f"Finetuning model on {len(task_examples)} examples...")
    
    # Create dataset
    dataset = SWEBenchDataset(
        task_examples, 
        tokenizer, 
        use_memory=use_memory,
        memory_examples=memory_examples
    )
    
    # Calculate warmup steps (10% of total steps)
    total_steps = min(
        (len(dataset) * num_epochs) // (batch_size * gradient_accumulation_steps),
        max_steps
    )
    warmup_steps = max(1, int(total_steps * 0.1))
    
    # Optimized training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        max_steps=max_steps,  # Safety limit
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=5,  # More frequent logging
        save_strategy="no",  # Disable checkpointing
        fp16=True,  # Mixed precision training
        bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if available
        remove_unused_columns=True,  # Saves memory
        optim="adamw_torch_fused",  # Fused optimizer
        report_to="none",  # Disable external logging
        gradient_checkpointing=True,  # Save memory
        dataloader_num_workers=2,  # Parallel data loading
        dataloader_pin_memory=True,  # Faster data transfer
    )
    
    # Initialize trainer with optimizations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    
    # Train model with timing
    start_time = time.time()
    trainer.train()
    training_time = (time.time() - start_time) / 60  # in minutes
    print(f"Training completed in {training_time:.1f} minutes")
    
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
        metrics_df: DataFrame containing metrics data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy over tasks
    plt.subplot(1, 2, 1)
    sns.lineplot(data=metrics_df, x="task_idx", y="accuracy", hue="task_name", 
                 marker="o", markersize=8, linewidth=2)
    plt.title("Accuracy Over Tasks")
    plt.xlabel("Task Index")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot forgetting
    plt.subplot(1, 2, 2)
    sns.lineplot(data=metrics_df, x="task_idx", y="forgetting", hue="task_name",
                 marker="o", markersize=8, linewidth=2)
    plt.title("Forgetting Over Tasks")
    plt.xlabel("Task Index")
    plt.ylabel("Forgetting")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"Saved metrics plot to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Finetune and evaluate a model on a continual learning task stream")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})")
    parser.add_argument("--task_stream_file", required=True, help="Path to task stream JSON file")
    parser.add_argument("--filter_repo", default="sympy/sympy", help="Repository to filter tasks for (default: sympy/sympy)")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, 
                        help=f"Learning rate for finetuning (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--output_dir", default="results", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, 
                        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS, 
                        help=f"Gradient accumulation steps (default: {DEFAULT_GRAD_ACCUM_STEPS})")
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_NUM_EPOCHS, 
                        help=f"Number of epochs per task (default: {DEFAULT_NUM_EPOCHS})")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum number of training steps per task (safety limit)")
    parser.add_argument("--use_4bit", action="store_true", default=DEFAULT_USE_4BIT, 
                        help=f"Use 4-bit quantization (faster, default: {DEFAULT_USE_4BIT})")
    parser.add_argument("--use_lora", action="store_true", default=True, 
                        help="Use LoRA for parameter-efficient fine-tuning (default: True)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Maximum number of examples per task (for faster training)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load task stream
    task_data = load_task_stream(args.task_stream_file, args.filter_repo)
    task_stream = task_data["tasks"]
    task_ordering = task_data["task_ordering"]
    
    # Initialize metrics storage
    metrics = []
    
    # Run experiments with and without memory
    for memory_enabled in [False, True]:
        print(f"\n{'='*80}")
        print(f"Running experiment with memory {'enabled' if memory_enabled else 'disabled'}")
        print(f"{'='*80}")
        
        # Load model and tokenizer with optimized settings
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            use_4bit=args.use_4bit,
            use_lora=args.use_lora,
            device=args.device
        )
        
        # Initialize memory storage if enabled
        stored_memories = []
        
        # Track accuracy for each task
        task_accuracies = {task: [] for task in task_ordering}
        
        # Iterate through tasks in order
        for task_idx, task_name in enumerate(task_ordering):
            print(f"\n{'='*80}")
            print(f"Task {task_idx+1}/{len(task_ordering)}: {task_name}")
            print(f"{'='*80}")
            
            # Get examples for current task
            task_examples = task_stream[task_name]
            print(f"Found {len(task_examples)} examples for task {task_name}")
            
            # Limit number of examples if specified
            if args.max_examples and len(task_examples) > args.max_examples:
                print(f"Limiting to {args.max_examples} examples for faster training")
                task_examples = task_examples[:args.max_examples]
            
            # Evaluate model on current task (before update)
            print("Evaluating on current task (before update)...")
            before_accuracy = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                task_examples=task_examples,
                device=args.device,
                batch_size=args.batch_size,
                use_memory=memory_enabled,
                memory_examples=stored_memories
            )
            
            # Finetune model on current task with optimized settings
            model = finetune_model(
                model=model,
                tokenizer=tokenizer,
                task_examples=task_examples,
                learning_rate=args.learning_rate,
                device=args.device,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                use_memory=memory_enabled,
                memory_examples=stored_memories,
                max_steps=args.max_steps
            )
            
            # Save model checkpoint
            checkpoint_dir = save_model_checkpoint(
                model, tokenizer, args.output_dir, 
                f"{'with_memory' if memory_enabled else 'no_memory'}_{task_name}"
            )
            
            # Evaluate model on current task (after update)
            print("Evaluating on current task (after update)...")
            after_accuracy = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                task_examples=task_examples,
                device=args.device,
                batch_size=args.batch_size,
                use_memory=memory_enabled,
                memory_examples=stored_memories
            )
            
            # Evaluate on all previous tasks to measure forgetting
            for prev_task_idx in range(task_idx):
                prev_task_name = task_ordering[prev_task_idx]
                prev_task_examples = task_stream.get(prev_task_name, [])
                if not prev_task_examples:
                    continue
                    
                print(f"Evaluating on previous task {prev_task_name}...")
                prev_accuracy = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    task_examples=prev_task_examples,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_memory=memory_enabled,
                    memory_examples=stored_memories
                )
                
                # Calculate forgetting
                forgetting = task_accuracies[prev_task_name][-1] - prev_accuracy if task_accuracies[prev_task_name] else 0.0
                
                # Store metrics
                metrics.append({
                    "task_name": prev_task_name,
                    "task_idx": prev_task_idx,
                    "current_task": task_name,
                    "accuracy": prev_accuracy,
                    "forgetting": max(0, forgetting),  # Don't allow negative forgetting
                    "memory_enabled": memory_enabled,
                    "is_current_task": False
                })
            
            # Store metrics for current task
            task_accuracies[task_name].append(after_accuracy)
            metrics.append({
                "task_name": task_name,
                "task_idx": task_idx,
                "current_task": task_name,
                "accuracy": after_accuracy,
                "forgetting": 0.0,  # No forgetting for current task yet
                "memory_enabled": memory_enabled,
                "is_current_task": True
            })
            
            # Store memory examples (if enabled)
            if memory_enabled:
                # Store a subset of examples for memory
                memory_size = min(10, len(task_examples) // 2)  # Store up to 10 examples
                stored_memories.extend(random.sample(task_examples, min(memory_size, len(task_examples))))
                print(f"Added {len(stored_memories)} examples to memory")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(args.output_dir, f"metrics_{'with_memory' if memory_enabled else 'no_memory'}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nSaved metrics to {metrics_file}")
        
        # Plot metrics
        plot_metrics(metrics_df, args.output_dir)

if __name__ == "__main__":
    main()
