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
import tempfile
import subprocess
import warnings
import shutil
from pathlib import Path
from tqdm import tqdm
import ast
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
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
DEFAULT_BATCH_SIZE = 16
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_NUM_EPOCHS = 3
DEFAULT_USE_4BIT = True

# Check if Docker is available for running SWE-Bench tests
def check_docker_available() -> bool:
    """Check if Docker is available for SWE-Bench test execution."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        warnings.warn(f"Docker not available: {e}. Functional tests will be skipped.")
        return False

# Global flag for Docker availability
DOCKER_AVAILABLE = check_docker_available()

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
    
    # The task stream is a dictionary where keys are task names (e.g., 'T_1', 'T_2', etc.)
    # and values are lists of examples, where each example is a dictionary with a 'repo' field
    
    # If no filter is specified, return the entire task stream
    if not filter_repo:
        task_ordering = sorted(task_stream.keys())
        print(f"\nLoaded {len(task_stream)} tasks with the following distribution:")
        for task in task_ordering:
            print(f"  {task}: {len(task_stream[task])} examples")
        
        return {
            "tasks": task_stream,
            "task_ordering": task_ordering
        }
    
    # Filter tasks by repository if specified
    print(f"Filtering tasks for repository: {filter_repo}")
    filtered_tasks = {}
    
    for task_name, examples in task_stream.items():
        # Skip if examples is not a list
        if not isinstance(examples, list):
            print(f"Warning: Task '{task_name}' does not contain a list of examples, skipping...")
            continue
            
        filtered = []
        for ex in examples:
            # Skip if the example is not a dictionary or doesn't have a 'repo' field
            if not isinstance(ex, dict) or 'repo' not in ex:
                print(f"Warning: Skipping malformed example in task '{task_name}' - not a dictionary or missing 'repo' field")
                continue
                
            # Check if the repository matches the filter
            if ex['repo'] == filter_repo:
                filtered.append(ex)
        
        # Only add the task if we found matching examples
        if filtered:
            filtered_tasks[task_name] = filtered
    
    # Get task ordering (alphabetical by default)
    task_ordering = sorted(filtered_tasks.keys())
    
    # Print task statistics
    print(f"\nLoaded {len(filtered_tasks)} tasks with the following distribution:")
    for task in task_ordering:
        print(f"  {task}: {len(filtered_tasks[task])} examples")
    
    # If no tasks were found, print available repositories
    if not filtered_tasks:
        print("\nNo tasks found for the specified repository. Available repositories:")
        all_repos = set()
        for examples in task_stream.values():
            if isinstance(examples, list):
                for ex in examples:
                    if isinstance(ex, dict) and 'repo' in ex:
                        all_repos.add(ex['repo'])
        for repo in sorted(all_repos):
            print(f"  - {repo}")
    
    return {
        "tasks": filtered_tasks,
        "task_ordering": task_ordering
    }

def classify_error(pred_tokens: str, true_tokens: str) -> str:
    """
    Classify the type of error in model predictions.
    
    Args:
        pred_tokens: Predicted token sequence
        true_tokens: Ground truth token sequence
        
    Returns:
        str: Error type classification
    """
    pred = pred_tokens.strip()
    true = true_tokens.strip()
    
    # No prediction
    if not pred:
        return "empty_prediction"
    
    # Exact match (should be filtered out before)
    if pred == true:
        return "correct"
    
    # Check for syntax errors
    try:
        ast.parse(pred)
    except SyntaxError:
        # Further classify syntax errors
        if any(op in pred for op in ['=', '==', '!=', '>=', '<=', '+', '-', '*', '/']):
            return "syntax_operator"
        elif any(kw in pred for kw in ['def ', 'class ', 'for ', 'while ', 'if ', 'else:']):
            return "syntax_control_flow"
        elif any(b in pred for b in ['(', ')', '[', ']', '{', '}']):
            return "syntax_brackets"
        else:
            return "syntax_general"
    
    # Check for indentation errors
    if pred.lstrip() != pred and len(pred.splitlines()) > 1:
        return "indentation"
    
    # Check for missing imports
    if 'import ' in true and 'import ' not in pred:
        return "missing_import"
    
    # Check for API misuse (common patterns)
    api_patterns = [
        (r'\.set_?[A-Z]', 'api_setter_missing'),
        (r'\.get_?[A-Z]', 'api_getter_missing'),
        (r'\.is_?[a-z]+\s*\(', 'api_method_missing'),
    ]
    
    for pattern, error_type in api_patterns:
        if re.search(pattern, true) and not re.search(pattern, pred):
            return error_type
    
    # Check for logical errors (approximate)
    if len(pred.split()) > len(true.split()) * 1.5:
        return "overly_verbose"
    elif len(pred.split()) < len(true.split()) * 0.5:
        return "too_concise"
    
    # Default classification
    return "semantic"

def plot_error_analysis(error_analysis: Dict[str, Any], output_dir: str, task_name: str):
    """
    Generate plots for error analysis.
    
    Args:
        error_analysis: Dictionary containing error analysis data
        output_dir: Directory to save plots
        task_name: Name of the current task
    """
    if not error_analysis.get('error_distribution'):
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Error distribution pie chart
    error_dist = error_analysis['error_distribution']
    if error_dist:
        plt.figure(figsize=(10, 6))
        labels, counts = zip(*sorted(error_dist.items(), key=lambda x: x[1], reverse=True))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f'Error Distribution - {task_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'error_distribution_{task_name}.png'))
        plt.close()
    
    # 2. Error examples table (saved as markdown)
    if error_analysis.get('error_examples'):
        error_file = os.path.join(output_dir, f'error_examples_{task_name}.md')
        with open(error_file, 'w') as f:
            f.write(f"# Error Examples - {task_name}\n\n")
            f.write("| Error Type | Prediction | Target |\n")
            f.write("|------------|------------|---------|\n")
            for ex in error_analysis['error_examples'][:10]:  # Limit to top 10 examples
                pred = ex['prediction'].replace('|', '\\|').replace('\n', ' ')[:100]
                target = ex['target'].replace('|', '\\|').replace('\n', ' ')[:100]
                f.write(f"| {ex['error_type']} | `{pred}` | `{target}` |\n")


def run_swebench_tests(issue: Dict[str, Any], patch_path: str, timeout: int = 12) -> bool:
    """
    Run SWE-Bench tests for the given issue using Docker.
    
    Args:
        issue: Issue dictionary containing repo, FAIL_TO_PASS and PASS_TO_PASS tests
        patch_path: Path to the generated patch file
        timeout: Timeout in seconds for test execution
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    if not DOCKER_AVAILABLE:
        warnings.warn("Docker not available. Skipping test execution.")
        return False
    
    try:
        repo_name = issue.get('repo', '')
        repo_path = Path(repo_name)
        
        # Check if we have the patch file
        if not os.path.exists(patch_path):
            warnings.warn(f"Patch file not found: {patch_path}")
            return False
        
        # Docker command to build and run tests
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{patch_path}:/patch.diff",
            "-v", f"{repo_path}:/repo",
            "swebench/tester:latest",
            "--repo", repo_name,
            "--patch", "/patch.diff",
            "--timeout", str(timeout)
        ]
        
        # Run Docker command with timeout
        process = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5  # Add buffer to Docker timeout
        )
        
        # Check if all tests passed
        output = process.stdout + process.stderr
        f2p_passed = "FAIL_TO_PASS: PASS" in output
        p2p_passed = "PASS_TO_PASS: PASS" in output
        
        return f2p_passed and p2p_passed
    
    except subprocess.TimeoutExpired:
        warnings.warn(f"Test execution timed out after {timeout} seconds")
        return False
    except Exception as e:
        warnings.warn(f"Error running tests: {e}")
        return False


def evaluate_success(model, dataset, batch_size=4):
    """
    Evaluate model success based on SWE-Bench test execution.
    
    Args:
        model: The model to evaluate
        dataset: List of issue examples to test
        batch_size: Batch size for processing
    
    Returns:
        success_rate: float - fraction of issues whose FAIL_TO_PASS tests
                          now pass and whose original PASS_TO_PASS
                          still pass.
        successes: list[bool] - per-issue pass/fail flags.
    
    Notes:
        * Pull tokenizer/model.device from the model itself (no extra args).
        * Call helper `run_swebench_tests(issue, patch_path, timeout=12)`.
        * Provide a stub that returns False and prints a warning if Docker
          is unavailable, so CPU-only CI still runs.
    """
    # Get tokenizer and device from model
    tokenizer = model.tokenizer if hasattr(model, 'tokenizer') else AutoTokenizer.from_pretrained(model.config._name_or_path)
    device = next(model.parameters()).device
    
    successes = []
    
    # If Docker is not available, return dummy results with warning
    if not DOCKER_AVAILABLE:
        warnings.warn("Docker not available. Returning dummy success rate.")
        return 0.0, [False] * len(dataset)
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        for issue in batch:
            try:
                # Format the prompt
                prompt = issue.get("prompt", "")
                
                # Generate patch
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
                    outputs = model.generate(
                        **inputs,
                        max_length=1024,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the patch
                # Assuming the model outputs a patch in diff format
                patch_content = generated_text  # This should be refined to extract only diff
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.diff') as f:
                    f.write(patch_content)
                    patch_path = f.name
                
                # Run tests
                success = run_swebench_tests(issue, patch_path, timeout=12)
                successes.append(success)
                
                # Clean up temp file
                os.unlink(patch_path)
                
            except Exception as e:
                warnings.warn(f"Error evaluating issue: {e}")
                successes.append(False)
    
    # Calculate success rate
    success_rate = sum(successes) / len(successes) if successes else 0.0
    
    return success_rate, successes


def evaluate_model(model, tokenizer, task_examples, device, batch_size=4, use_memory=False, memory_examples=None):
    """
    Evaluate model on a task with enhanced metrics and memory management.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer to use
        task_examples: List of examples for the task
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        use_memory: Whether to include memory in evaluation examples
        memory_examples: Examples from previous tasks to use as memory
    
    Returns:
        Dictionary containing evaluation metrics (accuracy, exact_match, f1, error_analysis)
    """
    if not task_examples:
        return {
            "accuracy": 0.0, 
            "exact_match": 0.0, 
            "f1": 0.0,
            "error_analysis": {
                'error_distribution': {},
                'error_examples': [],
                'total_errors': 0
            }
        }
    
    # Clear CUDA cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    dataset = SWEBenchDataset(
        task_examples, 
        tokenizer, 
        use_memory=use_memory,
        memory_examples=memory_examples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize metrics
    metrics = {
        'correct_tokens': 0,
        'total_tokens': 0,
        'exact_matches': 0,
        'total_examples': 0,
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Initialize error tracking
    error_types = defaultdict(int)
    error_examples = []
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
                try:
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
                    labels = batch["labels"]
                    
                    # Calculate token-level metrics
                    mask = labels != -100  # Ignore padding tokens
                    metrics['correct_tokens'] += ((preds == labels) & mask).sum().item()
                    metrics['total_tokens'] += mask.sum().item()
                    
                    # Calculate example-level metrics and track errors
                    for i in range(len(batch["input_ids"])):
                        # Get non-padding tokens
                        example_mask = mask[i].bool()
                        if not example_mask.any():
                            continue
                            
                        example_preds = preds[i][example_mask]
                        example_labels = labels[i][example_mask]
                        
                        # Check for exact match
                        is_exact_match = torch.equal(example_preds, example_labels)
                        if is_exact_match:
                            metrics['exact_matches'] += 1
                        else:
                            # Log error details
                            error_type = classify_error(
                                pred_tokens=tokenizer.decode(example_preds.tolist()),
                                true_tokens=tokenizer.decode(example_labels.tolist())
                            )
                            error_types[error_type] += 1
                            
                            # Store example of each error type
                            if error_types[error_type] <= 5:  # Store up to 5 examples per error type
                                error_examples.append({
                                    'prediction': tokenizer.decode(example_preds.tolist()),
                                    'target': tokenizer.decode(example_labels.tolist()),
                                    'error_type': error_type,
                                    'task_name': task_examples[i].get('task_name', 'unknown')
                                })
                        
                        # Update F1 metrics
                        metrics['true_positives'] += (example_preds == example_labels).sum().item()
                        metrics['false_positives'] += (example_preds != example_labels).sum().item()
                        metrics['false_negatives'] += (example_preds != example_labels).sum().item()
                        metrics['total_examples'] += 1
                
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f"OOM at batch {batch_idx}, reducing batch size and retrying...")
                        torch.cuda.empty_cache()
                        return evaluate_model(model, tokenizer, task_examples, device, 
                                           batch_size=max(1, batch_size // 2), 
                                           use_memory=use_memory, 
                                           memory_examples=memory_examples)
                    raise
        
        # Calculate final metrics
        accuracy = metrics['correct_tokens'] / metrics['total_tokens'] if metrics['total_tokens'] > 0 else 0.0
        exact_match = metrics['exact_matches'] / metrics['total_examples'] if metrics['total_examples'] > 0 else 0.0
        
        # Calculate F1 score
        precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'] + 1e-10)
        recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'] + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Prepare error analysis results
        error_analysis = {
            'error_distribution': dict(error_types),
            'error_examples': error_examples[:20],  # Limit number of saved examples
            'total_errors': sum(error_types.values())
        }
        
        # Save error analysis
        if error_analysis['total_errors'] > 0:
            task_name = task_examples[0].get('task_name', 'unknown') if task_examples else 'unknown'
            # Use a default output directory since args is not available here
            output_dir = "./error_analysis"
            os.makedirs(output_dir, exist_ok=True)
            plot_error_analysis(error_analysis, output_dir, task_name)
        
        print(f"  Evaluation metrics:")
        print(f"  - Token Accuracy: {accuracy:.4f}")
        print(f"  - Exact Match: {exact_match:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        if error_analysis['total_errors'] > 0:
            print(f"  - Error Distribution: {dict(error_types)}")
        
        return {
            "accuracy": accuracy,
            "exact_match": exact_match,
            "f1": f1,
            "error_analysis": error_analysis
        }
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "accuracy": 0.0, 
            "exact_match": 0.0, 
            "f1": 0.0,
            "error_analysis": {
                'error_distribution': {},
                'error_examples': [],
                'total_errors': 0
            }
        }

def finetune_model(model, tokenizer, task_examples, learning_rate, device, 
                  num_epochs=DEFAULT_NUM_EPOCHS, 
                  batch_size=DEFAULT_BATCH_SIZE, 
                  gradient_accumulation_steps=DEFAULT_GRAD_ACCUM_STEPS, 
                  use_memory=False, memory_examples=None,
                  max_steps=100,
                  use_4bit=DEFAULT_USE_4BIT):
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
    # Choose precision based on hardware and use_4bit flag
    # Only one of fp16 or bf16 can be True
    use_fp16 = not use_4bit and not torch.cuda.is_bf16_supported()
    use_bf16 = not use_4bit and torch.cuda.is_bf16_supported()
    
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
        fp16=use_fp16,  # Mixed precision training if not using 4-bit or bf16
        bf16=use_bf16,  # Use bfloat16 if supported and not using 4-bit
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

def plot_metrics(metrics_df, output_dir, log_token_metrics=False):
    """
    Plot continual learning metrics with focus on functional success rates.
    
    Args:
        metrics_df: DataFrame containing metrics data
        output_dir: Directory to save plots
        log_token_metrics: Whether to also plot token-level metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter metrics for after-training evaluations
    after_df = metrics_df[metrics_df['evaluation_type'] == 'after_training']
    
    # Get unique tasks
    tasks = sorted(metrics_df['task_name'].unique())
    task_indices = sorted(metrics_df['task_idx'].unique())
    task_indices = [idx for idx in task_indices if idx >= 0]
    num_tasks = len(task_indices)
    
    if num_tasks == 0:
        return
    
    # 1. Create heatmap of success rates
    results_matrix = np.zeros((num_tasks, num_tasks))
    
    # Fill in results matrix from metrics
    for i, curr_task_idx in enumerate(task_indices):
        curr_task = tasks[curr_task_idx]
        
        # For each task we've seen up to the current one
        for j, eval_task_idx in enumerate(task_indices[:i+1]):
            eval_task = tasks[eval_task_idx]
            
            # Get the success rate after training on curr_task for eval_task
            task_rows = after_df[(after_df['task_name'] == eval_task) & 
                                (after_df['current_task'] == curr_task)]
            
            if not task_rows.empty and 'success_rate' in task_rows.columns:
                results_matrix[i, j] = task_rows['success_rate'].values[0]
    
    # Plot heatmap of results
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(results_matrix, annot=True, fmt='.2f', cmap='viridis',
                    xticklabels=tasks, yticklabels=tasks)
    plt.xlabel('Task Evaluated')
    plt.ylabel('After Training on Task')
    plt.title('Success Rate Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_heatmap.png'))
    plt.close()
    
    # 2. Stability-Plasticity line plot
    plt.figure(figsize=(12, 6))
    
    # Plot success rate of each task over time
    for j, eval_task_idx in enumerate(task_indices):
        eval_task = tasks[eval_task_idx]
        
        # Get success rates for this task over time
        success_rates = []
        x_labels = []
        
        for i, curr_task_idx in enumerate(task_indices):
            if i >= j:  # Only plot after the task has been introduced
                curr_task = tasks[curr_task_idx]
                task_rows = after_df[(after_df['task_name'] == eval_task) & 
                                    (after_df['current_task'] == curr_task)]
                
                if not task_rows.empty and 'success_rate' in task_rows.columns:
                    success_rates.append(task_rows['success_rate'].values[0])
                    x_labels.append(curr_task)
        
        plt.plot(x_labels, success_rates, marker='o', label=f"Task {eval_task}")
    
    plt.xlabel('After Training on Task')
    plt.ylabel('Success Rate')
    plt.title('Stability-Plasticity: Task Success Over Time')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_plasticity.png'))
    plt.close()
    
    # 3. Plot forward transfer (FWT)
    fwt_values = []
    for i, task_idx in enumerate(task_indices[1:], 1):  # Skip first task
        task = tasks[task_idx]
        task_rows = after_df[(after_df['task_name'] == task) & 
                           (after_df['is_current_task'] == True)]
        
        if not task_rows.empty and 'fwt' in task_rows.columns:
            fwt_values.append({
                'task': task,
                'fwt': task_rows['fwt'].values[0]
            })
    
    if fwt_values:
        fwt_df = pd.DataFrame(fwt_values)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(fwt_df['task'], fwt_df['fwt'], color='skyblue')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Task')
        plt.ylabel('Forward Transfer (FWT)')
        plt.title('Forward Transfer by Task')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forward_transfer.png'))
        plt.close()
    
    # 4. Plot backward transfer (BWT)
    bwt_values = []
    for task_idx in task_indices[:-1]:  # Exclude the last task
        task = tasks[task_idx]
        task_rows = after_df[(after_df['task_name'] == task) & 
                           (after_df['current_task'] == tasks[-1])]
        
        if not task_rows.empty and 'bwt' in task_rows.columns:
            bwt_values.append({
                'task': task,
                'bwt': task_rows['bwt'].values[0]
            })
    
    if bwt_values:
        bwt_df = pd.DataFrame(bwt_values)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bwt_df['task'], bwt_df['bwt'], color='lightgreen')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Task')
        plt.ylabel('Backward Transfer (BWT)')
        plt.title('Backward Transfer by Task')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backward_transfer.png'))
        plt.close()
    
    # 5. Token-level metrics (only if enabled)
    if log_token_metrics:
        # Plot token accuracy over time
        plt.figure(figsize=(12, 6))
        for task in tasks:
            task_df = after_df[after_df['task_name'] == task]
            if 'token_accuracy' in task_df.columns:
                plt.plot(task_df['current_task'], task_df['token_accuracy'], marker='o', label=f"Task {task}")
        
        plt.xlabel('Current Training Task')
        plt.ylabel('Token Accuracy')
        plt.title('Token Accuracy Over Task Sequence')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'token_accuracy_over_time.png'))
        plt.close()

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
    parser.add_argument("--log_token_metrics", action="store_true", default=False,
                        help="Log token-level metrics in addition to functional success metrics")
    parser.add_argument("--skip_cache_warmup", action="store_true", default=False,
                        help="Skip Docker cache warmup for SWE-Bench tests")
    parser.add_argument("--test_timeout", type=int, default=12,
                        help="Timeout in seconds for SWE-Bench test execution (default: 12)")
    parser.add_argument("--functional_eval", action="store_true", default=True,
                        help="Use functional evaluation (unit test passes) instead of token metrics for evaluation")
    
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
    
    # Run experiments with and without memory if memory_enabled is True
    for memory_enabled in [False, True] if args.memory_enabled else [False]:
        print("\n" + "=" * 80)
        print(f"Running experiment with memory {'enabled' if memory_enabled else 'disabled'}")
        print("=" * 80)

        # Initialize model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            use_4bit=args.use_4bit,
            use_lora=args.use_lora,
            device=args.device
        )
        
        # Metrics tracking
        metrics = []
        task_accuracies = defaultdict(list)  # For tracking forgetting
        initial_accuracies = {}  # For initial task performances
        stored_memories = []  # For memory mechanism
        
        # CL metrics tracking
        num_tasks = len(task_ordering)
        results = np.zeros((num_tasks, num_tasks))  # Matrix to store success rates [task_after, task_evaluated]
        peak = defaultdict(float)  # Track peak performance per task
        
        # Determine evaluation function based on args
        def evaluate_task(model, task_examples):
            if args.functional_eval:
                # Use functional success evaluation
                success_rate, successes = evaluate_success(model, task_examples, batch_size=args.batch_size)
                return {"accuracy": success_rate, "successes": successes}
            else:
                # Use token-level evaluation
                return evaluate_model(
                    model=model,
                    tokenizer=tokenizer, 
                    task_examples=task_examples,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_memory=memory_enabled,
                    memory_examples=stored_memories
                )

        # Initial evaluation on all tasks (for forward transfer baseline)
        print("\nInitial evaluation on all tasks (forward transfer baseline)...")
        for task_name, task_examples in task_stream.items():
            if not task_examples:
                continue
                
            print(f"Evaluating {task_name} (initial)...")
            initial_accuracy = evaluate_task(model, task_examples)
            
            # Store metrics
            initial_accuracies[task_name] = initial_accuracy
            task_idx = task_ordering.index(task_name) if task_name in task_ordering else -1
            
            # Update results matrix with initial scores (time step 0)
            if task_idx >= 0:
                results[0, task_idx] = initial_accuracy["accuracy"]
            
            # Record metrics
            metrics.append({
                "task_name": task_name,
                "task_idx": task_idx,
                "current_task": "initial",
                "success_rate": initial_accuracy["accuracy"],
                "evaluation_type": "before_training",
                "memory_enabled": memory_enabled,
                "is_current_task": False
            })
            
            # Log token metrics if requested
            if args.log_token_metrics and not args.functional_eval:
                metrics[-1]["token_accuracy"] = initial_accuracy["accuracy"]
        
        # Train on each task sequentially
        for task_idx, task_name in enumerate(tqdm(task_ordering, desc="Training on tasks")):
            task_examples = task_stream.get(task_name, [])
            if not task_examples:
                print(f"Warning: No examples found for task {task_name}, skipping...")
                continue
                
            print(f"\n{'='*80}")
            print(f"Task {task_idx}: {task_name} ({len(task_examples)} examples)")
            print(f"{'='*80}")
            
            # First, evaluate on current task (before training)
            print("Evaluating on current task (before update)...")
            before_accuracy = evaluate_task(model, task_examples)
            
            # Finetune the model on the current task
            model = finetune_model(
                model=model,
                tokenizer=tokenizer,
                task_examples=task_examples,
                learning_rate=args.learning_rate,
                device=args.device,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_steps=args.max_steps,
                use_memory=memory_enabled,
                memory_examples=stored_memories,
                use_4bit=args.use_4bit
            )
            
            # Save checkpoint
            checkpoint_dir = save_model_checkpoint(
                model, tokenizer, args.output_dir, 
                f"{'with_memory' if memory_enabled else 'no_memory'}_{task_name}"
            )
            
            # Evaluate model on current task (after update)
            print("Evaluating on current task (after update)...")
            after_accuracy = evaluate_task(model, task_examples)
            
            # Update results matrix for current task
            current_acc = after_accuracy["accuracy"]
            results[task_idx+1, task_idx] = current_acc  # Current task after training
            
            # Update peak performance for current task
            peak[task_idx] = max(peak[task_idx], current_acc)
            
            # Calculate forward transfer for current task
            initial_acc = results[0, task_idx]  # Initial performance
            forward_transfer = current_acc - initial_acc
                        
            # Evaluate on all previous tasks to measure forgetting
            for prev_task_idx in range(task_idx):
                prev_task_name = task_ordering[prev_task_idx]
                prev_task_examples = task_stream.get(prev_task_name, [])
                if not prev_task_examples:
                    continue
                    
                print(f"Evaluating on previous task {prev_task_name}...")
                prev_accuracy = evaluate_task(model, prev_task_examples)
                
                # Update results matrix for previous task
                prev_acc = prev_accuracy["accuracy"]
                results[task_idx+1, prev_task_idx] = prev_acc
                
                # Update peak performance for previous task
                peak[prev_task_idx] = max(peak[prev_task_idx], prev_acc)
                
                # Calculate forgetting (drop from peak)
                highest_prev = max([results[t+1, prev_task_idx] for t in range(task_idx)])
                forgetting = max(0, highest_prev - prev_acc)  # Don't allow negative forgetting
                
                # Calculate backward transfer (compared to initial accuracy)
                initial_prev_acc = results[0, prev_task_idx]
                backward_transfer = prev_acc - initial_prev_acc
                
                # Store metrics
                metrics.append({
                    "task_name": prev_task_name,
                    "task_idx": prev_task_idx,
                    "current_task": task_name,
                    "success_rate": prev_acc,
                    "forgetting": forgetting,
                    "bwt": backward_transfer,
                    "evaluation_type": "after_training",
                    "memory_enabled": memory_enabled,
                    "is_current_task": False
                })
                
                # Add token metrics if enabled
                if args.log_token_metrics and not args.functional_eval:
                    metrics[-1]["token_accuracy"] = prev_accuracy["accuracy"]
            
            # Store metrics for current task
            metrics.append({
                "task_name": task_name,
                "task_idx": task_idx,
                "current_task": task_name,
                "success_rate": current_acc,
                "forgetting": 0.0,  # No forgetting for current task
                "fwt": forward_transfer,
                "evaluation_type": "after_training",
                "memory_enabled": memory_enabled,
                "is_current_task": True
            })
            
            # Add token-based metrics if requested
            if args.log_token_metrics and not args.functional_eval:
                metrics[-1]["token_accuracy"] = after_accuracy["accuracy"]
                metrics[-1]["error_distribution"] = after_accuracy.get('error_analysis', {}).get('error_distribution', {})
                metrics[-1]["total_errors"] = after_accuracy.get('error_analysis', {}).get('total_errors', 0)
            
            # Store memory examples (if enabled)
            if memory_enabled:
                # Store a subset of examples for memory
                memory_size = min(10, len(task_examples) // 2)  # Store up to 10 examples
                stored_memories.extend(random.sample(task_examples, min(memory_size, len(task_examples))))
                print(f"Added {memory_size} examples to memory")
        
        # Calculate overall CL metrics after all tasks are complete
        T = len(task_ordering)
        if T > 0:
            # Average Accuracy (AA): Mean final success over all tasks
            # Take the last row of the results matrix (after all tasks trained)
            AA = np.mean(results[T-1]) if T > 0 else 0.0
            
            # Forgetting (F): Avg. drop from peak success
            forgetting_values = []
            for i in range(T-1):  # Exclude the last task
                forgetting_values.append(peak[i] - results[T-1, i])
            F = np.mean(forgetting_values) if forgetting_values else 0.0
            
            # Forward Transfer (FWT): Benefit for unseen task after preceding tasks
            fwt_values = []
            for i in range(1, T):  # Start from the second task
                fwt_values.append(results[i-1, i] - results[0, i])
            FWT = np.mean(fwt_values) if fwt_values else 0.0
            
            # Backward Transfer (BWT): Effect of later learning on earlier tasks
            bwt_values = []
            for i in range(T-1):  # Exclude the last task
                bwt_values.append(results[T-1, i] - results[i, i])
            BWT = np.mean(bwt_values) if bwt_values else 0.0
            
            # Create summary dictionary
            summary = {
                "AA": float(AA),
                "F": float(F),
                "FWT": float(FWT),
                "BWT": float(BWT),
                "wall_clock_minutes": float((time.time() - start_time) / 60),
                "memory_enabled": memory_enabled
            }
            
            # Add GPU information if available
            if torch.cuda.is_available():
                gpu_hours = (time.time() - start_time) / 3600
                summary["gpu_hours"] = float(gpu_hours)
            
            # Save summary to JSON
            summary_file = os.path.join(args.output_dir, f"summary_{'with_memory' if memory_enabled else 'no_memory'}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nContinual Learning Metrics:")
            print(f"Average Accuracy (AA): {AA:.4f}")
            print(f"Forgetting (F): {F:.4f}")
            print(f"Forward Transfer (FWT): {FWT:.4f}")
            print(f"Backward Transfer (BWT): {BWT:.4f}")
            print(f"\nSaved summary to {summary_file}")
        
        # Save metrics to CSV with the new format
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(args.output_dir, f"metrics_{'with_memory' if memory_enabled else 'no_memory'}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nSaved metrics to {metrics_file}")
        
        # Plot metrics
        plot_metrics(metrics_df, args.output_dir, args.log_token_metrics)

if __name__ == "__main__":
    main()
