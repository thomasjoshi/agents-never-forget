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
import ast
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
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


def evaluate_success(model, dataset, batch_size=4):
    """
    Evaluates model success based on test execution instead of token matching.
    
    Args:
        model: The model to evaluate
        dataset: List of examples from the task (each containing prompt, FAIL_TO_PASS and PASS_TO_PASS tests)
        batch_size: Batch size for generation
    
    Returns:
        success_rate : float  – fraction of issues whose FAIL_TO_PASS tests
                                now pass and whose original PASS_TO_PASS
                                still pass.
        successes    : list[bool] – per-issue pass/fail flags.
    
    Notes:
        * Pull tokenizer/model.device from the model itself (no extra args).
        * Call helper `run_swebench_tests(issue, patch_path, timeout=12)`.
        * Provide a stub that returns False and prints a warning if Docker
          is unavailable, so CPU-only CI still runs.
    """
    import tempfile
    import os
    import subprocess
    import shutil
    
    # Check if Docker is available
    def is_docker_available():
        try:
            result = subprocess.run(["docker", "--version"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   check=False)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    if not is_docker_available():
        print("WARNING: Docker is not available. Test-based evaluation disabled.")
        print("Returning dummy results for CI compatibility.")
        return 0.0, [False] * len(dataset)
    
    # Extract tokenizer from model (may be wrapped in PEFT model)
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'tokenizer'):
        tokenizer = model.base_model.tokenizer
    elif hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
    else:
        # Fallback to loading tokenizer from model path
        try:
            from transformers import AutoTokenizer
            model_name = model.config._name_or_path
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Could not determine tokenizer: {e}")
            return 0.0, [False] * len(dataset)
    
    # Determine device
    device = next(model.parameters()).device
    
    # Helper function to run tests for a single issue
    def run_swebench_tests(issue, patch_path, timeout=12):
        """
        Run SWE-Bench tests for a single issue using the patch file.
        
        Args:
            issue: Dictionary containing issue details including tests
            patch_path: Path to the generated patch file
            timeout: Timeout in seconds for test execution
            
        Returns:
            Boolean indicating if all tests passed
        """
        try:
            # Create a temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare repo and apply patch
                repo_path = os.path.join(temp_dir, "repo")
                os.makedirs(repo_path, exist_ok=True)
                
                # Initialize repository with base commit
                base_commit = issue.get("base_commit")
                repo_name = issue.get("repo")
                if not base_commit or not repo_name:
                    print(f"Missing base_commit or repo for issue {issue.get('instance_id')}")
                    return False
                
                # Clone repository at specific commit
                clone_cmd = [
                    "git", "clone", 
                    f"https://github.com/{repo_name}.git",
                    repo_path
                ]
                subprocess.run(clone_cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
                
                # Checkout base commit
                checkout_cmd = ["git", "checkout", base_commit]
                subprocess.run(checkout_cmd, 
                              cwd=repo_path, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              check=False)
                
                # Apply patch
                with open(patch_path, "r") as f:
                    patch_content = f.read()
                
                apply_cmd = ["git", "apply", "-"]
                apply_proc = subprocess.Popen(apply_cmd, 
                                            cwd=repo_path,
                                            stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
                apply_proc.communicate(input=patch_content.encode())
                
                # Run tests in Docker container
                # Prepare test commands for FAIL_TO_PASS tests
                f2p_tests = issue.get("FAIL_TO_PASS_list", [])
                p2p_tests = issue.get("PASS_TO_PASS_list", [])
                
                # Build Docker image
                docker_build_cmd = [
                    "docker", "build", 
                    "-t", f"swebench-{issue.get('instance_id')}", 
                    "."
                ]
                build_result = subprocess.run(docker_build_cmd, 
                                           cwd=repo_path,
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, 
                                           timeout=60,
                                           check=False)
                if build_result.returncode != 0:
                    print(f"Docker build failed for {issue.get('instance_id')}")
                    return False
                
                # Run tests in container
                all_tests_pass = True
                
                # Run FAIL_TO_PASS tests (should now pass)
                for test in f2p_tests:
                    test_cmd = [
                        "docker", "run", 
                        f"swebench-{issue.get('instance_id')}", 
                        "pytest", test, "-v"
                    ]
                    test_result = subprocess.run(test_cmd, 
                                              stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE, 
                                              timeout=timeout,
                                              check=False)
                    if test_result.returncode != 0:
                        all_tests_pass = False
                        break
                
                # Run PASS_TO_PASS tests (should still pass)
                if all_tests_pass:
                    for test in p2p_tests:
                        test_cmd = [
                            "docker", "run", 
                            f"swebench-{issue.get('instance_id')}", 
                            "pytest", test, "-v"
                        ]
                        test_result = subprocess.run(test_cmd, 
                                                  stdout=subprocess.PIPE, 
                                                  stderr=subprocess.PIPE, 
                                                  timeout=timeout,
                                                  check=False)
                        if test_result.returncode != 0:
                            all_tests_pass = False
                            break
                
                return all_tests_pass
                
        except Exception as e:
            print(f"Error running tests: {e}")
            return False
    
    # Process each example in the dataset
    successes = []
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for i, issue in enumerate(dataset):
            print(f"Evaluating issue {i+1}/{len(dataset)}: {issue.get('instance_id', 'unknown')}")
            
            # Generate patch with model
            try:
                # Prepare input for the model
                prompt = issue.get("prompt") or f"Fix the following code:\n\n{issue.get('patch', '')}\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # Generate the patch
                generation_config = {
                    "max_new_tokens": 1024,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                
                # Handle different model types (PEFT vs regular)
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Decode the generated patch
                generated_patch = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract just the part after the prompt
                generated_patch = generated_patch[len(prompt):].strip()
                
                # Write the generated patch to a temp file
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".patch") as patch_file:
                    patch_file.write(generated_patch)
                    patch_path = patch_file.name
                
                # Run tests with the generated patch
                issue_success = run_swebench_tests(issue, patch_path, timeout=12)
                successes.append(issue_success)
                
                # Clean up the temp file
                os.unlink(patch_path)
                
            except Exception as e:
                print(f"Error generating patch: {e}")
                successes.append(False)
    
    # Calculate overall success rate
    success_rate = sum(successes) / len(successes) if successes else 0.0
    
    print(f"Overall success rate: {success_rate:.4f} ({sum(successes)}/{len(successes)} issues resolved)")
    return success_rate, successes

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

def plot_metrics(metrics_df, output_dir, results=None, memory_enabled=False):
    """
    Plot continual learning metrics with enhanced visualization including heat-maps and stability-plasticity plots.
    
    Args:
        metrics_df: DataFrame containing metrics data
        output_dir: Directory to save plots
        results: numpy array of success rates for each task combination (task_curr x task_seen)
        memory_enabled: Whether memory was enabled for these results
    """
    os.makedirs(output_dir, exist_ok=True)
    memory_suffix = "with_memory" if memory_enabled else "no_memory"
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Only generate token-based plots if requested and available
    if 'token_accuracy' in metrics_df.columns:
        # Create a 2x2 grid of plots for token metrics
        plt.figure(figsize=(16, 12))
        
        # 1. Plot token accuracy over tasks
        plt.subplot(2, 2, 1)
        sns.lineplot(data=metrics_df, 
                    x="task_id", y="token_accuracy", hue="seen_task", 
                    markers=True, dashes=False)
        plt.title("Token Accuracy by Task")
        plt.xlabel("Current Task ID")
        plt.ylabel("Token Accuracy")
        plt.legend(title="Task Being Evaluated", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Plot exact match rate
        if 'exact_match' in metrics_df.columns:
            plt.subplot(2, 2, 2)
            sns.lineplot(data=metrics_df, 
                        x="task_id", y="exact_match", hue="seen_task", 
                        markers=True, dashes=False)
            plt.title("Exact Match Rate by Task")
            plt.xlabel("Current Task ID")
            plt.ylabel("Exact Match Rate")
            plt.legend(title="Task Being Evaluated", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Plot F1 score
        if 'f1_score' in metrics_df.columns:
            plt.subplot(2, 2, 3)
            sns.lineplot(data=metrics_df, 
                        x="task_id", y="f1_score", hue="seen_task", 
                        markers=True, dashes=False)
            plt.title("F1 Score by Task")
            plt.xlabel("Current Task ID")
            plt.ylabel("F1 Score")
            plt.legend(title="Task Being Evaluated", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"token_metrics_{memory_suffix}.png"), dpi=300)
        plt.close()
    
    # Plot functional success metrics (main visualization)
    
    # 1. Heat-map of success rates
    if results is not None and results.size > 0:
        plt.figure(figsize=(10, 8))
        
        # Get number of tasks
        num_tasks = results.shape[0]
        
        # Create task labels
        task_labels = [f"T_{i}" for i in range(num_tasks)]
        
        # Create heatmap
        sns.heatmap(results, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=task_labels, yticklabels=task_labels)
        plt.title(f"Success Rate Heatmap ({memory_suffix})")
        plt.xlabel("Task Evaluated")
        plt.ylabel("After Training on Task")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heatmap_{memory_suffix}.png"), dpi=300)
        plt.close()
    
    # 2. Stability-Plasticity Line Plot
    plt.figure(figsize=(12, 6))
    
    # Filter data for success rate plotting
    current_task_df = metrics_df[metrics_df['is_current'] == True].copy()
    prev_task_df = metrics_df[metrics_df['is_current'] == False].copy()
    
    # Plot success on current tasks (plasticity)
    ax = sns.lineplot(data=current_task_df, x='task_id', y='success_rate', 
                     marker='o', markersize=10, label="Current Task (Plasticity)")
    
    # Plot average success on previous tasks (stability)
    if not prev_task_df.empty:
        stability_data = []
        for task_id in sorted(prev_task_df['task_id'].unique()):
            if task_id > 0:  # Skip initial evaluation
                task_data = prev_task_df[prev_task_df['task_id'] == task_id]
                avg_success = task_data['success_rate'].mean()
                stability_data.append({'task_id': task_id, 'avg_success_rate': avg_success})
        
        if stability_data:
            stability_df = pd.DataFrame(stability_data)
            sns.lineplot(data=stability_df, x='task_id', y='avg_success_rate', 
                        marker='s', markersize=10, label="Previous Tasks (Stability)")
    
    plt.title(f"Stability-Plasticity Trade-off ({memory_suffix})")
    plt.xlabel("Current Task ID")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"stability_plasticity_{memory_suffix}.png"), dpi=300)
    plt.close()
    
    # 3. Individual metrics plots
    plt.figure(figsize=(15, 5))
    
    # Plot forgetting
    non_current_df = metrics_df[metrics_df['is_current'] == False]
    if 'forgetting' in non_current_df.columns and not non_current_df.empty:
        plt.subplot(1, 3, 1)
        task_forgetting = []
        for task_id in sorted(non_current_df['task_id'].unique()):
            if task_id > 0:  # Skip initial task
                task_data = non_current_df[non_current_df['task_id'] == task_id]
                avg_forgetting = task_data['forgetting'].mean()
                task_forgetting.append({'task_id': task_id, 'avg_forgetting': avg_forgetting})
        
        if task_forgetting:
            forgetting_df = pd.DataFrame(task_forgetting)
            sns.lineplot(data=forgetting_df, x='task_id', y='avg_forgetting', marker='o')
            plt.title("Mean Forgetting")
            plt.xlabel("Current Task ID")
            plt.ylabel("Forgetting")
    
    # Plot forward transfer
    if 'fwt' in metrics_df.columns:
        plt.subplot(1, 3, 2)
        fwt_df = metrics_df[(metrics_df['is_current'] == True) & (metrics_df['fwt'].notna())]
        if not fwt_df.empty:
            sns.lineplot(data=fwt_df, x='task_id', y='fwt', marker='o')
            plt.title("Forward Transfer")
            plt.xlabel("Task ID")
            plt.ylabel("Forward Transfer")
    
    # Plot backward transfer
    if 'bwt' in metrics_df.columns:
        plt.subplot(1, 3, 3)
        bwt_df = metrics_df[(metrics_df['is_current'] == False) & (metrics_df['bwt'].notna())]
        if not bwt_df.empty:
            task_bwt = []
            for task_id in sorted(bwt_df['task_id'].unique()):
                if task_id > 0:  # Skip initial task
                    task_data = bwt_df[bwt_df['task_id'] == task_id]
                    avg_bwt = task_data['bwt'].mean()
                    task_bwt.append({'task_id': task_id, 'avg_bwt': avg_bwt})
            
            if task_bwt:
                bwt_summary = pd.DataFrame(task_bwt)
                sns.lineplot(data=bwt_summary, x='task_id', y='avg_bwt', marker='o')
                plt.title("Backward Transfer")
                plt.xlabel("Current Task ID")
                plt.ylabel("Backward Transfer")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cl_metrics_{memory_suffix}.png"), dpi=300)
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
                        help="Skip Docker cache warmup before evaluation (slower first evaluation)")
    
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
        
        # Track task accuracies for transfer analysis
        task_accuracies = {task_name: [] for task_name in task_ordering}
        initial_accuracies = {task_name: None for task_name in task_ordering}
        
        # Initialize CL metric tracking
        num_tasks = len(task_ordering)
        results = np.zeros((num_tasks, num_tasks))  # Task x Task matrix for success rates
        peak = defaultdict(float)  # Track peak performance on each task
        
        # Initial evaluation on all tasks to measure forward transfer potential
        print("\nInitial evaluation on all tasks (forward transfer baseline)...")
        for task_idx, task_name in enumerate(task_ordering):
            task_examples = task_stream.get(task_name, [])
            if not task_examples:
                continue
                
            print(f"Evaluating {task_name} (initial)...")
            
            # Always get functional test success metrics
            success_rate, successes = evaluate_success(
                model=model,
                dataset=task_examples,
                batch_size=args.batch_size
            )
            
            # Store the initial success rate in results matrix (row 0)
            results[0, task_idx] = success_rate
            
            # Optionally get token-level metrics if requested
            token_metrics = None
            if args.log_token_metrics:
                token_metrics = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    task_examples=task_examples,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_memory=memory_enabled,
                    memory_examples=stored_memories
                )
            
            # Track initial performance for both metrics types
            initial_accuracies[task_name] = {
                "success_rate": success_rate,
                "token_metrics": token_metrics
            }
            
            # Add to metrics tracking
            metrics_entry = {
                "task_id": 0,  # Initial evaluation
                "seen_task": task_idx,
                "success_rate": success_rate,
                "is_current": False,
                "task_name": task_name,
                "evaluation_type": "before_training",
                "memory_enabled": memory_enabled
            }
            
            # Add token metrics if available
            if token_metrics:
                metrics_entry["token_accuracy"] = token_metrics.get("accuracy", 0.0)
                metrics_entry["exact_match"] = token_metrics.get("exact_match", 0.0)
                metrics_entry["f1_score"] = token_metrics.get("f1", 0.0)
            
            metrics.append(metrics_entry)
        
        # Train on each task sequentially
        for task_idx, task_name in enumerate(tqdm(task_ordering, desc="Training on tasks")):
            task_examples = task_stream.get(task_name, [])
            if not task_examples:
                print(f"Warning: No examples found for task {task_name}, skipping...")
                continue
                
            print(f"\n{'='*80}")
            print(f"Task {task_idx}: {task_name} ({len(task_examples)} examples)")
            print(f"{'='*80}")
            
            # Evaluate on current task before update using functional test success
            print("Evaluating on current task (before update)...")
            before_success_rate, before_successes = evaluate_success(
                model=model,
                dataset=task_examples,
                batch_size=args.batch_size
            )
            
            # Optionally evaluate with token-level metrics if requested
            before_token_metrics = None
            if args.log_token_metrics:
                before_token_metrics = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    task_examples=task_examples,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_memory=memory_enabled,
                    memory_examples=stored_memories
                )
            
            # Limit examples for faster training if specified
            if args.max_examples and len(task_examples) > args.max_examples:
                task_examples = random.sample(task_examples, args.max_examples)
                print(f"Limiting to {args.max_examples} examples for training")
            
            # Fine-tune model on current task
            print(f"Finetuning model on {len(task_examples)} examples...")
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
                max_steps=args.max_steps,
                use_4bit=args.use_4bit
            )
            
            # Save model checkpoint
            checkpoint_dir = save_model_checkpoint(
                model, tokenizer, args.output_dir, 
                f"{'with_memory' if memory_enabled else 'no_memory'}_{task_name}"
            )
            
            # Evaluate model on current task (after update) using functional test success
            print("Evaluating on current task (after update)...")
            after_success_rate, after_successes = evaluate_success(
                model=model,
                dataset=task_examples,
                batch_size=args.batch_size
            )
            
            # Store results for current task
            results[task_idx, task_idx] = after_success_rate
            peak[task_idx] = max(peak[task_idx], after_success_rate)  # Update peak performance
            
            # Optionally evaluate with token-level metrics if requested
            after_token_metrics = None
            if args.log_token_metrics:
                after_token_metrics = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    task_examples=task_examples,
                    device=args.device,
                    batch_size=args.batch_size,
                    use_memory=memory_enabled,
                    memory_examples=stored_memories
                )
            
            # Calculate forward transfer for current task
            forward_transfer = None
            if task_idx > 0:  # Only calculate for tasks after the first one
                forward_transfer = results[task_idx-1, task_idx] - results[0, task_idx]
            
            # Add metrics for current task
            metrics_entry = {
                "task_id": task_idx,
                "seen_task": task_idx,
                "success_rate": after_success_rate,
                "is_current": True,
                "task_name": task_name,
                "memory_enabled": memory_enabled,
                "forward_transfer": forward_transfer
            }
            
            if args.log_token_metrics and after_token_metrics:
                metrics_entry["token_accuracy"] = after_token_metrics.get("accuracy", 0.0)
                metrics_entry["exact_match"] = after_token_metrics.get("exact_match", 0.0)
                metrics_entry["f1_score"] = after_token_metrics.get("f1", 0.0)
            
            # Store metrics
            metrics.append(metrics_entry)
            
            # Evaluate on all previous tasks to measure forgetting and backward transfer
            for prev_task_idx in range(task_idx):
                prev_task_name = task_ordering[prev_task_idx]
                prev_task_examples = task_stream.get(prev_task_name, [])
                if not prev_task_examples:
                    continue
                    
                print(f"Evaluating on previous task {prev_task_name}...")
                
                # Evaluate using functional test success
                prev_success_rate, prev_successes = evaluate_success(
                    model=model,
                    dataset=prev_task_examples,
                    batch_size=args.batch_size
                )
                
                # Store in results matrix
                results[task_idx, prev_task_idx] = prev_success_rate
                
                # Calculate metrics
                forgetting = peak[prev_task_idx] - prev_success_rate
                backward_transfer = prev_success_rate - results[prev_task_idx, prev_task_idx]
                
                # Optionally get token-level metrics
                prev_token_metrics = None
                if args.log_token_metrics:
                    prev_token_metrics = evaluate_model(
                        model=model,
                        tokenizer=tokenizer,
                        task_examples=prev_task_examples,
                        device=args.device,
                        batch_size=args.batch_size,
                        use_memory=memory_enabled,
                        memory_examples=stored_memories
                    )
                
                # Store metrics
                metrics_entry = {
                    "task_id": task_idx,
                    "seen_task": prev_task_idx,
                    "success_rate": prev_success_rate,
                    "is_current": False,
                    "task_name": prev_task_name,
                    "memory_enabled": memory_enabled,
                    "forgetting": max(0, forgetting),  # Don't allow negative forgetting
                    "backward_transfer": backward_transfer
                }
                
                if args.log_token_metrics and prev_token_metrics:
                    metrics_entry["token_accuracy"] = prev_token_metrics.get("accuracy", 0.0)
                    metrics_entry["exact_match"] = prev_token_metrics.get("exact_match", 0.0)
                    metrics_entry["f1_score"] = prev_token_metrics.get("f1", 0.0)
                
                metrics.append(metrics_entry)
                
            # Store memory examples (if enabled)
            if memory_enabled:
                # Store a subset of examples for memory
                memory_size = min(10, len(task_examples) // 2)  # Store up to 10 examples
                stored_memories.extend(random.sample(task_examples, min(memory_size, len(task_examples))))
                print(f"Added {min(memory_size, len(task_examples))} examples to memory (total: {len(stored_memories)})")
        
        # After all tasks, compute the continual learning metrics
        num_tasks = len(task_ordering)
        if num_tasks > 0:
            # 1. Average Accuracy: Mean final success over all tasks
            avg_accuracy = np.mean([results[num_tasks-1, i] for i in range(num_tasks)])
            
            # 2. Mean Forgetting: Avg. drop from peak success
            forgetting_values = [peak[i] - results[num_tasks-1, i] for i in range(num_tasks-1)]
            mean_forgetting = np.mean(forgetting_values) if forgetting_values else 0.0
            
            # 3. Forward Transfer: Benefit for unseen task after preceding tasks
            forward_values = [results[i-1, i] - results[0, i] for i in range(1, num_tasks)]
            forward_transfer = np.mean(forward_values) if forward_values else 0.0
            
            # 4. Backward Transfer: Effect of later learning on earlier tasks
            backward_values = [results[num_tasks-1, i] - results[i, i] for i in range(num_tasks-1)]
            backward_transfer = np.mean(backward_values) if backward_values else 0.0
            
            # Calculate wall clock and GPU time
            wall_clock_minutes = 0  # TODO: Track actual time
            gpu_hours = 0  # TODO: Track GPU usage
            
            # Create summary metrics
            summary = {
                "Average Accuracy": float(avg_accuracy),
                "Mean Forgetting": float(mean_forgetting),
                "Forward Transfer": float(forward_transfer),
                "Backward Transfer": float(backward_transfer),
                "wall_clock_minutes": wall_clock_minutes,
                "gpu_hours": gpu_hours,
                "memory_enabled": memory_enabled
            }
            
            # Save summary to JSON
            summary_file = os.path.join(args.output_dir, f"summary_{'with_memory' if memory_enabled else 'no_memory'}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary metrics to {summary_file}")
            
            # Print summary
            print("\nContinual Learning Summary:")
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            print(f"Mean Forgetting: {mean_forgetting:.4f}")
            print(f"Forward Transfer: {forward_transfer:.4f}")
            print(f"Backward Transfer: {backward_transfer:.4f}")
        
        # Save metrics to CSV in the required format
        # Format: task_id,seen_task,success_rate,is_current,forgetting,bwt,fwt
        metrics_rows = []
        for entry in metrics:
            # Create a clean row with only the required fields
            row = {
                'task_id': entry.get('task_id', 0),
                'seen_task': entry.get('seen_task', 0),
                'success_rate': entry.get('success_rate', 0.0),
                'is_current': entry.get('is_current', False),
                'forgetting': entry.get('forgetting', ''),  # Empty if not defined
                'bwt': entry.get('backward_transfer', ''),  # Empty if not defined
                'fwt': entry.get('forward_transfer', '')    # Empty if not defined
            }
            
            # Add token metrics if available and requested
            if args.log_token_metrics:
                if 'token_accuracy' in entry:
                    row['token_accuracy'] = entry['token_accuracy']
                if 'exact_match' in entry:
                    row['exact_match'] = entry['exact_match']
                if 'f1_score' in entry:
                    row['f1_score'] = entry['f1_score']
            
            metrics_rows.append(row)
        
        # Create DataFrame and save to CSV
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_file = os.path.join(args.output_dir, f"metrics_{'with_memory' if memory_enabled else 'no_memory'}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nSaved metrics to {metrics_file}")
        
        # Plot metrics
        plot_metrics(metrics_df, args.output_dir, results=results, memory_enabled=memory_enabled)

if __name__ == "__main__":
    main()
