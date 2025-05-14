#!/usr/bin/env python
"""Finetune and evaluate a model on a sequence of continual learning tasks.

This script loads TinyLlama-1.1B-Chat with 4-bit quantization, finetunes it sequentially on SWE-Bench-CL tasks,
and evaluates catastrophic forgetting and forward transfer.
"""

import sys
import os
import time

import os
import json
import random
import time
import argparse
import pickle
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


print(f"\n{'='*80}")
print(f"SCRIPT STARTING: {os.path.basename(__file__)}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PID: {os.getpid()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Working directory: {os.getcwd()}")
sys.stdout.flush()  # Force output buffer to flush
print(f"{'='*80}\n")

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
    gpu_mem = {"allocated": 0, "reserved": 0}
    cpu_percent = 0
    
    # Get GPU memory info if available
    if torch.cuda.is_available():
        gpu_mem = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # Convert to GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # Convert to GB
        }
    
    # Get CPU usage
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except (ImportError, Exception):
        cpu_percent = 0
    
    return {
        "gpu": gpu_mem,
        "cpu_percent": cpu_percent
    }

class SWEBenchDataset(Dataset):
    """Dataset for SWE-Bench tasks with optimized caching."""
    
    # Class-level cache for datasets to avoid redundant processing
    _cached_datasets = {}
    _cached_encodings = {}
    
    @classmethod
    def from_cached(cls, examples, tokenizer, max_length=1024, use_memory=False, memory_examples=None):
        """
        Create a dataset, using cached version if available.
        
        Args:
            examples: List of examples
            tokenizer: Tokenizer to use
            max_length: Maximum token length
            use_memory: Whether to include memory examples
            memory_examples: Examples from memory to include
            
        Returns:
            SWEBenchDataset instance (cached if possible)
        """
        # Generate cache key based on inputs
        cache_key = hash(f"{id(examples)}_{id(tokenizer)}_{max_length}_{use_memory}_{id(memory_examples)}")
        
        if cache_key in cls._cached_datasets:
            log_step(f"Using cached dataset (key: {cache_key})")
            return cls._cached_datasets[cache_key]
        
        # Create new dataset if not cached
        dataset = cls(examples, tokenizer, max_length, use_memory, memory_examples)
        cls._cached_datasets[cache_key] = dataset
        return dataset
    
    def __init__(self, examples, tokenizer, max_length=1024, use_memory=False, memory_examples=None):
        """
        Initialize dataset.
        
        Args:
            examples: List of examples
            tokenizer: Tokenizer to use
            max_length: Maximum token length
            use_memory: Whether to include memory examples
            memory_examples: Examples from memory to include
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_memory = use_memory
        self.memory_examples = memory_examples or []
        
        # Pre-tokenize all examples at once for efficiency
        log_step(f"Preparing dataset with {len(examples)} examples")
        self.encodings = self._prepare_encodings()
        
    def _prepare_encodings(self):
        """Pre-tokenize all examples for better performance."""
        start_time = time.time()
        
        # Format all examples
        prompts = []
        memory_text = ""
        
        # Prepare memory text once if needed
        if self.use_memory and self.memory_examples:
            memory_text = format_memory(self.memory_examples)
        
        # Process examples in small batches for memory efficiency
        batch_size = 16  # Process 16 examples at a time
        all_encodings = []
        
        for i in range(0, len(self.examples), batch_size):
            batch = self.examples[i:i+batch_size]
            batch_prompts = []
            
            for example in batch:
                # Format example
                prompt = format_example(example, self.tokenizer)
                
                # Add memory if needed
                if self.use_memory and self.memory_examples:
                    prompt = memory_text + "\n\n" + prompt
                
                batch_prompts.append(prompt)
            
            # Tokenize batch (more efficient than one-by-one)
            batch_encodings = [tokenize_example(prompt, self.tokenizer, self.max_length) 
                              for prompt in batch_prompts]
            all_encodings.extend(batch_encodings)
            
            # Free memory
            del batch_prompts
        
        log_step(f"Dataset preparation completed in {time.time() - start_time:.2f} seconds")
        return all_encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]

def format_example(example, tokenizer):
    """Format an example for the model."""
    return f"Fix the following code:\n\n{example['patch']}"


def format_memory(memory_examples):
    """Format memory examples for the model."""
    memory_items = []
    for mem in memory_examples:
        memory_items.append(f"Remember:\n{mem['patch']}\n\nTest Case:\n{mem.get('test_patch', '')}")
    return "\n\n".join(memory_items)


def tokenize_example(text, tokenizer, max_length):
    """Tokenize an example for the model."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Shift labels for causal language modeling
    input_ids = encoding["input_ids"].squeeze()
    labels = input_ids.clone()
    
    # Set padding tokens to -100 so they're ignored in the loss
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"].squeeze(),
        "labels": labels
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
    log_step(f"Initializing model: {model_name}")
    log_step(f"Configuration: 4-bit={use_4bit}, LoRA={use_lora}, device={device}")
    print(f"DEBUG: Model initialization started at {time.strftime('%H:%M:%S')}")
    print(f"DEBUG: Model path: {model_name}")
    sys.stdout.flush()
    
    # Configure quantization if enabled
    bnb_config = None
    if use_4bit:
        print(f"DEBUG: Setting up 4-bit quantization at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        log_step("Configuring 4-bit quantization")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        log_step(f"Using compute dtype: {compute_dtype}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model with optimizations
    model_start_time = time.time()
    print(f"DEBUG: Beginning model loading at {time.strftime('%H:%M:%S')}")
    print(f"DEBUG: Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")
    sys.stdout.flush()
    log_step("Loading model weights...")
    try:
        print(f"DEBUG: Calling from_pretrained() at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        
        # Add more granular progress when loading model
        hf_hub_download_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Disable cache for faster training
        )
        
        print(f"DEBUG: from_pretrained() completed in {time.time() - hf_hub_download_start:.2f} seconds")
        print(f"DEBUG: Total model loading time: {time.time() - model_start_time:.2f} seconds")
        sys.stdout.flush()
        log_step(f"Model loaded successfully on device: {model.device}")
        
        # Log model size and memory usage
        print(f"DEBUG: Calculating model parameters at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_step(f"Model parameters: {param_count:,} total, {trainable_params:,} trainable")
        
        print(f"DEBUG: Model parameter counts: {param_count:,} total, {trainable_params:,} trainable")
        sys.stdout.flush()
        
        if torch.cuda.is_available():
            print(f"DEBUG: GPU info at {time.strftime('%H:%M:%S')}")
            memory_allocated = torch.cuda.memory_allocated()/1024**2
            memory_reserved = torch.cuda.memory_reserved()/1024**2
            total_memory = torch.cuda.get_device_properties(0).total_memory/1024**2
            log_step(f"GPU Memory allocated: {memory_allocated:.2f} MB")
            log_step(f"GPU Memory reserved: {memory_reserved:.2f} MB")
            
            print(f"DEBUG: GPU Memory: {memory_allocated:.2f}MB allocated, {memory_reserved:.2f}MB reserved, {total_memory:.2f}MB total")
            print(f"DEBUG: GPU utilization: {memory_allocated/total_memory*100:.2f}%")
            sys.stdout.flush()
            
    except Exception as e:
        log_step(f"ERROR: Failed to load model: {str(e)}")
        if "out of memory" in str(e).lower():
            log_step("Out of memory error! Try reducing batch size or using a smaller model.")
        raise
    
    # Prepare model for k-bit training if using quantization
    if use_4bit:
        print(f"DEBUG: Beginning k-bit preparation at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        log_step("Preparing model for 4-bit training...")
        
        kbit_start = time.time()
        model = prepare_model_for_kbit_training(model)
        print(f"DEBUG: k-bit preparation completed in {time.time() - kbit_start:.2f} seconds")
        sys.stdout.flush()
    
    # Configure LoRA if enabled
    if use_lora:
        print(f"DEBUG: Beginning LoRA configuration at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        log_step("Configuring LoRA for parameter-efficient fine-tuning")
        try:
            lora_start = time.time()
            print(f"DEBUG: Creating LoRA config at {time.strftime('%H:%M:%S')}")
            sys.stdout.flush()
            
            lora_config = LoraConfig(
                r=16,  # Reduced rank for faster training
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            print(f"DEBUG: Applying LoRA adapter to model")
            sys.stdout.flush()
            model = get_peft_model(model, lora_config)
            print(f"DEBUG: LoRA configuration completed in {time.time() - lora_start:.2f} seconds")
            sys.stdout.flush()
            log_step("LoRA configuration applied successfully")
            model.print_trainable_parameters()
        except Exception as e:
            log_step(f"ERROR: Failed to apply LoRA: {str(e)}")
            raise
    
    # Load tokenizer
    print(f"DEBUG: Beginning tokenizer loading at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    log_step("Loading tokenizer...")
    try:
        tokenizer_start = time.time()
        print(f"DEBUG: Calling tokenizer.from_pretrained()")
        sys.stdout.flush()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            use_fast=True,  # Use fast tokenizer for better performance
            trust_remote_code=True,
        )
        
        print(f"DEBUG: Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds")
        sys.stdout.flush()
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token
        log_step(f"Tokenizer loaded with vocab size: {len(tokenizer):,}")
    except Exception as e:
        log_step(f"ERROR: Failed to load tokenizer: {str(e)}")
        raise
    
    # Enable gradient checkpointing to save memory
    print(f"DEBUG: Enabling gradient checkpointing at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    log_step("Enabling gradient checkpointing...")
    model.gradient_checkpointing_enable()
    
    total_init_time = time.time() - model_start_time
    print(f"DEBUG: Total model initialization completed in {total_init_time:.2f} seconds")
    print(f"DEBUG: Final GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.get_device_properties(0).total_memory/1024**2:.2f}MB")
    sys.stdout.flush()
    
    log_step("Model and tokenizer loaded successfully")
    return model, tokenizer

def load_task_stream(task_stream_file, filter_repo=None):
    """
    Load the task stream from a file and optionally filter for specific repo.
    Supports both JSON and preprocessed pickle formats for faster loading.
    
    Args:
        task_stream_file: Path to task stream file (JSON or pickle)
        filter_repo: Optional repository name to filter tasks (e.g., 'sympy/sympy')
    
    Returns:
        Dictionary containing task stream data and task ordering
    """
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] LOADING TASK STREAM: {task_stream_file}")
    sys.stdout.flush()
    
    log_step(f"Attempting to load task stream from: {os.path.abspath(task_stream_file)}")
    
    # Check if file exists
    if not os.path.exists(task_stream_file):
        raise FileNotFoundError(f"Task stream file not found: {os.path.abspath(task_stream_file)}")
    
    # Determine which file to load
    file_to_load = task_stream_file
    use_pickle = task_stream_file.endswith('.pkl')
    
    # If JSON was specified but a pickle file exists and is newer, use that instead
    if not use_pickle and task_stream_file.endswith('.json'):
        pickle_file = os.path.splitext(task_stream_file)[0] + ".pkl"
        if os.path.exists(pickle_file):
            # Check modification time only if both files exist
            if os.path.getmtime(pickle_file) > os.path.getmtime(task_stream_file):
                file_to_load = pickle_file
                use_pickle = True
                log_step(f"Found newer preprocessed file: {os.path.basename(pickle_file)}")
            else:
                log_step(f"Found preprocessed file but it's older than JSON, using JSON")
    
    # Timer for performance measurement
    start_time = time.time()
    print(f"DEBUG: Begin file loading at {time.strftime('%H:%M:%S')}, path: {file_to_load}")
    print(f"DEBUG: File size: {os.path.getsize(file_to_load) / (1024 * 1024):.2f} MB")
    sys.stdout.flush()
    
    # Load the data with fine-grained debugging
    try:
        if use_pickle:
            print(f"DEBUG: About to open pickle file at {time.strftime('%H:%M:%S')}")
            sys.stdout.flush()
            
            # Try opening the file first to isolate file IO issues
            try:
                open_start = time.time()
                with open(file_to_load, 'rb') as f:
                    print(f"DEBUG: File opened successfully in {time.time() - open_start:.3f} seconds, now loading pickle data")
                    sys.stdout.flush()
                    
                    # Try reading a small amount to check file validity
                    try:
                        f.seek(0)
                        header = f.read(100)  # Read just a bit to check if file seems valid
                        f.seek(0)  # Reset position
                        print(f"DEBUG: Successfully read header bytes, file appears readable")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"DEBUG: Error reading file header: {e}")
                        sys.stdout.flush()
                    
                    # Now try the actual pickle load with timing
                    try:
                        pickle_start = time.time()
                        print(f"DEBUG: Beginning pickle.load() at {time.strftime('%H:%M:%S')}")
                        sys.stdout.flush()
                        task_stream = pickle.load(f)
                        print(f"DEBUG: pickle.load() completed in {time.time() - pickle_start:.3f} seconds")
                        sys.stdout.flush()
                    except Exception as e:
                        print(f"DEBUG: Error during pickle.load(): {e}")
                        sys.stdout.flush()
                        raise
            except Exception as e:
                print(f"DEBUG: Error opening file: {e}")
                sys.stdout.flush()
                raise
                
            log_step(f"Loading preprocessed data from {os.path.basename(file_to_load)}")
        else:
            print(f"DEBUG: About to open JSON file at {time.strftime('%H:%M:%S')}")
            sys.stdout.flush()
            log_step(f"Loading JSON data from {os.path.basename(file_to_load)}")
            
            try:
                json_start = time.time()
                with open(file_to_load, 'r') as f:
                    print(f"DEBUG: File opened, now parsing JSON")
                    sys.stdout.flush()
                    task_stream = json.load(f)
                print(f"DEBUG: JSON parsing completed in {time.time() - json_start:.3f} seconds")
                sys.stdout.flush()
            except Exception as e:
                print(f"DEBUG: Error reading/parsing JSON: {e}")
                sys.stdout.flush()
                raise
                
        loading_time = time.time() - start_time
        log_step(f"Data loaded in {loading_time:.2f} seconds")
        
        print(f"DEBUG: Data type received: {type(task_stream)}")
        print(f"DEBUG: First-level keys or indices: {str(list(task_stream.keys())[:5]) if isinstance(task_stream, dict) else 'Not a dict'}")
        sys.stdout.flush()
        
        # Extract metadata if present
        print(f"DEBUG: Extracting metadata at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        metadata = task_stream.pop('metadata', {})
        print(f"DEBUG: Metadata extracted, size: {len(metadata)}")
        sys.stdout.flush()
        
        if metadata:
            log_step(f"Found metadata: {len(metadata)} entries")
            print(f"DEBUG: Metadata keys: {list(metadata.keys())}")
            sys.stdout.flush()
            
            if 'task_ordering' in metadata:
                log_step(f"Using task ordering from metadata")
                task_ordering = metadata['task_ordering']
                print(f"DEBUG: Got task_ordering from metadata, length: {len(task_ordering)}")
                sys.stdout.flush()
            else:
                print(f"DEBUG: No task_ordering in metadata, sorting keys")
                sys.stdout.flush()
                task_ordering = sorted(task_stream.keys())
                print(f"DEBUG: Created task_ordering by sorting, length: {len(task_ordering)}")
                sys.stdout.flush()
        else:
            print(f"DEBUG: No metadata found, sorting keys at {time.strftime('%H:%M:%S')}")
            sys.stdout.flush()
            sort_start = time.time()
            task_ordering = sorted(task_stream.keys())
            print(f"DEBUG: Keys sorted in {time.time() - sort_start:.3f} seconds, length: {len(task_ordering)}")
            sys.stdout.flush()
            
        # Count total examples with timing
        print(f"DEBUG: Counting total examples at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        count_start = time.time()
        
        try:
            total_examples = sum(len(examples) for examples in task_stream.values() 
                               if isinstance(examples, list))
            print(f"DEBUG: Counted {total_examples} total examples in {time.time() - count_start:.3f} seconds")
        except Exception as e:
            print(f"DEBUG: Error counting examples: {e}")
            print(f"DEBUG: Sample task_stream values: {str(list(task_stream.values())[:2])}")
        sys.stdout.flush()
        
        log_step(f"Loaded {len(task_stream)} tasks with {total_examples} total examples")
        
    except (json.JSONDecodeError, pickle.UnpicklingError) as e:
        raise ValueError(f"Failed to parse file {file_to_load}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_to_load}: {str(e)}")
    
    # If no filter is specified, return the entire task stream
    if not filter_repo:
        log_step(f"No repository filter applied, returning all tasks")
        
        # Brief task distribution logging
        for task in task_ordering[:5]:  # Show first 5 tasks
            if task in task_stream and isinstance(task_stream[task], list):
                log_step(f"  {task}: {len(task_stream[task])} examples")
        
        if len(task_ordering) > 5:
            log_step(f"  ... and {len(task_ordering) - 5} more tasks")
        
        return {
            "tasks": task_stream,
            "task_ordering": task_ordering
        }
    
    # Filter tasks by repository if specified - use multiprocessing for large datasets
    log_step(f"Filtering tasks for repository: {filter_repo}")
    print(f"DEBUG: Starting repository filtering for {filter_repo} at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    
    # Log structure of task stream to help debug potential issues
    print(f"DEBUG: Task stream contains {len(task_stream)} keys, task_ordering has {len(task_ordering)} items")
    print(f"DEBUG: First 3 keys in task_stream: {list(task_stream.keys())[:3]}")
    print(f"DEBUG: First 3 task_ordering items: {task_ordering[:3]}")
    sys.stdout.flush()
    
    # Determine if multiprocessing would be beneficial (for large datasets)
    use_multiprocessing = total_examples > 1000
    print(f"DEBUG: total_examples = {total_examples}, using multiprocessing: {use_multiprocessing}")
    sys.stdout.flush()
    
    filter_start = time.time()
    if use_multiprocessing:
        print(f"DEBUG: Starting parallel filtering at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        result = _filter_tasks_parallel(task_stream, task_ordering, filter_repo)
    else:
        print(f"DEBUG: Starting sequential filtering at {time.strftime('%H:%M:%S')}")
        sys.stdout.flush()
        result = _filter_tasks_sequential(task_stream, task_ordering, filter_repo)
    
    print(f"DEBUG: Filtering completed in {time.time() - filter_start:.2f} seconds")
    sys.stdout.flush()
    return result


def _filter_tasks_sequential(task_stream, task_ordering, filter_repo):
    """Filter tasks sequentially by repository."""
    print(f"DEBUG: Inside _filter_tasks_sequential for repo: {filter_repo}")
    sys.stdout.flush()
    
    filtered_tasks = {}
    total_examples = 0
    skipped_tasks = 0
    task_count = len(task_ordering)
    
    # Print start time for benchmarking
    start_time = time.time()
    print(f"DEBUG: Sequential filtering start time: {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    
    for i, task_name in enumerate(task_ordering):
        # Print progress every 20% or at least for the first few tasks
        if i < 5 or i % max(1, int(task_count * 0.2)) == 0:
            print(f"DEBUG: Processing task {i+1}/{task_count}: {task_name} at {time.strftime('%H:%M:%S')}")
            sys.stdout.flush()
            
        # Skip if task doesn't exist or examples is not a list
        if task_name not in task_stream or not isinstance(task_stream[task_name], list):
            if i < 5:  # Only log details for first few tasks to avoid log spam
                print(f"DEBUG: Skipping task {task_name}: exists={task_name in task_stream}, is_list={task_name in task_stream and isinstance(task_stream[task_name], list)}")
                sys.stdout.flush()
            skipped_tasks += 1
            continue
            
        examples = task_stream[task_name]
        if i < 5:  # Log details for first few tasks
            print(f"DEBUG: Task {task_name} has {len(examples)} examples")
            sys.stdout.flush()
            
        filtered = []
        
        for ex in examples:
            # Skip if the example is not a dictionary or doesn't have a 'repo' field
            if not isinstance(ex, dict) or 'repo' not in ex:
                continue
                
            # Check if the repository matches the filter
            if ex['repo'] == filter_repo:
                filtered.append(ex)
        
        # Only add the task if we found matching examples
        if filtered:
            filtered_tasks[task_name] = filtered
            total_examples += len(filtered)
            if i < 5 or len(filtered) > 0:  # Log for first few tasks or if matches found
                print(f"DEBUG: Found {len(filtered)} matching examples for task {task_name}")
                sys.stdout.flush()
    
    # Log results
    filtering_time = time.time() - start_time
    print(f"DEBUG: Sequential filtering completed in {filtering_time:.2f} seconds")
    print(f"DEBUG: Found {len(filtered_tasks)}/{task_count} tasks with {total_examples} examples, skipped {skipped_tasks} tasks")
    sys.stdout.flush()
    
    log_step(f"Filtering complete. Found {len(filtered_tasks)} tasks with {total_examples} examples matching '{filter_repo}'")
    
    # If no tasks were found, list available repositories
    if not filtered_tasks:
        print(f"DEBUG: No matching tasks found for {filter_repo}, checking available repositories")
        sys.stdout.flush()
        _log_available_repositories(task_stream)
    
    return {
        "tasks": filtered_tasks,
        "task_ordering": sorted(filtered_tasks.keys())
    }


def _filter_tasks_parallel(task_stream, task_ordering, filter_repo):
    """Filter tasks in parallel by repository."""
    import multiprocessing as mp
    from functools import partial
    
    print(f"DEBUG: Inside _filter_tasks_parallel for repo: {filter_repo} at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    
    log_step(f"Using parallel processing for filtering large dataset")
    
    def filter_task(task_data, repo):
        task_name, examples = task_data
        
        # Skip if examples is not a list
        if not isinstance(examples, list):
            return (task_name, [])
            
        filtered = []
        for ex in examples:
            # Skip if the example is not a dictionary or doesn't have a 'repo' field
            if not isinstance(ex, dict) or 'repo' not in ex:
                continue
                
            # Check if the repository matches the filter
            if ex['repo'] == repo:
                filtered.append(ex)
        
        return (task_name, filtered)
    
    # Prepare data for parallel processing
    tasks_to_process = [(name, task_stream[name]) for name in task_ordering if name in task_stream]
    
    # Determine number of processes (based on CPU cores, but limit to avoid excessive overhead)
    num_processes = min(mp.cpu_count(), 8, len(tasks_to_process))
    
    # Process in parallel
    start_time = time.time()
    with mp.Pool(num_processes) as pool:
        results = pool.map(partial(filter_task, repo=filter_repo), tasks_to_process)
    
    # Process results
    filtered_tasks = {name: examples for name, examples in results if examples}
    total_examples = sum(len(examples) for examples in filtered_tasks.values())
    
    # Log results
    log_step(f"Parallel filtering complete in {time.time() - start_time:.2f}s. Found {len(filtered_tasks)} tasks with {total_examples} examples.")
    
    # If no tasks were found, list available repositories
    if not filtered_tasks:
        _log_available_repositories(task_stream)
    
    return {
        "tasks": filtered_tasks,
        "task_ordering": sorted(filtered_tasks.keys())
    }


def _log_available_repositories(task_stream):
    """Log available repositories in the dataset."""
    log_step("No tasks found for the specified repository. Searching for available repositories...")
    all_repos = set()
    
    for examples in task_stream.values():
        if not isinstance(examples, list):
            continue
            
        for ex in examples:
            if isinstance(ex, dict) and 'repo' in ex:
                all_repos.add(ex['repo'])
    
    if all_repos:
        log_step(f"Found {len(all_repos)} unique repositories in the dataset:")
        for repo in sorted(list(all_repos)[:10]):  # Show first 10 repositories
            log_step(f"  - {repo}")
        
        if len(all_repos) > 10:
            log_step(f"  ... and {len(all_repos) - 10} more repositories")
    else:
        log_step("No valid repositories found in the dataset")

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
    import time
    
    # Check if Docker is available
    def is_docker_available():
        """Check if Docker is available on the system."""
        try:
            subprocess.run(["docker", "--version"], 
                          check=True, 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def docker_image_exists(image_name):
        """Check if a Docker image exists locally.
        
        Args:
            image_name: Name of the Docker image to check
        
        Returns:
            Boolean indicating if the image exists
        """
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    
    if not is_docker_available():
        print("ERROR: Docker is not available. Test-based evaluation cannot proceed.")
        print("Docker is required for SWE-Bench test execution. Please install Docker and try again.")
        raise RuntimeError("Docker is not available but required for evaluation.")
    
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
        container_id = None
        try:
            # Create a temporary directory for test execution
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare repo and apply patch
                repo_path = os.path.join(temp_dir, "repo")
                os.makedirs(repo_path, exist_ok=True)
                
                # Initialize repository with base commit
                base_commit = issue.get("base_commit")
                repo_name = issue.get("repo")
                instance_id = issue.get('instance_id')
                
                if not base_commit or not repo_name or not instance_id:
                    print(f"Missing required fields (base_commit, repo, or instance_id) for issue {instance_id}")
                    return False
                
                # Check if we have a cached Docker image for this repo/commit
                image_name = f"swebench-{repo_name.replace('/', '_')}-{base_commit[:8]}"
                
                # If image doesn't exist, build it
                if not docker_image_exists(image_name):
                    print(f"Docker image {image_name} not found in cache. Building...")
                    # Clone repository at specific commit
                    clone_cmd = [
                        "git", "clone", 
                        f"https://github.com/{repo_name}.git",
                        repo_path
                    ]
                    clone_result = subprocess.run(clone_cmd, 
                                                stdout=subprocess.PIPE, 
                                                stderr=subprocess.PIPE, 
                                                check=False)
                    
                    if clone_result.returncode != 0:
                        print(f"Failed to clone {repo_name}: {clone_result.stderr.decode()}")
                        return False
                    
                    # Checkout base commit
                    checkout_cmd = ["git", "checkout", base_commit]
                    checkout_result = subprocess.run(checkout_cmd, 
                                                  cwd=repo_path, 
                                                  stdout=subprocess.PIPE, 
                                                  stderr=subprocess.PIPE, 
                                                  check=False)
                    
                    if checkout_result.returncode != 0:
                        print(f"Failed to checkout {base_commit} for {repo_name}: {checkout_result.stderr.decode()}")
                        return False
                    
                    # Build Docker image with cache
                    docker_build_cmd = [
                        "docker", "build", 
                        "--no-cache=false",  # Use cache layers
                        "-t", image_name,
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
                
                # Clean up the cloned repo after building the image
                shutil.rmtree(repo_path, ignore_errors=True)
                os.makedirs(repo_path, exist_ok=True)
                
                # Re-clone for applying the patch
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
                subprocess.run(["git", "checkout", base_commit], 
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
                
                # Prepare test commands for FAIL_TO_PASS tests
                f2p_tests = issue.get("FAIL_TO_PASS_list", [])
                p2p_tests = issue.get("PASS_TO_PASS_list", [])
                
                # Run tests in the container with proper cleanup
                container_name = f"swebench-{instance_id}-{int(time.time())}"
                
                # Start the container in detached mode
                docker_run_cmd = [
                    "docker", "run",
                    "-d",  # Run in detached mode
                    "--rm",  # Automatically remove when done
                    "-v", f"{os.path.abspath(repo_path)}:/app",
                    "--name", container_name,
                    image_name,
                    "sleep", "3600"  # Keep container running for test execution
                ]
                
                try:
                    # Start container
                    run_result = subprocess.run(docker_run_cmd,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             check=False)
                    
                    if run_result.returncode != 0:
                        print(f"Failed to start container: {run_result.stderr.decode()}")
                        return False
                    
                    container_id = run_result.stdout.decode().strip()
                    
                    # Run FAIL_TO_PASS tests
                    all_passed = True
                    for test in f2p_tests:
                        test_cmd = [
                            "docker", "exec", container_id,
                            "pytest", test
                        ]
                        test_result = subprocess.run(test_cmd,
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE,
                                                  timeout=timeout,
                                                  check=False)
                        if test_result.returncode != 0:
                            all_passed = False
                            break
                    
                    # If FAIL_TO_PASS tests passed, check PASS_TO_PASS tests
                    if all_passed and p2p_tests:
                        for test in p2p_tests:
                            test_cmd = [
                                "docker", "exec", container_id,
                                "pytest", test
                            ]
                            test_result = subprocess.run(test_cmd,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE,
                                                      timeout=timeout,
                                                      check=False)
                            if test_result.returncode != 0:
                                all_passed = False
                                break
                    
                    return all_passed
                    
                except subprocess.TimeoutExpired:
                    print(f"Test execution timed out after {timeout} seconds")
                    return False
                except Exception as e:
                    print(f"Error running tests: {str(e)}")
                    return False
                finally:
                    # Ensure container is cleaned up
                    if container_id:
                        cleanup_docker_containers([container_id])
                
        except Exception as e:
            print(f"Error in test execution setup: {str(e)}")
            return False
    
    # Process each example in the dataset
    successes = []
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for i, issue in enumerate(dataset):
            print(f"Evaluating issue {i+1}/{len(dataset)}: {issue.get('instance_id', 'unknown')}")
            
            # Generate patch with model
            try:
                # Prepare input for the model using SWE Bench paper format
                prompt = (
                    "You will be provided with a partial code base and an issue statement "
                    "explaining a problem to resolve.\n\n"
                    f"<issue>\n{issue.get('problem_statement', issue.get('prompt', ''))}\n</issue>\n"
                    f"<code>\n"
                )
                
                # Add code files if available in the issue
                if 'code_files' in issue and isinstance(issue['code_files'], dict):
                    for file_path, file_content in issue['code_files'].items():
                        prompt += f"[start of {file_path}]\n{file_content}\n[end of {file_path}]\n"
                
                prompt += (
                    "</code>\n\n"
                    "Here is an example of a patch file. It consists of changes to the code "
                    "base. It specifies the file names, the line numbers of each change, "
                    "and the removed and added lines. A single patch file can contain "
                    "changes to multiple files.\n"
                    "<patch>\n"
                    "--- a/example.py\n"
                    "+++ b/example.py\n"
                    "@@ -1,2 +1,5 @@\n"
                    " def example():\n"
                    "-    return \"old code\"\n"
                    "+    # This is an example patch\n"
                    "+    return \"new code\"\n"
                    "+    # End of changes\n"
                    "</patch>\n\n"
                    "I need you to solve the provided issue by generating a single patch file "
                    "that I can apply directly to this repository using git apply. Please "
                    "respond with a single patch file in the format shown above.\n\n"
                    "Respond below:"
                )
                
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
    # Direct console output for immediate feedback
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING FINE-TUNING ON {len(task_examples)} EXAMPLES")
    print(f"Training config: batch={batch_size}, lr={learning_rate}, epochs={num_epochs}, device={device}")
    sys.stdout.flush()
    log_step(f"Starting model fine-tuning on {len(task_examples)} examples")
    log_step(f"Training parameters: batch_size={batch_size}, lr={learning_rate}, "
            f"epochs={num_epochs}, grad_accum_steps={gradient_accumulation_steps}")
    
    # Log model architecture and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_step(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable "
            f"({trainable_params/total_params*100:.2f}%)")
    
    # Log memory usage before training
    memory_usage = get_memory_usage()
    log_step(f"Memory before training - GPU: {memory_usage['gpu']['allocated']:.2f} GB "
            f"(reserved: {memory_usage['gpu']['reserved']:.2f} GB)")
    
    # Create dataset
    log_step("Preparing training dataset...")
    try:
        dataset = SWEBenchDataset(
            task_examples, 
            tokenizer, 
            use_memory=use_memory,
            memory_examples=memory_examples
        )
        log_step(f"Created dataset with {len(dataset)} examples")
    except Exception as e:
        log_step(f"ERROR: Failed to create dataset: {str(e)}")
        raise
    
    # Calculate training steps
    steps_per_epoch = len(dataset) // (batch_size * gradient_accumulation_steps)
    total_steps = min(steps_per_epoch * num_epochs, max_steps)
    warmup_steps = max(1, int(total_steps * 0.1))
    
    log_step(f"Training configuration: {total_steps} total steps, "
            f"{warmup_steps} warmup steps, "
            f"{steps_per_epoch} steps/epoch")
    
    # Configure training precision
    use_fp16 = not use_4bit and not torch.cuda.is_bf16_supported()
    use_bf16 = not use_4bit and torch.cuda.is_bf16_supported()
    
    log_step(f"Precision settings - 4-bit: {use_4bit}, FP16: {use_fp16}, BF16: {use_bf16}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=True,
        optim="adamw_torch_fused",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        logging_dir="./logs",
        log_level="info",
    )
    
    # Custom progress callback
    class TrainingProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.time()
            self.current_epoch = 0
            self.current_step = 0
            self.total_steps = total_steps
            self.steps_per_epoch = steps_per_epoch
            
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.current_epoch = state.epoch
            log_step(f"Starting epoch {int(self.current_epoch)}/{num_epochs}")
            
        def on_step_end(self, args, state, control, **kwargs):
            self.current_step = state.global_step
            if state.log_history:
                logs = state.log_history[-1]
                if 'loss' in logs:
                    step = logs.get('step', self.current_step)
                    loss = logs.get('loss', 0)
                    learning_rate = logs.get('learning_rate', 0)
                    
                    # Log progress every 10% of an epoch
                    if step % max(1, self.steps_per_epoch // 10) == 0:
                        epoch_progress = (step % self.steps_per_epoch) / self.steps_per_epoch
                        total_progress = step / self.total_steps
                        eta = (time.time() - self.start_time) * (1/total_progress - 1) if total_progress > 0 else 0
                        
                        log_step(f"Step {step}/{self.total_steps} "
                               f"(Epoch {int(self.current_epoch)}.{int(epoch_progress*100):02d}%) - "
                               f"Loss: {loss:.4f}, LR: {learning_rate:.2e}, "
                               f"ETA: {eta/60:.1f} min")
                        
                        # Log memory usage periodically
                        if step % (self.steps_per_epoch // 5) == 0:  # 5 times per epoch
                            memory_usage = get_memory_usage()
                            log_step(f"Memory usage - GPU: {memory_usage['gpu']['allocated']:.2f} GB, "
                                   f"CPU: {memory_usage['cpu_percent']:.1f}%")
    
    # Initialize trainer with optimizations
    log_step("Initializing trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
            callbacks=[TrainingProgressCallback()],
        )
        
        # Train model with timing
        log_step("Starting training...")
        start_time = time.time()
        
        try:
            trainer.train()
        except Exception as e:
            log_step(f"ERROR during training: {str(e)}")
            log_step(f"Current step: {trainer.state.global_step}/{total_steps}")
            raise
            
        training_time = (time.time() - start_time) / 60  # in minutes
        log_step(f"Training completed in {training_time:.1f} minutes")
        
        # Log final metrics
        if trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            log_step("Final training metrics: " + 
                    ", ".join(f"{k}: {v:.4f}" for k, v in final_metrics.items() 
                              if isinstance(v, (int, float))))
        
        # Report final memory usage
        memory_usage = get_memory_usage()
        log_step(f"Memory after training - GPU: {memory_usage['gpu']['allocated']:.2f} GB "
                f"(reserved: {memory_usage['gpu']['reserved']:.2f} GB), "
                f"CPU: {memory_usage['cpu_percent']:.1f}%")
        
        return model
        
    except Exception as e:
        log_step(f"ERROR: Failed to initialize or run trainer: {str(e)}")
        raise

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

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log_step(step_name):
    print(f"\n[{get_timestamp()}] {step_name}...")

def main():
    start_time = time.time()
    print(f"\n[{get_timestamp()}] Starting script with PID: {os.getpid()}")
    log_step("Initializing argument parser")
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
    log_step(f"Setting random seed to {args.seed}")
    set_seed(args.seed)
    
    # Check GPU availability
    log_step("Checking GPU availability")
    if args.device == "cuda":
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "No GPU found"
            log_step(f"Using GPU: {gpu_name}")
            log_step(f"CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
            log_step(f"PyTorch version: {torch.__version__}")
        else:
            log_step("WARNING: CUDA is not available. Falling back to CPU.")
            args.device = "cpu"
    else:
        log_step("Using CPU for computation")
    
    # Create output directory
    log_step(f"Creating output directory: {os.path.abspath(args.output_dir)}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load task stream
    log_step(f"Loading task stream from {args.task_stream_file}")
    log_step(f"Filtering repository: {args.filter_repo}")
    try:
        task_data = load_task_stream(args.task_stream_file, args.filter_repo)
        task_stream = task_data["tasks"]
        task_ordering = task_data["task_ordering"]
        log_step(f"Loaded {len(task_stream)} tasks with {len(task_ordering)} task orderings")
    except Exception as e:
        log_step(f"ERROR: Failed to load task stream: {str(e)}")
        raise
    
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
