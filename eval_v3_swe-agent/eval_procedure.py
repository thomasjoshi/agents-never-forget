# %% [markdown]
#  # SWE-Agent-CL Evaluation Framework (LangGraph + Semantic Memory)
# 
# 
# 
#  This notebook presents a comprehensive framework for evaluating large language models (LLMs) on SWE-Bench-CL, our continual learning adaptation of SWE-Bench. This framework integrates concepts from the **SWE-agent** project ([Yang et al., 2024](https://arxiv.org/abs/2405.15793)) with a novel **semantic memory** system to assess continual learning capabilities in software engineering tasks.
# 
# 
# 
#  **Key Features:**
# 
# 
# 
#  1.  **Multi-Model Evaluation:** Supports various closed and open-source models (e.g., `Claude-3.7-Sonnet`, `GPT-4o`, `gemma3`, `llama4`, `Qwen`) via a flexible `get_llm` function.
# 
#  2.  **SWE-agent Inspired ACI:** Implements an Agent-Computer Interface (ACI) based on SWE-agent principles using LangGraph. This includes:
# 
#      *   LM-friendly tools for file navigation (`open`, `scroll_up`, `scroll_down`, `goto`), searching (`find_file`, `search_file`, `search_dir`), editing (`edit` with integrated linter), and execution (`run_tests`).
# 
#      *   Concise, informative feedback mechanisms.
# 
#      *   Prompts adapted from SWE-agent's design (System, Instance, Error handling).
# 
#  3.  **Sophisticated Semantic Memory:** A novel memory system combining:
# 
#      *   **Semantic Memory (RAG):** Stores and retrieves past task experiences (problem, solution, success status, rationale) using vector embeddings (FAISS). This allows the agent to learn from past successes and failures within a sequence.
# 
#      *   **Context Management:** Integrates retrieved memories into the agent's prompt context.
# 
#  4.  **Continual Learning Metrics:** Defines and enables the calculation of metrics crucial for CL evaluation (Success Rate, Tool Use Efficiency, Forward Transfer potential).
# 
#  5.  **Experimental Design:** Facilitates experiments comparing performance with and without memory (0-shot vs. memory-augmented) across sequences.
# 
#  6.  **Self-Contained & Runnable:** Uses standard Python libraries (`os`, `subprocess`, `pathlib`) and LangChain/LangGraph for a runnable evaluation setup.
# 
# 
# 
#  This framework aims to evaluate how effectively LLM agents can leverage past experiences (semantic memory) within a structured, LM-friendly environment (ACI) to solve evolving software engineering problems, mimicking a developer's learning process on a project.

# %% [markdown]
#  ## 1. Setup and Config

# %%
# Core libraries
import os
import json
import random
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable, Annotated, Tuple
import subprocess
import logging
import operator
import re # For parsing linter output
import shutil

# Pydantic and LangChain/LangGraph
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Model providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

# Embeddings for Semantic Memory
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# Linter integration
try:
    import flake8.api.legacy as flake8_api
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False
    print("Warning: flake8 not installed. Edit tool linter functionality will be disabled. Install with: pip install flake8")


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# %% [markdown]
#  ## 2. Load and Configure Dataset

# %%
# Load the SWE-Bench-CL dataset
dataset_path = "../data/SWE-Bench-CL-Curriculum.json" # Adjust path as needed

try:
    with open(dataset_path, 'r') as f:
        swe_bench_cl = json.load(f)
    logger.info(f"Loaded SWE-Bench-CL dataset with {swe_bench_cl['metadata']['num_sequences']} sequences and {swe_bench_cl['metadata']['total_tasks']} tasks from {dataset_path}")

    # Display dataset metadata
    print("\nRepositories in dataset:")
    for repo in swe_bench_cl['metadata']['repositories']:
        print(f"- {repo}")

    # Examine the first sequence
    first_sequence = swe_bench_cl['sequences'][0]
    print(f"\nFirst sequence: {first_sequence['id']}")
    print(f"Repository: {first_sequence.get('repo', 'N/A')}") # Use .get for safety
    print(f"Number of tasks: {first_sequence['num_tasks']}")
    # Add checks for keys before accessing
    if 'statistics' in first_sequence:
        print(f"Difficulty distribution: {first_sequence['statistics'].get('difficulty_distribution', 'N/A')}")
        print(f"Tasks with dependencies: {first_sequence['statistics'].get('tasks_with_dependencies', 'N/A')} ({first_sequence['statistics'].get('dependency_rate', 'N/A')}%)")
    else:
        print("Statistics not available for the first sequence.")

except FileNotFoundError:
    logger.error(f"Dataset file not found at {dataset_path}. Please ensure the file exists or run the dummy data generation cell.")
    swe_bench_cl = None # Ensure variable is None if loading fails
except json.JSONDecodeError:
    logger.error(f"Error decoding JSON from {dataset_path}. The file might be corrupted.")
    swe_bench_cl = None
except Exception as e:
    logger.error(f"An unexpected error occurred while loading the dataset: {e}")
    swe_bench_cl = None

# %% [markdown]
#  ## 2.5 Repository Management Utilities

# %%
# Repository Configuration
REPOS_BASE_DIR = Path("./cloned_repos") # Use a distinct directory
REPOS_BASE_DIR.mkdir(exist_ok=True)
logger.info(f"Repositories will be cloned/managed in: {REPOS_BASE_DIR.resolve()}")

def setup_repository(
    repo_identifier: str, # e.g., "astropy/astropy" or "local/dummy_project"
    commit_hash: str,
    base_clones_dir: Path,
    dummy_files_setup: Optional[Callable[[Path], None]] = None # For dummy repos
) -> Path:
    """
    Ensures the specified repository is cloned and checked out to the given commit.
    Returns the local path to the repository. Handles local/dummy setups.
    Resets the repository state to avoid contamination between tasks.
    """
    if repo_identifier.startswith("local/"):
        project_name = repo_identifier.split("/", 1)[1]
        local_repo_path = base_clones_dir / project_name
        local_repo_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Setting up local/dummy repository at: {local_repo_path}")
        # Clean directory before setting up dummy files to ensure consistent state
        for item in local_repo_path.iterdir():
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item) # Requires import shutil
        if dummy_files_setup:
            dummy_files_setup(local_repo_path)
        return local_repo_path.resolve()

    # For actual git repositories
    sanitized_repo_name = repo_identifier.replace("/", "__")
    local_repo_path = base_clones_dir / sanitized_repo_name

    try:
        if not local_repo_path.exists():
            logger.info(f"Repository {repo_identifier} not found locally. Cloning to {local_repo_path}...")
            local_repo_path.parent.mkdir(parents=True, exist_ok=True)
            clone_url = f"https://github.com/{repo_identifier}.git"
            # Clone with depth 1 initially if possible, then fetch specific commit if needed? No, need full history.
            subprocess.run(["git", "clone", clone_url, str(local_repo_path)], check=True, timeout=600, capture_output=True)
            logger.info(f"Cloned {repo_identifier}.")

        # Ensure we are at the correct commit and the working directory is clean
        logger.info(f"Setting repository {repo_identifier} to commit {commit_hash} and cleaning...")
        # Fetch latest changes in case the commit is newer than the clone
        subprocess.run(["git", "fetch"], cwd=local_repo_path, check=True, timeout=120, capture_output=True)
        # Reset hard to the specific commit (discards local changes, index, and working tree changes)
        subprocess.run(["git", "reset", "--hard", commit_hash], cwd=local_repo_path, check=True, timeout=60, capture_output=True)
        # Clean untracked files and directories (-fdx option: files, directories, ignored files)
        subprocess.run(["git", "clean", "-fdx"], cwd=local_repo_path, check=True, timeout=60, capture_output=True)
        logger.info(f"Repository {repo_identifier} set to commit {commit_hash} and cleaned.")

        return local_repo_path.resolve()

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "N/A"
        stdout = e.stdout.decode() if e.stdout else "N/A"
        logger.error(f"Git command failed for {repo_identifier} at {commit_hash}. Error: {e}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        raise
    except subprocess.TimeoutExpired as e:
        logger.error(f"Git command timed out for {repo_identifier}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error setting up repository {repo_identifier}: {e}")
        raise

def get_current_commit(repo_dir: Path) -> Optional[str]:
    """Gets the current commit hash of a git repository."""
    if not (repo_dir / ".git").exists():
        # logger.warning(f"No .git directory found in {repo_dir}, cannot get commit hash.")
        return None # Expected for dummy repos
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return process.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Error getting current commit for {repo_dir}: {e}")
        return None

# %% [markdown]
#  ## 2.6 Dummy Data Setup (for testing)

# %%
# This cell creates the dummy dataset file if USE_DUMMY_DATA is True
USE_DUMMY_DATA = True
dummy_dataset_path = dataset_path.split("/")[-1].split(".")[0] + "_dummy.json"

if USE_DUMMY_DATA:
    logger.warning(f"Creating/Overwriting dummy dataset for demonstration at {dummy_dataset_path}")
    dummy_swe_bench_cl = {
        "metadata": {
            "name": "SWE-Bench-CL-Dummy",
            "description": "A dummy dataset for SWE-Bench-CL testing.",
            "version": "1.0.0",
            "num_sequences": 1,
            "total_tasks": 1,
            "repositories": ["local/dummy_math_project"],
            "generation_date": "2024-07-27T10:00:00Z"
        },
        "evaluation_metrics": swe_bench_cl['evaluation_metrics'] if swe_bench_cl else {}, # Copy metrics if real dataset loaded
        "sequences": [
            {
                "id": "dummy_math_project_sequence",
                "repo": "local/dummy_math_project", # Sequence level repo identifier
                "num_tasks": 1,
                "statistics": { # Optional: Add dummy stats if needed by evaluator
                    "difficulty_distribution": {"easy": 1},
                    "tasks_with_dependencies": 0,
                    "dependency_rate": 0.0
                },
                "tasks": [
                    {
                        "metadata": {
                            "instance_id": "local__dummy_math_project_task_1",
                            "repo": "local/dummy_math_project", # Task level repo identifier
                            "base_commit": "initial_state", # Dummy commit hash
                            "created_at": "2024-01-01T12:00:00+00:00",
                            "difficulty": "<15 min fix" # Use CL difficulty levels
                        },
                        "task": {
                            "problem_statement": "The function `add(a, b)` in `math_utils.py` currently returns `a - b`. It should return `a + b`.",
                            "hints_text": "Check the return statement in the add function. Basic arithmetic is needed."
                        },
                        "evaluation": {
                            # Ground truth patch (useful for reference, not directly used by agent)
                            "patch": "diff --git a/math_utils.py b/math_utils.py\n--- a/math_utils.py\n+++ b/math_utils.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b",
                            # Test patch to apply *before* running tests to check the fix
                            "test_patch": "diff --git a/test_math_utils.py b/test_math_utils.py\n--- a/test_math_utils.py\n+++ b/test_math_utils.py\n@@ -1,5 +1,8 @@\n import unittest\n from math_utils import add\n+\n class TestMath(unittest.TestCase):\n-    def test_initial_behavior(self):\n-        self.assertEqual(add(2, 2), 0) # Current incorrect behavior\n+    def test_addition(self):\n+        # This test should fail initially and pass after the fix\n+        self.assertEqual(add(2, 2), 4)\n     def test_existing_behavior(self):\n         self.assertTrue(True)\n \n",
                            # Tests that should FAIL before the fix and PASS after
                            "FAIL_TO_PASS": ["test_math_utils.TestMath.test_addition"],
                            # Tests that should PASS before and PASS after (regression check)
                            "PASS_TO_PASS": ["test_math_utils.TestMath.test_existing_behavior"]
                        },
                        "continual_learning": {
                            "sequence_position": 1,
                            "difficulty_score": 1,
                            "dependencies": [],
                            "modified_files": ["math_utils.py"]
                        }
                    }
                ]
            }
        ]
    }
    with open(dummy_dataset_path, 'w') as f:
        json.dump(dummy_swe_bench_cl, f, indent=2)

    # Define the setup function for the dummy repository files
    def dummy_files_setup_for_test(project_root_path: Path):
        project_root_path.mkdir(parents=True, exist_ok=True)
        # Create the initial buggy file
        with open(project_root_path / "math_utils.py", "w") as f:
            f.write("def add(a, b):\n    return a - b\n")
        # Create the initial test file (before test_patch is applied)
        with open(project_root_path / "test_math_utils.py", "w") as f:
            f.write("import unittest\nfrom math_utils import add\n\nclass TestMath(unittest.TestCase):\n    def test_initial_behavior(self):\n        # This test reflects the initial incorrect state\n        self.assertEqual(add(2, 2), 0)\n    def test_existing_behavior(self):\n        # This test should always pass\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n")
        logger.info(f"Dummy files created/reset in {project_root_path}")

    # Perform initial setup using the utility function
    dummy_repo_id = dummy_swe_bench_cl["sequences"][0]["tasks"][0]["metadata"]["repo"]
    dummy_commit = dummy_swe_bench_cl["sequences"][0]["tasks"][0]["metadata"]["base_commit"]
    try:
        setup_repository(dummy_repo_id, dummy_commit, REPOS_BASE_DIR, dummy_files_setup=dummy_files_setup_for_test)
        logger.info(f"Initial setup for dummy repository '{dummy_repo_id}' complete.")
    except Exception as e_dummy_setup:
        logger.error(f"Failed initial setup of dummy repository: {e_dummy_setup}")


# Reload swe_bench_cl if we created the dummy file
if USE_DUMMY_DATA and dummy_swe_bench_cl:
     try:
        with open(dataset_path, 'r') as f:
            swe_bench_cl = json.load(f)
        logger.info(f"Successfully loaded dummy dataset from {dataset_path}")
     except Exception as e:
         logger.error(f"Failed to reload dummy dataset: {e}")
         # Execution should probably stop if dataset isn't loaded
         raise RuntimeError("Dataset could not be loaded.")

# %% [markdown]
#  ## 3. Model Configuration

# %%
# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv('/Users/Shayan/Library/CloudStorage/GoogleDrive-sc4040@columbia.edu/My Drive/Academics/Spring 2025/COMS 4995 - Neural Nets & Deep Learning/NNDL Final Project/agents-never-forget/.env')

# Model configuration
### IMPORTANT: Naming format for models is `provider/model_name`
MODELS = [
    ## Closed-source models
    # "google/gemini-2.5-pro-preview-05-06",
    # "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.0-flash",
    # "anthropic/claude-3-7-sonnet",
    # "openai/gpt-4o",
    ## Open-source models
    # "ollama/llama3.1:8b"
    # "ollama/llama4:scout", # 17B
    # "ollama/gemma3:27b",
    # "ollama/qwen-3:14b",
    # "ollama/deepseek-r1:14b",
]
TEMPERATURE = 0.2
MAX_TOKENS = 4096 # Max tokens for model output, not context window

# Function to initialize model based on provider
def get_llm(model):
    provider = model.split("/")[0]
    model_name = model.split("/")[1]
    
    if provider == "anthropic":
        return ChatAnthropic(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=TEMPERATURE,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Initialize models that are configured and have API keys
initialized_models = {}
for name in MODELS:
    llm = get_llm(name)
    if llm: initialized_models[name] = llm
    else: logger.warning(f"Could not initialize model: {name}. It will be skipped.")

MODELS = list(initialized_models.keys()) # Update MODELS to only include successfully initialized ones
if not initialized_models: logger.error("No models were successfully initialized. Please check your API keys and model names.")
else: logger.info(f"Successfully initialized models: {list(initialized_models.keys())}")


# %%
# # Load API keys from .env file
# from dotenv import load_dotenv
# # Specify the path to your .env file explicitly if it's not in the default location
# dotenv_path = Path('/Users/Shayan/Library/CloudStorage/GoogleDrive-sc4040@columbia.edu/My Drive/Academics/Spring 2025/COMS 4995 - Neural Nets & Deep Learning/NNDL Final Project/agents-never-forget/.env') # Adjust this path
# if dotenv_path.exists():
#     load_dotenv(dotenv_path=dotenv_path)
#     logger.info(f"Loaded environment variables from {dotenv_path}")
# else:
#     logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables.")


# # Model configuration
# ### IMPORTANT: Naming format for models is `provider/model_name`
# MODELS = [
#     # Closed-source models
#     # "anthropic/claude-3-opus-20240229",
#     # "anthropic/claude-3-sonnet-20240229",
#     # "anthropic/claude-3-haiku-20240307",
#     # "openai/gpt-4o",
#     # "openai/gpt-4-turbo",
#     # "openai/gpt-3.5-turbo",
#     "google/gemini-1.5-pro-latest",
#     # "google/gemini-1.5-flash-latest",

#     # Open-source models (via Ollama) - Ensure Ollama server is running with these models pulled
#     # "ollama/llama3", # Default llama3 model
#     # "ollama/codellama",
#     # "ollama/mistral",
# ]
# TEMPERATURE = 0.1 # Lower temperature for more deterministic behavior in SWE tasks
# MAX_TOKENS = 4096 # Max tokens for model *output*, not context window

# # Function to initialize model based on provider
# def get_llm(model_id: str):
#     """Initializes LangChain ChatModel based on model_id string."""
#     if not isinstance(model_id, str) or "/" not in model_id:
#          raise ValueError(f"Invalid model_id format: {model_id}. Expected 'provider/model_name'.")

#     provider, model_name = model_id.split("/", 1)

#     try:
#         if provider == "anthropic":
#             api_key = os.getenv("ANTHROPIC_API_KEY")
#             if not api_key: raise ValueError("ANTHROPIC_API_KEY not found in environment.")
#             return ChatAnthropic(
#                 model=model_name,
#                 temperature=TEMPERATURE,
#                 max_tokens=MAX_TOKENS,
#                 api_key=api_key,
#             )
#         elif provider == "openai":
#             api_key = os.getenv("OPENAI_API_KEY")
#             if not api_key: raise ValueError("OPENAI_API_KEY not found in environment.")
#             return ChatOpenAI(
#                 model=model_name,
#                 temperature=TEMPERATURE,
#                 max_tokens=MAX_TOKENS,
#                 api_key=api_key,
#             )
#         elif provider == "google":
#             api_key = os.getenv("GOOGLE_API_KEY") # Often named GOOGLE_API_KEY or GEMINI_API_KEY
#             if not api_key: raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment.")
#             # Ensure the correct model name format for Google (e.g., 'gemini-1.5-pro-latest')
#             # The split already gives us the correct model_name part
#             return ChatGoogleGenerativeAI(
#                 model=model_name, # Use the part after 'google/'
#                 temperature=TEMPERATURE,
#                 max_tokens=MAX_TOKENS,
#                 google_api_key=api_key,
#                 # convert_system_message_to_human=True # Sometimes needed for older models/versions
#             )
#         elif provider == "ollama":
#             # Assumes Ollama server is running at default http://localhost:11434
#             # Ensure the model_name (e.g., 'llama3') is pulled in Ollama
#             return ChatOllama(
#                 model=model_name,
#                 temperature=TEMPERATURE,
#                 # max_tokens might not be directly supported or needed for Ollama Chat models in the same way
#             )
#         else:
#             raise ValueError(f"Unsupported provider: {provider}")
#     except Exception as e:
#         logger.error(f"Failed to initialize model {model_id}: {e}")
#         return None # Return None on failure

# # Initialize models that are configured and have API keys/Ollama setup
# initialized_models = {}
# for name in MODELS:
#     llm = get_llm(name)
#     if llm:
#         initialized_models[name] = llm
#         logger.info(f"Successfully initialized model: {name}")
#     else:
#         logger.warning(f"Could not initialize model: {name}. It will be skipped.")

# MODELS = list(initialized_models.keys()) # Update MODELS to only include successfully initialized ones
# if not initialized_models:
#     logger.error("No models were successfully initialized. Please check API keys, .env path, Ollama server, and model names.")
#     # Optionally raise an error if no models are essential
#     # raise RuntimeError("Model initialization failed. Cannot proceed.")
# else:
#     logger.info(f"Successfully initialized models available for evaluation: {MODELS}")

# %%
# Test an initialized model (if any exist)
if initialized_models:
    try:
        test_model_id = next(iter(initialized_models))
        test_llm = initialized_models[test_model_id]
        test_prompt = "Explain the concept of an Agent-Computer Interface (ACI) in 50 words."
        logger.info(f"Testing model: {test_model_id} with prompt: '{test_prompt}'")
        response = test_llm.invoke([HumanMessage(content=test_prompt)])
        logger.info(f"Test response from {test_model_id}: {response.content[:200]}...")
    except Exception as e:
        logger.error(f"Error testing model {test_model_id}: {e}", exc_info=True)
else:
    logger.warning("Skipping model test as no models were initialized.")

# %% [markdown]
#  ## 4. SWE-agent Tools Implementation
# 
# 
# 
#  Implement the LM-friendly tools described in the SWE-agent paper for file navigation, searching, editing (with linting), and execution. These tools operate on the `repo_path` provided in the agent's state.

# %%
# --- Tool Schemas (Pydantic Models) ---

class FindFileSchema(BaseModel):
    """Search for files by name within the repository."""
    filename: str = Field(description="The name or pattern of the file to find (e.g., 'test_*.py', 'settings.py').")
    repo_path: str = Field(description="The local filesystem path to the root of the repository.")
    # directory: Optional[str] = Field(default=".", description="Directory relative to repo_path to start the search.") # Simplified: always search from root

class SearchSchema(BaseModel):
    """Search for a string pattern within files or directories."""
    query: str = Field(description="The string or regex pattern to search for.")
    repo_path: str = Field(description="The local filesystem path to the root of the repository.")
    target: Optional[str] = Field(default=None, description="Optional: Path to a specific file or directory relative to repo_path to search within. If None, searches the current open file (if any) or the whole repo.")
    search_type: str = Field(description="Must be 'file' or 'dir'. Specifies whether to search a single file or a directory.")

class FileViewerSchema(BaseModel):
    """Interact with the file viewer: open a file, scroll, or go to a line."""
    action: str = Field(description="Action to perform: 'open', 'scroll_up', 'scroll_down', 'goto'.")
    repo_path: str = Field(description="The local filesystem path to the root of the repository.")
    path: Optional[str] = Field(default=None, description="Path to the file relative to repo_path (required for 'open').")
    line_number: Optional[int] = Field(default=None, description="Target line number (required for 'goto', optional for 'open').")

class EditSchema(BaseModel):
    """Edit the currently open file by replacing a range of lines."""
    start_line: int = Field(description="The 1-indexed starting line number of the range to replace (inclusive).")
    end_line: int = Field(description="The 1-indexed ending line number of the range to replace (inclusive). To insert before line N, use start_line=N, end_line=N-1. To delete lines N-M, provide empty replacement_text.")
    replacement_text: str = Field(description="The new text to insert. Use '\\n' for newlines. Must include correct indentation.")
    repo_path: str = Field(description="The local filesystem path to the root of the repository (used for context, edit happens on open file state).")
    # current_open_file is implicitly taken from agent state

class RunTestsSchema(BaseModel):
    """Run a shell command, typically for executing tests."""
    command: str = Field(description="The shell command to execute (e.g., 'python -m pytest', 'make test').")
    repo_path: str = Field(description="The local filesystem path to the root of the repository where the command should be run.")

# --- Pydantic Models for ReAct-style Planning, Reflection, and Solution (Inspired by v1) ---
class PlanStep(BaseModel):
    """A step in the plan to solve the problem."""
    step_id: int = Field(description="A unique identifier for this step (e.g., 1, 2, 3).")
    description: str = Field(description="Detailed description of what needs to be done in this step.")
    tools_to_consider: List[str] = Field(description="Tools likely to be useful for this step (e.g., 'find_file', 'edit', 'run_tests').")
    expected_outcome: str = Field(description="What is the expected result or state after this step is successfully completed?")

class AgentPlan(BaseModel):
    """A comprehensive plan to address the software engineering problem."""
    problem_understanding: str = Field(description="Your concise understanding of the core problem to be solved based on the issue.")
    overall_strategy: str = Field(description="A brief overview of the general approach to tackle the problem.")
    plan_steps: List[PlanStep] = Field(description="A list of detailed, sequential, and actionable steps to implement the solution.")
    verification_strategy: str = Field(description="How the final solution will be verified (e.g., which tests to run, what to check).")

class AgentReflection(BaseModel):
    """Reflection on the execution of a plan step and overall progress."""
    step_id_reflected_on: int = Field(description="The ID of the plan step being reflected upon.")
    progress_assessment: str = Field(description="Assessment of progress made on this specific step. Was the expected outcome achieved?")
    challenges_encountered: List[str] = Field(description="Any challenges, errors, or unexpected outcomes during the step's execution.")
    is_step_complete: bool = Field(description="Based on the assessment, is this plan step now considered complete?")
    plan_adjustment_suggestions: Optional[str] = Field(default=None, description="If the step is not complete or the plan needs changes, suggest adjustments or next actions for this step or the overall plan. If complete and on track, state 'None'.")
    confidence_score: float = Field(description="Confidence (0.0 to 1.0) in the current plan's trajectory towards solving the overall problem.")

class AgentSolution(BaseModel):
    """The final structured solution proposed by the agent after completing the plan."""
    solution_summary: str = Field(description="A concise summary of what was done to solve the problem.")
    code_changes_description: List[str] = Field(description="Brief descriptions of key code changes made (e.g., 'Modified add function in math_utils.py to correctly sum numbers.').")
    tests_passed_status: str = Field(description="Status of tests after applying the solution (e.g., 'All specified tests passed', 'Some tests failed', 'Tests not conclusively run'). Include details if tests failed or were inconclusive.")
    final_rationale: str = Field(description="Rationale for why this solution correctly addresses the problem statement and is robust.")
    remaining_issues_or_concerns: Optional[str] = Field(default=None, description="Any unresolved issues, potential side effects, or concerns about the solution. State 'None' if there are no concerns.")


# --- Tool Implementations ---

# Constants for tools
FILE_VIEWER_WINDOW_SIZE = 100
MAX_SEARCH_RESULTS = 50 # As per SWE-agent paper

def format_file_viewer_output(file_path_rel: str, lines: List[str], start_line_idx: int, window_size: int, total_lines: int) -> str:
    """Formats the file content for the agent's view."""
    end_line_idx = min(start_line_idx + window_size, total_lines)
    window_lines = lines[start_line_idx:end_line_idx]

    output = f"[File: {file_path_rel} ({total_lines} lines total)]\n"
    if start_line_idx > 0:
        output += f"({start_line_idx} lines above)\n"

    for i, line in enumerate(window_lines):
        output += f"{start_line_idx + 1 + i}: {line.rstrip()}\n" # Display 1-indexed line numbers

    lines_below = total_lines - end_line_idx
    if lines_below > 0:
        output += f"({lines_below} lines below)\n"
    return output

@tool("find_file", args_schema=FindFileSchema)
def find_file(filename: str, repo_path: str) -> str:
    """Finds files matching the filename pattern within the repository."""
    repo_abs_path = Path(repo_path).resolve()
    if not repo_abs_path.is_dir():
        return f"Error: Repository path {repo_path} does not exist."

    try:
        # Use git ls-files for potentially faster searching in git repos, fallback to find
        cmd_git = ["git", "ls-files", f"*{filename}*"] # Use wildcards for broader matching
        process_git = subprocess.run(cmd_git, cwd=repo_abs_path, capture_output=True, text=True, timeout=30)

        if process_git.returncode == 0 and process_git.stdout:
            results = process_git.stdout.strip().splitlines()
        else:
            # Fallback to 'find' command
            cmd_find = ["find", ".", "-name", filename, "-type", "f"]
            process_find = subprocess.run(cmd_find, cwd=repo_abs_path, capture_output=True, text=True, timeout=30)
            if process_find.returncode == 0:
                 # Strip './' prefix from find results for cleaner relative paths
                results = [p[2:] if p.startswith('./') else p for p in process_find.stdout.strip().splitlines()]
            else:
                logger.warning(f"'git ls-files' and 'find' failed for {filename} in {repo_path}. Error: {process_find.stderr or process_git.stderr}")
                results = []

        if not results:
            return f"No files found matching '{filename}' in {repo_path}."

        if len(results) > MAX_SEARCH_RESULTS:
             return f"Found {len(results)} files matching '{filename}'. Please refine your search. Showing first {MAX_SEARCH_RESULTS}:\n" + "\n".join(results[:MAX_SEARCH_RESULTS])
        else:
             return f"Found {len(results)} files matching '{filename}':\n" + "\n".join(results)

    except subprocess.TimeoutExpired:
        return f"Error: File search command timed out for '{filename}'."
    except Exception as e:
        return f"An unexpected error occurred during find_file: {str(e)}"

@tool("search", args_schema=SearchSchema)
def search(query: str, repo_path: str, search_type: str, target: Optional[str] = None) -> str:
    """Searches for a query string in a specific file or directory using ripgrep (rg) or grep."""
    repo_abs_path = Path(repo_path).resolve()
    if not repo_abs_path.is_dir():
        return f"Error: Repository path {repo_path} does not exist."

    search_path = "." # Default to searching the whole repo relative path
    if target:
        target_abs_path = (repo_abs_path / target).resolve()
        # Security check: ensure target is within repo
        if repo_abs_path not in target_abs_path.parents and target_abs_path != repo_abs_path:
             return f"Error: Target path '{target}' is outside the repository '{repo_path}'."
        if not target_abs_path.exists():
            return f"Error: Target path '{target}' does not exist."
        # Use relative path from repo root for search command
        search_path = str(target_abs_path.relative_to(repo_abs_path))

    if search_type == 'file' and target and not target_abs_path.is_file():
        return f"Error: Target '{target}' is not a file, but search_type is 'file'."
    if search_type == 'dir' and target and not target_abs_path.is_dir():
         return f"Error: Target '{target}' is not a directory, but search_type is 'dir'."
    if search_type == 'file' and not target:
        # TODO: Integrate with file viewer state - search current open file
        return "Error: Searching current open file not yet implemented via this tool. Provide a specific file path for 'target'."

    try:
        # Use ripgrep (rg) if available, fallback to grep
        cmd = ["rg", "-n", "--glob", ("*" if search_type == 'dir' else target) if target else "*", query, search_path]
        # Simpler command structure: rg -n query [path]
        cmd = ["rg", "-n", query, search_path]

        process = subprocess.run(cmd, cwd=repo_abs_path, capture_output=True, text=True, timeout=30)

        if process.returncode == 0: # Found matches
            results = process.stdout.strip().splitlines()
            output = f"Found {len(results)} matches for '{query}' in '{search_path}':\n"
            if len(results) > MAX_SEARCH_RESULTS:
                output += f"(Showing first {MAX_SEARCH_RESULTS})\n"
                results = results[:MAX_SEARCH_RESULTS]
            # Format results: file:line:match
            formatted_results = [f"- {line}" for line in results]
            return output + "\n".join(formatted_results)
        elif process.returncode == 1: # No matches found
            return f"No results found for '{query}' in '{search_path}'."
        else: # Error occurred
            logger.warning(f"ripgrep failed (code {process.returncode}): {process.stderr}. Trying grep...")
            # Fallback to grep - simpler grep command
            grep_cmd_str = f"grep -rnH -E '{query}' {search_path}"
            process_grep = subprocess.run(grep_cmd_str, shell=True, cwd=repo_abs_path, capture_output=True, text=True, timeout=30)

            if process_grep.returncode == 0:
                results = process_grep.stdout.strip().splitlines()
                output = f"Found {len(results)} matches for '{query}' in '{search_path}' (using grep):\n"
                if len(results) > MAX_SEARCH_RESULTS:
                    output += f"(Showing first {MAX_SEARCH_RESULTS})\n"
                    results = results[:MAX_SEARCH_RESULTS]
                formatted_results = [f"- {line}" for line in results]
                return output + "\n".join(formatted_results)
            elif process_grep.returncode == 1:
                return f"No results found for '{query}' in '{search_path}' (using grep)."
            else:
                return f"Error using grep (code {process_grep.returncode}): {process_grep.stderr}"

    except FileNotFoundError:
        return "Error: ripgrep (rg) command not found. Please ensure it's installed and in PATH for efficient search."
    except subprocess.TimeoutExpired:
        return f"Error: Search command timed out for '{query}'."
    except Exception as e:
        return f"An unexpected error occurred during search: {str(e)}"

# Note: The file viewer tool needs access to the agent's state (current file, lines, window position).
# LangGraph tools are typically stateless. We handle this by passing the relevant state parts
# *from* the AgentState *into* the tool call within the graph execution logic.
# The tool function itself will receive these as arguments.
# The *return* value of the tool should include the updated state parts.

@tool("file_viewer", args_schema=FileViewerSchema)
def file_viewer(action: str, repo_path: str, path: Optional[str] = None, line_number: Optional[int] = None,
                # State passed from AgentState
                current_open_file: Optional[str] = None,
                current_file_lines: Optional[List[str]] = None,
                current_window_start_line: int = 0
                ) -> Dict[str, Any]:
    """Opens, scrolls, or jumps within a file viewer. Returns new view and updated state."""
    repo_abs_path = Path(repo_path).resolve()
    output_state = { # Dictionary to return updated state fields
        "current_open_file": current_open_file,
        "current_file_lines": current_file_lines,
        "current_window_start_line": current_window_start_line,
        "viewer_output": "" # The formatted string to show the agent
    }

    if action == "open":
        if not path: return {"viewer_output": "Error: 'path' is required for 'open' action."}
        target_abs_path = (repo_abs_path / path).resolve()
        if repo_abs_path not in target_abs_path.parents and target_abs_path != repo_abs_path:
            return {"viewer_output": f"Error: Path '{path}' is outside the repository."}
        if not target_abs_path.is_file():
            return {"viewer_output": f"Error: Path '{path}' is not a file."}

        try:
            lines = target_abs_path.read_text(encoding='utf-8', errors='ignore').splitlines(True) # Keep newlines
            total_lines = len(lines)
            start_line_idx = 0
            if line_number:
                start_line_idx = max(0, min(line_number - 1, total_lines - 1)) # Go to specific line (0-indexed)
                # Center window if possible
                start_line_idx = max(0, start_line_idx - FILE_VIEWER_WINDOW_SIZE // 2)

            output_state["current_open_file"] = path # Store relative path
            output_state["current_file_lines"] = lines
            output_state["current_window_start_line"] = start_line_idx
            output_state["viewer_output"] = format_file_viewer_output(path, lines, start_line_idx, FILE_VIEWER_WINDOW_SIZE, total_lines)
            return output_state
        except Exception as e:
            return {"viewer_output": f"Error opening file '{path}': {str(e)}"}

    # Actions requiring an open file
    if not current_open_file or current_file_lines is None:
        return {"viewer_output": "Error: No file is currently open. Use 'open' first."}

    total_lines = len(current_file_lines)
    start_line_idx = current_window_start_line

    if action == "scroll_down":
        start_line_idx = min(current_window_start_line + FILE_VIEWER_WINDOW_SIZE, total_lines - 1)
    elif action == "scroll_up":
        start_line_idx = max(0, current_window_start_line - FILE_VIEWER_WINDOW_SIZE)
    elif action == "goto":
        if not line_number: return {"viewer_output": "Error: 'line_number' is required for 'goto' action."}
        target_line_idx = max(0, min(line_number - 1, total_lines - 1))
        # Center window if possible
        start_line_idx = max(0, target_line_idx - FILE_VIEWER_WINDOW_SIZE // 2)
    else:
        return {"viewer_output": f"Error: Invalid file_viewer action '{action}'."}

    output_state["current_window_start_line"] = start_line_idx
    output_state["viewer_output"] = format_file_viewer_output(current_open_file, current_file_lines, start_line_idx, FILE_VIEWER_WINDOW_SIZE, total_lines)
    return output_state


def run_linter(file_path: Path, file_content: str) -> Tuple[bool, str]:
    """Runs flake8 linter on file content and returns (has_errors, error_message)."""
    if not FLAKE8_AVAILABLE:
        logger.info("[Linter] Check skipped: flake8 not installed.")
        return False, "Linter check skipped: flake8 not installed."

    logger.info(f"[Linter] Running flake8 on temporary content for: {file_path.name}")
    temp_file_path = file_path.parent / f".~{file_path.name}.tmp"
    try:
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(file_content)

        select_codes = ["E111", "E112", "E113", "E999", "F821", "F822", "F831"]
        
        # First, try the API for a quick check (might not give all details or catch all subprocess behaviors)
        # style_guide = flake8_api.get_style_guide(select=select_codes)
        # report = style_guide.check_files([str(temp_file_path)])
        # errors_api = report.get_statistics('E') + report.get_statistics('F')
        # if errors_api:
        # logger.debug(f"[Linter] API check found codes: {errors_api}")

        # Always use subprocess for consistent and detailed error messages as per SWE-Agent style
        cmd_flake8 = ["flake8", "--select=" + ",".join(select_codes), str(temp_file_path)]
        logger.debug(f"[Linter] Executing command: {' '.join(cmd_flake8)}")
        proc = subprocess.run(cmd_flake8, capture_output=True, text=True, timeout=10)

        if proc.stdout: # Flake8 outputs errors to stdout
            error_messages = proc.stdout.strip().splitlines()
            lint_result_message = "[Linter] Errors Found:\n" + "\n".join(error_messages)
            logger.warning(f"[Linter] Found errors for {file_path.name}:\n{proc.stdout.strip()}")
            return True, lint_result_message
        elif proc.returncode != 0 and proc.stderr: # Should not happen if stdout is empty for errors
            # This case is less common as flake8 usually puts errors on stdout
            error_messages = proc.stderr.strip().splitlines()
            lint_result_message = "[Linter] Errors Found (from stderr):\n" + "\n".join(error_messages)
            logger.warning(f"[Linter] Found errors (from stderr) for {file_path.name}:\n{proc.stderr.strip()}")
            return True, lint_result_message
        else:
            logger.info(f"[Linter] No critical linting errors found for {file_path.name}.")
            return False, "[Linter] No critical linting errors found."
            
    except Exception as e:
        logger.error(f"[Linter] Error during linting process for {file_path.name}: {e}", exc_info=True)
        return False, f"[Linter] Check failed with exception: {e}"
    finally:
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                logger.warning(f"[Linter] Could not remove temporary lint file: {temp_file_path}")


@tool("edit", args_schema=EditSchema)
def edit(start_line: int, end_line: int, replacement_text: str, repo_path: str,
         # State passed from AgentState
         current_open_file: Optional[str] = None,
         current_file_lines: Optional[List[str]] = None,
         current_window_start_line: int = 0
         ) -> Dict[str, Any]:
    """Edits the currently open file, runs linter, and returns new view/state or error."""
    output_state = {
        "current_open_file": current_open_file,
        "current_file_lines": current_file_lines,
        "current_window_start_line": current_window_start_line,
        "viewer_output": ""
    }

    if not current_open_file or current_file_lines is None:
        output_state["viewer_output"] = "Error: No file is currently open. Use 'open' first."
        return output_state

    repo_abs_path = Path(repo_path).resolve()
    target_file_abs_path = (repo_abs_path / current_open_file).resolve()
    if repo_abs_path not in target_file_abs_path.parents and target_file_abs_path != repo_abs_path:
         output_state["viewer_output"] = f"Error: File path '{current_open_file}' seems outside the repository '{repo_path}'."
         return output_state
    if not target_file_abs_path.is_file(): # Should exist if open
         output_state["viewer_output"] = f"Error: Open file '{current_open_file}' not found or is not a file."
         return output_state

    lines = list(current_file_lines) # Make a mutable copy
    total_lines = len(lines)

    # Validate line numbers (1-indexed input)
    # Adjust for 0-indexed list access
    start_idx = start_line - 1
    # end_idx is the index *after* the last line to remove/replace (exclusive)
    # For insertion (end_line = start_line - 1), end_idx = start_idx
    end_idx = end_line # If replacing line N, end_line=N, end_idx=N (slices up to N)

    if start_line < 1:
        output_state["viewer_output"] = f"Error: start_line ({start_line}) must be 1 or greater."
        return output_state
    # Allow insertion at end: start_line = total_lines + 1, end_line = total_lines
    if start_line > total_lines + 1:
         output_state["viewer_output"] = f"Error: start_line ({start_line}) is out of bounds for file with {total_lines} lines."
         return output_state
    # Insertion check: end_line == start_line - 1 is valid
    if end_line < start_line - 1:
         output_state["viewer_output"] = f"Error: end_line ({end_line}) cannot be less than start_line - 1 ({start_line - 1})."
         return output_state
    if end_line > total_lines:
         output_state["viewer_output"] = f"Error: end_line ({end_line}) is out of bounds for file with {total_lines} lines."
         return output_state

    # Perform the edit in memory
    new_lines_list = replacement_text.splitlines(True) # Keep trailing newlines
    try:
        modified_lines = lines[:start_idx] + new_lines_list + lines[end_idx:]
        modified_content = "".join(modified_lines)
    except IndexError:
         output_state["viewer_output"] = f"Error: Line index calculation failed for range {start_line}-{end_line}."
         return output_state

    # Run linter on the modified content
    has_errors, lint_message = run_linter(target_file_abs_path, modified_content)

    if has_errors:
        # Format error message as per SWE-agent Appendix Figure 11
        original_snippet = "".join(lines[max(0, start_idx-2):min(total_lines, end_idx+2)]) # Context around edit
        proposed_snippet = "".join(modified_lines[max(0, start_idx-2):min(len(modified_lines), start_idx+len(new_lines_list)+2)])

        error_output = (
            f"Your proposed edit has introduced new syntax error(s). Please understand the fixes and retry your edit command.\n"
            f"ERRORS:\n{lint_message}\n\n"
            f"This is how your edit would have looked if applied (showing context):\n---\n{proposed_snippet}---\n\n"
            f"This is the original code before your edit (showing context):\n---\n{original_snippet}---\n\n"
            f"Your changes have NOT been applied. Please fix your edit command and try again.\n"
            f"DO NOT re-run the same failed edit command. Running it again will lead to the same error."
        )
        output_state["viewer_output"] = error_output
        # State remains unchanged as edit was rejected
        return output_state
    else:
        # Linter passed, apply changes to file and state
        try:
            with open(target_file_abs_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            output_state["current_file_lines"] = modified_lines
            # Keep window start line the same, let user navigate if needed
            new_total_lines = len(modified_lines)
            output_state["viewer_output"] = f"Edit successful. {lint_message}\n" + \
                                            format_file_viewer_output(current_open_file, modified_lines, current_window_start_line, FILE_VIEWER_WINDOW_SIZE, new_total_lines)
            return output_state
        except Exception as e:
            output_state["viewer_output"] = f"Edit applied in memory but failed to write file '{current_open_file}': {str(e)}"
            # Revert state if write fails? Or keep in-memory state? Let's keep state for now.
            output_state["current_file_lines"] = modified_lines
            return output_state


@tool("run_tests", args_schema=RunTestsSchema)
def run_tests(command: str, repo_path: str) -> str:
    """Runs a shell command (usually tests) in the repository path."""
    repo_abs_path = Path(repo_path).resolve()
    if not repo_abs_path.is_dir():
        return f"Error: Repository path {repo_path} does not exist."

    logger.info(f"[run_tests] Executing command in '{repo_abs_path}': {command}")
    try:
        # Use shell=True carefully, ensure command isn't directly from LLM without validation if possible
        # For SWE-Bench, the test commands are usually predefined or simple patterns.
        process = subprocess.run(command, shell=True, cwd=repo_abs_path, capture_output=True, text=True, timeout=600) # 10 min timeout for tests
        
        # Provide concise feedback - now logging full output
        output = f"Command: {command}\nExit Code: {process.returncode}\n"
        stdout_full = process.stdout.strip() 
        stderr_full = process.stderr.strip()

        if stdout_full:
            output += f"--- STDOUT ---\n{stdout_full}\n"
        if stderr_full:
             output += f"--- STDERR ---\n{stderr_full}\n"

        if not stdout_full and not stderr_full:
             output += "(No output produced)\n"

        # Simple success indication based on exit code
        if process.returncode == 0:
            output += "Command executed successfully (exit code 0)."
            logger.info(f"[run_tests] Command '{command}' executed successfully.")
        else:
            output += "Command failed or produced errors (non-zero exit code)."
            logger.warning(f"[run_tests] Command '{command}' failed. Exit Code: {process.returncode}. STDERR: {stderr_full[:500]}...") # Log snippet of stderr for quick glance

        return output
    except subprocess.TimeoutExpired:
        logger.error(f"[run_tests] Command '{command}' timed out after 10 minutes.")
        return f"Error: Command '{command}' timed out after 10 minutes."
    except Exception as e:
        logger.error(f"[run_tests] An error occurred running command '{command}': {str(e)}", exc_info=True)
        return f"An error occurred running command '{command}': {str(e)}"

# List of all SWE-agent style tools
swe_agent_tools = [find_file, search, file_viewer, edit, run_tests]

# %% [markdown]
#  ## 5. Semantic Memory System Integration
# 
# 
# 
#  Implement the semantic memory system using FAISS for RAG and integrate it into the agent workflow. This memory stores past task solutions and experiences, allowing the agent to retrieve relevant context for new tasks, especially within the same sequence (simulating learning).

# %%
class SemanticMemory:
    """Stores and retrieves task experiences using vector embeddings."""
    def __init__(self, embedding_model: Any, k_results: int = 3):
        self.embedding_model = embedding_model
        self.k_results = k_results
        self.index = None
        self.documents: List[Document] = []
        self.doc_counter = 0 # Simple ID for documents

    def add_entry(self, task_id: str, sequence_id: str, content: str, success: bool, metadata: Optional[Dict] = None):
        """Adds a task experience to the memory."""
        meta = metadata or {}
        meta["task_id"] = task_id
        meta["sequence_id"] = sequence_id
        meta["success"] = success
        meta["doc_id"] = self.doc_counter
        self.doc_counter += 1

        # Prepend status to content for better retrieval signal
        status_prefix = "[SUCCESSFUL SOLUTION]" if success else "[ATTEMPTED SOLUTION (Failed)]"
        full_content = f"{status_prefix} for Task {task_id} in Sequence {sequence_id}:\n{content}"

        doc = Document(page_content=full_content, metadata=meta)
        self.documents.append(doc)

        # Update FAISS index incrementally (or rebuild)
        if self.documents:
            try:
                if self.index:
                    # FAISS doesn't easily support incremental additions without saving/loading
                    # Rebuilding is simpler for this scale. For larger scales, consider alternatives.
                    self.index = FAISS.from_documents(self.documents, self.embedding_model)
                else:
                    self.index = FAISS.from_documents(self.documents, self.embedding_model)
            except Exception as e:
                logger.error(f"Error updating FAISS index: {e}")
                # Optionally remove the last added document if index update fails?
                # self.documents.pop()
                # self.doc_counter -=1

    def retrieve_relevant(self, query: str, sequence_id_filter: Optional[str] = None, num_results: Optional[int] = None) -> List[Dict]:
        """Retrieves relevant experiences, optionally filtering by sequence."""
        if not self.index:
            logger.warning("Semantic memory index not initialized or empty.")
            return []

        k = num_results if num_results is not None else self.k_results

        try:
            # FAISS basic search doesn't support metadata filtering directly in similarity_search.
            # We retrieve more results and filter afterwards.
            # Retrieve more candidates, e.g., k * 5 or a fixed larger number
            candidate_k = k * 5
            results_with_scores = self.index.similarity_search_with_score(query, k=max(candidate_k, len(self.documents)))

            filtered_results = []
            seen_task_ids = set() # Avoid duplicate task entries if content is similar
            for doc, score in results_with_scores:
                # Apply sequence filter if provided
                if sequence_id_filter and doc.metadata.get("sequence_id") != sequence_id_filter:
                    continue
                # Avoid duplicates
                task_id = doc.metadata.get("task_id", "unknown")
                if task_id in seen_task_ids:
                    continue

                formatted = {
                    "task_id": task_id,
                    "sequence_id": doc.metadata.get("sequence_id", "unknown"),
                    "content": doc.page_content, # Already includes status prefix
                    "success": doc.metadata.get("success", False),
                    "score": float(score) # Lower score is better in FAISS L2 distance
                }
                filtered_results.append(formatted)
                seen_task_ids.add(task_id)

                if len(filtered_results) >= k:
                    break # Stop once we have enough filtered results

            # Sort by score (ascending for L2 distance)
            filtered_results.sort(key=lambda x: x["score"])
            return filtered_results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def clear(self):
        """Clears the memory."""
        self.index = None
        self.documents = []
        self.doc_counter = 0
        logger.info("Semantic memory cleared.")

class MemorySystem:
    """Manages semantic memory and context building for the agent."""
    def __init__(self, semantic_memory: SemanticMemory, max_context_tokens: int = 8000):
        self.semantic_memory = semantic_memory
        self.max_context_tokens = max_context_tokens # Rough token limit for context

    def add_experience_to_memory(self, task_id: str, sequence_id: str, solution_data: Dict):
        """Adds a completed task experience to semantic memory."""
        summary = solution_data.get("solution_summary", "N/A")
        rationale = solution_data.get("final_rationale", "N/A")
        # Include tool usage info?
        tool_calls = solution_data.get("tool_calls_count", 0)
        success = solution_data.get("tests_passed", False) # Expecting a boolean

        content_to_store = (
            f"Problem Summary (Task {task_id}): {solution_data.get('problem_statement', 'N/A')[:200]}...\n"
            f"Solution Summary: {summary}\n"
            f"Rationale: {rationale}\n"
            f"Tool Calls: {tool_calls}\n"
            f"Outcome: {'Success' if success else 'Failure'}"
        )

        self.semantic_memory.add_entry(task_id, sequence_id, content_to_store, success=success)
        logger.info(f"Added {'successful' if success else 'failed'} experience for task {task_id} (Seq: {sequence_id}) to semantic memory.")

    def get_relevant_context_for_prompt(self, current_task_prompt: str, current_sequence_id: str, num_memories: int = 3) -> str:
        """Builds a context string including relevant memories."""
        # Retrieve memories, prioritizing those from the same sequence
        retrieved_memories = self.semantic_memory.retrieve_relevant(
            current_task_prompt,
            sequence_id_filter=current_sequence_id,
            num_results=num_memories
        )

        # If not enough memories from the same sequence, retrieve globally (optional)
        # if len(retrieved_memories) < num_memories:
        #     global_memories = self.semantic_memory.retrieve_relevant(
        #         current_task_prompt,
        #         num_results=num_memories - len(retrieved_memories)
        #     )
        #     # Add global memories if they are not already included
        #     existing_ids = {mem['task_id'] for mem in retrieved_memories}
        #     for mem in global_memories:
        #         if mem['task_id'] not in existing_ids:
        #             retrieved_memories.append(mem)
        #     retrieved_memories.sort(key=lambda x: x["score"]) # Re-sort after adding global

        context_str = ""
        if retrieved_memories:
            context_str += "\n\n--- Relevant Past Experiences (from Semantic Memory) ---\n"
            # Simple heuristic for token counting (split by space) - use a proper tokenizer in production
            current_token_count = 0
            base_prompt_token_count = len(current_task_prompt.split())

            for mem in retrieved_memories:
                # Content already includes status prefix
                mem_text = f"Experience (Score: {mem['score']:.2f}):\n{mem['content']}\n---\n"
                mem_token_count = len(mem_text.split())

                # Check token limit (approximate)
                # Need to account for base prompt tokens as well
                if base_prompt_token_count + current_token_count + mem_token_count <= self.max_context_tokens:
                    context_str += mem_text
                    current_token_count += mem_token_count
                else:
                    logger.info(f"Memory context limit reached ({self.max_context_tokens} tokens), truncating memories.")
                    break
            context_str += "--- End of Past Experiences ---\n"
        return context_str

    def clear_memory(self):
        """Clears the underlying semantic memory."""
        self.semantic_memory.clear()

# --- Initialize Memory System ---
# Choose embedding model
# embedding_model_name = "openai/text-embedding-3-small"
embedding_model_name = "ollama/nomic-embed-text" # Example for local embeddings via Ollama

active_embedding_model = None
memory_system = None

try:
    if embedding_model_name.startswith("openai/"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: logger.warning("OpenAI API key not found for embeddings.")
        else: active_embedding_model = OpenAIEmbeddings(model=embedding_model_name.split("/")[1], openai_api_key=api_key)
    elif embedding_model_name.startswith("ollama/"):
        # Ensure Ollama server is running and has the embedding model
        try:
            active_embedding_model = OllamaEmbeddings(model=embedding_model_name.split("/")[1])
            # Test connection
            active_embedding_model.embed_query("test")
            logger.info(f"Ollama embedding model '{embedding_model_name.split('/')[1]}' seems available.")
        except Exception as ollama_err:
             logger.error(f"Failed to connect or use Ollama embedding model '{embedding_model_name.split('/')[1]}'. Is Ollama running and the model pulled? Error: {ollama_err}")
             active_embedding_model = None
    else:
        logger.error(f"Unsupported embedding model provider for: {embedding_model_name}")

    if active_embedding_model:
        semantic_memory_instance = SemanticMemory(embedding_model=active_embedding_model)
        memory_system = MemorySystem(semantic_memory_instance)
        logger.info(f"Memory system initialized with {embedding_model_name}.")
    else:
        logger.error("Failed to initialize embedding model. Semantic memory system will be disabled.")

except Exception as e:
    logger.error(f"Error initializing memory system: {e}. Memory system will be disabled.")
    memory_system = None

# %% [markdown]
#  ## 6. Agentic Workflow Implementation (LangGraph + SWE-agent Principles)
# 
# 
# 
#  Define the agent state, adapt prompts from SWE-agent, and build the LangGraph workflow incorporating the new tools and semantic memory.

# %%
# --- Agent State ---
class AgentState(BaseModel):
    """State for the SWE-Agent-CL."""
    # Task specific
    problem_statement: str = Field(description="The problem description from the dataset")
    hints: Optional[str] = Field(default=None, description="Hints provided for the task")
    repo_path: str = Field(description="Absolute path to the locally cloned repository")
    task_id: str = Field(description="Unique identifier for the current task")
    sequence_id: str = Field(description="Identifier for the sequence the task belongs to")
    task_data: Dict = Field(description="Full data dictionary for the current task") # Includes evaluation details

    # Model and Memory
    model_id: str = Field(description="Identifier of the LLM being used")
    memory_enabled: bool = Field(default=True, description="Whether semantic memory is enabled")

    # Agent's internal state
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list, description="History of messages exchanged with the LLM for the current step execution")
    execution_log: Annotated[List[str], operator.add] = Field(default_factory=list, description="Human-readable log of agent actions and outcomes")
    final_solution: Optional[Dict] = Field(default=None, description="The final structured solution proposed by the agent, as a dictionary.")

    # File Viewer State
    current_open_file: Optional[str] = Field(default=None, description="Relative path of the file currently open in the viewer")
    current_file_lines: Optional[List[str]] = Field(default=None, description="Content of the open file, split into lines")
    current_window_start_line: int = Field(default=0, description="0-indexed start line of the current view window")

    # Metrics
    tool_calls_count: int = Field(default=0, description="Total number of tool calls made")
    successful_tool_calls: int = Field(default=0, description="Number of tool calls that did not return an explicit error")
    errors_encountered: Annotated[List[str], operator.add] = Field(default_factory=list, description="Log of errors encountered during execution")
    turn_count: int = Field(default=0, description="Number of agent turns (LLM interactions for execution). Planning/Reflection are separate.")
    max_turns_per_step: int = Field(default=10, description="Maximum turns for the executor on a single plan step before mandatory reflection.")
    turns_on_current_step: int = Field(default=0, description="Number of turns spent by the executor on the current plan step.")

    # ReAct specific state
    plan: Optional[AgentPlan] = Field(default=None, description="The overall plan generated by the planner.")
    current_plan_step_index: int = Field(default=0, description="The 0-indexed current step in the plan being executed.")
    reflections: Annotated[List[AgentReflection], operator.add] = Field(default_factory=list, description="History of reflections made by the agent.")


# --- Prompts (Adapted from SWE-agent Appendix C) ---

# Utility to format tool documentation for the prompt
def get_tool_documentation(tools: List[Callable]) -> str:
    """Generates markdown documentation for available tools."""
    doc_str = ""
    for tool_func in tools:
        schema = tool_func.args_schema
        # Extract signature, docstring, args from schema
        signature = f"{tool_func.name} <args>" # Simplified signature
        docstring = schema.model_json_schema().get('description', 'No description available.')

        args_desc = []
        if schema.model_json_schema().get('properties'):
            for name, prop in schema.model_json_schema()['properties'].items():
                 # Skip repo_path and state fields in user-facing docs
                if name in ['repo_path', 'current_open_file', 'current_file_lines', 'current_window_start_line']:
                    continue
                req = "required" if name in schema.model_json_schema().get('required', []) else "optional"
                prop_type = prop.get('type', 'any')
                prop_desc = prop.get('description', '')
                args_desc.append(f"  - {name} ({prop_type}, {req}): {prop_desc}")

        doc_str += f"COMMAND: `{signature}`\n"
        doc_str += f"DESCRIPTION: {docstring}\n"
        if args_desc:
            doc_str += f"ARGUMENTS:\n" + "\n".join(args_desc) + "\n"
        doc_str += "---\n"
    return doc_str

TOOL_DOCUMENTATION = get_tool_documentation(swe_agent_tools)

# System Prompt for the PLANNER Agent
PLANNER_SYSTEM_PROMPT_TEMPLATE = """You are an expert software engineering AI. Your task is to create a detailed, step-by-step plan to solve the given GitHub issue.
The repository is cloned locally at {repo_path}. You have access to the problem statement and hints.

Available tools for plan execution (for your awareness, you don't call them now):
{tool_documentation}

Based on the problem statement and any hints, devise a comprehensive plan.
Your plan should clearly outline:
1.  Your understanding of the problem.
2.  The overall strategy to tackle it.
3.  A sequence of actionable steps. Each step should have a unique ID, a clear description, tools that might be useful, and the expected outcome of that step.
4.  How the final solution will be verified (e.g., specific tests to run from `FAIL_TO_PASS` and `PASS_TO_PASS` lists if provided in task data, or general test suite execution).

Output your plan ONLY as a JSON object that conforms to the AgentPlan Pydantic schema.
Ensure step_id starts from 1.
Example AgentPlan schema:
```json
{{
  "problem_understanding": "string",
  "overall_strategy": "string",
  "plan_steps": [
    {{
      "step_id": 1,
      "description": "string",
      "tools_to_consider": ["string"],
      "expected_outcome": "string"
    }}
  ],
  "verification_strategy": "string"
}}
```
Do not add any commentary before or after the JSON object.
"""

# System Prompt for the EXECUTOR Agent (Modified from original SYSTEM_PROMPT_TEMPLATE)
EXECUTOR_SYSTEM_PROMPT_TEMPLATE = """SETTING: You are SWE-Agent-CL, an autonomous software engineering agent. You are working directly in a bash command line interface with a special file viewing/editing interface to solve GitHub issues.
The repository is cloned locally, and you need to navigate, search, view, edit files, and run tests to implement the required changes.

Your current goal is to execute a specific step from a pre-defined plan.
Overall Problem: {problem_statement}
Current Plan Step (ID: {current_step_id}):
Description: {current_step_description}
Expected Outcome: {current_step_expected_outcome}

SPECIAL INTERFACE COMMANDS:
{tool_documentation}

STANDARD BASH COMMANDS: You also have access to standard bash commands like `ls`, `cd`, `cat`, `mkdir`, `echo`, etc. However, the environment does NOT support interactive session commands (e.g., `python` REPL, `vim`). Write scripts and execute them instead.

IMPORTANT TIPS (Review Carefully!):
1.  **Focus on the Current Step:** Your primary objective is to complete the current plan step.
2.  **Check Environment:** Pay attention to the current working directory and the currently open file shown in the prompt.
3.  **Editing:** Use `edit`. If the linter fails, the edit is *not applied*. Fix your `edit` command.
4.  **Error Recovery:** If a command fails, try a different command or modify arguments.
5.  **Single Command:** ONE command per turn.

RESPONSE FORMAT:
Your shell prompt is formatted as: (Open file: <path_or_None>) (Current directory: <path>) $
You MUST provide your response as a JSON object conforming to the AgentLLMResponse Pydantic schema:
```json
{{
    "discussion": "string (Your reasoning for the command, how it relates to the current plan step, and analysis of previous outputs.)",
    "command": "string (The command to execute. See details below.)"
}}
```
Details for the "command" field:
*   **Special Commands for Executor**:
    *   `SUBMIT_FOR_REFLECTION`: Use this if you believe the current step is complete OR if you are stuck, have tried multiple approaches for the current step without success, or need to reassess the plan for this step.
    *   `SUBMIT_TASK_COMPLETE`: Use this ONLY if you believe the ENTIRE task (all plan steps) is complete and successful according to the overall plan's verification strategy. This will trigger final solution generation.
*   For **standard bash commands** (like `ls`, `cd`), provide the command string directly. E.g., `"python -m unittest discover test_math_utils.py"`
*   For **special interface commands** (`find_file`, `search`, `file_viewer`, `edit`, `run_tests`), provide a JSON *string* with "tool_name" and "args". E.g., `"{{ \\"tool_name\\": \\"file_viewer\\", \\"args\\": {{ \\"action\\": \\"open\\", \\"path\\": \\"math_utils.py\\" }} }}"`
    (Inner JSON for command must be a string itself). Do NOT include `repo_path` or state arguments like `current_open_file` in `args`; these are handled internally.

Your entire output MUST be a single JSON object.
"""

# System Prompt for the REFLECTOR Agent
REFLECTOR_SYSTEM_PROMPT_TEMPLATE = """You are a critical and thoughtful software engineering AI. Your task is to reflect on the execution of a plan step.
Overall Problem: {problem_statement}
Overall Plan: {plan_json}
Current Step Attempted (ID: {current_step_id}): {current_step_description}
Expected Outcome for this step: {current_step_expected_outcome}
Execution History for this step (messages and tool outputs):
{step_execution_history}

Review the step's goal, expected outcome, and the execution history.
1.  Assess if the expected outcome for this step was achieved.
2.  Identify any challenges or errors.
3.  Determine if this specific step (ID: {current_step_id}) is now complete.
4.  If the step is not complete, or if the overall plan might need adjustment based on this step's outcome, provide suggestions.
5.  Provide a confidence score for the overall plan's success.

Output your reflection ONLY as a JSON object that conforms to the AgentReflection Pydantic schema.
Example AgentReflection schema:
```json
{{
  "step_id_reflected_on": {current_step_id},
  "progress_assessment": "string",
  "challenges_encountered": ["string"],
  "is_step_complete": boolean,
  "plan_adjustment_suggestions": "string or null",
  "confidence_score": number (0.0 to 1.0)
}}
```
Do not add any commentary before or after the JSON object.
"""

# System Prompt for the SOLVER Agent
SOLVER_SYSTEM_PROMPT_TEMPLATE = """You are a meticulous software engineering AI. The agent has completed its plan (or decided to submit its work).
Your task is to synthesize all information and generate a final, structured solution summary.
Overall Problem: {problem_statement}
Overall Plan: {plan_json}
Full Execution Log:
{full_execution_log}
Reflections Log:
{reflections_log}

Based on all the above, provide a comprehensive summary of the solution.
1.  Summarize the solution implemented.
2.  Describe the key code changes made.
3.  Report the status of tests (especially considering `FAIL_TO_PASS` and `PASS_TO_PASS` criteria if mentioned in the original task data or plan's verification strategy).
4.  Provide a clear rationale for why this solution is correct.
5.  Mention any remaining issues or concerns.

Output your response ONLY as a JSON object that conforms to the AgentSolution Pydantic schema.
Example AgentSolution schema:
```json
{{
  "solution_summary": "string",
  "code_changes_description": ["string"],
  "tests_passed_status": "string",
  "final_rationale": "string",
  "remaining_issues_or_concerns": "string or null"
}}
```
Do not add any commentary before or after the JSON object.
"""

# Instance Prompt Content (to be formatted in the planning node)
INSTANCE_PROMPT_TEMPLATE = """We're currently solving the following issue within the '{repo_name}' repository.
Repository is locally cloned at: {repo_path}

ISSUE DETAILS:
Problem Statement:
{problem_statement}

Hints:
{hints}

Your task is to understand the issue, identify the necessary code changes, implement them using the available tools, and verify the fix (e.g., by running provided tests or the reproduction steps). Remember the IMPORTANT TIPS. Start by planning your first step.
"""

# --- Pydantic Model for LLM Response ---
class AgentLLMResponse(BaseModel):
    """Defines the expected structured output from the LLM."""
    discussion: str = Field(description="The agent's reasoning, analysis, and plan for the next step.")
    command: str = Field(description="The command to execute. This should be a string, which can be either a direct bash command or a JSON string representing a tool call (tool_name and args).")


# --- Node Functions ---

def plan_generation_node(state: AgentState) -> Dict[str, Any]:
    """Generates the initial plan for solving the task."""
    logger.info(f"--- Plan Generation Node for Task {state.task_id} (Seq: {state.sequence_id}) ---")
    llm = initialized_models[state.model_id]

    planner_prompt = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        repo_path=state.repo_path,
        tool_documentation=TOOL_DOCUMENTATION
    )

    # Construct the human message for the planner
    # Include problem statement, hints, and memory context
    human_message_content = f"Problem Statement:\n{state.problem_statement}\n\n"
    if state.hints:
        human_message_content += f"Hints:\n{state.hints}\n\n"

    # Add memory context if enabled
    memory_context_str = ""
    if memory_system and state.memory_enabled:
        query_text = f"Problem: {state.problem_statement}\nHints: {state.hints or ''}"
        memory_context_str = memory_system.get_relevant_context_for_prompt(query_text, state.sequence_id)
        if memory_context_str:
            human_message_content += f"Relevant Past Experiences (from Semantic Memory):\n{memory_context_str}\n\n"

    human_message_content += "Please generate a plan based on the above information and the system prompt guidelines."

    messages_for_planner = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content=human_message_content)
    ]

    updated_execution_log = list(state.execution_log) # Make a copy
    updated_execution_log.append(f"Starting task {state.task_id}. Repo: {state.repo_path}. Memory: {'Enabled' if state.memory_enabled else 'Disabled'}.")
    if memory_context_str:
        updated_execution_log.append("Retrieved relevant context from memory for planning.")

    try:
        structured_llm_planner = llm.with_structured_output(AgentPlan)
        logger.debug("Invoking Planner LLM with structured output for AgentPlan.")
        generated_plan = structured_llm_planner.invoke(messages_for_planner)
        
        logger.info(f"Plan generated successfully for task {state.task_id}.")
        logger.debug(f"Generated Plan: {generated_plan.model_dump_json(indent=2)}")
        updated_execution_log.append(f"Planner generated a plan with {len(generated_plan.plan_steps)} steps. Full plan logged.")
        # Log the entire plan for the user as requested
        logger.info(f"Full Generated Plan for Task {state.task_id}:\n{generated_plan.model_dump_json(indent=2)}")


        # Executor will start with a clean slate of messages for its first step.
        # The plan itself is in state.plan.
        # The initial human message for the executor will be constructed by the executor node.
        return {
            "plan": generated_plan,
            "current_plan_step_index": 0,
            "messages": [], # Executor starts with fresh messages for the first step
            "execution_log": updated_execution_log,
            "turns_on_current_step": 0
        }
    except Exception as e:
        logger.error(f"Error in plan_generation_node: {e}", exc_info=True)
        updated_execution_log.append(f"CRITICAL: Planning failed: {e}")
        # If planning fails, we can't proceed. Set final_solution to indicate failure.
        return {
            "plan": None,
            "messages": [],
            "execution_log": updated_execution_log,
            "final_solution": {
                "solution_summary": "Planning failed.",
                "tests_passed_status": "N/A",
                "final_rationale": f"Could not generate a plan due to error: {e}",
            "tool_calls_count": state.tool_calls_count,
            "problem_statement": state.problem_statement
            },
            "errors_encountered": [f"Planning failed: {e}"]
        }


def executor_node(state: AgentState) -> Dict[str, Any]:
    """Executes the current step of the plan or decides to reflect/submit."""
    logger.info(f"--- Executor Node (Turn {state.turn_count +1}, Step Index {state.current_plan_step_index}) ---")

    if not state.plan or not state.plan.plan_steps:
        logger.warning("Executor: No plan found. Cannot execute. Triggering final solution for error.")
        return {
            "messages": [],
            "final_solution": {"solution_summary": "Execution error: No plan provided.", "tests_passed_status": "N/A", "final_rationale": "No plan to execute."},
            "errors_encountered": ["Executor: No plan found."],
            "turn_count": state.turn_count + 1
        }

    if state.current_plan_step_index >= len(state.plan.plan_steps):
        logger.info("Executor: All plan steps seem to be completed by index. Triggering solver.")
        # This state should ideally be caught by reflection_node first.
        # If executor reaches here, it means reflection node decided to proceed to a non-existent step.
        # Or, this is the first entry after plan generation and plan was empty.
        return {"messages": [], "command_for_router": "TRIGGER_SOLVER"}


    current_step = state.plan.plan_steps[state.current_plan_step_index]
    current_turn_total = state.turn_count + 1
    turns_on_this_step = state.turns_on_current_step + 1

    logger.debug(f"State at START of executor_node (Overall Turn {current_turn_total}, Step Turn {turns_on_this_step}): "
                 f"current_open_file='{state.current_open_file}', "
                 f"current_plan_step_id='{current_step.step_id}'")

    if turns_on_this_step > state.max_turns_per_step:
        logger.warning(f"Max turns ({state.max_turns_per_step}) reached for step {current_step.step_id}. Forcing reflection.")
        # The command_for_router will be used by the conditional edge
        return {"messages": [], "command_for_router": "SUBMIT_FOR_REFLECTION", "turn_count": current_turn_total, "turns_on_current_step": turns_on_this_step, "errors_encountered": [f"Max turns for step {current_step.step_id} reached."]}

    llm = initialized_models[state.model_id]
    
    executor_system_prompt = EXECUTOR_SYSTEM_PROMPT_TEMPLATE.format(
        problem_statement=state.problem_statement,
        current_step_id=current_step.step_id,
        current_step_description=current_step.description,
        current_step_expected_outcome=current_step.expected_outcome,
        tool_documentation=TOOL_DOCUMENTATION
    )

    # Messages for the LLM for this specific execution turn
    # Start with system prompt, then history of messages for THIS STEP'S execution
    messages_for_llm = [SystemMessage(content=executor_system_prompt)]
    messages_for_llm.extend(state.messages) # state.messages should contain the history for the current step attempt

    # Add current environment state as a HumanMessage
    cwd_rel = Path(state.repo_path).name # Simplified CWD for prompt
    open_file_rel = state.current_open_file if state.current_open_file else "None"
    env_prompt_addition = f"\n(Open file: {open_file_rel}) (Current directory: {cwd_rel}) $"

    if messages_for_llm and isinstance(messages_for_llm[-1], HumanMessage):
        messages_for_llm[-1] = HumanMessage(content=messages_for_llm[-1].content + env_prompt_addition)
    elif messages_for_llm and (isinstance(messages_for_llm[-1], AIMessage) or isinstance(messages_for_llm[-1], ToolMessage) or isinstance(messages_for_llm[-1], SystemMessage)):
        messages_for_llm.append(HumanMessage(content=env_prompt_addition))
    else: # Should not happen if messages_for_llm always starts with SystemMessage
        messages_for_llm.append(HumanMessage(content=env_prompt_addition))

    updated_execution_log = []
    updated_errors_encountered = []
    
    try:
        structured_llm_executor = llm.with_structured_output(AgentLLMResponse)
        logger.debug(f"Invoking Executor LLM for step {current_step.step_id}.")
        logger.debug(f"Messages for executor: {messages_for_llm}")
        response_model = structured_llm_executor.invoke(messages_for_llm)

        discussion = response_model.discussion
        command_str = response_model.command.strip()

        logger.debug(f"Executor LLM Response: discussion='{discussion[:100]}...', command='{command_str}'")
        updated_execution_log.append(f"Executor (Step {current_step.step_id}, Turn {turns_on_this_step}): Discussion: {discussion}")
        updated_execution_log.append(f"Executor (Step {current_step.step_id}, Turn {turns_on_this_step}): Command: {command_str}")

        # This will be used by the conditional edge function
        output_payload = {"command_for_router": command_str} 

        ai_message_content = response_model.model_dump() # This is what gets stored in AIMessage

        if command_str == "SUBMIT_FOR_REFLECTION" or command_str == "SUBMIT_TASK_COMPLETE":
            # No tool call, just passing the command to the router
            # Reset step-specific messages as we are transitioning.
            return {
                "messages": [AIMessage(content=json.dumps(ai_message_content))], # Log the decision
                **output_payload, 
                "tool_calls_count": state.tool_calls_count, # No new tool call here
                "turn_count": current_turn_total,
                "turns_on_current_step": turns_on_this_step, # Accumulate turns on this step
                "execution_log": updated_execution_log,
                "errors_encountered": updated_errors_encountered
            }
        
        # Attempt to parse as special interface command
        parsed_tool_call_from_json = None
        try:
            potential_json_command = json.loads(command_str)
            if isinstance(potential_json_command, dict) and "tool_name" in potential_json_command and "args" in potential_json_command:
                parsed_tool_call_from_json = potential_json_command
        except json.JSONDecodeError:
            pass # Will be treated as bash

        current_tool_calls_total = state.tool_calls_count + 1 # Increment for this command if it's a tool/bash

        if parsed_tool_call_from_json:
            tool_name = parsed_tool_call_from_json["tool_name"]
            tool_args = parsed_tool_call_from_json["args"]
            tool_args["repo_path"] = state.repo_path # Ensure repo_path

            known_tool_names = [t.name for t in swe_agent_tools]
            if tool_name in known_tool_names:
                if tool_name in ["file_viewer", "edit"]: # Inject stateful args for these tools
                    tool_args["current_open_file"] = state.current_open_file
                    tool_args["current_file_lines"] = state.current_file_lines
                    tool_args["current_window_start_line"] = state.current_window_start_line

                ai_message_with_tool_call = AIMessage(
                    content=json.dumps(ai_message_content),
                    tool_calls=[{
                        "id": f"tool_{tool_name}_{current_tool_calls_total}",
                        "name": tool_name,
                        "args": tool_args
                    }]
                )
                # Append AIMessage with tool call to current step's messages
                # The 'messages' field in output will be picked by operator.add
                return {
                    "messages": [ai_message_with_tool_call], 
                    **output_payload, 
                    "tool_calls_count": current_tool_calls_total,
                    "turn_count": current_turn_total,
                    "turns_on_current_step": turns_on_this_step,
                    "execution_log": updated_execution_log,
                    "errors_encountered": updated_errors_encountered
                 }
            else: # Unknown tool
                logger.warning(f"Executor specified unknown tool '{tool_name}'.")
                updated_errors_encountered.append(f"Unknown tool specified: {tool_name}")
                # Feedback to LLM about the error, then force reflection
                error_feedback_msg = f"Error: Tool '{tool_name}' is not a known special interface command. Please choose from the documented tools or use a standard bash command, or SUBMIT_FOR_REFLECTION."
                # The AIMessage is the LLM's original attempt. The ToolMessage (simulated) contains the error.
                # Next turn, LLM sees this error. For now, let's force reflection.
                return {
                    "messages": [AIMessage(content=json.dumps(ai_message_content)), ToolMessage(content=error_feedback_msg, tool_call_id="error_unknown_tool")],
                    "command_for_router": "SUBMIT_FOR_REFLECTION", # Force reflection
                    "tool_calls_count": current_tool_calls_total, # Count the attempt
                    "turn_count": current_turn_total,
                    "turns_on_current_step": turns_on_this_step,
                    "execution_log": updated_execution_log,
                    "errors_encountered": updated_errors_encountered
                }
        else: # Treat as bash command
            logger.info(f"Executor treating command as bash: {command_str}")
            bash_tool_args = {"command": command_str, "repo_path": state.repo_path}
            ai_message_with_bash_call = AIMessage(
                content=json.dumps(ai_message_content),
                tool_calls=[{
                    "id": f"tool_bash_{current_tool_calls_total}",
                    "name": "run_tests", # Bash commands are routed through run_tests
                    "args": bash_tool_args
                }]
            )
            return {
                "messages": [ai_message_with_bash_call], 
                **output_payload, 
                "tool_calls_count": current_tool_calls_total,
                "turn_count": current_turn_total,
                "turns_on_current_step": turns_on_this_step,
                "execution_log": updated_execution_log,
                "errors_encountered": updated_errors_encountered
            }

    except Exception as e:
        logger.error(f"Error in executor_node LLM call or parsing: {e}", exc_info=True)
        error_msg = f"Executor Error: {e}"
        updated_errors_encountered.append(error_msg)
        # Force reflection on error
        return {
            "messages": [AIMessage(content=f"Error processing command: {e}"), HumanMessage(content="An unexpected error occurred. Please reflect on this situation.")], # Provide error message to history
            "command_for_router": "SUBMIT_FOR_REFLECTION", 
            "turn_count": current_turn_total, 
            "turns_on_current_step": turns_on_this_step,
            "errors_encountered": updated_errors_encountered,
            "execution_log": updated_execution_log + [error_msg]
        }


def reflection_node(state: AgentState) -> Dict[str, Any]:
    """Reflects on the current step's execution and decides next action."""
    logger.info(f"--- Reflection Node for Step Index {state.current_plan_step_index} ---")
    llm = initialized_models[state.model_id]

    if not state.plan or state.current_plan_step_index >= len(state.plan.plan_steps):
        logger.warning("Reflection: Plan is missing or step index is out of bounds. Triggering solver.")
        return {"next_node_router": "solver", "messages": [], "reflections": state.reflections, "errors_encountered": ["Reflection: Invalid plan/step state."]}

    current_step = state.plan.plan_steps[state.current_plan_step_index]

    # Consolidate execution history for the current step
    # state.messages contains the AIMessage from executor and subsequent ToolMessage
    step_execution_history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in state.messages])

    reflector_prompt = REFLECTOR_SYSTEM_PROMPT_TEMPLATE.format(
        problem_statement=state.problem_statement,
        plan_json=state.plan.model_dump_json(indent=2),
        current_step_id=current_step.step_id,
        current_step_description=current_step.description,
        current_step_expected_outcome=current_step.expected_outcome,
        step_execution_history=step_execution_history_str
    )
    
    updated_execution_log = []
    updated_reflections = list(state.reflections)

    try:
        structured_llm_reflector = llm.with_structured_output(AgentReflection)
        logger.debug(f"Invoking Reflector LLM for step {current_step.step_id}.")
        reflection_output = structured_llm_reflector.invoke([
            SystemMessage(content=reflector_prompt),
            HumanMessage(content=f"Please provide your reflection on step {current_step.step_id} based on the execution history and guidelines.")
        ])
        
        updated_reflections.append(reflection_output)
        logger.info(f"Reflection for step {current_step.step_id}: Complete={reflection_output.is_step_complete}, Confidence={reflection_output.confidence_score}")
        logger.debug(f"Full Reflection: {reflection_output.model_dump_json(indent=2)}")
        updated_execution_log.append(f"Reflector (Step {current_step.step_id}): Step complete: {reflection_output.is_step_complete}. Suggestion: {reflection_output.plan_adjustment_suggestions}")

        new_plan_step_index = state.current_plan_step_index
        if reflection_output.is_step_complete:
            new_plan_step_index += 1
            logger.info(f"Step {current_step.step_id} marked complete by reflector. Moving to step index {new_plan_step_index}.")
        else:
            logger.info(f"Step {current_step.step_id} not marked complete by reflector. Will retry or follow suggestions.")
            # Executor will retry the same step with reflection output as context.

        next_node_decision = "executor" # Default to executor (either next step or retry current)
        if new_plan_step_index >= len(state.plan.plan_steps):
            logger.info("All plan steps completed according to reflector. Triggering solver.")
            next_node_decision = "solver"
        
        # Messages for the next executor turn should be fresh or include reflection.
        # For now, executor will start with its system prompt and the new step details.
        # Reflection is stored in state.reflections and can be used by future planner/reflector calls.
        return {
            "next_node_router": next_node_decision,
            "current_plan_step_index": new_plan_step_index,
            "reflections": updated_reflections, # Use Annotated to add
            "messages": [], # Clear messages for the next step/retry
            "turns_on_current_step": 0, # Reset turns for the new/retried step
            "execution_log": updated_execution_log
        }

    except Exception as e:
        logger.error(f"Error in reflection_node: {e}", exc_info=True)
        updated_execution_log.append(f"Reflection failed for step {current_step.step_id}: {e}")
        # If reflection fails, proceed to solver to at least summarize what happened.
        return {
            "next_node_router": "solver", 
            "reflections": updated_reflections, # Use Annotated
            "messages": [],
            "errors_encountered": [f"Reflection failed: {e}"],
            "execution_log": updated_execution_log
        }

def solver_node(state: AgentState) -> Dict[str, Any]:
    """Generates the final solution summary for the task."""
    logger.info(f"--- Solver Node for Task {state.task_id} ---")
    llm = initialized_models[state.model_id]

    plan_json_str = state.plan.model_dump_json(indent=2) if state.plan else "No plan was generated or followed."
    reflections_log_str = "\n---\n".join([r.model_dump_json(indent=2) for r in state.reflections]) if state.reflections else "No reflections were made."
    full_execution_log_str = "\n".join(state.execution_log)

    solver_prompt = SOLVER_SYSTEM_PROMPT_TEMPLATE.format(
        problem_statement=state.problem_statement,
        plan_json=plan_json_str,
        full_execution_log=full_execution_log_str,
        reflections_log=reflections_log_str
    )

    updated_execution_log = []

    try:
        structured_llm_solver = llm.with_structured_output(AgentSolution)
        logger.debug("Invoking Solver LLM.")
        final_solution_obj = structured_llm_solver.invoke([
            SystemMessage(content=solver_prompt),
            HumanMessage(content="Please generate the final solution summary based on all provided information and guidelines.")
        ])
        
        logger.info(f"Solver generated solution summary for task {state.task_id}.")
        logger.debug(f"Final Solution: {final_solution_obj.model_dump_json(indent=2)}")
        updated_execution_log.append(f"Solver: Generated final solution. Tests: {final_solution_obj.tests_passed_status}")

        final_solution_dict = final_solution_obj.model_dump()
        # Augment with other necessary fields for evaluation/memory
        final_solution_dict["tool_calls_count"] = state.tool_calls_count
        final_solution_dict["successful_tool_calls"] = state.successful_tool_calls
        final_solution_dict["problem_statement"] = state.problem_statement
        final_solution_dict["errors_encountered_during_run"] = state.errors_encountered
        final_solution_dict["final_plan_followed"] = plan_json_str 

        return {
            "final_solution": final_solution_dict, # This will trigger END in the main graph
            "messages": [], # Clear messages
            "execution_log": updated_execution_log
        }
    except Exception as e:
        logger.error(f"Error in solver_node: {e}", exc_info=True)
        updated_execution_log.append(f"CRITICAL: Solver failed: {e}")
        # Populate a basic error solution
        error_solution = {
            "solution_summary": "Solver failed to generate a structured solution.",
            "tests_passed_status": "Unknown due to solver error.",
            "final_rationale": f"Solver error: {e}",
            "tool_calls_count": state.tool_calls_count,
            "problem_statement": state.problem_statement,
            "errors_encountered_during_run": state.errors_encountered + [f"Solver failed: {e}"],
        }
        return {
            "final_solution": error_solution,
            "messages": [],
            "execution_log": updated_execution_log,
            "errors_encountered": [f"Solver failed: {e}"]
        }


def tool_node(state: AgentState) -> Dict[str, Any]:
    """Executes the tool call requested by the agent."""
    if not state.messages or not isinstance(state.messages[-1], AIMessage) or not state.messages[-1].tool_calls:
        # Should not happen if graph logic is correct
        logger.warning("Tool node called without a preceding tool call AIMessage.")
        return {} # Return empty dict or raise error? Empty dict will stop graph.

    tool_call = state.messages[-1].tool_calls[0]
    tool_name = tool_call["name"]
    tool_args_from_llm = tool_call["args"] # Arguments provided by the LLM
    tool_id = tool_call["id"]

    logger.info(f"--- Tool Node: Executing '{tool_name}' with LLM args: {tool_args_from_llm} ---")
    # state.execution_log.append(f"Executing Tool: {tool_name} with args: {tool_args_from_llm}") # Annotated

    tool_map = {tool.name: tool for tool in swe_agent_tools}
    
    updated_errors_encountered = []
    updated_execution_log = [f"Executing Tool: {tool_name} with LLM args: {tool_args_from_llm}"]
    
    if tool_name not in tool_map:
        error_msg = f"Error: Tool '{tool_name}' not found."
        updated_errors_encountered.append(error_msg)
        tool_message = ToolMessage(content=error_msg, tool_call_id=tool_id)
        # successful_tool_calls is not incremented
        return {"messages": [tool_message], "successful_tool_calls": state.successful_tool_calls, 
                "errors_encountered": updated_errors_encountered, "execution_log": updated_execution_log}

    selected_tool = tool_map[tool_name]
    output_state_update = {} # For current_open_file etc.
    current_successful_tool_calls = state.successful_tool_calls

    # Prepare the final arguments for the tool, combining LLM args and state args
    final_tool_args = tool_args_from_llm.copy() # Start with LLM-provided args

    # Inject stateful arguments for specific tools
    if tool_name == "file_viewer" or tool_name == "edit":
        final_tool_args["current_open_file"] = state.current_open_file
        final_tool_args["current_file_lines"] = state.current_file_lines
        final_tool_args["current_window_start_line"] = state.current_window_start_line
        logger.debug(f"Injected state for '{tool_name}': "
                     f"current_open_file='{state.current_open_file}', "
                     f"current_file_lines is {'set' if state.current_file_lines else 'None'}")

    try:
        # Execute the tool by calling its underlying function directly
        # This bypasses some of the Tool.invoke() logic that might filter args
        if hasattr(selected_tool, 'func') and callable(selected_tool.func):
            logger.debug(f"Calling tool '{tool_name}' via .func(**final_tool_args)")
            tool_output = selected_tool.func(**final_tool_args)
        else:
            # Fallback to invoke if .func is not available (shouldn't happen for @tool)
            logger.debug(f"Calling tool '{tool_name}' via .invoke(final_tool_args)")
            tool_output = selected_tool.invoke(final_tool_args)
            
        logger.debug(f"Raw tool_output from '{tool_name}': {tool_output}")

        # Handle tools that return state updates (file_viewer, edit)
        if isinstance(tool_output, dict) and "viewer_output" in tool_output:
            output_content = tool_output["viewer_output"]
            # Update agent state from tool output IF the keys are present in tool_output
            if "current_open_file" in tool_output:
                output_state_update["current_open_file"] = tool_output["current_open_file"]
                logger.debug(f"Tool '{tool_name}' trying to update 'current_open_file' to: {tool_output['current_open_file']}")
            if "current_file_lines" in tool_output:
                output_state_update["current_file_lines"] = tool_output["current_file_lines"]
            if "current_window_start_line" in tool_output:
                output_state_update["current_window_start_line"] = tool_output["current_window_start_line"]
        elif isinstance(tool_output, str):
             output_content = tool_output
        else:
             output_content = json.dumps(tool_output) # Default to JSON string

        # Check for errors indicated by the tool output string
        if "Error:" in output_content or "failed" in output_content.lower() or "Errno" in output_content or "Traceback" in output_content or "not found" in output_content.lower():
             updated_errors_encountered.append(f"Tool Error ({tool_name}): {output_content[:200]}")
             logger.warning(f"Tool '{tool_name}' indicated an error: {output_content[:200]}...")
        else:
             current_successful_tool_calls += 1

        updated_execution_log.append(f"Tool '{tool_name}' Output: {output_content[:300]}...")
        tool_message = ToolMessage(content=output_content, tool_call_id=tool_id)

    except Exception as e:
        error_msg = f"Error executing tool '{tool_name}': {e}"
        logger.error(error_msg, exc_info=True)
        updated_errors_encountered.append(error_msg)
        tool_message = ToolMessage(content=error_msg, tool_call_id=tool_id)
        # successful_tool_calls is not incremented

    # Return the tool message and any state updates
    # Merge specific state updates like current_open_file into the return dict
    final_return_dict = {
        "messages": [tool_message], 
        "successful_tool_calls": current_successful_tool_calls,
        "errors_encountered": updated_errors_encountered,
        "execution_log": updated_execution_log,
        **output_state_update # This is crucial for current_open_file
    }
    logger.debug(f"Tool_node returning: { {k: v for k, v in final_return_dict.items() if k not in ['messages', 'current_file_lines'] } }") # Log without large messages/content
    return final_return_dict


# --- Conditional Edges ---

def route_after_executor(state: AgentState) -> str:
    """Determines the next node after the executor based on its command."""
    command = ""
    try:
        command = state.command_for_router
    except AttributeError:
        logger.debug("route_after_executor: 'command_for_router' not found in state, defaulting to empty string.")
        pass # command remains ""
    
    if command == "SUBMIT_TASK_COMPLETE":
        logger.info("Executor submitted task as complete. Routing to solver.")
        return "solver"
    elif command == "SUBMIT_FOR_REFLECTION":
        logger.info("Executor submitted for reflection. Routing to reflection_node.")
        return "reflection"
    
    # Check if the last message from executor has tool calls
    last_message = state.messages[-1] if state.messages else None
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Executor requested a tool. Routing to tool_executor.")
        return "tool_executor"
    
    # Fallback or if executor produced text without specific command (should be rare with new structure)
    logger.warning(f"Executor output unhandled command '{command}' or no specific routing action. Defaulting to reflection.")
    return "reflection"


def route_after_reflection(state: AgentState) -> str:
    """Determines the next node after reflection."""
    next_node_decision = "executor" # Default
    try:
        next_node_decision = state.next_node_router
    except AttributeError:
        logger.debug("route_after_reflection: 'next_node_router' not found in state, defaulting to 'executor'.")
        pass # next_node_decision remains "executor"

    if next_node_decision == "solver":
        logger.info("Reflection complete, all plan steps done. Routing to solver.")
        return "solver"
    else: # Default is "executor"
        logger.info(f"Reflection complete. Routing to executor for step index {state.current_plan_step_index}.")
        return "executor"

def check_if_task_finished(state: AgentState) -> str:
    """Checks if final_solution is set, indicating task completion or critical failure."""
    if state.final_solution:
        logger.info("Final solution is set or task critically failed. Ending graph.")
        return END
    # New condition: if plan generation failed and set final_solution.
    if state.plan is None and state.final_solution: # Plan generation failed
        logger.info("Planning failed critically. Ending graph.")
        return END
    logger.debug("Task not yet finished. Continuing.")
    return "continue_task"


# --- Build Graph ---

def build_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("plan_generator", plan_generation_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("tool_executor", tool_node) 
    workflow.add_node("reflector", reflection_node)
    workflow.add_node("solver", solver_node)

    workflow.set_entry_point("plan_generator")

    # Conditional routing from plan_generator (handles planning failure)
    workflow.add_conditional_edges(
        "plan_generator",
        check_if_task_finished, # Checks if final_solution was set due to planning failure
        {
            END: END,
            "continue_task": "executor"
        }
    )
    
    # Main execution loop
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "tool_executor": "tool_executor",
            "reflection": "reflector",
            "solver": "solver"
        }
    )
    
    workflow.add_edge("tool_executor", "executor") # Tool output goes back to executor to process

    workflow.add_conditional_edges(
        "reflector",
        route_after_reflection,
        {
            "executor": "executor",
            "solver": "solver"
        }
    )
    
    # Solver always leads to END because it sets final_solution
    workflow.add_edge("solver", END)


    # Compile the graph
    app = workflow.compile()
    logger.info("SWE-Agent-CL ReAct workflow compiled.")
    return app

# Build the workflow only if models are available
agent_workflow = build_agent_workflow() if initialized_models else None

# %%
# --- Test the agent workflow on a single task ---
# Ensure swe_bench_cl is loaded (either real or dummy)
logger.info("--- Testing Agent Workflow on First Task ---")

try:
    # Select the first task from the first sequence
    # test_sequence = swe_bench_cl["sequences"][0]
    test_sequence = dummy_swe_bench_cl["sequences"][0]
    test_task = test_sequence["tasks"][0]
    test_task_id = test_task["metadata"]["instance_id"]
    test_sequence_id = test_sequence["id"]

    # Select a model (prefer faster/cheaper if available)
    test_model_id = MODELS[0] # Use the first available initialized model
    logger.info(f"Using model '{test_model_id}' for the test run.")

    # Setup the repository for the test task
    task_repo_identifier = test_task["metadata"]["repo"]
    task_base_commit = test_task["metadata"]["base_commit"]

    # Determine dummy setup function if needed
    dummy_setup_func = None
    if task_repo_identifier.startswith("local/"):
        # Ensure the dummy setup function is available in this scope
        if 'dummy_files_setup_for_test' in globals():
            dummy_setup_func = dummy_files_setup_for_test
        else:
            logger.error("Dummy setup function 'dummy_files_setup_for_test' not found.")
            raise RuntimeError("Missing dummy setup function.")

    task_specific_repo_path = setup_repository(
        task_repo_identifier,
        task_base_commit,
        REPOS_BASE_DIR,
        dummy_files_setup=dummy_setup_func
    )
    logger.info(f"Repository for test task '{test_task_id}' prepared at: {task_specific_repo_path}")

    # Prepare initial state
    initial_state_test = AgentState(
        problem_statement=test_task["task"]["problem_statement"],
        hints=test_task["task"].get("hints_text"),
        repo_path=str(task_specific_repo_path.resolve()),
        task_id=test_task_id,
        sequence_id=test_sequence_id,
        task_data=test_task,
        model_id=test_model_id,
        memory_enabled=(True if memory_system else False), # Enable memory if system exists
        # Other fields default
    )

    # Clear memory before test run if memory system exists and is enabled
    if memory_system and initial_state_test.memory_enabled:
        memory_system.clear_memory()
        logger.info("Cleared semantic memory before test run.")

    logger.info(f"Invoking agent workflow for test task: {test_task_id}")
    logger.info(f"Problem: {initial_state_test.problem_statement[:100]}...")

    # Configuration for the graph run
    config = {"recursion_limit": 60} # Increase recursion limit for more steps

    # Run the workflow
    final_test_state_dict = agent_workflow.invoke(initial_state_test, config=config)

    # Convert final state dict back to AgentState object for easier access
    final_test_state = AgentState(**final_test_state_dict)

    logger.info("--- Agent Workflow Test Completed ---")

    # Log final state details
    logger.info(f"Final turn count: {final_test_state.turn_count}")
    logger.info(f"Total tool calls: {final_test_state.tool_calls_count}")
    logger.info(f"Successful tool calls: {final_test_state.successful_tool_calls}")
    logger.info(f"Errors encountered: {len(final_test_state.errors_encountered)}")
    if final_test_state.errors_encountered:
            logger.warning(f"Last error: {final_test_state.errors_encountered[-1]}")

    if final_test_state.final_solution:
        logger.info(f"Final Solution Summary: {final_test_state.final_solution.get('solution_summary', 'N/A')}")
        logger.info(f"Final Rationale: {final_test_state.final_solution.get('final_rationale', 'N/A')}")
        # Note: tests_passed is evaluated *after* the run by the evaluator
    else:
        logger.warning("No final solution was generated (agent might have errored or timed out).")

    # Check dummy file content if applicable
    if USE_DUMMY_DATA:
        dummy_file_path = task_specific_repo_path / "math_utils.py"
        if dummy_file_path.exists():
            math_utils_content_after = dummy_file_path.read_text()
            logger.info(f"Content of math_utils.py after agent run:\n{math_utils_content_after}")
            if "return a + b" in math_utils_content_after:
                logger.info("Agent seems to have correctly edited math_utils.py for the dummy task!")
            else:
                logger.warning("Agent did not correctly edit math_utils.py for the dummy task.")
        else:
            logger.warning(f"math_utils.py not found in {task_specific_repo_path} after test run.")

except Exception as e:
    logger.error(f"Error during agent workflow test: {e}", exc_info=True)

# %% [markdown]
#  ## 7. Evaluation Framework
# 
# 
# 
#  Implement the evaluation logic, including applying patches, running tests against `FAIL_TO_PASS` and `PASS_TO_PASS` criteria, and calculating metrics like Success Rate and Tool Use Efficiency.

# %%
import patch as patch_parser # Use the 'patch' library: pip install python-patch

def apply_patch(patch_content: str, repo_path: Path) -> bool:
    """Applies a patch file content to the repository."""
    # Use the 'patch' library
    try:
        pset = patch_parser.fromstring(patch_content.encode())
        if not pset:
             logger.warning("Patch string could not be parsed.")
             return False
        success = pset.apply(root=str(repo_path))
        if success:
            logger.info("Patch applied successfully using python-patch.")
            return True
        else:
            # Try git apply as fallback? Requires patch file.
            logger.warning("python-patch failed to apply patch. Trying git apply...")
            patch_file = repo_path / ".eval_temp.patch"
            with open(patch_file, "w") as f:
                f.write(patch_content)
            # Check patch applicability without applying
            check_cmd = ["git", "apply", "--check", str(patch_file)]
            check_proc = subprocess.run(check_cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)
            if check_proc.returncode != 0:
                 logger.error(f"git apply --check failed for patch:\n{check_proc.stderr or check_proc.stdout}")
                 patch_file.unlink()
                 return False

            # Apply the patch
            apply_cmd = ["git", "apply", str(patch_file)]
            apply_proc = subprocess.run(apply_cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)
            patch_file.unlink() # Clean up temp file

            if apply_proc.returncode == 0:
                logger.info("Patch applied successfully using git apply.")
                return True
            else:
                logger.error(f"git apply failed:\n{apply_proc.stderr or apply_proc.stdout}")
                # Attempt to reverse if possible? Difficult. Best effort.
                # subprocess.run(["git", "apply", "--reverse", str(patch_file)], cwd=repo_path, capture_output=True)
                return False

    except Exception as e:
        logger.error(f"Error applying patch: {e}", exc_info=True)
        return False


def run_evaluation_tests(repo_path: Path, test_command: str = "python -m unittest discover") -> Tuple[bool, str, str]:
    """Runs the test suite command and captures output."""
    try:
        process = subprocess.run(test_command, shell=True, cwd=repo_path, capture_output=True, text=True, timeout=600)
        passed = process.returncode == 0
        return passed, process.stdout, process.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Test command '{test_command}' timed out.")
        return False, "", "Test execution timed out."
    except Exception as e:
        logger.error(f"Error running evaluation tests: {e}")
        return False, "", f"Error running tests: {e}"

def check_test_outcomes(stdout: str, stderr: str, fail_to_pass: List[str], pass_to_pass: List[str]) -> bool:
    """
    Checks if the test outcomes match the expected FAIL_TO_PASS and PASS_TO_PASS criteria.
    This is a simplified check based on stderr typically containing failure info.
    A more robust approach would parse specific test runner output formats (unittest, pytest).
    """
    output = stdout + "\n" + stderr
    all_passed = True

    # Check FAIL_TO_PASS: These should NOT appear as failing in the output
    for test_case in fail_to_pass:
        # Simple check: if test case name appears with "FAIL" or "ERROR" nearby in stderr
        if re.search(rf"(FAIL|ERROR): {re.escape(test_case)}", stderr, re.IGNORECASE):
            logger.warning(f"FAIL_TO_PASS check failed: Test '{test_case}' still seems to be failing.")
            all_passed = False
            break # One failure is enough
        # Check if it passed (absence of failure is weak evidence, but best effort here)

    if not all_passed: return False

    # Check PASS_TO_PASS: These should also NOT appear as failing
    for test_case in pass_to_pass:
        if re.search(rf"(FAIL|ERROR): {re.escape(test_case)}", stderr, re.IGNORECASE):
            logger.warning(f"PASS_TO_PASS check failed: Test '{test_case}' seems to have failed (regression).")
            all_passed = False
            break

    # Additionally, check for overall failure indicators like "FAILED (" or "Ran ... tests ... failures=..."
    if re.search(r"FAILED \(", stderr) or re.search(r"failures=\d+", stderr, re.IGNORECASE) and not re.search(r"failures=0", stderr, re.IGNORECASE):
         # This might catch failures not explicitly listed, but could be overly strict
         # Let's rely primarily on the specific test case checks for now.
         # Consider adding a check if *any* unexpected test failed.
         pass


    # If stderr is empty and exit code was 0, assume success if specific checks passed
    if not stderr.strip() and all_passed:
         logger.info("Test outcome check: No failures found in stderr, assuming success based on specific checks.")
         return True

    # If specific checks passed but there's other failure info, log it but maybe still return True?
    # Let's be strict: if any specific check failed, return False. Otherwise, True.
    return all_passed


def evaluate_task_solution(task_data: Dict, repo_path: Path) -> bool:
    """
    Evaluates the agent's solution for a single task by applying patches and running tests.
    Returns True if the solution is considered successful, False otherwise.
    """
    logger.info(f"--- Evaluating Task: {task_data['metadata']['instance_id']} ---")
    eval_details = task_data["evaluation"]
    test_patch_content = eval_details.get("test_patch")
    fail_to_pass_tests = eval_details.get("FAIL_TO_PASS", [])
    pass_to_pass_tests = eval_details.get("PASS_TO_PASS", [])

    # 1. Apply the test patch to set up the evaluation environment
    if test_patch_content:
        logger.info("Applying test patch...")
        if not apply_patch(test_patch_content, repo_path):
            logger.error("Failed to apply test patch. Cannot evaluate accurately.")
            # Should we return False or raise an error? Let's return False.
            return False
        logger.info("Test patch applied successfully.")
    else:
        logger.info("No test patch provided for this task.")

    # 2. Run the tests (using a default command, could be customized)
    # TODO: Potentially extract test command from dataset if available
    test_command = "python -m unittest discover" # Default, adjust if needed
    logger.info(f"Running evaluation tests with command: '{test_command}'")
    tests_passed_exit_code, stdout, stderr = run_evaluation_tests(repo_path, test_command)

    # 3. Check outcomes against FAIL_TO_PASS and PASS_TO_PASS
    logger.info("Checking test outcomes...")
    # Basic check: Did the command exit successfully?
    if not tests_passed_exit_code:
         logger.warning(f"Test command failed (exit code non-zero). Solution likely incorrect.")
         # Log snippets for debugging
         logger.debug(f"STDOUT Snippet:\n{stdout[-500:]}")
         logger.debug(f"STDERR Snippet:\n{stderr[-1000:]}")
         # Even if exit code is bad, run specific checks in case only *some* tests failed as expected
         # return False # Strict check

    # Perform detailed check based on stdout/stderr and expected test lists
    final_success = check_test_outcomes(stdout, stderr, fail_to_pass_tests, pass_to_pass_tests)

    if final_success:
        logger.info(f"Evaluation PASSED for task {task_data['metadata']['instance_id']}.")
    else:
        logger.info(f"Evaluation FAILED for task {task_data['metadata']['instance_id']}.")
        # Log snippets if failure wasn't caught by exit code
        if tests_passed_exit_code:
             logger.debug(f"STDOUT Snippet:\n{stdout[-500:]}")
             logger.debug(f"STDERR Snippet:\n{stderr[-1000:]}")


    return final_success


class SWEAgentCLEvaluator:
    def __init__(self, dataset, workflow, memory_system):
        if not dataset:
            raise ValueError("Dataset not loaded or invalid.")
        self.dataset = dataset
        self.agent_workflow = workflow
        self.memory_system = memory_system
        self.results = {} # Store results per model, per sequence

    def run_evaluation(self, model_id: str, sequence_ids: Optional[List[str]] = None, memory_enabled: bool = True):
        """Runs evaluation for a given model on specified sequences."""
        if not self.agent_workflow:
            logger.error("Agent workflow is not compiled. Cannot run evaluation.")
            return None
        if model_id not in initialized_models:
             logger.error(f"Model {model_id} is not initialized. Cannot run evaluation.")
             return None

        if sequence_ids is None:
            sequence_ids = [seq["id"] for seq in self.dataset["sequences"]]

        model_results = {
            "model_id": model_id,
            "memory_enabled": memory_enabled,
            "sequences": {}
        }

        # Clear memory before starting evaluation run if the system exists
        if self.memory_system:
            self.memory_system.clear_memory()
            logger.info(f"Cleared semantic memory before evaluation run for {model_id} (Memory: {memory_enabled}).")

        for seq_id in sequence_ids:
            sequence = next((s for s in self.dataset["sequences"] if s["id"] == seq_id), None)
            if not sequence:
                logger.warning(f"Sequence {seq_id} not found in dataset. Skipping.")
                continue

            logger.info(f"--- Evaluating Model '{model_id}' on Sequence '{seq_id}' (Memory: {memory_enabled}) ---")
            sequence_results = {
                "tasks_total": sequence["num_tasks"],
                "tasks_attempted": 0,
                "tasks_succeeded": 0,
                "total_tool_calls": 0,
                "total_successful_tool_calls": 0,
                "task_details": {}
            }

            # Sort tasks by sequence position
            tasks_in_sequence = sorted(sequence["tasks"], key=lambda t: t["continual_learning"]["sequence_position"])

            for task in tqdm(tasks_in_sequence, desc=f"Tasks in {seq_id}"):
                task_id = task["metadata"]["instance_id"]
                repo_identifier = task["metadata"]["repo"]
                base_commit = task["metadata"]["base_commit"]

                # Determine dummy setup function
                dummy_setup_func = None
                if repo_identifier.startswith("local/"):
                     if 'dummy_files_setup_for_test' in globals(): dummy_setup_func = dummy_files_setup_for_test
                     else: raise RuntimeError("Missing dummy setup function for local repo.")

                try:
                    # 1. Setup repository to the correct base commit (clean state)
                    repo_path = setup_repository(repo_identifier, base_commit, REPOS_BASE_DIR, dummy_files_setup=dummy_setup_func)

                    # 2. Prepare initial state for the agent
                    initial_state = AgentState(
                        problem_statement=task["task"]["problem_statement"],
                        hints=task["task"].get("hints_text"),
                        repo_path=str(repo_path.resolve()),
                        task_id=task_id,
                        sequence_id=seq_id,
                        task_data=task,
                        model_id=model_id,
                        memory_enabled=memory_enabled and bool(self.memory_system) # Only enable if system exists
                    )

                    # 3. Run the agent workflow
                    config = {"recursion_limit": 60} # Max agent steps
                    final_state_dict = self.agent_workflow.invoke(initial_state, config=config)
                    final_state = AgentState(**final_state_dict) # Convert back to object

                    sequence_results["tasks_attempted"] += 1
                    sequence_results["total_tool_calls"] += final_state.tool_calls_count
                    sequence_results["total_successful_tool_calls"] += final_state.successful_tool_calls

                    # 4. Evaluate the final state of the repository
                    # The agent should have made edits directly.
                    task_success = evaluate_task_solution(task, repo_path)

                    if task_success:
                        sequence_results["tasks_succeeded"] += 1

                    # 5. Record results and add experience to memory (if enabled)
                    task_result_detail = {
                        "success": task_success,
                        "tool_calls": final_state.tool_calls_count,
                        "successful_tool_calls": final_state.successful_tool_calls,
                        "turns": final_state.turn_count,
                        "errors": final_state.errors_encountered,
                        "final_solution_summary": final_state.final_solution.get('solution_summary', 'N/A') if final_state.final_solution else 'N/A'
                    }
                    sequence_results["task_details"][task_id] = task_result_detail

                    if self.memory_system and memory_enabled:
                        # Prepare data for memory storage
                        memory_data = final_state.final_solution or {} # Use final solution if available
                        memory_data["tests_passed"] = task_success # Add evaluation result
                        memory_data["tool_calls_count"] = final_state.tool_calls_count # Ensure counts are included
                        memory_data["successful_tool_calls"] = final_state.successful_tool_calls
                        memory_data["problem_statement"] = initial_state.problem_statement # Add problem statement

                        self.memory_system.add_experience_to_memory(task_id, seq_id, memory_data)

                except Exception as e:
                    logger.error(f"Error evaluating task {task_id}: {e}", exc_info=True)
                    sequence_results["task_details"][task_id] = {"success": False, "error": str(e)}

            # Calculate sequence summary metrics
            seq_success_rate = (sequence_results["tasks_succeeded"] / sequence_results["tasks_attempted"]) if sequence_results["tasks_attempted"] > 0 else 0
            seq_avg_tool_calls = (sequence_results["total_tool_calls"] / sequence_results["tasks_attempted"]) if sequence_results["tasks_attempted"] > 0 else 0
            seq_tool_efficiency = (sequence_results["total_successful_tool_calls"] / sequence_results["total_tool_calls"]) if sequence_results["total_tool_calls"] > 0 else 0

            sequence_results["summary"] = {
                "success_rate": seq_success_rate,
                "avg_tool_calls": seq_avg_tool_calls,
                "tool_use_efficiency": seq_tool_efficiency
            }
            model_results["sequences"][seq_id] = sequence_results
            logger.info(f"Sequence {seq_id} completed. Success Rate: {seq_success_rate*100:.1f}%, Avg Tools: {seq_avg_tool_calls:.1f}, Tool Efficiency: {seq_tool_efficiency*100:.1f}%")

        # Calculate overall metrics for the model run
        total_succeeded = sum(s["tasks_succeeded"] for s in model_results["sequences"].values())
        total_attempted = sum(s["tasks_attempted"] for s in model_results["sequences"].values())
        overall_success_rate = (total_succeeded / total_attempted) if total_attempted > 0 else 0
        overall_tool_calls = sum(s["total_tool_calls"] for s in model_results["sequences"].values())
        overall_successful_tools = sum(s["total_successful_tool_calls"] for s in model_results["sequences"].values())
        overall_tool_efficiency = (overall_successful_tools / overall_tool_calls) if overall_tool_calls > 0 else 0

        model_results["overall_summary"] = {
            "overall_success_rate": overall_success_rate,
            "average_tool_calls_per_task": (overall_tool_calls / total_attempted) if total_attempted > 0 else 0,
            "overall_tool_use_efficiency": overall_tool_efficiency
        }
        logger.info(f"--- Overall Results for Model '{model_id}' (Memory: {memory_enabled}) ---")
        logger.info(f"Overall Success Rate: {overall_success_rate*100:.1f}% ({total_succeeded}/{total_attempted})")
        logger.info(f"Overall Tool Use Efficiency: {overall_tool_efficiency*100:.1f}%")

        self.results[f"{model_id}_{'mem' if memory_enabled else 'no_mem'}"] = model_results
        return model_results

# Initialize evaluator (if workflow and dataset are ready)
evaluator = None
if agent_workflow and swe_bench_cl:
    evaluator = SWEAgentCLEvaluator(swe_bench_cl, agent_workflow, memory_system)
    logger.info("Evaluation framework initialized.")
else:
    logger.warning("Evaluator not initialized because agent workflow or dataset is missing.")



# %% [markdown]
#  ## 8. Experimental Design and Execution
# 
# 
# 
#  Define and run experiments to compare model performance with and without semantic memory, assessing continual learning capabilities like forward transfer.

# %%
# Define experiments
# Ensure MODELS list is up-to-date with initialized models
experiments = []
if MODELS:
    # Experiment 1: Baseline (No Memory) - Run on first sequence
    experiments.append({
        "name": f"Baseline_{MODELS[0].replace('/', '_')}_NoMemory",
        "description": f"Evaluate {MODELS[0]} without memory on the first sequence.",
        "model_id": MODELS[0],
        "sequence_ids": [swe_bench_cl["sequences"][0]["id"]] if swe_bench_cl else [],
        "memory_enabled": False
    })
    # Experiment 2: Memory-Augmented - Run on first sequence
    experiments.append({
        "name": f"Memory_{MODELS[0].replace('/', '_')}_WithMemory",
        "description": f"Evaluate {MODELS[0]} with memory on the first sequence.",
        "model_id": MODELS[0],
        "sequence_ids": [swe_bench_cl["sequences"][0]["id"]] if swe_bench_cl else [],
        "memory_enabled": True
    })
    # Add more experiments for other models or sequences if desired
    # Example: Evaluate a second model
    if len(MODELS) > 1:
         experiments.append({
            "name": f"Baseline_{MODELS[1].replace('/', '_')}_NoMemory",
            "description": f"Evaluate {MODELS[1]} without memory on the first sequence.",
            "model_id": MODELS[1],
            "sequence_ids": [swe_bench_cl["sequences"][0]["id"]] if swe_bench_cl else [],
            "memory_enabled": False
         })
         experiments.append({
            "name": f"Memory_{MODELS[1].replace('/', '_')}_WithMemory",
            "description": f"Evaluate {MODELS[1]} with memory on the first sequence.",
            "model_id": MODELS[1],
            "sequence_ids": [swe_bench_cl["sequences"][0]["id"]] if swe_bench_cl else [],
            "memory_enabled": True
         })

else:
    logger.warning("No models initialized, cannot define experiments.")

# Function to run experiments
def run_experiments(experiments_to_run, evaluator_instance):
    """Runs the defined experiments using the evaluator."""
    if not evaluator_instance:
        logger.error("Evaluator is not initialized. Cannot run experiments.")
        return None
    if not experiments_to_run:
        logger.warning("No experiments defined to run.")
        return {}

    all_results = {}
    for exp in experiments_to_run:
        # Check if sequence ID is valid before running
        if not exp["sequence_ids"]:
             logger.warning(f"Skipping experiment '{exp['name']}' due to missing sequence IDs (dataset likely not loaded).")
             continue

        logger.info(f"\n--- Running Experiment: {exp['name']} ---")
        logger.info(exp['description'])

        exp_results = evaluator_instance.run_evaluation(
            model_id=exp['model_id'],
            sequence_ids=exp['sequence_ids'],
            memory_enabled=exp['memory_enabled']
        )
        all_results[exp['name']] = exp_results
        logger.info(f"--- Completed Experiment: {exp['name']} ---\n")

    return all_results

# Run the experiments
if evaluator and experiments:
    logger.info("Starting experimental runs...")
    experiment_results = run_experiments(experiments, evaluator)
    # Save results
    results_path = Path("./swe_agent_cl_results.json")
    try:
        with open(results_path, "w") as f:
            # Use default=str to handle non-serializable types like Path if they sneak in
            json.dump(experiment_results, f, indent=2, default=str)
        logger.info(f"Experiment results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save experiment results: {e}")

else:
    logger.warning("Skipping experiment execution as evaluator or experiments are not ready.")
    experiment_results = None



# %% [markdown]
#  ## 9. Results Analysis and Visualization
# 
# 
# 
#  Analyze the collected results, focusing on comparing performance with and without memory to understand the impact of the semantic memory system and calculate forward transfer potential.

# %%
import pandas as pd # For easier data manipulation

def analyze_results(results_data):
    """Analyzes and visualizes the experimental results."""
    if not results_data:
        logger.warning("No experiment results to analyze.")
        return {"summary": "No results available."}

    analysis = {}
    summary_data = []

    # Process results into a flat structure for easier analysis
    for exp_name, exp_result in results_data.items():
        if not exp_result: continue # Skip failed/empty experiments

        model_id = exp_result.get("model_id", "N/A")
        memory_enabled = exp_result.get("memory_enabled", "N/A")
        overall_summary = exp_result.get("overall_summary", {})
        success_rate = overall_summary.get("overall_success_rate", 0)
        tool_efficiency = overall_summary.get("overall_tool_use_efficiency", 0)

        # Extract sequence-level data if needed (e.g., for learning curves)
        sequence_results = exp_result.get("sequences", {})
        for seq_id, seq_data in sequence_results.items():
             seq_summary = seq_data.get("summary", {})
             summary_data.append({
                 "experiment": exp_name,
                 "model_id": model_id,
                 "memory_enabled": memory_enabled,
                 "sequence_id": seq_id,
                 "seq_success_rate": seq_summary.get("success_rate", 0),
                 "seq_avg_tool_calls": seq_summary.get("avg_tool_calls", 0),
                 "seq_tool_efficiency": seq_summary.get("tool_use_efficiency", 0),
                 "overall_success_rate": success_rate, # Add overall for context
                 "overall_tool_efficiency": tool_efficiency
             })

    if not summary_data:
        logger.warning("No valid summary data extracted from results.")
        return {"summary": "No valid summary data found."}

    df = pd.DataFrame(summary_data)

    # --- Visualizations ---
    output_dir = Path("./swe_agent_cl_plots")
    output_dir.mkdir(exist_ok=True)

    # 1. Success Rate Comparison (Memory vs. No Memory per Model)
    plt.figure(figsize=(12, 7))
    pivot_success = df.pivot_table(index='model_id', columns='memory_enabled', values='overall_success_rate', aggfunc='mean')
    if not pivot_success.empty:
        pivot_success.plot(kind='bar', figsize=(12, 7))
        plt.title('Overall Success Rate Comparison (Memory vs. No Memory)')
        plt.ylabel('Success Rate')
        plt.xlabel('Model ID')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Memory Enabled')
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate_memory_comparison.png')
        plt.show()
        analysis["success_rate_plot"] = 'success_rate_memory_comparison.png'

    # 2. Tool Use Efficiency Comparison
    plt.figure(figsize=(12, 7))
    pivot_tool = df.pivot_table(index='model_id', columns='memory_enabled', values='overall_tool_efficiency', aggfunc='mean')
    if not pivot_tool.empty:
        pivot_tool.plot(kind='bar', figsize=(12, 7))
        plt.title('Overall Tool Use Efficiency Comparison (Memory vs. No Memory)')
        plt.ylabel('Tool Use Efficiency (Successful/Total Calls)')
        plt.xlabel('Model ID')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Memory Enabled')
        plt.tight_layout()
        plt.savefig(output_dir / 'tool_efficiency_memory_comparison.png')
        plt.show()
        analysis["tool_efficiency_plot"] = 'tool_efficiency_memory_comparison.png'

    # --- Metric Calculation ---
    analysis["metrics"] = {}
    for model_id in df['model_id'].unique():
        model_df = df[df['model_id'] == model_id]
        # Calculate Forward Transfer Potential (Simple version)
        # Compare avg success rate with memory vs without memory on the *same* sequences
        # Requires results from both memory=True and memory=False runs for the model.
        mem_true_df = model_df[model_df['memory_enabled'] == True]
        mem_false_df = model_df[model_df['memory_enabled'] == False]

        if not mem_true_df.empty and not mem_false_df.empty:
            # Average over sequences if multiple were run
            avg_success_mem = mem_true_df['seq_success_rate'].mean()
            avg_success_no_mem = mem_false_df['seq_success_rate'].mean()
            forward_transfer_potential = avg_success_mem - avg_success_no_mem

            analysis["metrics"][model_id] = {
                "avg_success_rate_with_memory": avg_success_mem,
                "avg_success_rate_without_memory": avg_success_no_mem,
                "forward_transfer_potential": forward_transfer_potential,
                "avg_tool_efficiency_with_memory": mem_true_df['seq_tool_efficiency'].mean(),
                 "avg_tool_efficiency_without_memory": mem_false_df['seq_tool_efficiency'].mean(),
            }
            print(f"\nMetrics for {model_id}:")
            print(f"  Avg Success Rate (No Memory): {avg_success_no_mem:.3f}")
            print(f"  Avg Success Rate (With Memory): {avg_success_mem:.3f}")
            print(f"  Forward Transfer Potential: {forward_transfer_potential:.3f}")
        else:
             analysis["metrics"][model_id] = {"notes": "Insufficient data for comparison (missing memory or no-memory run)."}


    analysis["summary"] = "Analysis complete. Plots saved to ./swe_agent_cl_plots/. Forward transfer potential calculated."
    analysis["dataframe_summary"] = df.to_dict('records') # Include processed data

    return analysis

# Analyze the results (if available)
if experiment_results:
    analysis_summary = analyze_results(experiment_results)
    # Save analysis summary
    analysis_path = Path("./swe_agent_cl_analysis.json")
    try:
        with open(analysis_path, "w") as f:
             json.dump(analysis_summary, f, indent=2, default=str)
        logger.info(f"Analysis summary saved to {analysis_path}")
    except Exception as e:
        logger.error(f"Failed to save analysis summary: {e}")

else:
    logger.warning("No experiment results found to analyze.")



# %% [markdown]
#  ## 10. Findings and Conclusions (Example)
# 
# 
# 
#  *Based on the results (replace with actual findings after running):*
# 
# 
# 
#  *   **Memory Impact:** The semantic memory system demonstrated a [positive/negative/negligible] impact on overall success rates. For model X, the success rate improved by Y% when memory was enabled, suggesting [effective learning/no significant learning] from past tasks within the sequence.
# 
#  *   **Forward Transfer:** The calculated forward transfer potential was [positive/negative/zero] for most models, indicating that accessing solutions to earlier tasks [helped/hindered/did not affect] performance on later tasks. Model Z showed the highest positive transfer.
# 
#  *   **Tool Use:** Memory augmentation [did/did not] significantly affect tool use efficiency. Agents [used tools more effectively / showed similar efficiency] when memory was available.
# 
#  *   **Model Comparison:** Model A consistently outperformed Model B, both with and without memory, suggesting inherent differences in problem-solving capabilities for these tasks.
# 
#  *   **SWE-agent Adaptation:** The implemented ACI allowed the agents to interact with the environment effectively, using tools like `edit` with linting and structured file viewing. Challenges remain in [mention specific areas, e.g., complex multi-step edits, efficient search query formulation].
# 
# 
# 
#  **Conclusion:** This framework successfully integrates SWE-agent's ACI principles with a semantic memory system for continual learning evaluation. The results highlight the potential [and challenges] of using memory to improve agent performance on evolving software engineering tasks. Future work should focus on more sophisticated memory retrieval strategies, handling longer sequences, and refining the CL metrics.

# %% [markdown]
#  ## 11. Future Work and Extensions
# 
# 
# 
#  *   **Refined CL Metrics:** Implement robust calculation for Forgetting Rate and Backward Transfer, requiring more complex experimental setups (e.g., re-testing).
# 
#  *   **Advanced Memory:** Explore different memory structures (e.g., episodic memory, code-specific embeddings), retrieval strategies (e.g., filtering by code similarity), and context management techniques.
# 
#  *   **Longer Sequences:** Test on longer sequences to better evaluate long-term learning and forgetting.
# 
#  *   **Tool Improvement:** Enhance tool robustness, particularly argument parsing for complex commands. Implement search within the currently open file.
# 
#  *   **Agent Reasoning:** Improve agent planning and reflection, potentially adding explicit nodes for plan adjustment based on memory or reflection.
# 
#  *   **Evaluation Detail:** Parse test runner output more precisely for `check_test_outcomes` instead of relying solely on stderr patterns.
# 
#  *   **Fine-tuning:** Experiment with fine-tuning models on SWE-Bench-CL sequences to directly improve CL capabilities.

# %% [markdown]
#  ## 12. References and Acknowledgments
# 
# 
# 
#  *   Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*. arXiv preprint arXiv:2405.15793.
# 
#  *   Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. R. (2024). *SWE-bench: Can Language Models Resolve Real-world GitHub Issues?* Proceedings of the Twelfth International Conference on Learning Representations (ICLR).
# 
#  *   Anthropic Research. (2024). *Building Effective Agents with Foundation Models*. Retrieved from [https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)
# 
#  *   LangChain & LangGraph documentation.
# 
#  *   SWE-Bench-CL Dataset (This project).
# 
# 
# 
#  **Acknowledgments:** Based on the original `eval_procedure.py` structure provided by the user. Incorporates ideas and methodologies from the SWE-agent paper.


