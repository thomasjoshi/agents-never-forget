# %% [markdown]
#    # SWE-Bench-CL: Comprehensive Continual Learning Evaluation for LLMs (with Resumability)
# 
#    This notebook implements a framework for evaluating coding LLMs on the SWE-Bench-CL dataset, 
#    focusing on continual learning capabilities. It includes:
#    - Sequential task processing with an optional semantic memory (RAG) system.
#    - Automated execution of the `swebench.harness.run_evaluation` script.
#    - Calculation of various continual learning metrics (ACC, F, FT, BWT, AULC, CL-Score).
#    - State management for resumability of long evaluation runs.
#    - A quick test block for debugging the pipeline.

# %% [markdown]
#    ## 1. Setup and Dependencies

# %%
# !pip install -q datasets pandas python-dotenv langchain langchain_anthropic langchain_openai langchain_google_genai langchain-ollama faiss-cpu tiktoken numpy

# %%
import os
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Any, List, Dict, Optional, Tuple
import logging
import time
import subprocess # For running harness
import shutil # For cleaning up testbeds
import numpy as np
import torch # For checking MPS availability
from pathlib import Path # For rglob and path manipulation
import re
from tqdm import tqdm # Import tqdm

# LangChain components
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage

# Embeddings and Vector Stores
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Automatic SWE-bench setup ---
def setup_swe_bench(repo_path: str):
    """Checks for SWE-bench repository and attempts to set it up if not found."""
    swe_bench_dir = Path(repo_path)
    if swe_bench_dir.is_dir() and (swe_bench_dir / ".git").is_dir():
        logger.info(f"SWE-bench repository found at {repo_path}.")
        # Optionally, check if pip install -e . was run (e.g., by checking for swebench.egg-info)
        # For simplicity, we'll assume if the repo is there, it might be set up.
        # A more robust check could try importing swebench.
        try:
            import swebench
            logger.info("SWE-bench package is importable.")
            return True
        except ImportError:
            logger.warning("SWE-bench repository found, but package not importable. Attempting to install.")
            try:
                subprocess.run(["pip", "install", "-e", "."], cwd=repo_path, check=True, capture_output=True, text=True)
                logger.info(f"Successfully ran 'pip install -e .' in {repo_path}.")
                # Verify import again
                import swebench
                logger.info("SWE-bench package is now importable.")
                return True
            except Exception as e:
                logger.error(f"Failed to install SWE-bench from {repo_path}: {e.stderr if hasattr(e, 'stderr') else e}")
                logger.error("Please manually install SWE-bench: cd into the SWE-bench directory and run 'pip install -e .'")
                return False
    else:
        logger.warning(f"SWE-bench repository not found at {repo_path}. Attempting to clone and install.")
        try:
            # Create parent directory if it doesn't exist, to avoid clone errors if repo_path is like "./foo/SWE-bench"
            swe_bench_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning SWE-bench into {repo_path}...")
            subprocess.run(
                ["git", "clone", "git@github.com:princeton-nlp/SWE-bench.git", str(swe_bench_dir)], 
                check=True, capture_output=True, text=True
            )
            logger.info(f"Successfully cloned SWE-bench to {repo_path}.")
            logger.info(f"Installing SWE-bench from {repo_path}...")
            subprocess.run(
                ["pip", "install", "-e", "."], 
                cwd=repo_path, check=True, capture_output=True, text=True
            )
            logger.info(f"Successfully ran 'pip install -e .' in {repo_path}.")
            # Verify import
            import swebench
            logger.info("SWE-bench package is now importable.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during SWE-bench setup: {e.stderr}")
            logger.error("Please ensure Git is installed and you have SSH access to GitHub if using git@github.com URLs.")
            logger.error("Alternatively, clone SWE-bench manually and install with 'pip install -e .'")
            return False
        except ImportError:
            logger.error(f"Even after attempted install, SWE-bench package is not importable from {repo_path}.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during SWE-bench setup: {e}")
            return False

# Call the setup function early
# Note: SWE_BENCH_REPO_PATH is defined in the Configuration section below.
# We will call setup_swe_bench() after SWE_BENCH_REPO_PATH is defined.


# %% [markdown]
#    ## 2. Configuration
#    Set paths, model configurations, and experiment parameters here.

# %%
# --- Paths ---
# NOTE: Update these paths according to your local setup
DOTENV_PATH = '../.env'
SWE_BENCH_CL_DATASET_PATH = "../data/SWE-Bench-CL-Curriculum.json" # Path to your SWE-Bench-CL JSON file
SWE_BENCH_REPO_PATH = "./SWE-bench"  # Path to your cloned swe-bench repository (for running the harness)
BASE_OUTPUT_DIR = "eval_results" # Base directory for all outputs
STATE_FILE_PATH = os.path.join(BASE_OUTPUT_DIR, "eval_state.json")

# --- Attempt to set up SWE-bench ---
if not setup_swe_bench(SWE_BENCH_REPO_PATH):
    logger.error("SWE-bench setup failed. The script may not run correctly. Please set up SWE-bench manually.")
    # Depending on strictness, you might want to raise an error here:
    # raise RuntimeError("SWE-bench setup failed. Halting.")
else:
    logger.info("SWE-bench setup check complete.")


# --- LLM Configuration ---
MODELS_TO_EVALUATE = [
    # "google/gemini-1.5-flash-latest",
    "google/gemini-2.0-flash",
    # "openai/gpt-4o-mini",
    # "ollama/llama3"
]
TEMPERATURE = 0.2
MAX_TOKENS_OUTPUT = 4096

# --- Semantic Memory Configuration ---
EMBEDDING_MODEL_CONFIG = {
    "name": "ollama/nomic-embed-text",
    "max_context_tokens_for_retrieval": 2000,
    "num_retrieved_memories": 3
}

# --- Experiment Configuration ---
EXPERIMENT_CONDITIONS = {
    "memory_enabled": {"memory_enabled": True},
    "memory_disabled": {"memory_enabled": False},
}
SEQUENCES_TO_RUN = None # Example: ["django_django_sequence"] or None for all
TASKS_PER_SEQUENCE_LIMIT = None # Example: 2 or None for all

# --- Harness Configuration ---
HARNESS_TIMEOUT_PER_TASK = 900
HARNESS_MAX_WORKERS = 4
HARNESS_RUN_ID_PREFIX = ""

# --- CL Metrics Configuration ---
CL_SCORE_WEIGHTS = {
    "lambda_F": 1.0, "lambda_FT": 1.0, "lambda_BWT": 1.0, "lambda_AULC": 1.0,
}

# Create base output directory if it doesn't exist
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# %% [markdown]
#    ## 2.1. State Management for Resumability

# %%
def load_or_initialize_state(force_fresh_start=False) -> Dict:
    """Loads evaluation state from STATE_FILE_PATH or initializes a new one."""
    if os.path.exists(STATE_FILE_PATH) and not force_fresh_start:
        try:
            with open(STATE_FILE_PATH, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded existing state from {STATE_FILE_PATH} for run {state.get('evaluation_run_timestamp')}")
            
            # Check if dataset path matches, if not, force fresh state for safety
            if state.get('current_swe_bench_cl_dataset_path') != os.path.abspath(SWE_BENCH_CL_DATASET_PATH):
                logger.warning("Dataset path in state file does not match current configuration. Forcing a fresh state.")
                return load_or_initialize_state(force_fresh_start=True) # Recursive call for fresh start
            
            # Ensure backward compatibility for task_eval_progress structure if loading older states
            # No explicit change needed here for adding a new key, as older entries just won't have it.
            # Code reading from it should use .get("gold_patch") for safety.
            return state
        except Exception as e:
            logger.error(f"Error loading state file {STATE_FILE_PATH}: {e}. Initializing fresh state.")
            # Fall through to initialize fresh state
    
    # Initialize fresh state
    new_timestamp = time.strftime("%Y%m%d-%H%M%S")
    logger.info(f"Initializing fresh evaluation state with run timestamp: {new_timestamp}")
    state = {
        "evaluation_run_timestamp": new_timestamp,
        "current_swe_bench_cl_dataset_path": os.path.abspath(SWE_BENCH_CL_DATASET_PATH),
        "task_eval_progress": {}, # model_name -> condition_name -> {instance_id: {"model_patch": str, "harness_result": bool, "raw_llm_output": Optional[str], "gold_patch": Optional[str]}}
        "parsed_harness_results_data": {},# model -> condition -> {seq_id: {instance_id: pass_status}}
        "overall_results_list": [],      # List of dicts for final summary
        "overall_results_summary_df_path": None
    }
    save_state(state)
    return state

def save_state(state: Dict):
    """Saves the current evaluation state to STATE_FILE_PATH."""
    try:
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state, f, indent=4)
        logger.debug(f"Saved state to {STATE_FILE_PATH}")
    except Exception as e:
        logger.error(f"Error saving state to {STATE_FILE_PATH}: {e}")

# Load or initialize global state
current_evaluation_state = load_or_initialize_state()
EVALUATION_RUN_TIMESTAMP = current_evaluation_state["evaluation_run_timestamp"]





# %% [markdown]
#   ## 3. LLM and Embedding Model Initialization

# %%
# Load API keys
try:
    if os.path.exists(DOTENV_PATH):
        load_dotenv(DOTENV_PATH)
        logger.info(f"Loaded .env file from: {DOTENV_PATH}")
    else:
        logger.warning(f".env file not found at {DOTENV_PATH}. API calls to proprietary models might fail if keys not in environment.")
except Exception as e:
    logger.error(f"Error loading .env file: {e}")

# Function to initialize LLM (adapted from your provided code)
def get_llm(model_str: str, temp: float, max_tokens: int):
    provider, model_name = model_str.split("/", 1)
    try:
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: logger.warning(f"ANTHROPIC_API_KEY not found for {model_str}."); return None
            return ChatAnthropic(model=model_name, temperature=temp, max_tokens=max_tokens, api_key=api_key)
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: logger.warning(f"OPENAI_API_KEY not found for {model_str}."); return None
            return ChatOpenAI(model=model_name, temperature=temp, max_tokens=max_tokens, api_key=api_key)
        elif provider == "google":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key: logger.warning(f"GEMINI_API_KEY/GOOGLE_API_KEY not found for {model_str}."); return None
            return ChatGoogleGenerativeAI(model=model_name, temperature=temp, max_output_tokens=max_tokens, google_api_key=api_key)
        elif provider == "ollama":
            return ChatOllama(model=model_name, temperature=temp) # max_tokens often part of ollama run options or model file
        else:
            logger.error(f"Unsupported provider: {provider}"); return None
    except Exception as e:
        logger.error(f"Failed to initialize LLM {model_str}: {e}"); return None

# Initialize configured LLMs
initialized_llms = {}
for name in MODELS_TO_EVALUATE:
    llm = get_llm(name, TEMPERATURE, MAX_TOKENS_OUTPUT)
    if llm:
        initialized_llms[name] = llm
        logger.info(f"Successfully initialized LLM: {name}")
    else:
        logger.warning(f"Could not initialize LLM: {name}. It will be skipped.")

if not initialized_llms:
    raise RuntimeError("No LLMs were successfully initialized. Halting.")
logger.info(f"LLMs ready for evaluation: {list(initialized_llms.keys())}")


# Initialize Embedding Model and Tokenizer
active_embedding_model = None
tokenizer_for_counting = None
try:
    emb_config_name = EMBEDDING_MODEL_CONFIG["name"]
    if emb_config_name.startswith("openai/"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key: logger.warning("OpenAI API key not found for embeddings.")
        else: active_embedding_model = OpenAIEmbeddings(model=emb_config_name.split("/")[1], openai_api_key=api_key)
    elif emb_config_name.startswith("ollama/"):
        model_ollama_name = emb_config_name.split("/")[1]
        try:
            active_embedding_model = OllamaEmbeddings(model=model_ollama_name)
            active_embedding_model.embed_query("test connection")
            logger.info(f"Ollama embedding model '{model_ollama_name}' initialized.")
        except Exception as ollama_err:
            logger.error(f"Failed to use Ollama embedding model '{model_ollama_name}'. Error: {ollama_err}")
            active_embedding_model = None
    else:
        logger.error(f"Unsupported embedding model provider for: {emb_config_name}")

    if active_embedding_model:
        try:
            import tiktoken
            tokenizer_for_counting = tiktoken.get_encoding("cl100k_base")
            logger.info("Using tiktoken for precise token counting in memory context.")
        except Exception as e:
            logger.warning(f"Could not initialize tiktoken: {e}. Using space-based split for token counting.")
    else:
        logger.error("Embedding model not initialized. Semantic memory will be effectively disabled.")

except Exception as e:
    logger.error(f"Error initializing embedding model components: {e}")
    active_embedding_model = None




# %% [markdown]
#   ## 4. Load SWE-Bench-CL Dataset

# %%
def load_swe_bench_cl_dataset_content(file_path: str) -> Optional[Dict]: # Renamed to avoid conflict
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded SWE-Bench-CL dataset from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading SWE-Bench-CL dataset from {file_path}: {e}")
        return None

swe_bench_cl_full_data = load_swe_bench_cl_dataset_content(SWE_BENCH_CL_DATASET_PATH)
if not swe_bench_cl_full_data:
    raise RuntimeError("SWE-Bench-CL dataset could not be loaded. Halting.")

if SEQUENCES_TO_RUN:
    swe_bench_cl_data_sequences = [
        seq for seq in swe_bench_cl_full_data.get("sequences", []) if seq.get("id") in SEQUENCES_TO_RUN
    ]
    if not swe_bench_cl_data_sequences:
        raise ValueError(f"None of the specified SEQUENCES_TO_RUN found: {SEQUENCES_TO_RUN}")
    logger.info(f"Running on {len(swe_bench_cl_data_sequences)} selected sequences: {SEQUENCES_TO_RUN}")
else:
    swe_bench_cl_data_sequences = swe_bench_cl_full_data.get("sequences", [])
    logger.info(f"Running on all {len(swe_bench_cl_data_sequences)} sequences from the dataset.")




# %% [markdown]
#   ## 5. Semantic Memory System

# %%
class SemanticMemory:
    """Stores and retrieves task experiences using vector embeddings."""
    def __init__(self, embedding_model: Any, k_results: int = 3):
        self.embedding_model = embedding_model
        self.k_results = k_results
        self.index = None
        self.documents: List[Document] = []
        self.doc_counter = 0

    def add_entry(self, task_id: str, sequence_id: str, problem_statement: str, attempted_solution: str, harness_result: Optional[bool], metadata: Optional[Dict] = None):
        if not self.embedding_model: return 
        meta = metadata or {}
        meta.update({"task_id": task_id, "sequence_id": sequence_id, "harness_result": harness_result, "doc_id": self.doc_counter})
        self.doc_counter += 1
        
        if harness_result is True:
            status_prefix = "[SUCCESSFUL SOLUTION (Verified by Harness)]"
        elif harness_result is False:
            status_prefix = "[FAILED ATTEMPT (Verified by Harness)]"
        else: # harness_result is None (e.g. harness error, or not run yet)
            status_prefix = "[ATTEMPT (Outcome Unknown/Not Verified)]"
            
        full_content = (f"{status_prefix} for Task {task_id} in Sequence {sequence_id}:\n"
                        f"Problem: {problem_statement[:500]}...\n"
                        f"Attempted Solution (Patch):\n{attempted_solution[:1000]}...\n")
        doc = Document(page_content=full_content, metadata=meta)
        self.documents.append(doc)
        try:
            texts_for_faiss = [d.page_content for d in self.documents]
            metadatas_for_faiss = [d.metadata for d in self.documents]
            self.index = FAISS.from_texts(texts_for_faiss, self.embedding_model, metadatas=metadatas_for_faiss)
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            if self.documents: self.documents.pop() # Remove the problematic document
            self.doc_counter -=1

    def retrieve_relevant(self, query: str, sequence_id_filter: Optional[str] = None, num_results: Optional[int] = None) -> List[Dict]:
        if not self.index or not self.embedding_model: return []
        k = num_results if num_results is not None else self.k_results
        try:
            candidate_k = max(k * 5 if sequence_id_filter else k, k) # Fetch more candidates if filtering by sequence
            actual_k_for_search = min(candidate_k, len(self.documents))
            if actual_k_for_search == 0: return []
            
            results_with_scores = self.index.similarity_search_with_score(query, k=actual_k_for_search)
            
            filtered_results = []
            seen_task_ids = set() 
            for doc, score in results_with_scores:
                # Filter by sequence_id if provided
                if sequence_id_filter and doc.metadata.get("sequence_id") != sequence_id_filter:
                    continue
                
                task_id = doc.metadata.get("task_id", "unknown_task")
                # Ensure we don't include multiple memories of the same task unless explicitly allowed by k
                # (Current logic adds most recent attempt for a task to FAISS, overwriting is not how FAISS works here, so multiple versions could exist if add_entry is called multiple times for same task_id)
                # This logic picks the highest scoring (most similar) unique task_id.
                if task_id in seen_task_ids: 
                    continue

                filtered_results.append({
                    "task_id": task_id, 
                    "sequence_id": doc.metadata.get("sequence_id", "unknown_seq"),
                    "content": doc.page_content, 
                    "harness_result": doc.metadata.get("harness_result"), # Changed from success_placeholder
                    "score": float(score) 
                })
                seen_task_ids.add(task_id)
                if len(filtered_results) >= k:
                    break
            
            return sorted(filtered_results, key=lambda x: x["score"]) # Sort by relevance score (lower is better for FAISS L2, higher for cosine)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}"); return []

    def clear(self):
        self.index = None; self.documents = []; self.doc_counter = 0
        logger.debug("Semantic memory cleared.")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer: return len(self.tokenizer.encode(text))
        return len(text.split()) # Fallback

    def add_experience_to_memory(self, task_id: str, sequence_id: str, problem_statement: str, generated_patch: str, harness_result: Optional[bool]):
        if not self.embedding_model: return
        self.embedding_model.add_documents([Document(page_content=f"{problem_statement}\n{generated_patch}", metadata={"task_id": task_id, "sequence_id": sequence_id, "harness_result": harness_result})])

    def get_relevant_context_for_prompt(self, current_task_problem_statement: str, current_sequence_id: str, num_memories: int) -> str:
        if not self.embedding_model: return ""
        retrieved = self.retrieve_relevant(current_task_problem_statement, current_sequence_id, num_memories)
        context_str = ""
        if retrieved:
            context_str += "\n\n--- Relevant Past Experiences (from Semantic Memory) ---\n"
            current_tokens = 0
            for mem in retrieved:
                mem_text = (f"Past Experience (Task: {mem['task_id']}, Relevance Score: {mem['score']:.2f}):\n"
                            f"{mem['content']}\n---\n")
                mem_tokens = self._count_tokens(mem_text)
                if current_tokens + mem_tokens <= self.embedding_model.max_tokens:
                    context_str += mem_text; current_tokens += mem_tokens
                else:
                    logger.debug(f"Memory context limit ({self.embedding_model.max_tokens} tokens) reached."); break
            context_str += "--- End of Past Experiences ---\n"
        return context_str

class MemorySystem:
    """Manages semantic memory and context building for the agent."""
    def __init__(self, semantic_memory: SemanticMemory, max_context_tokens: int, tokenizer: Optional[Any]):
        self.semantic_memory = semantic_memory
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer: return len(self.tokenizer.encode(text))
        return len(text.split()) # Fallback

    def add_experience_to_memory(self, task_id: str, sequence_id: str, problem_statement: str, generated_patch: str, harness_result: Optional[bool]):
        if not self.semantic_memory.embedding_model: return
        self.semantic_memory.add_entry(task_id, sequence_id, problem_statement, generated_patch, harness_result)

    def get_relevant_context_for_prompt(self, current_task_problem_statement: str, current_sequence_id: str, num_memories: int) -> str:
        if not self.semantic_memory.embedding_model: return ""
        retrieved = self.semantic_memory.retrieve_relevant(current_task_problem_statement, current_sequence_id, num_memories)
        context_str = ""
        if retrieved:
            context_str += "\n\n--- Relevant Past Experiences (from Semantic Memory) ---\n"
            current_tokens = 0
            for mem in retrieved:
                mem_text = (f"Past Experience (Task: {mem['task_id']}, Relevance Score: {mem['score']:.2f}):\n"
                            f"{mem['content']}\n---\n")
                mem_tokens = self._count_tokens(mem_text)
                if current_tokens + mem_tokens <= self.max_context_tokens:
                    context_str += mem_text; current_tokens += mem_tokens
                else:
                    logger.debug(f"Memory context limit ({self.max_context_tokens} tokens) reached."); break
            context_str += "--- End of Past Experiences ---\n"
        return context_str

    def clear_memory(self): self.semantic_memory.clear()




# %% [markdown]
#   ## 6. Prompt Definition

# %%
PATCH_EXAMPLE = """--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points"""

prompt_template_str_with_memory = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.
The goal is to generate a code patch in the **unified diff format** that resolves the issue.

<issue>
{problem_statement}
</issue>

Relevant context from the repository ({repo} at commit {base_commit}):
<code>
{retrieved_context}

**Hints (if any from the original issue):**
{hints_text}

**Files to consider (based on gold solution, try to identify which files to modify):**
{text_files}
</code>

Here is an example of a patch file. It consists of changes to the codebase. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files. 

<patch>
{patch_example_content}
</patch>

I need you to solve the provded issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the format shown above.

Respond below:""".replace("{patch_example_content}", PATCH_EXAMPLE)

output_parser = StrOutputParser()



# %% [markdown]
#   ## 7. Patch Generation Function

# %%
def clean_llm_generated_patch(patch_text: str) -> str:
    """Cleans the LLM-generated patch text."""
    # Regex to capture content within ``` variations
    # It handles ```, ```diff, ```patch, etc.
    # It uses a non-greedy match for the content between the fences.
    # It also handles optional language specifier on the same line as ```
    # e.g. ```python or ```patch
    fence_match = re.search(r"^```(?:[a-zA-Z0-9_]*\n)?(.*?)```$", patch_text, re.DOTALL | re.MULTILINE)
    
    if fence_match:
        content_inside_fences = fence_match.group(1)
    else:
        # If no fences, assume the whole text is the patch.
        content_inside_fences = patch_text

    # Strip leading/trailing whitespace from the extracted (or original) content
    # This helps if there's whitespace before the actual diff content or after it (before a closing fence)
    current_patch_content = content_inside_fences.strip()

    # If the LLM still includes "patch" or "diff" as the first line *inside* the fences (or if no fences)
    lines = current_patch_content.split('\n')
    if lines and lines[0].strip().lower() in ["patch", "diff"] and \
       not lines[0].strip().startswith("---") and \
       not lines[0].strip().startswith("diff --git"): # Avoid stripping actual diff header lines
        # Remove this introductory line ("patch" or "diff")
        cleaned_patch_str = '\n'.join(lines[1:])
    else:
        cleaned_patch_str = current_patch_content
        
    # Final strip to remove any extraneous newlines or spaces an LLM might add at the very beginning/end of the patch itself
    cleaned_patch_str = cleaned_patch_str.strip()

    # Ensure the patch ends with a single newline if it's not empty
    if cleaned_patch_str: # Only add newline if patch is not empty
        if not cleaned_patch_str.endswith('\n'):
            cleaned_patch_str += '\n'
    # If the patch was empty to begin with (e.g., LLM returned empty string after cleaning),
    # it should remain an empty string, not just "\n".
    # An empty patch is valid for SWE-bench (means model predicts no change).
            
    return cleaned_patch_str




# %% [markdown]
#   ## 8. SWE-bench Harness Execution and Result Parsing

# %%
def run_swe_bench_harness(
    predictions_path: str,
    model_name: str,
    experiment_condition_name: str, 
    run_id_suffix: str = "" # Unique suffix for this specific harness invocation
) -> Optional[str]:
    if not os.path.exists(SWE_BENCH_REPO_PATH):
        logger.error(f"SWE-bench repository not found at {SWE_BENCH_REPO_PATH}. Cannot run harness.")
        return None
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True, timeout=10)
    except Exception:
        logger.error("Docker does not appear to be running/installed or responsive. Harness requires Docker.")
        return None

    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    # Use the global EVALUATION_RUN_TIMESTAMP for the main log folder, then specific run_id for subfolder
    harness_log_base_dir = os.path.join(BASE_OUTPUT_DIR, "harness_logs", EVALUATION_RUN_TIMESTAMP)
    
    # This specific invocation's ID for subfolder within the main log dir
    if HARNESS_RUN_ID_PREFIX:
        current_harness_invocation_id = f"{HARNESS_RUN_ID_PREFIX}_{safe_model_name}_{experiment_condition_name}_{run_id_suffix}"
    else:
        current_harness_invocation_id = f"{safe_model_name}_{experiment_condition_name}_{run_id_suffix}"
    
    log_dir_for_this_run = os.path.join(harness_log_base_dir, current_harness_invocation_id)
    # For per-task runs, we want to ensure the parent of the specific invocation ID dir exists,
    # but the specific invocation ID dir itself should be clean or newly created by the harness script if it handles it.
    # If harness doesn't create current_harness_invocation_id, we should.
    # The `report_dir` argument to the harness implies it will create subdirectories within it.
    # So, we ensure harness_log_base_dir exists. The harness should handle creation under that.
    # Let's adjust: log_dir_for_this_run IS the --report_dir. The harness will put its outputs (like run_logs) INSIDE this.
    
    # Ensure the base for all reports for this timestamp exists
    os.makedirs(harness_log_base_dir, exist_ok=True) 
    # If the specific invocation's log dir already exists (e.g. from a previous failed attempt for this task),
    # it might be better to clear it to ensure fresh logs for this attempt.
    if os.path.exists(log_dir_for_this_run): 
        logger.warning(f"Report directory {log_dir_for_this_run} already exists. Clearing it for a fresh harness run.")
        shutil.rmtree(log_dir_for_this_run)
    os.makedirs(log_dir_for_this_run, exist_ok=True) # Create it clean

    logger.info(f"Running SWE-bench harness for: {model_name} ({experiment_condition_name}), Invocation ID (Task): {run_id_suffix}")
    logger.info(f"Report directory for this task: {log_dir_for_this_run}")

    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--predictions_path", os.path.abspath(predictions_path),
        "--dataset_name", "princeton-nlp/SWE-bench", 
        "--split", "test", 
        "--report_dir", os.path.abspath(log_dir_for_this_run),
        "--timeout", str(HARNESS_TIMEOUT_PER_TASK),
        "--max_workers", str(HARNESS_MAX_WORKERS),
        "--run_id", current_harness_invocation_id,
    ]
    if torch.backends.mps.is_available():
        cmd.extend(["--namespace", ""])

    logger.info(f"Executing harness command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(cmd, cwd=SWE_BENCH_REPO_PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Define the target path where the copied/renamed aggregate report will be placed for parsing
        target_results_file_for_parsing = os.path.join(log_dir_for_this_run, "instance_results.jsonl")

        # Determine the expected filename of the aggregate JSON report generated by the harness in its CWD
        # Harness uses: <model_slug_from_prediction_file>.<run_id_passed_to_harness>.json
        # model_slug_from_prediction_file is model_name_or_path.replace("/", "__")
        # In our case, model_name_or_path is effectively the 'model_name' argument to this function.
        harness_model_slug = model_name.replace('/', '__').replace(':', '_') 
        expected_aggregate_json_filename = f"{harness_model_slug}.{current_harness_invocation_id}.json"
        source_aggregate_json_path = os.path.join(SWE_BENCH_REPO_PATH, expected_aggregate_json_filename)
        
        if process.returncode == 0:
            logger.info(f"SWE-bench harness process completed with return code 0.")
            if os.path.exists(source_aggregate_json_path):
                logger.info(f"Found harness aggregate report at: {source_aggregate_json_path}")
                try:
                    shutil.copy(source_aggregate_json_path, target_results_file_for_parsing)
                    logger.info(f"Copied aggregate report to {target_results_file_for_parsing} for parsing.")
                    # Remove the original from SWE-bench CWD to avoid clutter if desired, optional
                    # os.remove(source_aggregate_json_path) 
                    logger.info(f"Check for detailed instance logs (e.g., run_logs/) which might be in: {log_dir_for_this_run} (if --report_dir is respected for them) or {SWE_BENCH_REPO_PATH}")
                    return target_results_file_for_parsing
                except Exception as e_copy:
                    logger.error(f"Failed to copy aggregate report from {source_aggregate_json_path} to {target_results_file_for_parsing}: {e_copy}")
                    return None
            else:
                logger.error(f"SWE-bench harness exited successfully (ret: 0), BUT the expected aggregate JSON report was NOT found at {source_aggregate_json_path}.")
                logger.error(f"Please check harness logs. Detailed logs (run_logs/) might be in {log_dir_for_this_run} (if --report_dir worked for those) or within {SWE_BENCH_REPO_PATH}.")
                return None
        else: # process.returncode != 0
            logger.error(f"SWE-bench harness failed with return code: {process.returncode}.")
            logger.error(f"Look for specific error logs. Detailed logs (run_logs/) might be in {log_dir_for_this_run} (if --report_dir worked for those) or within {SWE_BENCH_REPO_PATH}.")
            logger.error(f"Harness STDOUT:\n{stdout}...")
            logger.error(f"Harness STDERR:\n{stderr}...")
            return None
    except Exception as e:
        logger.error(f"An exception occurred while running the harness: {e}")
        return None

def parse_harness_results(results_file_path: str) -> Dict[str, bool]:
    if not results_file_path or not os.path.exists(results_file_path):
        logger.error(f"Cannot parse harness results: file not found at {results_file_path}")
        return {}
    
    instance_results = {}
    try:
        with open(results_file_path, 'r') as f:
            # Load the single JSON object (copied aggregate report)
            report_data = json.load(f)
        
        resolved_ids = set(report_data.get("resolved_ids", []))
        # Use "submitted_ids" as the list of tasks the harness was asked to process from the predictions file.
        # Fallback to "completed_ids" if "submitted_ids" is not present or empty, though "submitted_ids" from your example seems correct.
        tasks_processed_by_harness = report_data.get("submitted_ids")
        if not tasks_processed_by_harness: # Fallback
            tasks_processed_by_harness = report_data.get("completed_ids", [])
            if tasks_processed_by_harness:
                 logger.warning(f"Parsing results from {results_file_path}: 'submitted_ids' was empty/missing, using 'completed_ids' as list of processed tasks.")
            else:
                logger.warning(f"Parsing results from {results_file_path}: Both 'submitted_ids' and 'completed_ids' are empty/missing in the report. No results can be parsed.")
                return {}


        error_ids = set(report_data.get("error_ids", []))
        unresolved_ids = set(report_data.get("unresolved_ids", [])) # Tasks that completed but didn't pass tests

        for instance_id in tasks_processed_by_harness:
            passed = instance_id in resolved_ids
            instance_results[instance_id] = passed
            
            if not passed:
                if instance_id in error_ids:
                    logger.warning(f"Instance {instance_id} from {results_file_path} reported as an 'error' in the harness summary (did not complete evaluation).")
                elif instance_id in unresolved_ids:
                    logger.info(f"Instance {instance_id} from {results_file_path} was completed but 'unresolved' (failed tests).")
                # If it's in tasks_processed_by_harness but not resolved, and not error/unresolved, it's still a fail.
                # This covers cases where it might be in "completed_ids" but not explicitly in "unresolved_ids" or "error_ids"
                # (though ideally, it should be in one of those if not resolved).
                elif instance_id in report_data.get("completed_ids", []): # Check if it at least completed
                     logger.info(f"Instance {instance_id} from {results_file_path} completed but was not resolved and not in error/unresolved lists. Marking as failed.")
                else: # Was submitted but didn't even make it to completed_ids (very unlikely if submit succeeded)
                    logger.warning(f"Instance {instance_id} from {results_file_path} was submitted but not found in completed, resolved, unresolved, or error lists. Marking as failed.")


        if not instance_results and report_data.get("submitted_instances", 0) > 0 and not tasks_processed_by_harness :
             logger.warning(f"Report {results_file_path} indicates {report_data.get('submitted_instances')} submitted instance(s), but no instance_ids could be determined for parsing from 'submitted_ids' or 'completed_ids'. Resulting map will be empty.")

        logger.info(f"Parsed {len(instance_results)} results from the aggregate report: {results_file_path}")
        return instance_results
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from aggregate report {results_file_path}: {e}. File content might not be a valid single JSON object.")
        return {}
    except Exception as e:
        logger.error(f"Error parsing aggregate harness results file {results_file_path}: {e}")
        return {}


# %% [markdown]
#    ## 8.1. Quick Test and Debugging Block
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#    This block runs a single task from the first sequence with the first model to test the pipeline.

# %%
RUN_DEBUG_TEST = False
if RUN_DEBUG_TEST:
    logger.info("\n\n" + "="*20 + " Starting Quick Test and Debugging " + "="*20)
    if not initialized_llms:
        logger.error("No LLMs initialized. Skipping quick test.")
    if not swe_bench_cl_data_sequences:
        logger.error("No sequences loaded. Skipping quick test.")

    test_model_name = list(initialized_llms.keys())[0]
    test_llm_instance = initialized_llms[test_model_name]

    test_sequence_data = swe_bench_cl_data_sequences[0]
    if not test_sequence_data.get("tasks"):
        logger.error(f"First sequence {test_sequence_data.get('id')} has no tasks. Skipping quick test.")

    # Test with memory disabled for simplicity
    test_memory_system = None
    test_memory_enabled = False

    logger.info(f"Quick Test: Model={test_model_name}, Sequence={test_sequence_data['id']}, First Task, Memory Disabled.")

    # --- THIS BLOCK IS HEAVILY MODIFIED FOR PER-TASK ---
    # Get the first task
    test_task_detail = test_sequence_data["tasks"][0]
    test_task_meta = test_task_detail.get("metadata", {})
    test_instance_id = test_task_meta.get("instance_id")
    test_base_commit = test_task_meta.get("base_commit")
    test_repo_name = test_sequence_data["repo"]
    
    test_task_content = test_task_detail.get("task", {})
    test_problem_statement = test_task_content.get("problem_statement")
    test_hints_text = test_task_content.get("hints_text", "No hints provided.")
    test_gold_patch_text = test_task_detail.get("evaluation", {}).get("patch", "")

    if not all([test_instance_id, test_problem_statement, test_repo_name, test_base_commit]):
        logger.error("Quick Test: Critical data missing for the test task. Skipping.")
    else:
        # 1. Generate patch for the single task
        test_retrieved_context_str = "" # Memory disabled for this simple test
        test_text_files_involved_str = "Not available." # Simplified for test
        if test_gold_patch_text:
            try:
                lines = test_gold_patch_text.split('\n')
                test_text_files_involved = list(set([line[len("--- a/"):] for line in lines if line.startswith("--- a/")] + \
                                           [line[len("+++ b/"):] for line in lines if line.startswith("+++ b/") and line[len("+++ b/"):] != "/dev/null"]))
                test_text_files_involved_str = "\n".join(test_text_files_involved) if test_text_files_involved else "No specific files identified."
            except Exception: test_text_files_involved_str = "Error parsing files."

        test_input_data_for_llm = {
            "repo": test_repo_name, "base_commit": test_base_commit, "retrieved_context": test_retrieved_context_str,
            "problem_statement": test_problem_statement, "hints_text": test_hints_text or "None", "text_files": test_text_files_involved_str
        }
        
        generated_patch_raw = ""
        generated_patch_cleaned = ""
        try:
            formatted_prompt_content = prompt_template_str_with_memory.format(**test_input_data_for_llm)
            messages = [HumanMessage(content=formatted_prompt_content)]
            ai_response = test_llm_instance.invoke(messages)
            generated_patch_raw = output_parser.invoke(ai_response)
            generated_patch_cleaned = clean_llm_generated_patch(generated_patch_raw)
            logger.info(f"Quick Test: Generated patch for {test_instance_id}:\n{generated_patch_cleaned}")
        except Exception as e_gen:
            logger.error(f"Quick Test: Error generating patch for {test_instance_id}: {e_gen}")
            generated_patch_cleaned = f"Error: LLM failed. {e_gen}"

        # 2. Save this single prediction to a temporary file
        debug_preds_dir = os.path.join(BASE_OUTPUT_DIR, "debug_predictions", EVALUATION_RUN_TIMESTAMP)
        os.makedirs(debug_preds_dir, exist_ok=True)
        # Make filename unique per task for debug to avoid issues if run multiple times
        temp_debug_predictions_file = os.path.join(debug_preds_dir, f"debug_pred_task_{test_instance_id.replace('/','_')}.jsonl")
        
        with open(temp_debug_predictions_file, "w") as f:
            swe_bench_formatted_pred = {
                "instance_id": test_instance_id,
                "model_name_or_path": test_model_name, # The model identifier for harness
                "model_patch": generated_patch_cleaned
            }
            f.write(json.dumps(swe_bench_formatted_pred) + "\n")
        logger.info(f"Quick Test: Saved single prediction for task {test_instance_id} to {temp_debug_predictions_file}")

        # 3. Run harness for this single prediction
        # Use instance_id in run_id_suffix for unique log dir
        debug_harness_results_file = run_swe_bench_harness(
            temp_debug_predictions_file, 
            test_model_name, 
            "no_memory_debug_condition", 
            run_id_suffix=f"debug_{test_instance_id.replace('/', '_')}" # Unique suffix
        )

        # 4. Parse harness result
        if debug_harness_results_file:
            parsed_debug_results = parse_harness_results(debug_harness_results_file)
            pass_status = parsed_debug_results.get(test_instance_id, False) # Default to False if not found
            logger.info(f"Quick Test: Harness result for task {test_instance_id}: {'PASSED' if pass_status else 'FAILED/ERROR'}")
            
            # 5. (Optional for debug) Memory update
            # if test_memory_system: # (memory is disabled in this test block currently)
            #    test_memory_system.add_experience_to_memory(test_instance_id, test_sequence_data['id'], test_problem_statement, generated_patch_cleaned, pass_status)
            #    logger.info(f"Quick Test: (Simulated) memory update for {test_instance_id} with result: {pass_status}")

        else:
            logger.error(f"Quick Test: Harness run failed or results not found for task {test_instance_id}.")
        
        # Clean up the temporary single prediction file
        # if os.path.exists(temp_debug_predictions_file):
        #     os.remove(temp_debug_predictions_file)

    logger.info("="*20 + " Quick Test Finished " + "="*20 + "\n\n")

# %% [markdown]
#    ## Utility Cell: Backfill Gold Patches into Existing State
#    This cell can be run once to update an existing `eval_state.json` file 
#    to include the `gold_patch` for tasks that were processed before this field was added.
#    Make sure `SWE_BENCH_CL_DATASET_PATH` and `STATE_FILE_PATH` are correctly defined above.

# %%
RUN_BACKFILL_GOLD_PATCHES = True # Set to True to run this utility

if RUN_BACKFILL_GOLD_PATCHES:
    logger.info("Starting utility to backfill gold patches into eval_state.json...")

    # 1. Load the dataset to get gold patches
    if not swe_bench_cl_full_data:
        logger.error("SWE-Bench-CL dataset is not loaded. Cannot backfill gold patches.")
    else:
        gold_patch_map = {}
        for seq in swe_bench_cl_full_data.get("sequences", []):
            for task_detail in seq.get("tasks", []):
                instance_id = task_detail.get("metadata", {}).get("instance_id")
                gold_patch = task_detail.get("evaluation", {}).get("patch")
                if instance_id and gold_patch is not None: # Ensure gold_patch exists, even if empty string
                    gold_patch_map[instance_id] = gold_patch
        
        logger.info(f"Created gold patch map with {len(gold_patch_map)} entries.")

        # 2. Load the current state (or re-load to be sure)
        # current_evaluation_state is already loaded globally, but for safety, one might reload.
        # For this utility, we'll assume the global current_evaluation_state is what we want to modify.
        if not current_evaluation_state or not os.path.exists(STATE_FILE_PATH):
            logger.error(f"Evaluation state not loaded or state file not found at {STATE_FILE_PATH}. Cannot backfill.")
        else:
            updated_count = 0
            missing_in_map_count = 0
            
            task_eval_progress = current_evaluation_state.get("task_eval_progress", {})
            
            for model_name, conditions in task_eval_progress.items():
                for condition_name, instances in conditions.items():
                    # Typically, you'd be interested in "memory_enabled" and "memory_disabled"
                    # if condition_name in ["memory_enabled", "memory_disabled"]: # Or iterate all
                    for instance_id, task_data in instances.items():
                        if isinstance(task_data, dict) and (task_data.get("gold_patch") is None or task_data.get("gold_patch") == ""): # Check if None or empty
                            if instance_id in gold_patch_map:
                                task_data["gold_patch"] = gold_patch_map[instance_id]
                                updated_count += 1
                                logger.debug(f"Added gold patch for {model_name}/{condition_name}/{instance_id}")
                            else:
                                logger.warning(f"Gold patch not found in map for {instance_id} (model: {model_name}, condition: {condition_name}). Skipping.")
                                missing_in_map_count +=1
            
            if updated_count > 0 or missing_in_map_count > 0 : # Save only if changes were made or warnings occurred
                logger.info(f"Backfill complete. Added/updated gold_patch for {updated_count} task entries.")
                if missing_in_map_count > 0:
                    logger.warning(f"Could not find gold patch in the dataset map for {missing_in_map_count} task entries in the state file.")
                
                logger.info("Saving updated state...")
                save_state(current_evaluation_state)
                logger.info("State saved successfully.")
            else:
                logger.info("No task entries in the state file needed gold_patch backfilling.")

    logger.info("Gold patch backfill utility finished.")

# %% [markdown]
#    ## 9. Continual Learning Metrics Calculation Functions

# %%
def calculate_accuracy(task_results_for_sequence: Dict[str, bool]) -> float:
    if not task_results_for_sequence: return 0.0
    return sum(1 for passed in task_results_for_sequence.values() if passed) / len(task_results_for_sequence)

def calculate_aulc(task_results_for_sequence: Dict[str, bool], sequence_task_order: List[str]) -> float:
    if not task_results_for_sequence or not sequence_task_order: return 0.0
    N = len(sequence_task_order); cumulative_aulc_terms = 0
    if N == 0: return 0.0
    for i in range(1, N + 1):
        current_i_tasks = sequence_task_order[:i]
        sum_akk_for_current_i = sum(1 for task_id_k in current_i_tasks if task_results_for_sequence.get(task_id_k, False))
        cumulative_aulc_terms += (1 / i) * sum_akk_for_current_i
    return cumulative_aulc_terms / N

def calculate_forward_transfer(task_results_mem_enabled: Dict[str, bool], task_results_no_mem: Dict[str, bool], sequence_task_order: List[str]) -> float:
    if not task_results_mem_enabled or not task_results_no_mem or not sequence_task_order: return 0.0
    N = len(sequence_task_order)
    if N <= 1: return 0.0
    ft_sum = 0
    for i in range(N - 1):
        task_i_plus_1_id = sequence_task_order[i+1]
        s_mem_i_plus_1 = 1 if task_results_mem_enabled.get(task_i_plus_1_id, False) else 0
        s_no_mem_i_plus_1 = 1 if task_results_no_mem.get(task_i_plus_1_id, False) else 0
        ft_sum += (s_mem_i_plus_1 - s_no_mem_i_plus_1)
    return ft_sum / (N - 1)

def calculate_forgetting(task_results_initial_pass: Dict[str, bool], task_results_final_state: Dict[str, bool], sequence_task_order: List[str]) -> float:
    if not task_results_initial_pass or not sequence_task_order: return 0.0
    N = len(sequence_task_order)
    if N <= 1: return 0.0
    final_state_results_to_use = task_results_final_state if task_results_final_state else task_results_initial_pass
    forgetting_sum = 0
    for j_idx in range(N - 1):
        task_j_id = sequence_task_order[j_idx]
        max_perf_on_task_j = 1 if task_results_initial_pass.get(task_j_id, False) else 0
        final_perf_on_task_j = 1 if final_state_results_to_use.get(task_j_id, False) else 0
        forgetting_sum += (max_perf_on_task_j - final_perf_on_task_j)
    return forgetting_sum / (N - 1) if (N-1) > 0 else 0.0

def calculate_backward_transfer(task_results_initial_pass: Dict[str, bool], task_results_final_state: Dict[str, bool], sequence_task_order: List[str]) -> float:
    if not task_results_initial_pass or not sequence_task_order: return 0.0
    N = len(sequence_task_order)
    if N <= 1: return 0.0
    final_state_results_to_use = task_results_final_state if task_results_final_state else task_results_initial_pass
    bwt_sum = 0
    for i_idx in range(N - 1):
        task_i_id = sequence_task_order[i_idx]
        final_perf_on_task_i = 1 if final_state_results_to_use.get(task_i_id, False) else 0
        initial_perf_on_task_i = 1 if task_results_initial_pass.get(task_i_id, False) else 0
        bwt_sum += (final_perf_on_task_i - initial_perf_on_task_i)
    return bwt_sum / (N - 1) if (N-1) > 0 else 0.0

def calculate_cl_score(acc: float, f: float, ft: float, bwt: float, aulc: float, weights: Dict) -> float:
    return (acc - weights["lambda_F"] * f + weights["lambda_FT"] * ft + weights["lambda_BWT"] * bwt + weights["lambda_AULC"] * aulc)

def calculate_tool_use_efficiency(harness_log_dir: str) -> float:
    logger.warning("Tool Use Efficiency (TUE) calculation is a placeholder.")
    return 0.0


# %% [markdown]
#    ## 10. Main Experiment Orchestration

# %%
# Load overall results list from state for appending
overall_results_summary_list_from_state = current_evaluation_state.get("overall_results_list", [])

# This dictionary will map (model_name, sequence_id) to a list of already processed metric dicts
# to avoid re-adding if script is resumed after metric calculation for some sequences.
processed_metrics_tracker = {}
for item in overall_results_summary_list_from_state:
    key = (item["model_name"], item["sequence_id"])
    if key not in processed_metrics_tracker:
        processed_metrics_tracker[key] = []
    processed_metrics_tracker[key].append(item)


for model_name, llm_instance in tqdm(initialized_llms.items(), desc="Models"):
    logger.info(f"\n\n{'='*20} Orchestrating for Model: {model_name} {'='*20}")

    # Ensure model entry exists in state dictionaries
    current_evaluation_state.setdefault("task_eval_progress", {}).setdefault(model_name, {})
    current_evaluation_state["parsed_harness_results_data"].setdefault(model_name, {})


    # Store results for this model across conditions (loaded from state if available)
    # These will be populated task-by-task now.
    model_results_no_mem_from_state = current_evaluation_state["parsed_harness_results_data"][model_name].get("memory_disabled", {})
    model_results_mem_enabled_from_state = current_evaluation_state["parsed_harness_results_data"][model_name].get("memory_enabled", {})
    
    for condition_name, condition_params in tqdm(EXPERIMENT_CONDITIONS.items(), desc=f"Conditions for {model_name}", leave=False):
        is_memory_enabled_for_run = condition_params["memory_enabled"]
        logger.info(f"\n--- Running Condition: {condition_name} (Memory: {is_memory_enabled_for_run}) for Model: {model_name} ---")

        current_evaluation_state["task_eval_progress"][model_name].setdefault(condition_name, {})
        current_evaluation_state["parsed_harness_results_data"][model_name].setdefault(condition_name, {})

        # Initialize memory system for the current condition
        current_memory_system = None
        if is_memory_enabled_for_run and active_embedding_model:
            semantic_mem_instance = SemanticMemory(active_embedding_model, EMBEDDING_MODEL_CONFIG["num_retrieved_memories"])
            current_memory_system = MemorySystem(semantic_mem_instance, EMBEDDING_MODEL_CONFIG["max_context_tokens_for_retrieval"], tokenizer_for_counting)
            current_memory_system.clear_memory() # Clear memory at the start of each condition
            # Repopulate memory from historical successes/failures *within this condition* if resuming
            # This requires iterating through already processed tasks *in order* for this condition
            # and adding them to memory if they were evaluated.
            logger.info(f"Initializing memory for {model_name}/{condition_name}. Attempting to populate from prior tasks in this condition run.")
            
            # Create a list of all tasks in their original order for this condition
            ordered_tasks_for_memory_repopulation = []
            for seq_data_mem_repop in swe_bench_cl_data_sequences:
                seq_id_mem_repop = seq_data_mem_repop["id"]
                tasks_in_seq_mem_repop = seq_data_mem_repop.get("tasks", [])
                if TASKS_PER_SEQUENCE_LIMIT is not None:
                    tasks_in_seq_mem_repop = tasks_in_seq_mem_repop[:TASKS_PER_SEQUENCE_LIMIT]
                for task_detail_mem_repop in tasks_in_seq_mem_repop:
                    instance_id_mem_repop = task_detail_mem_repop["metadata"]["instance_id"]
                    ordered_tasks_for_memory_repopulation.append({
                        "instance_id": instance_id_mem_repop,
                        "sequence_id": seq_id_mem_repop,
                        "problem_statement": task_detail_mem_repop["task"]["problem_statement"]
                        # "patch" and "harness_result" will come from task_eval_progress
                    })
            
            repopulated_memory_count = 0
            for task_to_repop in ordered_tasks_for_memory_repopulation:
                task_prog = current_evaluation_state["task_eval_progress"][model_name][condition_name].get(task_to_repop["instance_id"])
                if task_prog and "model_patch" in task_prog and "harness_result" in task_prog:
                    current_memory_system.add_experience_to_memory(
                        task_to_repop["instance_id"],
                        task_to_repop["sequence_id"],
                        task_to_repop["problem_statement"], # Need original problem statement
                        task_prog["model_patch"],
                        task_prog["harness_result"]
                    )
                    repopulated_memory_count += 1
            if repopulated_memory_count > 0:
                logger.info(f"Repopulated memory with {repopulated_memory_count} experiences for {model_name}/{condition_name}.")

        elif is_memory_enabled_for_run and not active_embedding_model:
            logger.warning(f"Memory for {model_name}/{condition_name} requested, but no embedding model. Running effectively memory-disabled for RAG.")

        # PER-TASK PREDICTION, EVALUATION, AND MEMORY UPDATE LOOP
        for sequence_data in tqdm(swe_bench_cl_data_sequences, desc=f"Sequences for {model_name}/{condition_name}", leave=False):
            sequence_id = sequence_data["id"]
            repo_name = sequence_data["repo"]
            
            current_evaluation_state["parsed_harness_results_data"][model_name][condition_name].setdefault(sequence_id, {})

            tasks_to_process_in_sequence = sequence_data.get("tasks", [])
            if TASKS_PER_SEQUENCE_LIMIT is not None:
                tasks_to_process_in_sequence = tasks_to_process_in_sequence[:TASKS_PER_SEQUENCE_LIMIT]

            # If memory is per-sequence (rather than per-condition), clear it here.
            # Current setup: memory is per-condition, initialized and cleared once above.
            # If per-sequence:
            # if is_memory_enabled_for_run and current_memory_system:
            #     current_memory_system.clear_memory()
            #     logger.debug(f"Memory cleared for new sequence {sequence_id} under {model_name}/{condition_name}")


            for task_idx, task_detail in tqdm(enumerate(tasks_to_process_in_sequence), desc=f"Tasks in {sequence_id}", total=len(tasks_to_process_in_sequence), leave=False):
                task_meta = task_detail.get("metadata", {})
                instance_id = task_meta.get("instance_id")

                # --- Check if task already processed (prediction + evaluation) ---
                task_progress_entry = current_evaluation_state["task_eval_progress"][model_name][condition_name].get(instance_id)
                if task_progress_entry: # Check if entry exists
                    logger.info(f"Skipping task {instance_id} in {sequence_id} ({model_name}/{condition_name}), already processed and evaluated.")
                    # Ensure its result is in parsed_harness_results_data if resuming after metrics
                    current_evaluation_state["parsed_harness_results_data"][model_name][condition_name][sequence_id][instance_id] = task_progress_entry.get("harness_result", False)
                    continue
                
                base_commit = task_meta.get("base_commit")
                task_content = task_detail.get("task", {})
                problem_statement = task_content.get("problem_statement")
                hints_text = task_content.get("hints_text", "No hints provided.")
                gold_patch_text = task_detail.get("evaluation", {}).get("patch", "") # For hints AND NOW FOR STORING

                if not all([instance_id, problem_statement, repo_name, base_commit]):
                    logger.warning(f"Skipping task in {sequence_id} ({instance_id or 'Unknown ID'}) due to missing critical data.")
                    continue

                # --- 1. Retrieve context from memory (if enabled) ---
                retrieved_context_str = ""
                if is_memory_enabled_for_run and current_memory_system:
                    current_task_query = f"Problem: {problem_statement}\nHints: {hints_text}"
                    retrieved_context_str = current_memory_system.get_relevant_context_for_prompt(
                        current_task_query, sequence_id, EMBEDDING_MODEL_CONFIG["num_retrieved_memories"]
                    )
                
                text_files_involved_str = "Not available."
                if gold_patch_text: # Use gold patch only for file hints, not for patch content
                    try:
                        lines = gold_patch_text.split('\n')
                        text_files_involved = list(set([line[len("--- a/"):] for line in lines if line.startswith("--- a/")] + \
                                                       [line[len("+++ b/"):] for line in lines if line.startswith("+++ b/") and line[len("+++ b/"):] != "/dev/null"]))
                        text_files_involved_str = "\n".join(text_files_involved) if text_files_involved else "No specific files identified."
                    except Exception: text_files_involved_str = "Error parsing files."

                input_data_for_llm = {
                    "repo": repo_name, "base_commit": base_commit, "retrieved_context": retrieved_context_str,
                    "problem_statement": problem_statement, "hints_text": hints_text or "None", "text_files": text_files_involved_str
                }
                
                # --- 2. Generate Patch ---
                generated_patch_text_cleaned = ""
                raw_llm_output_for_state = None
                logger.debug(f"Generating patch for Task: {instance_id} ({model_name}/{condition_name})")
                try:
                    start_time_task_gen = time.time()
                    formatted_prompt_content = prompt_template_str_with_memory.format(**input_data_for_llm)
                    messages = [HumanMessage(content=formatted_prompt_content)]
                    
                    ai_response = llm_instance.invoke(messages)
                    raw_llm_output_for_state = output_parser.invoke(ai_response) # Save raw output
                    generated_patch_text_cleaned = clean_llm_generated_patch(raw_llm_output_for_state)
                    
                    duration_task_gen = time.time() - start_time_task_gen
                    logger.debug(f"LLM generated patch for {instance_id} in {duration_task_gen:.2f}s.")
                    logger.info(f"Generated patch for {instance_id} (cleaned):\n{generated_patch_text_cleaned[:500]}...") # Log snippet
                except Exception as e_gen:
                    logger.error(f"Error generating patch for {instance_id}: {e_gen}")
                    generated_patch_text_cleaned = f"Error: LLM failed. {e_gen}" # Store error as patch
                    raw_llm_output_for_state = f"Error: LLM failed. {e_gen}"


                # --- 3. Save Single Prediction to Temporary File ---
                temp_preds_dir = os.path.join(BASE_OUTPUT_DIR, "temp_predictions", EVALUATION_RUN_TIMESTAMP, model_name.replace("/","_"), condition_name)
                os.makedirs(temp_preds_dir, exist_ok=True)
                temp_prediction_file_for_task = os.path.join(temp_preds_dir, f"pred_task_{instance_id.replace('/','_')}.jsonl")
                
                with open(temp_prediction_file_for_task, "w") as f_task_pred:
                    pred_to_write = {
                        "instance_id": instance_id,
                        "model_patch": generated_patch_text_cleaned,
                        "model_name_or_path": model_name # Harness needs this
                    }
                    f_task_pred.write(json.dumps(pred_to_write) + "\n")

                # --- 4. Run Harness for this Single Task ---
                task_harness_result_file = None
                task_pass_status = False # Default to False
                if os.path.exists(temp_prediction_file_for_task):
                    logger.info(f"Running harness for single task {instance_id} using {temp_prediction_file_for_task}")
                    # Ensure run_id_suffix is unique for the task to avoid log collision
                    task_run_id_suffix = f"{instance_id.replace('/', '_')}" 
                    task_harness_result_file = run_swe_bench_harness(
                        temp_prediction_file_for_task, model_name, condition_name, task_run_id_suffix
                    )
                else:
                    logger.error(f"Temporary prediction file for task {instance_id} not found. Cannot run harness.")

                # --- 5. Parse Harness Result for this Task ---
                if task_harness_result_file:
                    parsed_single_task_result = parse_harness_results(task_harness_result_file)
                    task_pass_status = parsed_single_task_result.get(instance_id, False)
                    logger.info(f"Harness result for task {instance_id}: {'PASSED' if task_pass_status else 'FAILED/ERROR'}")
                else:
                    logger.error(f"Harness run failed or no result file for task {instance_id}. Assuming failure for this task.")
                    task_pass_status = False # Explicitly set as False

                # --- 6. Update Semantic Memory (if enabled) ---
                if is_memory_enabled_for_run and current_memory_system:
                    current_memory_system.add_experience_to_memory(
                        instance_id, sequence_id, problem_statement, generated_patch_text_cleaned, task_pass_status
                    )
                    logger.debug(f"Updated memory with experience from task {instance_id}, result: {task_pass_status}")

                # --- 7. Store Task Outcome in State ---
                current_evaluation_state["task_eval_progress"][model_name][condition_name][instance_id] = {
                    "model_patch": generated_patch_text_cleaned,
                    "harness_result": task_pass_status,
                    "raw_llm_output": raw_llm_output_for_state, # Store the raw output
                    "gold_patch": gold_patch_text, # Store the gold patch
                    "timestamp": time.strftime("%Y%m%d-%H%M%S")
                }
                current_evaluation_state["parsed_harness_results_data"][model_name][condition_name].setdefault(sequence_id, {})[instance_id] = task_pass_status
                
                # --- 8. Save Overall State ---
                save_state(current_evaluation_state)
                logger.debug(f"State saved after processing task {instance_id}")

                # --- 9. Clean up temporary prediction file for the task ---
                if os.path.exists(temp_prediction_file_for_task):
                    try:
                        os.remove(temp_prediction_file_for_task)
                    except Exception as e_remove:
                        logger.warning(f"Could not remove temporary prediction file {temp_prediction_file_for_task}: {e_remove}")
            
            # End of tasks in a sequence
            # If memory is per-sequence, clear it here
            # if is_memory_enabled_for_run and current_memory_system and MEMORY_SCOPE == "sequence": # (Conceptual MEMORY_SCOPE)
            #    current_memory_system.clear_memory()


        # --- (End of all sequences for a condition) ---
        # The old logic for writing a single predictions file for the whole condition is no longer needed.
        # Also, the single harness run for the whole condition is replaced by per-task runs.
        # Parsing of a single large harness result file is also replaced.
        
        # Update model_results_no_mem or model_results_mem_enabled for immediate use in metric calculation if needed
        # This is now implicitly handled by populating current_evaluation_state["parsed_harness_results_data"] directly
        if is_memory_enabled_for_run:
            model_results_mem_enabled_from_state = current_evaluation_state["parsed_harness_results_data"][model_name].get(condition_name, {})
        else:
            model_results_no_mem_from_state = current_evaluation_state["parsed_harness_results_data"][model_name].get(condition_name, {})


    # --- 4. Calculate CL Metrics for the current model (after both conditions are processed) ---
    logger.info(f"\n--- Calculating CL Metrics for Model: {model_name} ---")
    
    # Use the results populated task-by-task during the run
    model_results_no_mem = current_evaluation_state["parsed_harness_results_data"][model_name].get("memory_disabled", {})
    model_results_mem_enabled = current_evaluation_state["parsed_harness_results_data"][model_name].get("memory_enabled", {})

    if not model_results_no_mem and not model_results_mem_enabled:
        logger.warning(f"No harness results available for model {model_name} under any condition. Skipping metric calculation.")
        continue

    # Determine sequence task orders once
    sequence_task_orders_for_metrics = {}
    for s_data in swe_bench_cl_data_sequences:
        s_id = s_data["id"]
        s_tasks = [t["metadata"]["instance_id"] for t in s_data.get("tasks", [])]
        if TASKS_PER_SEQUENCE_LIMIT: s_tasks = s_tasks[:TASKS_PER_SEQUENCE_LIMIT]
        sequence_task_orders_for_metrics[s_id] = s_tasks

    for seq_id in tqdm(sequence_task_orders_for_metrics.keys(), desc=f"Sequences (Metrics) for {model_name}", leave=False):
        # Check if metrics for this model/sequence are already in overall_results_list_from_state
        # This is a simple check; more robust would be to check based on a unique run ID for metrics.
        # For now, if (model_name, seq_id) is in processed_metrics_tracker, assume it's done.
        if (model_name, seq_id) in processed_metrics_tracker:
            logger.info(f"Metrics for {model_name}/{seq_id} already found in state. Skipping recalculation.")
            # Ensure these existing metrics are part of the final summary by keeping overall_results_summary_list_from_state as is.
            continue 

        logger.info(f"Calculating metrics for Sequence: {seq_id} of Model: {model_name}")
        seq_tasks_order = sequence_task_orders_for_metrics[seq_id]
        
        seq_res_no_mem = model_results_no_mem.get(seq_id, {})
        seq_res_mem = model_results_mem_enabled.get(seq_id, {})
        
        # If one condition failed to produce results, use an empty dict for it
        if not seq_res_no_mem and (model_results_no_mem or model_results_mem_enabled): # if model_results_no_mem is globally empty, this is fine
            logger.warning(f"No-memory results missing for {model_name}/{seq_id}. FT might be inaccurate.")
            seq_res_no_mem = {task_id: False for task_id in seq_tasks_order}
        if not seq_res_mem and (model_results_no_mem or model_results_mem_enabled):
            logger.warning(f"Memory-enabled results missing for {model_name}/{seq_id}. Some metrics might be zero.")
            seq_res_mem = {task_id: False for task_id in seq_tasks_order}

        acc_no_mem = calculate_accuracy(seq_res_no_mem)
        aulc_no_mem = calculate_aulc(seq_res_no_mem, seq_tasks_order)
        f_no_mem = calculate_forgetting(seq_res_no_mem, seq_res_no_mem, seq_tasks_order)
        bwt_no_mem = calculate_backward_transfer(seq_res_no_mem, seq_res_no_mem, seq_tasks_order)
        
        acc_mem = calculate_accuracy(seq_res_mem)
        aulc_mem = calculate_aulc(seq_res_mem, seq_tasks_order)
        ft_mem = calculate_forward_transfer(seq_res_mem, seq_res_no_mem, seq_tasks_order)
        f_mem = calculate_forgetting(seq_res_mem, seq_res_mem, seq_tasks_order)
        bwt_mem = calculate_backward_transfer(seq_res_mem, seq_res_mem, seq_tasks_order)

        cl_score_no_mem = calculate_cl_score(acc_no_mem, f_no_mem, 0, bwt_no_mem, aulc_no_mem, CL_SCORE_WEIGHTS)
        cl_score_mem = calculate_cl_score(acc_mem, f_mem, ft_mem, bwt_mem, aulc_mem, CL_SCORE_WEIGHTS)
        
        tue_no_mem, tue_mem = 0.0, 0.0 # Placeholder

        logger.info(f"  {model_name}/{seq_id} - Memory Disabled: ACC={acc_no_mem:.3f}, AULC={aulc_no_mem:.3f}, F={f_no_mem:.3f}*, BWT={bwt_no_mem:.3f}*, CLS={cl_score_no_mem:.3f}")
        logger.info(f"  {model_name}/{seq_id} - Memory Enabled:  ACC={acc_mem:.3f}, AULC={aulc_mem:.3f}, FT={ft_mem:.3f}, F={f_mem:.3f}*, BWT={bwt_mem:.3f}*, CLS={cl_score_mem:.3f}")

        current_metric_entry = {
            "model_name": model_name, "sequence_id": seq_id, "eval_timestamp": EVALUATION_RUN_TIMESTAMP,
            "acc_no_mem": acc_no_mem, "aulc_no_mem": aulc_no_mem, "f_no_mem": f_no_mem, "bwt_no_mem": bwt_no_mem, "cl_score_no_mem": cl_score_no_mem, "tue_no_mem": tue_no_mem,
            "acc_mem": acc_mem, "aulc_mem": aulc_mem, "ft_mem": ft_mem, "f_mem": f_mem, "bwt_mem": bwt_mem, "cl_score_mem": cl_score_mem, "tue_mem": tue_mem,
        }
        # Add to overall_results_summary_list_from_state (which is current_evaluation_state["overall_results_list"])
        # Avoid duplicates if resuming after this point
        found_in_state_list = False
        for idx, item in enumerate(current_evaluation_state["overall_results_list"]):
            if item["model_name"] == model_name and item["sequence_id"] == seq_id and item.get("eval_timestamp") == EVALUATION_RUN_TIMESTAMP:
                current_evaluation_state["overall_results_list"][idx] = current_metric_entry # Update if exists for this run
                found_in_state_list = True
                break
        if not found_in_state_list:
            current_evaluation_state["overall_results_list"].append(current_metric_entry)
        
        save_state(current_evaluation_state) # Save state after each sequence's metrics are calculated

    logger.warning("* F and BWT are calculated assuming no re-evaluation of tasks with final memory state. True values require a re-evaluation phase.")

# %% [markdown]
#    ## 11. Final Results Summary

# %%
if current_evaluation_state["overall_results_list"]:
    # Use the list from the state, which has been appended to during the run
    results_df = pd.DataFrame(current_evaluation_state["overall_results_list"])
    logger.info("\n\n======= Overall Continual Learning Metrics Summary =======")
    print(results_df.to_string())
    
    summary_file_path = os.path.join(BASE_OUTPUT_DIR, f"cl_evaluation_summary_{EVALUATION_RUN_TIMESTAMP}.csv")
    results_df.to_csv(summary_file_path, index=False)
    logger.info(f"Overall summary saved to: {summary_file_path}")
    current_evaluation_state["overall_results_summary_df_path"] = summary_file_path
    save_state(current_evaluation_state) # Save final state with summary path
else:
    logger.info("No results to summarize.")

logger.info(f"\nEvaluation run {EVALUATION_RUN_TIMESTAMP} complete. Outputs are in {BASE_OUTPUT_DIR}")
logger.info(f"State file is at: {STATE_FILE_PATH}")




# %% [markdown]
#    --- End of Notebook ---
