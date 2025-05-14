# %% [markdown]
# # Analysis of SWE-Bench-CL Evaluation Results
#
# This notebook analyzes the results stored in `eval_state.json` from the
# `eval_v1/eval_procedure.py` script. It focuses on comparing the performance
# of models under different conditions, primarily "memory_enabled" vs. "memory_disabled".
#
# Key analyses include:
# - Pass rate comparisons.
# - Patch similarity metrics (Levenshtein distance) between model-generated patches and gold patches.
# - Visualizations of these comparisons.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk # For Levenshtein distance and tokenization

# Download punkt tokenizer data if not already present (needed for word_tokenize)
# You only need to run this once.
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# %% [markdown]
# ## 2. Configuration
#
# Update these paths to point to your files.

# %%
# Path to the evaluation state file
# Assumes this script is in the project root, and eval_state.json is in eval_v1/eval_results/
STATE_FILE_PATH = "eval_v1/eval_results/eval_state.json"

# Optional: Path to the original dataset for additional metadata (e.g., sequence task order for learning curves)
DATASET_FILE_PATH = "data/SWE-Bench-CL-Curriculum.json"
LOAD_FULL_DATASET = True # Set to False if you don't need full dataset context

# Models and conditions to focus on (if None, will try to process all found)
MODELS_TO_ANALYZE = None # Example: ["google/gemini-2.0-flash"] or None for all
CONDITIONS_TO_COMPARE = ["memory_enabled", "memory_disabled"]

# %% [markdown]
# ## 3. Load Data

# %%
if not os.path.exists(STATE_FILE_PATH):
    logger.error(f"State file not found at: {STATE_FILE_PATH}")
    raise FileNotFoundError(f"State file not found at: {STATE_FILE_PATH}")

with open(STATE_FILE_PATH, 'r') as f:
    eval_state = json.load(f)

logger.info(f"Successfully loaded evaluation state from {STATE_FILE_PATH}")
evaluation_run_timestamp = eval_state.get("evaluation_run_timestamp", "unknown_run")

swe_bench_cl_full_data = None
if LOAD_FULL_DATASET and os.path.exists(DATASET_FILE_PATH):
    try:
        with open(DATASET_FILE_PATH, 'r') as f:
            swe_bench_cl_full_data = json.load(f)
        logger.info(f"Successfully loaded full dataset from {DATASET_FILE_PATH}")
    except Exception as e:
        logger.warning(f"Could not load full dataset from {DATASET_FILE_PATH}: {e}. Some contextual analyses might be limited.")
elif LOAD_FULL_DATASET:
    logger.warning(f"Dataset file not found at {DATASET_FILE_PATH}. Some contextual analyses might be limited.")

# %% [markdown]
# ## 4. Data Preparation
#
# Extract task evaluation progress and structure it into a Pandas DataFrame for easier analysis.

# %%
all_task_results = []
task_eval_progress = eval_state.get("task_eval_progress", {})

if not task_eval_progress:
    logger.warning("No 'task_eval_progress' found in the state file. Cannot proceed with analysis.")
else:
    for model_name, conditions_data in task_eval_progress.items():
        if MODELS_TO_ANALYZE and model_name not in MODELS_TO_ANALYZE:
            continue
        for condition_name, instances_data in conditions_data.items():
            if condition_name not in CONDITIONS_TO_COMPARE: # Focus on specified conditions
                continue
            for instance_id, task_data in instances_data.items():
                if isinstance(task_data, dict): # Ensure task_data is a dictionary
                    all_task_results.append({
                        "model_name": model_name,
                        "condition": condition_name,
                        "instance_id": instance_id,
                        "harness_result": task_data.get("harness_result", False),
                        "model_patch": task_data.get("model_patch", ""),
                        "gold_patch": task_data.get("gold_patch", ""),
                        "raw_llm_output": task_data.get("raw_llm_output", ""),
                        "timestamp": task_data.get("timestamp")
                    })
                else:
                    logger.warning(f"Skipping malformed task_data for {model_name}/{condition_name}/{instance_id}")

if not all_task_results:
    logger.error("No valid task results found to analyze after filtering. Exiting.")
    # exit() # Or handle more gracefully

results_df = pd.DataFrame(all_task_results)
logger.info(f"Created DataFrame with {len(results_df)} task results.")
if not results_df.empty:
    logger.info(f"DataFrame head:\n{results_df.head()}")
    logger.info(f"\nModels found: {results_df['model_name'].unique()}")
    logger.info(f"Conditions found: {results_df['condition'].unique()}")
else:
    logger.error("Results DataFrame is empty. Please check your state file and configuration.")


# Add sequence information if dataset is loaded
# This helps in per-sequence analysis and learning curves
if swe_bench_cl_full_data and not results_df.empty:
    instance_to_sequence_map = {}
    instance_to_task_index_map = {}
    sequence_task_orders = {}

    for seq_idx, seq_data in enumerate(swe_bench_cl_full_data.get("sequences", [])):
        seq_id = seq_data.get("id", f"unknown_sequence_{seq_idx}")
        sequence_task_orders[seq_id] = []
        for task_idx, task_detail in enumerate(seq_data.get("tasks", [])):
            instance_id = task_detail.get("metadata", {}).get("instance_id")
            if instance_id:
                instance_to_sequence_map[instance_id] = seq_id
                instance_to_task_index_map[instance_id] = task_idx # 0-indexed task number in sequence
                sequence_task_orders[seq_id].append(instance_id)

    results_df["sequence_id"] = results_df["instance_id"].map(instance_to_sequence_map)
    results_df["task_index_in_sequence"] = results_df["instance_id"].map(instance_to_task_index_map)
    logger.info("Added sequence ID and task index to DataFrame.")
    logger.info(f"DataFrame head with sequence info:\n{results_df.head()}")


# %% [markdown]
# ## 5. Calculate Patch Similarity Metrics
#
# We'll calculate Levenshtein distance at token and character levels.
# Patches starting with "Error: LLM failed." will be assigned a very high distance.

# %%
def calculate_levenshtein_distance(s1: str, s2: str, level="char") -> int:
    """Calculates Levenshtein distance.
    Level can be 'char' or 'token'.
    Returns -1 if inputs are invalid (e.g., not strings).
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        return -1 # Or a very large number to indicate error/max distance

    # Handle cases where patches indicate LLM failure
    if s1.startswith("Error: LLM failed.") or s2.startswith("Error: LLM failed."):
        # Assign a very high distance if one is an error, scaled by typical patch length
        # Or could return a specific sentinel value if preferred for filtering later
        return max(len(s1), len(s2)) * 2 + 1000 # Arbitrarily large

    if level == "char":
        return nltk.edit_distance(s1, s2)
    elif level == "token":
        # Simple whitespace tokenization for patches
        tokens1 = s1.split()
        tokens2 = s2.split()
        # Using NLTK's edit_distance which works on sequences (lists of tokens here)
        return nltk.edit_distance(tokens1, tokens2)
    else:
        raise ValueError("Level must be 'char' or 'token'")

if not results_df.empty:
    logger.info("Calculating patch similarity metrics (this may take a moment)...")
    # Ensure patches are strings
    results_df["model_patch"] = results_df["model_patch"].astype(str)
    results_df["gold_patch"] = results_df["gold_patch"].astype(str)

    tqdm.pandas(desc="Char Levenshtein")
    results_df["char_levenshtein"] = results_df.progress_apply(
        lambda row: calculate_levenshtein_distance(row["model_patch"], row["gold_patch"], level="char"), axis=1
    )
    tqdm.pandas(desc="Token Levenshtein")
    results_df["token_levenshtein"] = results_df.progress_apply(
        lambda row: calculate_levenshtein_distance(row["model_patch"], row["gold_patch"], level="token"), axis=1
    )
    logger.info("Finished calculating Levenshtein distances.")
    logger.info(f"DataFrame with similarity metrics:\n{results_df[['instance_id', 'char_levenshtein', 'token_levenshtein']].head()}")
else:
    logger.warning("Results DataFrame is empty. Skipping similarity metric calculation.")


# %% [markdown]
# ## 6. Aggregate Results and Comparisons

# %% [markdown]
# ### 6.1. Pass Rate Analysis

# %%
if not results_df.empty:
    logger.info("\n--- Pass Rate Analysis ---")

    # Overall pass rates per model and condition
    overall_pass_rates = results_df.groupby(["model_name", "condition"])["harness_result"].agg(['mean', 'sum', 'count']).rename(
        columns={'mean': 'pass_rate', 'sum': 'passed_count', 'count': 'total_tasks'}
    ).reset_index()
    logger.info(f"\nOverall Pass Rates:\n{overall_pass_rates.to_string()}")

    # Pass rates per sequence (if sequence_id is available)
    if "sequence_id" in results_df.columns:
        sequence_pass_rates = results_df.groupby(["model_name", "condition", "sequence_id"])["harness_result"].agg(
            ['mean', 'sum', 'count']
        ).rename(
            columns={'mean': 'pass_rate', 'sum': 'passed_count', 'count': 'total_tasks'}
        ).reset_index()
        logger.info(f"\nPass Rates per Sequence:\n{sequence_pass_rates.to_string()}")
    else:
        logger.warning("Sequence ID not available in DataFrame, skipping per-sequence pass rate analysis.")
else:
    logger.warning("Results DataFrame is empty. Skipping pass rate analysis.")


# %% [markdown]
# ### 6.2. Patch Similarity Analysis
#
# We'll look at average similarity scores, especially for tasks that PASSED vs. FAILED.
# Lower Levenshtein distance is better.

# %%
if not results_df.empty and "char_levenshtein" in results_df.columns:
    logger.info("\n--- Patch Similarity Analysis (Levenshtein Distance) ---")

    # Filter out rows where distance calculation might have failed (e.g., returned -1 or the high error value)
    # or where gold_patch was empty (can skew averages if model_patch is also empty)
    valid_similarity_df = results_df[
        (results_df["char_levenshtein"] >= 0) & (results_df["char_levenshtein"] < 1000) & # Exclude error sentinels
        (results_df["token_levenshtein"] >= 0) & (results_df["token_levenshtein"] < 1000) &
        (results_df["gold_patch"].str.len() > 0) # Only consider if gold patch exists
    ].copy()


    if not valid_similarity_df.empty:
        avg_similarity_scores = valid_similarity_df.groupby(["model_name", "condition", "harness_result"])[
            ["char_levenshtein", "token_levenshtein"]
        ].mean().reset_index()
        logger.info(f"\nAverage Levenshtein Distances (lower is better):\n{avg_similarity_scores.to_string()}")

        avg_similarity_overall = valid_similarity_df.groupby(["model_name", "condition"])[
            ["char_levenshtein", "token_levenshtein"]
        ].mean().reset_index()
        logger.info(f"\nOverall Average Levenshtein Distances:\n{avg_similarity_overall.to_string()}")
    else:
        logger.warning("No valid similarity scores to analyze after filtering for errors or empty gold patches.")
else:
    logger.warning("Similarity metrics not calculated or DataFrame empty. Skipping patch similarity analysis.")


# %% [markdown]
# ## 7. Visualizations
#
# Using Seaborn for aesthetically pleasing plots.

# %%
sns.set_theme(style="whitegrid")
FIG_OUTPUT_DIR = f"analysis_plots_{evaluation_run_timestamp}"
os.makedirs(FIG_OUTPUT_DIR, exist_ok=True)
logger.info(f"Saving plots to {FIG_OUTPUT_DIR}/")

# %% [markdown]
# ### 7.1. Pass Rate Plots

# %%
if not results_df.empty and not overall_pass_rates.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=overall_pass_rates, x="model_name", y="pass_rate", hue="condition")
    plt.title("Overall Pass Rate by Model and Memory Condition")
    plt.ylabel("Pass Rate (Fraction)")
    plt.xlabel("Model Name")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUTPUT_DIR, "overall_pass_rates.png"))
    logger.info(f"Saved overall_pass_rates.png")
    plt.show()

    if "sequence_id" in results_df.columns and not sequence_pass_rates.empty:
        # Plot per-sequence pass rates for each model
        for model in results_df['model_name'].unique():
            model_seq_pass_rates = sequence_pass_rates[sequence_pass_rates['model_name'] == model]
            if not model_seq_pass_rates.empty:
                plt.figure(figsize=(12, 7))
                sns.barplot(data=model_seq_pass_rates, x="sequence_id", y="pass_rate", hue="condition")
                plt.title(f"Pass Rate per Sequence for Model: {model}\n(Comparison by Memory Condition)")
                plt.ylabel("Pass Rate (Fraction)")
                plt.xlabel("Sequence ID")
                plt.xticks(rotation=75, ha="right")
                plt.legend(title="Memory Condition")
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_OUTPUT_DIR, f"sequence_pass_rates_{model.replace('/','_')}.png"))
                logger.info(f"Saved sequence_pass_rates_{model.replace('/','_')}.png")
                plt.show()
else:
    logger.warning("Pass rate data not available for plotting.")

# %% [markdown]
# #### Learning Curves (Cumulative Pass Rate)
# This requires task order within sequences.

# %%
if "sequence_id" in results_df.columns and "task_index_in_sequence" in results_df.columns and not results_df.empty:
    logger.info("Generating learning curves...")
    # Sort by task index to ensure correct cumulative sum
    learning_df = results_df.sort_values(by=["model_name", "condition", "sequence_id", "task_index_in_sequence"])
    learning_df["cumulative_passed"] = learning_df.groupby(
        ["model_name", "condition", "sequence_id"]
    )["harness_result"].cumsum()
    learning_df["cumulative_task_count"] = learning_df.groupby(
        ["model_name", "condition", "sequence_id"]
    ).cumcount() + 1
    learning_df["cumulative_pass_rate"] = learning_df["cumulative_passed"] / learning_df["cumulative_task_count"]

    unique_models = learning_df["model_name"].unique()
    unique_sequences = learning_df["sequence_id"].dropna().unique()

    for model in unique_models:
        for seq_id in unique_sequences:
            plot_data = learning_df[(learning_df["model_name"] == model) & (learning_df["sequence_id"] == seq_id)]
            if not plot_data.empty:
                plt.figure(figsize=(12, 7))
                sns.lineplot(data=plot_data, x="cumulative_task_count", y="cumulative_pass_rate", hue="condition", marker="o")
                plt.title(f"Learning Curve for {model} on Sequence {seq_id}")
                plt.xlabel("Number of Tasks Seen in Sequence")
                plt.ylabel("Cumulative Pass Rate")
                plt.ylim(0, 1.05)
                plt.grid(True, which="both", ls="-", alpha=0.5)
                plt.legend(title="Memory Condition")
                plt.tight_layout()
                plt.savefig(os.path.join(FIG_OUTPUT_DIR, f"learning_curve_{model.replace('/','_')}_{seq_id}.png"))
                logger.info(f"Saved learning_curve_{model.replace('/','_')}_{seq_id}.png")
                plt.show()
else:
    logger.warning("Sequence or task index information not available. Skipping learning curve plots.")


# %% [markdown]
# ### 7.2. Patch Similarity Plots

# %%
if not results_df.empty and "char_levenshtein" in results_df.columns and not valid_similarity_df.empty:
    # Character Levenshtein Distance
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=valid_similarity_df, x="model_name", y="char_levenshtein", hue="condition")
    plt.title("Character Levenshtein Distance (Model vs. Gold Patch)\nLower is Better")
    plt.ylabel("Character Edit Distance")
    plt.xlabel("Model Name")
    plt.xticks(rotation=45, ha="right")
    # Consider y-axis scale if distances are very large for some
    # plt.ylim(0, valid_similarity_df["char_levenshtein"].quantile(0.95)) # Example: Cap at 95th percentile
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUTPUT_DIR, "char_levenshtein_boxplot.png"))
    logger.info(f"Saved char_levenshtein_boxplot.png")
    plt.show()

    # Token Levenshtein Distance
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=valid_similarity_df, x="model_name", y="token_levenshtein", hue="condition")
    plt.title("Token Levenshtein Distance (Model vs. Gold Patch)\nLower is Better")
    plt.ylabel("Token Edit Distance")
    plt.xlabel("Model Name")
    plt.xticks(rotation=45, ha="right")
    # plt.ylim(0, valid_similarity_df["token_levenshtein"].quantile(0.95))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUTPUT_DIR, "token_levenshtein_boxplot.png"))
    logger.info(f"Saved token_levenshtein_boxplot.png")
    plt.show()

    # --- Semantic Similarity (Placeholder) ---
    # If you calculate semantic similarity (e.g., cosine similarity from embeddings)
    # you would plot it here, likely with higher values being better.
    # Example:
    # if "semantic_similarity" in valid_similarity_df.columns:
    #     plt.figure(figsize=(12, 7))
    #     sns.boxplot(data=valid_similarity_df, x="model_name", y="semantic_similarity", hue="condition")
    #     plt.title("Semantic Similarity (Model vs. Gold Patch)\nHigher is Better")
    #     plt.ylabel("Cosine Similarity")
    #     plt.xlabel("Model Name")
    #     plt.xticks(rotation=45, ha="right")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FIG_OUTPUT_DIR, "semantic_similarity_boxplot.png"))
    #     plt.show()

else:
    logger.warning("Patch similarity data not available for plotting.")

# %% [markdown]
# ## 8. Further Analysis Ideas (Optional)
#
# - **Qualitative Analysis**: Examine specific instances where memory helped or hurt significantly. Look at the `raw_llm_output` or the patches themselves.
# - **Error Analysis**: Categorize types of errors made by the model patches when they fail.
# - **Patch Length Comparison**: Analyze if `memory_enabled` leads to longer/shorter patches and if this correlates with success or similarity.
# - **Statistical Significance**: Perform statistical tests (e.g., t-tests, ANOVA) to see if differences in pass rates or similarity scores are statistically significant.
# - **Impact of Retrieved Context**: If you stored the retrieved context in your `eval_state.json`, analyze its properties (e.g., length, relevance scores if available from memory system) and correlate with task success.

# %%
logger.info("Analysis script finished.")
