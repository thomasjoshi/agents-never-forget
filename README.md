# SWE-Bench-CL: Continual Learning for Coding Agents

**Final Project for COMS 4995: Neural Networks and Deep Learning w/ Prof. Richard Zemel + advised by Tom Zollo**
**Team Members:** Thomas Joshi, Shayan Chowdhury, Fatih Uysal

## Introduction & Motivation

Large Language Models (LLMs) have achieved remarkable success in a variety of code-related tasks, from autocompletion to generating entire code snippets from natural language descriptions. However, the lifecycle of real-world software projects is inherently dynamic and continuous. Repositories evolve daily: APIs are deprecated, libraries are upgraded, new bugs are discovered and fixed, and novel features are constantly requested. An adept software engineering agent must therefore not only generate correct code for an immediate request but also learn from its experiences, adapt to changes in the codebase, and, crucially, retain knowledge of how to handle past issues as the project grows and shifts. Consider the analogy of a human software engineer: one who has successfully resolved 100 bugs within a specific, complex codebase will be far more adept at tackling the 101st bug in that same repository than an equally skilled engineer encountering the codebase for the first time. This ability to accumulate and leverage experience is a hallmark of expertise and a critical capability for **agents that continuously learn**.

This project makes the following primary contributions:
1.  **A Novel Benchmark Dataset (SWE-Bench-CL):** We detail the construction and structure of SWE-Bench-CL, a reproducible, temporally organized benchmark designed to measure adaptation and memory retention in coding agents.
2.  **Preliminary Dataset Analysis:** We present an analysis of SWE-Bench-CL's structural characteristics, including inter-task similarity and contextual sensitivity. These findings highlight the unique challenges the benchmark poses for continual learning and inform the design of effective evaluation strategies and agent architectures.
3.  **A Proposed Agentic Evaluation Framework:** We propose a methodology for evaluating agents on SWE-Bench-CL. This framework centers on an interactive coding agent, which is notably augmented with a semantic memory module to facilitate learning from past experiences. It was developed to overcome challenges encountered with existing evaluation harnesses when applied to our continual learning setup, offering greater transparency and control.
4.  **Specialized Continual Learning Metrics:** We define a suite of evaluation metrics specifically tailored for assessing continual learning in the context of software engineering, addressing aspects like success rate, tool use efficiency, knowledge transfer, and forgetting.

## Project Structure

This repository is organized as follows:
*   [`data/`](./data/): Contains the core dataset files.
    *   [`SWE-Bench-CL-Curriculum.json`](./data/SWE-Bench-CL-Curriculum.json): The continual learning benchmark dataset derived from SWE-Bench Verified.
*   [`eval_v1/eval_procedure.py`](./eval_v1/eval_procedure.py): Naive implementation of continual learning experiments, generating patches, evaluating with the SWE-bench harness, and calculating metrics, evaluating LLMs on SWE-bench-CL using SWE-bench's own evaluation harness w/ docker containers
*   [`eval_v2_agent/eval_procedure.py`](./eval_v2_agent/eval_procedure.py): an agentic implementation via langgraph w/ basic file search, editing, and unit test running functionality + FAISS RAG for CL semantic memory
*   [`eval_v3_agent/eval_procedure.py`](./eval_v3_agent/eval_procedure.py): eval v3, an agentic evaluation of SWE-bench-CL inspired by SWE-agent + integrating continual learning methods + using LangGraph + semantic memory
*   [`scripts/`](./scripts/): Utility scripts for dataset construction, experimentation, and analysis
    *   [`SWE-Bench-CL_dataset_construction.py`](./scripts/SWE-Bench-CL_dataset_construction.py): The script used to generate the `SWE-Bench-CL-Curriculum.json` dataset from the original SWE-Bench data.
*   `requirements.txt`: Python dependencies for the project.
*   `.env`: (User-created) File for storing API keys.
*   `LICENSE`: Project license.
*   `Makefile`: Makefile for potential build/automation tasks.
*   `research-papers/`: Contains relevant research papers.

## SWE-Bench-CL Dataset
We developed **SWE-Bench-CL**, a continual learning adaptation of the [SWE-Bench-Verified](https://openai.com/index/introducing-swe-bench-verified/) dataset, a human-verified refinement of the original [SWE-Bench](https://arxiv.org/abs/2310.06770) dataset, designed to evaluate how effectively AI agents can learn and retain programming knowledge over time. Our new benchmark transforms the original task-independent format into sequential learning scenarios that simulate a developer's progression on real-world projects. The entire dataset is provided in JSON format at [`SWE-Bench-CL.json`](./SWE-Bench-CL.json).

### Dataset Structure
Transforming the original `SWE-Bench-Verified` dataset, we created 8 learning sequences, each associated with a different repository from the original dataset. Each sequence is designed to follow a curriculum, starting with simpler tasks and progressively introducing more complex problems.

We employed several strategies for how we sequenced the tasks within each repository:
1. **Chronological Ordering**: Tasks within each repository are primarily ordered by their creation date, simulating the natural evolution of a codebase.
2. **Curriculum Learning**: Within each sequence, tasks are further grouped by difficulty levels:
   - Level 1: <15 min fix
   - Level 2: 15 min - 1 hour
   - Level 3: 1-4 hours 
   - Level 4: >4 hours
3. **Dependency Awareness**: The dataset identifies potential dependencies between tasks based on file modifications, enabling evaluation of knowledge transfer between related problems.

### Dataset Statistics
| Repository | Tasks | Easy (<15m) | Medium (15m-1h) | Hard (1-4h) | Very Hard (>4h) | Tasks w/ Dependencies |
|------------|-------|-------------|-----------------|-------------|-----------------|------------------------|
| django/django | 50 | 50 | 0 | 0 | 0 | 25 (50%) |
| sympy/sympy | 50 | 25 | 25 | 0 | 0 | 12 (24%) |
| sphinx-doc/sphinx | 44 | 22 | 17 | 4 | 1 | 23 (52%) |
| matplotlib/matplotlib | 34 | 15 | 19 | 0 | 0 | 13 (38%) |
| scikit-learn/scikit-learn | 32 | 13 | 18 | 1 | 0 | 4 (13%) |
| astropy/astropy | 22 | 4 | 15 | 3 | 0 | 3 (14%) |
| pydata/xarray | 22 | 5 | 15 | 1 | 1 | 13 (59%) |
| pytest-dev/pytest | 19 | 8 | 8 | 3 | 0 | 7 (37%) |

## Evaluation Metrics (TODO: just some ideas for now!!)
We introduce specialized metrics for evaluating continual learning performance:
### Success Rate
- **Description**: Percentage of tasks successfully solved in a sequence
- **Formula**: `successCount / totalTasks`
### Forgetting Rate
- **Description**: Measures how much previously learned knowledge is forgotten
- **Formula**: `(recentSuccess - currentSuccess) / recentSuccess`
- **Details**: Calculated by periodically re-testing previously solved tasks. A value of 0 indicates no forgetting, while 1 indicates complete forgetting.
### Forward Transfer
- **Description**: Measures how learning on previous tasks improves performance on new tasks
- **Formula**: `performanceOnNewTasks(with_previous_training) - performanceOnNewTasks(without_previous_training)`
- **Details**: Positive values indicate positive transfer (previous learning helps with new tasks).
### Backward Transfer
- **Description**: Measures how learning on new tasks affects performance on previous tasks
- **Formula**: `performanceOnPreviousTasks(after_new_training) - performanceOnPreviousTasks(before_new_training)`
- **Details**: Positive values indicate that learning new tasks improves performance on old tasks.
### Tool Use Efficiency
- **Description**: Measures the efficiency of tool use during problem solving
- **Formula**: `successfulToolCalls / totalToolCalls`
- **Details**: Higher values indicate more efficient use of tools.
### Tool Use Adaptation
- **Description**: Measures how the agent's tool use evolves over time
- **Formula**: Qualitative analysis of tool use patterns across sequences
- **Details**: Assesses whether the agent learns to use more appropriate tools or develops more efficient tool use strategies.
### Learning Curve
- **Description**: Measures the rate of learning over the sequence
- **Formula**: Success rate as a function of task number
- **Details**: Steeper curves indicate faster learning.
### CL-Score (Comprehensive Metric)
- **Description**: A comprehensive continual learning score
- **Formula**: `successRate * (1 - forgettingRate) * (1 + 0.5 * forwardTransfer + 0.5 * backwardTransfer) * toolUseEfficiency`
- **Details**: Combines success, forgetting, transfer, and tool use metrics into a single score.

## Evaluation Procedure
1. **Sequential Learning**:
   - Train/evaluate the agent on each sequence in order
   - For each task in a sequence:
     - Present the task to the agent
     - Measure success (whether the agent's solution passes all tests)
     - Record tool usage and solution strategy
2. **Forgetting Assessment**:
   - Periodically re-test the agent on previously solved tasks
   - Calculate the forgetting rate based on performance degradation
3. **Transfer Evaluation**:
   - After completing a repository sequence, test the agent on tasks from other repositories
   - Measure cross-domain transfer and knowledge retention
4. **Reporting**:
   - Learning curve: Success rate as a function of task number
   - Forgetting curve: Performance on previously solved tasks over time
   - Transfer matrix: How learning on one repository affects performance on others
   - Tool usage patterns: How tool use evolves over time
   - CL-Score: Combined metric incorporating success, forgetting, transfer, and tool use