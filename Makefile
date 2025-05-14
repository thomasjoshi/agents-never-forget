# --------------------------------------------------------------------
# SWE-Bench-CL Project Makefile
# A benchmark for continual learning of coding agents
# --------------------------------------------------------------------

# ===================================================================
# VARIABLES AND CONFIGURATIONS
# ===================================================================
PYTHON ?= python3
VENV_DIR ?= .venv
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := . $(VENV_DIR)/bin/activate

# Project directories
DATA_DIR := data
RESULTS_DIR := results
MODELS_DIR := models
SCRIPTS_DIR := scripts

# Requirements files
REQS := requirements.txt
TEST_REQS := test-requirements.txt
DEV_REQS := dev-requirements.txt

# Model configuration
# Set these to point to your local model files
MODEL_PATH ?= $(HOME)/.cache/huggingface/models
MODEL_PRECISION ?= bfloat16  # Options: float32, float16, bfloat16
MODEL_BITS ?= 16  # Options: 8, 16 (for quantization)

# LoRA configuration
USE_LORA ?= true
LORA_RANK ?= 16
LORA_ALPHA ?= 32

# Training configuration
TRAIN_EPOCHS ?= 3  # Full training epochs
QUICK_EPOCHS ?= 1  # Quick test run epochs

# Evaluation configuration
BATCH_SIZE ?= 1
MAX_SEQ_LEN ?= 4096
NUM_TASKS ?= 5  # Number of tasks to evaluate per repository
REPO_FILTER ?= "django/django"  # Default repository to evaluate

# Experiment tracking
EXP_NAME ?= swe_bench_cl_$(shell date +%Y%m%d_%H%M%S)
EXP_DIR := $(RESULTS_DIR)/$(EXP_NAME)
SEED ?= 42  # Reproducibility seed

# Google Cloud configuration
# Multiple zones for better availability
GCP_ZONES ?= us-central1-c us-east4-c us-west4-c europe-west4-a asia-east1-a
GCP_ZONE ?= us-central1-c  # Default zone (first will be tried first)
# Machine types to try (will try in this order)
GCP_MACHINE_TYPES ?= n1-standard-4 n1-standard-8 n1-highmem-8
# GPU configuration - set GPU_ENABLED=false if you don't have GPU quota
GPU_ENABLED ?= true
# Set CPU_FALLBACK=true to create a CPU instance if GPU creation fails
CPU_FALLBACK ?= false
GCP_GPU_TYPE ?= nvidia-tesla-t4
GCP_GPU_COUNT ?= 1
GCP_DISK_SIZE ?= 200GB

# Google Cloud instance name
GCP_INSTANCE_NAME ?= swe-bench-cl

# Dependency tracking
DEPS_STAMP := $(VENV_DIR)/.deps_stamp

# ===================================================================
# MAIN TARGETS
# ===================================================================

# Default target shows help
.PHONY: help
help:
	@echo "SWE-Bench-CL Project Makefile"
	@echo "============================"
	@echo ""
	@echo "Setup targets:"
	@echo "  setup                Setup complete environment (venv + deps)"
	@echo "  venv                 Create virtual environment"
	@echo "  deps                 Install Python dependencies"
	@echo "  generate-requirements Generate requirements.txt file"
	@echo "  generate-test-requirements Generate test-requirements.txt file"
	@echo "  check-model          Verify model is properly set up"
	@echo ""
	@echo "Data processing targets:"
	@echo "  process-data         Process SWE-Bench-CL dataset for evaluation"
	@echo "  generate-embed       Generate embeddings for the tasks"
	@echo ""
	@echo "Evaluation targets:"
	@echo "  quick-eval           Run quick evaluation (single epoch)"
	@echo "  full-eval            Run full evaluation (multiple epochs)"
	@echo "  eval-zero-shot       Run zero-shot evaluation baseline"
	@echo "  eval-context-aug     Run evaluation with context augmentation"
	@echo "  eval-memory          Run evaluation with memory mechanism"
	@echo "  eval-all             Run all evaluation methods"
	@echo ""
	@echo "Analysis targets:"
	@echo "  analyze-results      Analyze and visualize results"
	@echo "  generate-report      Generate PDF report of results"
	@echo ""
	@echo "Reproducibility targets:"
	@echo "  generate-configs     Generate configuration files for reproducibility"
	@echo "  reproduce            Reproduce experiment from configuration"
	@echo ""
	@echo "Paper generation targets:"
	@echo "  setup-paper          Set up paper directory"
	@echo "  generate-paper       Generate paper draft from results"
	@echo "  compile-paper        Compile PDF paper"
	@echo ""
	@echo "Google Cloud targets:"
	@echo "  gcp-setup            Set up Google Cloud instance with A100 GPU"
	@echo "  gcp-start            Start Google Cloud instance"
	@echo "  gcp-stop             Stop Google Cloud instance"
	@echo "  gcp-ssh              Connect to Google Cloud instance"
	@echo ""
	@echo "Model configuration:"
	@echo "  register-model       Register model with metadata in model registry"
	@echo ""
	@echo "Configuration variables:"
	@echo "  MODEL=model-name            Set model (default: $(MODEL))"
	@echo "  TRAIN_EPOCHS=n              Set number of training epochs (default: $(TRAIN_EPOCHS))"
	@echo "  QUICK_EPOCHS=n              Set number of quick evaluation epochs (default: $(QUICK_EPOCHS))"
	@echo "  BATCH_SIZE=n                Set batch size (default: $(BATCH_SIZE))"
	@echo "  MAX_SEQ_LEN=n               Set max sequence length (default: $(MAX_SEQ_LEN))"
	@echo "  NUM_TASKS=n                 Set number of tasks to evaluate (default: $(NUM_TASKS))"
	@echo "  REPO_FILTER=repo/name       Set repository to evaluate (default: $(REPO_FILTER))"
	@echo "  MODEL_PRECISION=precision   Set model precision (default: $(MODEL_PRECISION))"

# ===================================================================
# SETUP TARGETS
# ===================================================================

.PHONY: setup
setup: venv deps check-model
	@echo "✓ Setup complete! Use 'source $(VENV_DIR)/bin/activate' to activate the environment."

.PHONY: venv
venv:
	@echo "Creating virtual environment..."
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@echo "✓ Virtual environment created at $(VENV_DIR)"

.PHONY: deps
deps: venv $(DEPS_STAMP)

$(DEPS_STAMP): $(REQS) $(TEST_REQS) $(wildcard pyproject.toml)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	@test -f $(REQS) && $(PIP) install -r $(REQS) || echo "No $(REQS) found, skipping..."
	@test -f $(TEST_REQS) && $(PIP) install -r $(TEST_REQS) || echo "No $(TEST_REQS) found, skipping..."
	@test -f $(DEV_REQS) && $(PIP) install -r $(DEV_REQS) || echo "No $(DEV_REQS) found, skipping..."
	@test -f pyproject.toml && $(PIP) install -e . || echo "No pyproject.toml found, skipping..."
	@mkdir -p $(DATA_DIR) $(RESULTS_DIR) $(MODELS_DIR) $(SCRIPTS_DIR)
	@touch $@
	@echo "✓ Dependencies installed"

.PHONY: check-model
check-model:
	@if [ ! -d "$(MODEL_PATH)" ]; then \
		echo "Error: Model not found at $(MODEL_PATH)"; \
		echo "Please set MODEL_PATH to point to your model directory"; \
		echo "Example: make setup MODEL_PATH=/path/to/your/model"; \
		exit 1; \
	fi
	@echo "✓ Model found at $(MODEL_PATH)"

# ===================================================================
# DATA PROCESSING TARGETS
# ===================================================================

.PHONY: process-data
process-data: deps
	@echo "Processing SWE-Bench-CL dataset..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/process_data.py \
		--input SWE-Bench-CL.json \
		--output $(DATA_DIR)/processed_tasks.json
	@echo "✓ Data processing complete"

.PHONY: generate-embed
generate-embed: deps process-data
	@echo "Generating embeddings for tasks..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/generate_embeddings.py \
		--input $(DATA_DIR)/processed_tasks.json \
		--output $(DATA_DIR)/task_embeddings.pkl
	@echo "✓ Embeddings generated"

# ===================================================================
# EVALUATION TARGETS
# ===================================================================

.PHONY: eval-zero-shot
eval-zero-shot: deps check-model process-data
	@echo "Running zero-shot evaluation with model from $(MODEL_PATH)..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/evaluate_zero_shot.py \
		--input $(DATA_DIR)/processed_tasks.json \
		--output $(RESULTS_DIR)/zero_shot_results.json \
		--model_path $(MODEL_PATH) \
		--batch_size $(BATCH_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--precision $(MODEL_PRECISION) \
		--num_tasks $(NUM_TASKS) \
		--repo_filter $(REPO_FILTER) \
		--seed $(SEED)
	@echo "✓ Zero-shot evaluation complete. Results saved to $(RESULTS_DIR)/zero_shot_results.json"

.PHONY: quick-eval
quick-eval: deps check-model
	@echo "Running quick evaluation with model from $(MODEL_PATH)..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/finetune_and_evaluate.py \
		--model_path $(MODEL_PATH) \
		--batch_size $(BATCH_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--precision $(MODEL_PRECISION) \
		--epochs $(QUICK_EPOCHS) \
		--output_dir $(RESULTS_DIR)/quick_eval_$(shell date +%Y%m%d_%H%M%S) \
		--seed $(SEED)
	@echo "✓ Quick evaluation complete. Results saved to $(RESULTS_DIR)/quick_eval_*/"

.PHONY: full-eval
full-eval: deps check-model
	@echo "Running full evaluation with model from $(MODEL_PATH)..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/finetune_and_evaluate.py \
		--model_path $(MODEL_PATH) \
		--batch_size $(BATCH_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--precision $(MODEL_PRECISION) \
		--epochs $(TRAIN_EPOCHS) \
		--output_dir $(RESULTS_DIR)/full_eval_$(shell date +%Y%m%d_%H%M%S) \
		--seed $(SEED)
	@echo "✓ Full evaluation complete. Results saved to $(RESULTS_DIR)/full_eval_*/"

.PHONY: eval-context-aug
eval-context-aug: deps check-model process-data generate-embed
	@echo "Running evaluation with context augmentation using model from $(MODEL_PATH)..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/evaluate_context_aug.py \
		--input $(DATA_DIR)/processed_tasks.json \
		--embeddings $(DATA_DIR)/task_embeddings.pkl \
		--output $(RESULTS_DIR)/context_aug_results.json \
		--model_path $(MODEL_PATH) \
		--batch_size $(BATCH_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--precision $(MODEL_PRECISION) \
		--num_tasks $(NUM_TASKS) \
		--repo_filter $(REPO_FILTER) \
		--top_k 3 \
		--seed $(SEED)
	@echo "✓ Context augmentation evaluation complete. Results saved to $(RESULTS_DIR)/context_aug_results.json"

.PHONY: eval-memory
eval-memory: deps check-model process-data generate-embed
	@echo "Running evaluation with memory mechanism using model from $(MODEL_PATH)..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/evaluate_memory.py \
		--input $(DATA_DIR)/processed_tasks.json \
		--embeddings $(DATA_DIR)/task_embeddings.pkl \
		--output $(RESULTS_DIR)/memory_results.json \
		--model_path $(MODEL_PATH) \
		--batch_size $(BATCH_SIZE) \
		--max_seq_len $(MAX_SEQ_LEN) \
		--precision $(MODEL_PRECISION) \
		--num_tasks $(NUM_TASKS) \
		--repo_filter $(REPO_FILTER) \
		--memory_size 100 \
		--seed $(SEED)
	@echo "✓ Memory mechanism evaluation complete. Results saved to $(RESULTS_DIR)/memory_results.json"

.PHONY: eval-all
eval-all: check-model
	@echo "Running all evaluations with model from $(MODEL_PATH)..."
	@$(MAKE) eval-zero-shot
	@$(MAKE) eval-context-aug
	@$(MAKE) eval-memory
	@echo "✓ All evaluations completed. Results saved to $(RESULTS_DIR)/"

# ===================================================================
# ANALYSIS TARGETS
# ===================================================================

.PHONY: analyze-results
analyze-results: deps
	@echo "Analyzing evaluation results..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/analyze_results.py \
		--zero_shot $(RESULTS_DIR)/zero_shot_results.json \
		--context_aug $(RESULTS_DIR)/context_aug_results.json \
		--memory $(RESULTS_DIR)/memory_results.json \
		--output $(RESULTS_DIR)/analysis
	@echo "✓ Analysis complete - results in $(RESULTS_DIR)/analysis/"

.PHONY: generate-report
generate-report: analyze-results
	@echo "Generating PDF report..."
	$(VENV_DIR)/bin/python $(SCRIPTS_DIR)/generate_report.py \
		--input $(RESULTS_DIR)/analysis \
		--output $(RESULTS_DIR)/report.pdf
	@echo "✓ Report generated at $(RESULTS_DIR)/report.pdf"

# ===================================================================
# REPRODUCIBILITY TARGETS
# ===================================================================

.PHONY: generate-configs
generate-configs: 
	@echo "Generating configuration files for reproducibility..."
	@mkdir -p $(EXP_DIR)/configs
	@echo "MODEL=$(MODEL)" > $(EXP_DIR)/configs/experiment.env
	@echo "MODEL_REVISION=$(MODEL_REVISION)" >> $(EXP_DIR)/configs/experiment.env
	@echo "PRECISION=$(MODEL_PRECISION)" >> $(EXP_DIR)/configs/experiment.env
	@echo "BATCH_SIZE=$(BATCH_SIZE)" >> $(EXP_DIR)/configs/experiment.env
	@echo "MAX_SEQ_LEN=$(MAX_SEQ_LEN)" >> $(EXP_DIR)/configs/experiment.env
	@echo "NUM_TASKS=$(NUM_TASKS)" >> $(EXP_DIR)/configs/experiment.env
	@echo "REPO_FILTER=$(REPO_FILTER)" >> $(EXP_DIR)/configs/experiment.env
	@echo "SEED=$(SEED)" >> $(EXP_DIR)/configs/experiment.env
	@echo "EXP_NAME=$(EXP_NAME)" >> $(EXP_DIR)/configs/experiment.env
	@echo "✓ Configuration files generated"

.PHONY: reproduce
reproduce: deps process-data
	@echo "Reproducing experiment from configuration..."
	@if [ ! -f "$(RESULTS_DIR)/configs/experiment.env" ]; then \
		echo "Error: No configuration file found. Run an experiment first."; \
		exit 1; \
	fi
	@. $(RESULTS_DIR)/configs/experiment.env && \
	$(MAKE) generate-embed eval-zero-shot eval-context-aug eval-memory analyze-results \
		MODEL=$$MODEL \
		MODEL_REVISION=$$MODEL_REVISION \
		MODEL_PRECISION=$$PRECISION \
		BATCH_SIZE=$$BATCH_SIZE \
		MAX_SEQ_LEN=$$MAX_SEQ_LEN \
		NUM_TASKS=$$NUM_TASKS \
		REPO_FILTER=$$REPO_FILTER \
		SEED=$$SEED \
		EXP_NAME=$$EXP_NAME
	@echo "✓ Experiment reproduced"

# ===================================================================
# TRACK EXPERIMENT (Simple CSV-based tracking)
# ===================================================================

.PHONY: log-experiment
log-experiment: 
	@echo "Logging experiment details to experiments.csv..."
	@mkdir -p $(RESULTS_DIR)
	@if [ ! -f "$(RESULTS_DIR)/experiments.csv" ]; then \
		echo "experiment_id,model,precision,batch_size,max_seq_len,num_tasks,repo_filter,seed,date" > $(RESULTS_DIR)/experiments.csv; \
	fi
	@echo "$(EXP_NAME),$(MODEL),$(MODEL_PRECISION),$(BATCH_SIZE),$(MAX_SEQ_LEN),$(NUM_TASKS),$(REPO_FILTER),$(SEED),$(shell date +%Y-%m-%d)" >> $(RESULTS_DIR)/experiments.csv
	@echo "✓ Experiment logged to experiments.csv"

# ===================================================================
# REQUIREMENTS.TXT GENERATION
# ===================================================================

.PHONY: venv
venv:
	@echo "Creating Python virtual environment..."
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQS)

.PHONY: generate-requirements
generate-requirements:
	@echo "The requirements.txt file is the source of truth for dependencies."
	@echo "To add a dependency, edit requirements.txt directly."
	@echo "Then run 'make venv' to update your environment."
	@echo ""
	@echo "Current requirements:"
	@cat $(REQS)

.PHONY: generate-test-requirements
generate-test-requirements:
	@echo "Generating test-requirements.txt..."
	@echo "# SWE-Bench-CL test dependencies" > $(TEST_REQS)
	@echo "pytest>=7.3.1" >> $(TEST_REQS)
	@echo "pytest-cov>=4.1.0" >> $(TEST_REQS)
	@echo "black>=23.3.0" >> $(TEST_REQS)
	@echo "isort>=5.12.0" >> $(TEST_REQS)
	@echo "flake8>=6.0.0" >> $(TEST_REQS)
	@echo "✓ Test-requirements.txt generated"

# ===================================================================
# GOOGLE CLOUD TARGETS
# ===================================================================

.PHONY: gcp-setup
gcp-setup:
	@if [ "$(GPU_ENABLED)" = "true" ]; then \
		echo "Setting up Google Cloud environment with $(GCP_GPU_TYPE) GPU..."; \
		echo "Trying machine types: $(GCP_MACHINE_TYPES)"; \
		echo "Across zones: $(GCP_ZONES)"; \
		success=false; \
		for machine in $(GCP_MACHINE_TYPES); do \
			echo "\nTrying machine type: $$machine"; \
			for zone in $(GCP_ZONES); do \
				echo "  - In zone: $$zone"; \
				if gcloud compute instances create $(GCP_INSTANCE_NAME) \
					--zone=$$zone \
					--machine-type=$$machine \
					--accelerator=type=$(GCP_GPU_TYPE),count=$(GCP_GPU_COUNT) \
					--boot-disk-size=$(GCP_DISK_SIZE) \
					--image-family=pytorch-latest-gpu \
					--image-project=deeplearning-platform-release \
					--maintenance-policy=TERMINATE \
					--scopes=https://www.googleapis.com/auth/cloud-platform \
					--metadata="install-nvidia-driver=True" \
					--restart-on-failure; then \
					echo "\n✓ Success! Created $$machine instance with $(GCP_GPU_TYPE) GPU in zone $$zone"; \
					echo "INSTANCE_ZONE=$$zone" > .gcp_zone; \
					echo "MACHINE_TYPE=$$machine" >> .gcp_zone; \
					echo "HAS_GPU=true" >> .gcp_zone; \
					success=true; \
					break 2; \
				fi; \
			done; \
		done; \
		if [ "$$success" != "true" ]; then \
			echo "\nWARNING: Failed to create GPU instance. You likely need to request GPU quota."; \
			echo "See: https://cloud.google.com/compute/quotas#requesting_additional_quota"; \
			if [ "$(CPU_FALLBACK)" = "true" ]; then \
				echo "\nAttempting to create CPU-only instance instead (CPU_FALLBACK=true)..."; \
				for machine in $(GCP_MACHINE_TYPES); do \
					for zone in $(GCP_ZONES); do \
						echo "  - Trying CPU-only $$machine in zone: $$zone"; \
						if gcloud compute instances create $(GCP_INSTANCE_NAME) \
							--zone=$$zone \
							--machine-type=$$machine \
							--boot-disk-size=$(GCP_DISK_SIZE) \
							--image-family=pytorch-latest-cpu \
							--image-project=deeplearning-platform-release \
							--maintenance-policy=TERMINATE \
							--scopes=https://www.googleapis.com/auth/cloud-platform \
							--restart-on-failure; then \
							echo "\n✓ Created CPU-only $$machine instance in zone $$zone"; \
							echo "INSTANCE_ZONE=$$zone" > .gcp_zone; \
							echo "MACHINE_TYPE=$$machine" >> .gcp_zone; \
							echo "HAS_GPU=false" >> .gcp_zone; \
							echo "\nNOTE: This is a CPU-only instance. Model training will be slow."; \
							echo "Request GPU quota at: https://cloud.google.com/compute/quotas#requesting_additional_quota"; \
							success=true; \
							break 2; \
						fi; \
					done; \
				done; \
			else \
				echo "\nAborting instance creation. Set CPU_FALLBACK=true to attempt creating a CPU-only instance."; \
			fi; \
		fi; \
	else \
		echo "Setting up CPU-only Google Cloud environment..."; \
		success=false; \
		for machine in $(GCP_MACHINE_TYPES); do \
			echo "\nTrying machine type: $$machine"; \
			for zone in $(GCP_ZONES); do \
				echo "  - In zone: $$zone"; \
				if gcloud compute instances create $(GCP_INSTANCE_NAME) \
					--zone=$$zone \
					--machine-type=$$machine \
					--boot-disk-size=$(GCP_DISK_SIZE) \
					--image-family=pytorch-latest-cpu \
					--image-project=deeplearning-platform-release \
					--maintenance-policy=TERMINATE \
					--scopes=https://www.googleapis.com/auth/cloud-platform \
					--restart-on-failure; then \
					echo "\n✓ Success! Created CPU-only $$machine instance in zone $$zone"; \
					echo "INSTANCE_ZONE=$$zone" > .gcp_zone; \
					echo "MACHINE_TYPE=$$machine" >> .gcp_zone; \
					echo "HAS_GPU=false" >> .gcp_zone; \
					success=true; \
					break 2; \
				fi; \
			done; \
		done; \
	fi; \
	if [ "$$success" != "true" ]; then \
		echo "\nFailed to create instance with any configuration."; \
		echo "Check your GCP project configuration and quotas."; \
		exit 1; \
	fi

.PHONY: gcp-start
gcp-start:
	@echo "Starting Google Cloud instance..."
	@if [ -f .gcp_zone ]; then \
		zone=$$(grep INSTANCE_ZONE .gcp_zone | cut -d'=' -f2); \
		echo "Using zone: $$zone"; \
		gcloud compute instances start $(GCP_INSTANCE_NAME) --zone=$$zone; \
	else \
		echo "No zone information found. Using default: $(GCP_ZONE)"; \
		gcloud compute instances start $(GCP_INSTANCE_NAME) --zone=$(GCP_ZONE); \
	fi
	@echo "✓ Instance started"

.PHONY: gcp-stop
gcp-stop:
	@echo "Stopping Google Cloud instance..."
	@if [ -f .gcp_zone ]; then \
		zone=$$(grep INSTANCE_ZONE .gcp_zone | cut -d'=' -f2); \
		echo "Using zone: $$zone"; \
		gcloud compute instances stop $(GCP_INSTANCE_NAME) --zone=$$zone; \
	else \
		echo "No zone information found. Using default: $(GCP_ZONE)"; \
		gcloud compute instances stop $(GCP_INSTANCE_NAME) --zone=$(GCP_ZONE); \
	fi
	@echo "✓ Instance stopped"

.PHONY: gcp-ssh
gcp-ssh:
	@echo "Connecting to Google Cloud instance..."
	@if [ -f .gcp_zone ]; then \
		zone=$$(grep INSTANCE_ZONE .gcp_zone | cut -d'=' -f2); \
		echo "Using zone: $$zone"; \
		gcloud compute ssh $(GCP_INSTANCE_NAME) --zone=$$zone; \
	else \
		echo "No zone information found. Using default: $(GCP_ZONE)"; \
		gcloud compute ssh $(GCP_INSTANCE_NAME) --zone=$(GCP_ZONE); \
	fi

# ===================================================================
# EVALUATION TARGETS
# ===================================================================

.PHONY: quick-eval
quick-eval: deps process-data generate-embed
	@echo "Running quick evaluation (single epoch)..."
	@mkdir -p $(EXP_DIR)/quick
	@echo "ENVIRONMENT=quick" > $(EXP_DIR)/env_config.txt
	$(MAKE) eval-zero-shot eval-context-aug eval-memory \
		MODEL=$(MODEL) \
		EPOCHS=$(QUICK_EPOCHS) \
		USE_LORA=$(USE_LORA) \
		LORA_RANK=$(LORA_RANK) \
		BATCH_SIZE=1 \
		MAX_SEQ_LEN=2048 \
		EXP_NAME=$(EXP_NAME)_quick
	@echo "✓ Quick evaluation complete"

.PHONY: full-eval
full-eval: deps process-data generate-embed
	@echo "Running full evaluation ($(TRAIN_EPOCHS) epochs)..."
	@mkdir -p $(EXP_DIR)/full
	@echo "ENVIRONMENT=full" > $(EXP_DIR)/env_config.txt
	$(MAKE) eval-zero-shot eval-context-aug eval-memory \
		MODEL=$(MODEL) \
		EPOCHS=$(TRAIN_EPOCHS) \
		USE_LORA=$(USE_LORA) \
		LORA_RANK=$(LORA_RANK) \
		BATCH_SIZE=1 \
		MAX_SEQ_LEN=4096 \
		EXP_NAME=$(EXP_NAME)_full
	@echo "✓ Full evaluation complete ($(TRAIN_EPOCHS) epochs)"

# ===================================================================
# CLEANUP TARGETS
# ===================================================================

.PHONY: clean-results
clean-results:
	@echo "Cleaning results..."
	rm -rf $(RESULTS_DIR)/*
	mkdir -p $(RESULTS_DIR)
	@echo "✓ Results cleaned"

.PHONY: clean
clean: clean-results
	@echo "Cleaning environment..."
	rm -rf $(VENV_DIR)
	rm -f $(DEPS_STAMP)
	@echo "✓ Environment cleaned"