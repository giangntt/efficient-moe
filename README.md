# Efficient MoE: Mixture-of-Experts Model Pruning and Analysis

A comprehensive research toolkit for analyzing, profiling, and pruning Mixture-of-Experts (MoE) models to improve their efficiency and understand their behavior. This project focuses on the **Qwen1.5-MoE-A2.7B** model and provides tools for expert activation analysis, router behavior profiling, correlation analysis, and model pruning with performance evaluation.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [Usage Guide](#-usage-guide)
- [Analysis Notebooks](#-analysis-notebooks)
- [Configuration](#-configuration)
- [Evaluation Tasks](#-evaluation-tasks)

## 🚀 Features

### Analysis & Profiling
- **Expert Activation Analysis**: Monitor and analyze expert activation patterns across layers
- **Router Behavior Profiling**: Collect and analyze router logits to understand routing decisions
- **Correlation Analysis**: Compute correlations between router activations and expert usage across different task categories
- **Statistical Analysis**: Compute comprehensive statistics including mean, variance, frequency, and probability distributions

### Pruning & Optimization
- **Multiple Pruning Strategies**: Support for masking and zeroing-based expert pruning
- **Flexible Expert Selection**: Prune least-used or most-used experts based on various criteria
- **Performance Evaluation**: Comprehensive evaluation framework using LM-Eval with support for multiple benchmarks

### Visualization & Tools
- **Advanced Plotting**: Generate correlation plots, usage matrices, and statistical visualizations
- **Data Processing**: Efficient text packing and dataset handling for large-scale analysis
- **MMLU Category Analysis**: Specialized tools for analyzing model behavior across MMLU categories

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Compatible GPU with CUDA support (recommended)
- **PyTorch**: With CUDA support

### Dependencies

Install the required packages:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers datasets lm-eval matplotlib seaborn tqdm numpy
```

## 🎯 Quick Start

### 1. Basic Model Evaluation

Evaluate the model on standard benchmarks:

```bash
python scripts/evaluation.py --tasks mmlu --batch_size 8 --limit 100 --device cuda --model_name Qwen/Qwen1.5-MoE-A2.7B
```

### 2. Profile Model and Determine Experts to Prune

Profile the model on MMLU prompts and generate expert pruning metadata:

```bash
python scripts/profile_and_prune.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \ 
    --mmlu_topic stem \
    --sample_size 5 \
    --output_file outputs/statistics/expert_stats.json \
    --device cuda
```

### 3. Evaluate Pruned Model

Evaluate a pruned model using pre-computed expert rankings:

```bash
python scripts/evaluation.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \ 
    --tasks mmlu \
    --batch_size 8 \
    --limit 100 \
    --use_pruned_model \
    --pruned_metadata outputs/statistics/expert_stats.json \
    --pruning_method zero \
    --k 20 \
    --device cuda \
    --output_file outputs/evaluation_results/pruned_results.json
```

### 4. Analyze Router-Expert Correlations

Run correlation analysis across MMLU categories:

```bash
python run_mmlu_categories_correlation.py
```

### 5. Using the Shell Script

For convenience, use the provided shell script:

```bash
bash scripts/run_evaluation.sh
```

## 📁 Project Structure

```
efficient_moe/
├── README.md                           # This file
├── run_mmlu_categories_correlation.py  # MMLU category correlation analysis
├── analyze_expert_dynamics.ipynb       # Expert activation dynamics
├── analyze_routing_statistics.ipynb     # Router behavior analysis
│
├── scripts/                            # Main executable scripts
│   ├── evaluation.py                   # Model evaluation script
│   ├── profile_and_prune.py            # Model profiling and pruning
│   └── run_evaluation.sh               # Evaluation runner script
│
├── utils/                              # Utility modules
│   ├── __init__.py
│   ├── analysis_utils.py               # Statistical analysis functions
│   ├── common_utils.py                 # Common helper functions
│   ├── data_utils.py                   # Dataset processing utilities
│   ├── hook_utils.py                   # Expert activation hook management
│   ├── model_utils.py                  # Model pruning utilities
│   ├── router_utils.py                 # Router logits collection
│   └── visualization_utils.py         # Plotting and visualization
│
└── outputs/                            # Generated outputs
    ├── evaluation_results/             # Evaluation results JSON files
    ├── plots/                          # Generated plots and visualizations
    ├── prune_experts/                  # Pre-computed expert rankings
    └── statistics/                     # Statistical analysis outputs
```

## 🔧 Core Components

### 1. Evaluation Script (`scripts/evaluation.py`)

Main script for evaluating models with or without pruning.

**Usage:**
```bash
python scripts/evaluation.py [OPTIONS]
```

**Key Options:**
- `--model_name`: Name or path of the model to profile & prune (default: `Qwen/Qwen1.5-MoE-A2.7B`)
- `--tasks`: List of evaluation tasks (e.g., `mmlu`, `gsm8k`, `wikitext`)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--limit`: Limit number of examples for quick testing
- `--use_pruned_model`: Enable pruned model evaluation
- `--pruned_metadata`: Path to pruned expert metadata JSON file
- `--k`: Maximum number of experts to prune per layer (default: 20)
- `--pruning_method`: Pruning method - `mask` or `zero` (default: `zero`)
- `--device`: Device for model - `cuda` or `cpu` (default: `cuda`)
- `--output_file`: File path to save evaluation results JSON

**Example:**
```bash
python scripts/evaluation.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \ 
    --tasks mmlu gsm8k \
    --batch_size 16 \
    --use_pruned_model \
    --pruned_metadata outputs/statistics/experts_to_prune.json \
    --pruning_method zero \
    --k 15 \
    --output_file results.json
```

### 2. Profile and Prune Script (`scripts/profile_and_prune.py`)

Profiles the model on various datasets and determines which experts to prune based on activation statistics.

**Usage:**
```bash
python scripts/profile_and_prune.py [OPTIONS]
```

**Key Options:**
- `--model_name`: Name or path of the model to profile & prune (default: `Qwen/Qwen1.5-MoE-A2.7B`)
- `--prompts_file`: Path to JSON file with prompt strings
- `--mmlu_topic`: MMLU topic category (`humanities`, `stem`, `social_sciences`, `other`)
- `--gsm8k`: Use GSM8K dataset for prompts
- `--sample_size`: Maximum samples per MMLU subject (default: 5)
- `--output_file`: Output file path for expert statistics
- `--device`: Device for model (default: `cuda`)

**Example:**
```bash
python scripts/profile_and_prune.py \
    --model_name Qwen/Qwen1.5-MoE-A2.7B \ 
    --mmlu_topic stem \
    --sample_size 10 \
    --output_file outputs/statistics/stem_experts.json \
    --device cuda
```

### 3. Correlation Analysis (`run_mmlu_categories_correlation.py`)

Analyzes correlations between router activations and expert usage across MMLU categories.

**Usage:**
```bash
python run_mmlu_categories_correlation.py
```

**Output:**
- Generates correlation plots in `outputs/plots/`:
  - `mmlu_router_activation_pearson_all_categories.png`
  - `mmlu_router_activation_spearman_all_categories.png`

### 4. Utility Modules

#### `utils/analysis_utils.py`
- Statistical computation functions
- Correlation analysis (Pearson, Spearman)
- Expert ranking and selection utilities

#### `utils/model_utils.py`
- Model pruning functions (`apply_pruning`)
- Expert masking and zeroing implementations

#### `utils/router_utils.py`
- Router logit collection
- Routing pattern analysis

#### `utils/hook_utils.py`
- Expert activation hook management
- Forward hook registration and data collection

#### `utils/data_utils.py`
- Dataset preparation utilities
- MMLU and GSM8K prompt preparation
- Text packing and dataloader creation

#### `utils/visualization_utils.py`
- Plotting functions for matrices, bar charts, correlations
- Visualization utilities for analysis results

#### `utils/common_utils.py`
- Common helper functions
- Expert metadata loading and processing

## 📊 Pruning Methods

### Masking (`--pruning_method mask`)
- Sets router logits to `-∞` for pruned experts
- Prevents tokens from being routed to pruned experts
- More aggressive pruning approach
- Completely removes pruned experts from routing decisions

### Zeroing (`--pruning_method zero`)
- Zeros out outputs from pruned experts
- Tokens may still be routed to pruned experts, but their outputs are nullified
- Gentler pruning approach
- Preserves routing structure while nullifying expert contributions

## 📈 Usage Examples

### Example 1: Complete Pruning Workflow

```bash
# Step 1: Profile the model
python scripts/profile_and_prune.py \
    --mmlu_topic stem \
    --sample_size 10 \
    --output_file outputs/statistics/stem_profile.json

# Step 2: Evaluate pruned model
python scripts/evaluation.py \
    --tasks mmlu \
    --use_pruned_model \
    --pruned_metadata outputs/statistics/stem_profile.json \
    --pruning_method zero \
    --k 20 \
    --output_file outputs/evaluation_results/pruned_mmlu.json
```

### Example 2: Python API Usage

```python
from utils.model_utils import apply_pruning
from utils.common_utils import get_experts_to_prune_from_json
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")

# Load expert pruning metadata
experts_to_prune = get_experts_to_prune_from_json(
    path="outputs/statistics/experts_to_prune.json",
    k=20
)

# Apply pruning
apply_pruning(model, experts_to_prune, mode="zero")
```

### Example 3: Data Processing

```python
from utils.data_utils import prepare_mmlu_prompts, create_packed_dataloader

# Prepare MMLU prompts
prompts = prepare_mmlu_prompts(
    topic="stem",
    max_samples_per_subject=5
)

# Create packed dataloader
loader = create_packed_dataloader(
    tokenizer=tokenizer,
    dataset_name="brando/small-c4-dataset",
    split="train",
    sample_size=512,
    max_length=512
)
```

### Example 4: Visualization

```python
from utils.visualization_utils import plot_matrix, plot_bar

# Plot expert usage matrix
plot_matrix(
    expert_usage_matrix,
    title="Expert Usage Patterns",
    xlabel="Expert ID",
    ylabel="Layer"
)

# Plot expert frequencies
plot_bar(
    expert_frequencies,
    title="Expert Usage Frequency",
    xlabel="Expert ID",
    ylabel="Frequency"
)
```

## 📓 Analysis Notebooks

The project includes Jupyter notebooks for interactive analysis:

- **`analyze_expert_dynamics.ipynb`**: 
  - Expert activation dynamics and patterns
  - Temporal analysis of expert usage
  - Activation monitoring across layers

- **`analyze_routing_statistics.ipynb`**: 
  - Router behavior analysis and statistics
  - Routing pattern visualization
  - Expert ranking by various criteria

These notebooks provide:
- Interactive exploration of expert behavior
- Custom analysis workflows
- Visualization of routing patterns
- Performance impact assessment

## ⚙️ Configuration

### Model Configuration

The project is configured for the **Qwen1.5-MoE-A2.7B** model by default. To use a different model:

1. Update the `model_name` variable in scripts or notebooks
2. Ensure the model has MoE layers with the expected structure
3. Adjust model loading parameters as needed

### Pruning Configuration

Expert pruning can be configured through:

- **Pre-computed rankings**: Use existing JSON files in `outputs/statistics/` or `outputs/prune_experts/`
- **Custom rankings**: Generate your own expert rankings using `profile_and_prune.py`
- **Pruning parameters**: 
  - `k`: Number of experts to prune per layer
  - `pruning_method`: `mask` or `zero`
  - Selection criteria: Based on mean, variance, frequency, or probability

### Output Directories

The project uses the following output structure:
- `outputs/evaluation_results/`: Evaluation result JSON files
- `outputs/plots/`: Generated plots and visualizations
- `outputs/statistics/`: Statistical analysis outputs and expert rankings
- `outputs/prune_experts/`: Pre-computed expert pruning metadata

## 📋 Evaluation Tasks

The evaluation script supports various tasks from the LM-Eval framework:

### Common Tasks
- **`mmlu`**: Massive Multitask Language Understanding
- **`gsm8k`**: Grade school math problems
- **`wikitext`**: Wikipedia text perplexity
- **`hellaswag`**: Commonsense reasoning
- **`arc`**: AI2 reasoning challenge
- **`winogrande`**: Common sense reasoning
- **`truthfulqa`**: Truthful question answering

### Usage
```bash
# Single task
python scripts/evaluation.py --tasks mmlu

# Multiple tasks
python scripts/evaluation.py --tasks mmlu gsm8k hellaswag arc
```

**Note**: This project is designed for research purposes. Ensure you have appropriate computational resources (GPU) for running evaluations and analyses on MoE models.
