# Efficient MoE: Mixture-of-Experts Model Pruning and Analysis

A research project for analyzing, pruning, and evaluating Mixture-of-Experts (MoE) models to improve their efficiency and performance. This project focuses on the Qwen1.5-MoE-A2.7B model and provides comprehensive analysis tools, pruning strategies, and evaluation frameworks for MoE model optimization.

## 🚀 Features

- **Expert Analysis**: Comprehensive analysis of router behavior and expert activation patterns
- **Pruning Strategies**: Implementation of multiple pruning methods (masking and zeroing)
- **Performance Evaluation**: Extensive evaluation framework using LM-Eval
- **Visualization Tools**: Advanced plotting and analysis of expert usage patterns
- **Data Processing**: Efficient text packing and dataset handling for large-scale analysis

## 📁 Repository Structure

```
efficient_moe/
├── README.md                           # This file
├── evaluation.py                       # Main evaluation script
├── model_utils.py                      # Model pruning utilities
├── common_utils.py                     # Common helper functions
├── data_utils.py                       # Dataset processing utilities
├── visualization_utils.py              # Plotting and visualization tools
├── run_evaluation.sh                   # Evaluation script runner
├── analyze_router_behavior.ipynb       # Router behavior analysis notebook
├── analyze_expert_activation.ipynb     # Expert activation analysis notebook
└── prune_experts/                      # Pre-computed expert rankings
    ├── sorted_by_freq.json            # Experts sorted by frequency
    ├── sorted_by_prob.json            # Experts sorted by probability
    ├── super_experts_ids.json         # Super expert identifiers
    └── super_experts.json             # Super expert data
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets lm-eval matplotlib seaborn tqdm
```

## 🎯 Quick Start

### 1. Basic Evaluation

Run evaluation on the Qwen1.5-MoE-A2.7B model:

```bash
python evaluation.py --tasks gsm8k --batch_size 8 --limit 100 --device cuda
```

### 2. Pruned Model Evaluation

Evaluate a pruned model using pre-computed expert rankings:

```bash
python evaluation.py \
    --tasks gsm8k \
    --batch_size 8 \
    --limit 100 \
    --use_pruned_model \
    --pruned_metadata "prune_experts/super_experts_ids.json" \
    --mode least \
    --pruning_method zero \
    --k 10 \
    --device cuda
```

### 3. Using the Shell Script

For convenience, use the provided shell script:

```bash
bash run_evaluation.sh
```

## 📊 Analysis Notebooks

### Router Behavior Analysis

The `analyze_router_behavior.ipynb` notebook provides:

- Router logit collection and analysis
- Expert usage frequency analysis
- Routing pattern visualization
- Expert ranking by various criteria

### Expert Activation Analysis

The `analyze_expert_activation.ipynb` notebook includes:

- Expert activation monitoring
- Activation pattern analysis
- Performance impact assessment
- Visualization of expert contributions

## 🔧 Project Components

### Evaluation Script

```bash
python evaluation.py [OPTIONS]
```

**Options:**
- `--tasks`: List of evaluation tasks (default: ['wikitext'])
- `--batch_size`: Batch size for evaluation (default: 8)
- `--limit`: Limit number of examples for testing (default: None)
- `--use_pruned_model`: Enable pruned model evaluation
- `--pruned_metadata`: Path to pruned expert metadata JSON
- `--mode`: Expert selection strategy ('least' or 'most')
- `--pruning_method`: Pruning method ('mask' or 'zero')
- `--k`: Number of experts to prune per layer (default: 5)
- `--device`: Device for model ('cuda' or 'cpu')
- `--output_file`: File to save results JSON

### Pruning Methods

#### Masking (`--pruning_method mask`)
- Sets router logits to -∞ for pruned experts
- Prevents tokens from being routed to pruned experts
- More aggressive pruning approach

#### Zeroing (`--pruning_method zero`)
- Zeros out outputs from pruned experts
- Tokens may still be routed to pruned experts
- Gentler pruning approach

### Expert Selection Modes

#### Least Used (`--mode least`)
- Prunes the least frequently used experts
- Based on routing frequency analysis

#### Most Used (`--mode most`)
- Prunes the most frequently used experts
- More aggressive approach

## 📈 Usage Examples

### 1. Analyze Expert Usage Patterns

```python
from common_utils import get_topk_experts_from_json

# Load expert rankings
experts_to_prune = get_topk_experts_from_json(
    path="prune_experts/sorted_by_freq.json",
    top_k=5,
    mode="least"
)
```

### 2. Apply Pruning to Model

```python
from model_utils import apply_pruning

# Apply zero-based pruning
apply_pruning(model, experts_to_prune, mode="zero")
```

### 3. Create Packed Dataset

```python
from data_utils import create_packed_dataloader

# Create efficient packed dataloader
loader = create_packed_dataloader(
    tokenizer=tokenizer,
    dataset_name="brando/small-c4-dataset",
    split="train",
    sample_size=512,
    max_length=512
)
```

### 4. Visualization

```python
from visualization_utils import plot_matrix, plot_bar

# Plot expert usage matrix
plot_matrix(expert_usage_matrix, title="Expert Usage Patterns")

# Plot expert frequencies
plot_bar(expert_frequencies, title="Expert Usage Frequency")
```

## 🎛️ Configuration

### Model Configuration

The repository is configured for the Qwen1.5-MoE-A2.7B model by default. To use a different model:

1. Update the `model_name` variable in `evaluation.py`
2. Ensure the model has MoE layers with the expected structure
3. Adjust the model loading parameters as needed

### Pruning Configuration

Expert pruning can be configured through:

- **Pre-computed rankings**: Use existing JSON files in `prune_experts/`
- **Custom rankings**: Generate your own expert rankings using the analysis notebooks
- **Pruning parameters**: Adjust `k` (number of experts to prune) and `mode` (selection strategy)

## 📋 Evaluation Tasks

The evaluation script supports various tasks from the LM-Eval framework:

- `gsm8k`: Grade school math problems
- `wikitext`: Wikipedia text perplexity
- `hellaswag`: Commonsense reasoning
- `arc`: AI2 reasoning challenge
- And many more...
