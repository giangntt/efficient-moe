"""
Checkpoint management utilities for DiEP
Handles saving and loading of models, masks, and training state
"""

import torch
import os
import json
from datetime import datetime
from typing import Dict, Optional, Any
import shutil
from pathlib import Path


class CheckpointManager:
    """
    Manages checkpoints for DiEP training.
    Handles saving/loading of model, masks, optimizer state, and metadata.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep (None = keep all)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        
        # Track saved checkpoints
        self.checkpoint_history = []
    
    def save_checkpoint(
        self,
        epoch: int,
        prunable_model,
        mask_controller,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        prefix: str = "checkpoint"
    ) -> str:
        """
        Save a complete checkpoint.
        
        Args:
            epoch: Current epoch number
            prunable_model: Prunable model instance
            mask_controller: Mask controller instance
            optimizer: Optimizer instance (optional)
            scheduler: Learning rate scheduler (optional)
            metrics: Training metrics (optional)
            is_best: Whether this is the best checkpoint
            prefix: Checkpoint filename prefix
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"{prefix}_epoch_{epoch}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'mask_controller_state': mask_controller.state_dict(),
            'mask_statistics': mask_controller.get_statistics(),
            'metrics': metrics or {},
        }
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint['optimizer_state'] = optimizer.state_dict()
        
        # Add scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save model separately (large file)
        model_path = self.checkpoint_dir / f"{checkpoint_name}_model"
        prunable_model.model.save_pretrained(model_path)
        print(f"Model saved: {model_path}")
        
        # Save metadata as JSON
        metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        self._save_metadata(metadata_path, checkpoint)
        
        # Track checkpoint
        self.checkpoint_history.append({
            'epoch': epoch,
            'checkpoint_path': str(checkpoint_path),
            'model_path': str(model_path),
            'is_best': is_best
        })
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            best_model_path = self.checkpoint_dir / "best_model"
            
            shutil.copy(checkpoint_path, best_path)
            if best_model_path.exists():
                shutil.rmtree(best_model_path)
            shutil.copytree(model_path, best_model_path)
            
            print(f"Best checkpoint updated: {best_path}")
        
        # Clean old checkpoints
        if self.keep_last_n is not None:
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        mask_controller,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        device: str = "cuda"
    ) -> Dict:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            mask_controller: Mask controller instance to load into
            optimizer: Optimizer instance to load into (optional)
            scheduler: Scheduler instance to load into (optional)
            device: Device to load tensors to
            
        Returns:
            Checkpoint dictionary with metadata
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load mask controller state
        mask_controller.load_state_dict(checkpoint['mask_controller_state'])
        print("  Mask controller state loaded")
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("  Optimizer state loaded")
        
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            print("  Scheduler state loaded")
        
        print(f"  Checkpoint from epoch {checkpoint['epoch']}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        
        return checkpoint
    
    def load_best_checkpoint(
        self,
        mask_controller,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
        device: str = "cuda"
    ) -> Optional[Dict]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pt"
        
        if not best_path.exists():
            print("No best checkpoint found")
            return None
        
        return self.load_checkpoint(
            str(best_path), mask_controller, optimizer, scheduler, device
        )
    
    def _save_metadata(self, path: Path, checkpoint: Dict):
        """Save checkpoint metadata as JSON."""
        metadata = {
            'epoch': checkpoint['epoch'],
            'timestamp': checkpoint['timestamp'],
            'mask_statistics': checkpoint['mask_statistics'],
            'metrics': checkpoint.get('metrics', {})
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return
        
        # Sort by epoch
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        
        # Keep best checkpoint and last N checkpoints
        to_remove = []
        for entry in sorted_history[:-self.keep_last_n]:
            if not entry.get('is_best', False):
                to_remove.append(entry)
        
        # Remove old checkpoints
        for entry in to_remove:
            try:
                # Remove checkpoint file
                if os.path.exists(entry['checkpoint_path']):
                    os.remove(entry['checkpoint_path'])
                
                # Remove model directory
                if os.path.exists(entry['model_path']):
                    shutil.rmtree(entry['model_path'])
                
                # Remove metadata
                metadata_path = entry['checkpoint_path'].replace('.pt', '_metadata.json')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                
                # Remove from history
                self.checkpoint_history.remove(entry)
                
                print(f"Removed old checkpoint: epoch {entry['epoch']}")
            except Exception as e:
                print(f"Error removing checkpoint: {e}")
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return latest['checkpoint_path']


def save_pruned_model(
    prunable_model,
    tokenizer,
    mask_controller,
    output_dir: str,
    save_format: str = "huggingface"
):
    """
    Save the final pruned model.
    
    Args:
        prunable_model: Pruned model instance
        tokenizer: Tokenizer instance
        mask_controller: Mask controller with final masks
        output_dir: Output directory
        save_format: Format to save (huggingface, pytorch, gguf)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving pruned model to: {output_dir}")
    
    # Get final statistics
    mask_stats = mask_controller.get_statistics()
    compression_ratio = prunable_model.get_compression_ratio()
    total_sparsity = prunable_model.get_total_sparsity()
    
    if save_format == "huggingface":
        # Save as HuggingFace model
        prunable_model.model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"  Model saved in HuggingFace format")
    
    elif save_format == "pytorch":
        # Save as PyTorch checkpoint
        torch.save({
            'model_state_dict': prunable_model.model.state_dict(),
            'mask_statistics': mask_stats,
            'compression_ratio': compression_ratio,
            'total_sparsity': total_sparsity
        }, output_path / "model.pt")
        tokenizer.save_pretrained(output_path)
        print(f"  Model saved in PyTorch format")
    
    else:
        raise ValueError(f"Unknown save format: {save_format}")
    
    # Save pruning configuration
    config_path = output_path / "pruning_config.json"
    pruning_config = {
        'mask_statistics': mask_stats,
        'compression_ratio': compression_ratio,
        'total_sparsity': total_sparsity,
        'layer_wise_stats': [
            {
                'layer_idx': stat['layer_idx'],
                'num_experts': stat['num_experts'],
                'num_active': stat['num_active'],
                'sparsity': stat['sparsity']
            }
            for stat in mask_stats['layer_stats']
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(pruning_config, f, indent=2)
    
    print(f"  Pruning config saved: {config_path}")
    
    # Save mask weights
    mask_path = output_path / "masks.pt"
    torch.save(mask_controller.state_dict(), mask_path)
    print(f"  Mask weights saved: {mask_path}")
    
    # Save README
    readme_path = output_path / "README.md"
    _create_model_readme(readme_path, pruning_config)
    print(f"  README saved: {readme_path}")
    
    print(f"\nPruned model saved successfully!")
    print(f"  Compression ratio: {compression_ratio:.2%}")
    print(f"  Total sparsity: {total_sparsity:.2%}")
    print(f"  Estimated speedup: {1 / compression_ratio:.2f}x")


def _create_model_readme(path: Path, config: Dict):
    """Create README for pruned model."""
    content = f"""# Pruned Qwen MoE Model

This model has been pruned using DiEP (Differentiable Expert Pruning).

## Pruning Statistics

- **Compression Ratio**: {config['compression_ratio']:.2%}
- **Total Sparsity**: {config['total_sparsity']:.2%}
- **Estimated Speedup**: {1 / config['compression_ratio']:.2f}x

## Layer-wise Statistics

| Layer | Total Experts | Active Experts | Sparsity |
|-------|---------------|----------------|----------|
"""
    
    for stat in config['layer_wise_stats']:
        content += f"| {stat['layer_idx']} | {stat['num_experts']} | {stat['num_active']} | {stat['sparsity']:.2%} |\n"
    
    content += """
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Use as normal
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
```

## Citation

If you use this pruned model, please cite the DiEP paper:

```bibtex
@article{diep2024,
  title={DiEP: Differentiable Expert Pruning for Mixture-of-Experts Models},
  author={...},
  journal={arXiv preprint arXiv:2509.16105},
  year={2024}
}
```
"""
    
    with open(path, 'w') as f:
        f.write(content)


def load_pruned_model(model_path: str, device: str = "cuda"):
    """
    Load a saved pruned model.
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model on
        
    Returns:
        (model, tokenizer, pruning_config)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading pruned model from: {model_path}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load pruning config
    config_path = Path(model_path) / "pruning_config.json"
    pruning_config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            pruning_config = json.load(f)
        print(f"  Compression ratio: {pruning_config['compression_ratio']:.2%}")
        print(f"  Total sparsity: {pruning_config['total_sparsity']:.2%}")
    
    return model, tokenizer, pruning_config