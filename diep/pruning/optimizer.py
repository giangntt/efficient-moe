"""
Two-Stage Optimization for DiEP
Stage 1: Learn pruning configuration (mask parameters)
Stage 2: Fine-tune pruned model (optional)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List
from tqdm import tqdm
import os
import json
from datetime import datetime
import bitsandbytes.optim as bnb_optim

class DiEPOptimizer:
    """
    Two-stage optimizer for Differentiable Expert Pruning.
    """
    
    def __init__(
        self,
        prunable_model,
        mask_controller,
        device: str = "cuda"
    ):
        """
        Args:
            prunable_model: PrunableQwenMoE instance
            mask_controller: LayerWiseMaskController instance
            device: Device for training
        """
        self.model = prunable_model
        self.mask_controller = mask_controller
        self.device = device
        
        # Training history
        self.history = {
            'stage1': {'task_loss': [], 'sparsity_loss': [], 'total_loss': [], 'temperature': []},
            'stage2': {'loss': [], 'perplexity': []}
        }
        
    def stage1_learn_pruning_config(
        self,
        train_dataloader,
        val_dataloader: Optional = None,
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        lambda_sparsity: float = 0.1,
        gradient_clip: float = 1.0,
        anneal_every: int = 1,
        save_dir: Optional[str] = None,
        eval_every: int = 1,
        early_stopping_patience: int = 5
    ) -> Dict:
        """
        Stage 1: Learn optimal pruning configuration via gradient descent.
        Only mask parameters are updated; model weights are frozen.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for mask parameters
            lambda_sparsity: Weight for sparsity loss
            gradient_clip: Gradient clipping threshold
            anneal_every: Anneal temperature every N epochs
            save_dir: Directory to save checkpoints
            eval_every: Evaluate every N epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training statistics dictionary
        """
        print("\n" + "=" * 60)
        print("Stage 1: Learning Pruning Configuration")
        print("=" * 60)
        
        # Freeze model parameters
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        # Only optimize mask parameters
        optimizer = torch.optim.Adam(
            self.mask_controller.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Training
            epoch_stats = self._train_epoch_stage1(
                train_dataloader,
                optimizer,
                lambda_sparsity,
                gradient_clip
            )
            
            # Log statistics
            print(f"  Task Loss: {epoch_stats['task_loss']:.4f}")
            print(f"  Sparsity Loss: {epoch_stats['sparsity_loss']:.4f}")
            print(f"  Total Loss: {epoch_stats['total_loss']:.4f}")
            print(f"  Temperature: {epoch_stats['temperature']:.4f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Store history
            self.history['stage1']['task_loss'].append(epoch_stats['task_loss'])
            self.history['stage1']['sparsity_loss'].append(epoch_stats['sparsity_loss'])
            self.history['stage1']['total_loss'].append(epoch_stats['total_loss'])
            self.history['stage1']['temperature'].append(epoch_stats['temperature'])
            
            # Get current mask statistics
            mask_stats = self.mask_controller.get_statistics()
            print(f"  Current Sparsity: {mask_stats['overall_sparsity']:.2%}")
            print(f"  Target Sparsity: {mask_stats['target_sparsity']:.2%}")
            print(f"  Active Experts: {mask_stats['total_active_experts']}/{mask_stats['total_experts']}")
            
            # Validation
            if val_dataloader and (epoch + 1) % eval_every == 0:
                val_stats = self._validate_stage1(val_dataloader, lambda_sparsity)
                print(f"  Val Loss: {val_stats['total_loss']:.4f}")
                
                # Early stopping check
                if val_stats['total_loss'] < best_loss:
                    best_loss = val_stats['total_loss']
                    patience_counter = 0
                    
                    # Save best checkpoint
                    if save_dir:
                        self._save_checkpoint(save_dir, epoch, 'best_stage1')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
            
            # Temperature annealing
            if (epoch + 1) % anneal_every == 0:
                self.mask_controller.anneal_temperature()
            
            # Step scheduler
            scheduler.step()
            
            # Save periodic checkpoint
            if save_dir and (epoch + 1) % 5 == 0:
                self._save_checkpoint(save_dir, epoch, f'epoch_{epoch+1}')
        
        # Unfreeze model parameters
        for param in self.model.model.parameters():
            param.requires_grad = True
        
        print("\nStage 1 Complete!")
        return self.history['stage1']
    
    def _train_epoch_stage1(
        self,
        dataloader,
        optimizer,
        lambda_sparsity: float,
        gradient_clip: float
    ) -> Dict:
        """Single training epoch for Stage 1."""
        self.model.model.train()
        
        total_task_loss = 0.0
        total_sparsity_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training Stage 1")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate soft masks
            masks = self.mask_controller(hard=False)
            self.model.set_all_masks(masks)
            
            # Forward pass
            outputs = self.model(**batch, labels=batch.get('input_ids'))
            task_loss = outputs.loss
            
            # Sparsity loss
            sparsity_loss = self.mask_controller.compute_sparsity_loss(masks)
            
            # Total loss
            loss = task_loss + lambda_sparsity * sparsity_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.mask_controller.parameters(),
                gradient_clip
            )
            
            # Update
            optimizer.step()
            
            # Accumulate statistics
            total_task_loss += task_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'task_loss': f'{task_loss.item():.4f}',
                'sparsity_loss': f'{sparsity_loss.item():.4f}',
                'total_loss': f'{loss.item():.4f}'
            })
        
        return {
            'task_loss': total_task_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'temperature': self.mask_controller.get_current_temperature()
        }
    
    def _validate_stage1(self, dataloader, lambda_sparsity: float) -> Dict:
        """Validation for Stage 1."""
        self.model.model.eval()
        
        total_task_loss = 0.0
        total_sparsity_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                masks = self.mask_controller(hard=False)
                self.model.set_all_masks(masks)
                
                outputs = self.model(**batch, labels=batch.get('input_ids'))
                task_loss = outputs.loss
                sparsity_loss = self.mask_controller.compute_sparsity_loss(masks)
                
                total_task_loss += task_loss.item()
                total_sparsity_loss += sparsity_loss.item()
                num_batches += 1
        
        return {
            'task_loss': total_task_loss / num_batches,
            'sparsity_loss': total_sparsity_loss / num_batches,
            'total_loss': (total_task_loss + lambda_sparsity * total_sparsity_loss) / num_batches
        }
    
    def stage2_finetune_pruned_model(
        self,
        train_dataloader,
        val_dataloader: Optional = None,
        num_epochs: int = 3,
        learning_rate: float = 1e-5,
        gradient_clip: float = 1.0,
        save_dir: Optional[str] = None,
        eval_every: int = 1
    ) -> Dict:
        """
        Stage 2: Fine-tune the pruned model with hard binary masks.
        All model parameters are updated.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate
            gradient_clip: Gradient clipping threshold
            save_dir: Directory to save checkpoints
            eval_every: Evaluate every N epochs
            
        Returns:
            Training statistics dictionary
        """
        print("\n" + "=" * 60)
        print("Stage 2: Fine-tuning Pruned Model")
        print("=" * 60)
        
        # Finalize masks to hard binary
        with torch.no_grad():
            hard_masks = self.mask_controller(hard=True)
            self.model.set_all_masks(hard_masks)
        
        # Freeze mask parameters
        for param in self.mask_controller.parameters():
            param.requires_grad = False
        
        # === BẮT ĐẦU CODE SỬA LỖI OOM GIAI ĐOẠN 2 ===
        print("Freezing model... preparing to unfreeze active experts.")
        
        # 1. Đóng băng TẤT CẢ tham số
        for param in self.model.model.parameters():
            param.requires_grad = False

        trainable_params = []
        non_expert_params_count = 0
        
        # 2. Mở băng các tham số chung (non-MoE)
        # Bao gồm: self_attn, layernorm, embed_tokens, lm_head, 
        # VÀ QUAN TRỌNG: mlp.gate, mlp.shared_expert, mlp.shared_expert_gate
        for name, param in self.model.model.named_parameters():
            # Bộ lọc "mlp.experts." (CÓ DẤU CHẤM) sẽ chỉ lọc ra
            # các tham số BÊN TRONG ModuleList 60 expert
            if "experts" not in name:
                param.requires_grad = True
                trainable_params.append(param)
                non_expert_params_count += param.numel()
        
        print(f"  Unfrozen {non_expert_params_count/1e6:.1f}M non-expert parameters.")
        
        # 3. Mở băng các expert đang hoạt động (active)
        active_expert_params_count = 0
        for layer_idx, layer in self.model.prunable_layers.items():
            mask = hard_masks[layer_idx]
            active_expert_indices = (mask > 0.5).nonzero(as_tuple=True)[0]
            
            for expert_idx in active_expert_indices:
                expert_module = layer.experts[expert_idx]
                for param in expert_module.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                    active_expert_params_count += param.numel()

        print(f"  Unfrozen {active_expert_params_count/1e6:.1f}M active expert parameters.")
        
        total_trainable_count = sum(p.numel() for p in trainable_params)
        print(f"Total trainable parameters: {total_trainable_count/1e6:.1f}M")
        
        optimizer = bnb_optim.AdamW8bit(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Training
            epoch_stats = self._train_epoch_stage2(
                train_dataloader,
                optimizer,
                gradient_clip
            )
            
            print(f"  Train Loss: {epoch_stats['loss']:.4f}")
            print(f"  Train Perplexity: {epoch_stats['perplexity']:.2f}")
            
            self.history['stage2']['loss'].append(epoch_stats['loss'])
            self.history['stage2']['perplexity'].append(epoch_stats['perplexity'])
            
            # Validation
            if val_dataloader and (epoch + 1) % eval_every == 0:
                val_stats = self._validate_stage2(val_dataloader)
                print(f"  Val Loss: {val_stats['loss']:.4f}")
                print(f"  Val Perplexity: {val_stats['perplexity']:.2f}")
                
                if save_dir:
                    self._save_checkpoint(save_dir, epoch, f'stage2_epoch_{epoch+1}')
            
            scheduler.step()
        
        print("\nStage 2 Complete!")
        return self.history['stage2']
    
    def _train_epoch_stage2(
        self,
        dataloader,
        optimizer,
        gradient_clip: float
    ) -> Dict:
        """Single training epoch for Stage 2."""
        self.model.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training Stage 2")
        for batch in pbar:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model(**batch, labels=batch.get('input_ids'))
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.model.parameters(),
                gradient_clip
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
    
    def _validate_stage2(self, dataloader) -> Dict:
        """Validation for Stage 2."""
        self.model.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch, labels=batch.get('input_ids'))
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
    
    def _save_checkpoint(self, save_dir: str, epoch: int, name: str):
        """Save training checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(save_dir, f'{name}.pt')
        
        checkpoint = {
            'epoch': epoch,
            'mask_controller_state': self.mask_controller.state_dict(),
            'history': self.history,
            'mask_statistics': self.mask_controller.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save mask statistics as JSON
        stats_path = os.path.join(save_dir, f'{name}_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(checkpoint['mask_statistics'], f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.mask_controller.load_state_dict(checkpoint['mask_controller_state'])
        self.history = checkpoint.get('history', self.history)
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'Unknown')}")
        
        return checkpoint


class ProgressiveSparsityScheduler:
    """
    Scheduler for progressive sparsity increase.
    Gradually increases target sparsity during training.
    """
    
    def __init__(
        self,
        mask_controller,
        initial_sparsity: float = 0.1,
        final_sparsity: float = 0.5,
        num_steps: int = 10
    ):
        """
        Args:
            mask_controller: Mask controller to update
            initial_sparsity: Starting sparsity
            final_sparsity: Target final sparsity
            num_steps: Number of steps to reach final sparsity
        """
        self.mask_controller = mask_controller
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_steps = num_steps
        self.current_step = 0
        
        # Calculate sparsity increment per step
        self.sparsity_increment = (final_sparsity - initial_sparsity) / num_steps
        
        # Set initial sparsity
        self.mask_controller.target_sparsity = initial_sparsity
    
    def step(self):
        """Increase target sparsity."""
        if self.current_step < self.num_steps:
            self.current_step += 1
            new_sparsity = self.initial_sparsity + self.current_step * self.sparsity_increment
            self.mask_controller.target_sparsity = min(new_sparsity, self.final_sparsity)
            
            return self.mask_controller.target_sparsity
        return self.final_sparsity
    
    def get_current_sparsity(self) -> float:
        """Get current target sparsity."""
        return self.mask_controller.target_sparsity