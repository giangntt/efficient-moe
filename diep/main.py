"""
Main entry point for DiEP training on Qwen MoE
Provides a unified interface for all pruning operations
"""

import argparse
import torch
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_prunable_qwen_model, create_mask_controller
from pruning import (
    ExpertImportanceScorer,
    DiEPOptimizer,
    visualize_importance_scores,
    ProgressiveSparsityScheduler
)
from utils import (
    prepare_calibration_dataloaders,
    CheckpointManager,
    save_pruned_model,
    load_pruned_model
)


def setup_environment():
    """Setup training environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("Using CPU")
    
    return device


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DiEP: Differentiable Expert Pruning for Qwen MoE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model_name', type=str, 
        default='/home/sora/llm/moe/ckpt/Qwen1.5',
        help='HuggingFace model name or path'
    )
    model_group.add_argument(
        '--target_sparsity', type=float, 
        default=0.5,
        help='Target expert sparsity (0.0-1.0)'
    )
    model_group.add_argument(
        '--use_layerwise', action='store_true',
        default=True,
        help='Use layer-wise adaptive pruning'
    )
    
    # Importance scoring arguments
    importance_group = parser.add_argument_group('Importance Scoring')
    importance_group.add_argument(
        '--importance_init', action='store_true',
        default=True,
        help='Initialize masks with importance scores'
    )
    importance_group.add_argument(
        '--importance_method', type=str,
        default='combined',
        choices=['router', 'gradient', 'activation', 'fisher', 'combined'],
        help='Method for computing expert importance'
    )
    importance_group.add_argument(
        '--importance_samples', type=int,
        default=500,
        help='Number of samples for importance scoring'
    )
    
    # Stage 1 training arguments
    stage1_group = parser.add_argument_group('Stage 1: Mask Learning')
    stage1_group.add_argument(
        '--stage1_epochs', type=int,
        default=10,
        help='Number of epochs for Stage 1'
    )
    stage1_group.add_argument(
        '--stage1_lr', type=float,
        default=1e-3,
        help='Learning rate for Stage 1'
    )
    stage1_group.add_argument(
        '--lambda_sparsity', type=float,
        default=0.1,
        help='Weight for sparsity loss'
    )
    stage1_group.add_argument(
        '--gradient_clip', type=float,
        default=1.0,
        help='Gradient clipping threshold'
    )
    stage1_group.add_argument(
        '--temperature_init', type=float,
        default=5.0,
        help='Initial Gumbel-Softmax temperature'
    )
    stage1_group.add_argument(
        '--temperature_min', type=float,
        default=0.1,
        help='Minimum Gumbel-Softmax temperature'
    )
    
    # Stage 2 training arguments
    stage2_group = parser.add_argument_group('Stage 2: Fine-tuning')
    stage2_group.add_argument(
        '--stage2_epochs', type=int,
        default=3,
        help='Number of epochs for Stage 2 (0 to skip)'
    )
    stage2_group.add_argument(
        '--stage2_lr', type=float,
        default=1e-5,
        help='Learning rate for Stage 2'
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--dataset', type=str,
        default='wikitext',
        help='Dataset name (wikitext, c4, or path to JSON)'
    )
    data_group.add_argument(
        '--num_train_samples', type=int,
        default=5000,
        help='Number of training samples'
    )
    data_group.add_argument(
        '--num_val_samples', type=int,
        default=1000,
        help='Number of validation samples'
    )
    data_group.add_argument(
        '--batch_size', type=int,
        default=1,
        help='Training batch size'
    )
    data_group.add_argument(
        '--max_length', type=int,
        default=512,
        help='Maximum sequence length'
    )
    data_group.add_argument(
        '--num_workers', type=int,
        default=2,
        help='Number of dataloader workers'
    )
    
    # Training control arguments
    training_group = parser.add_argument_group('Training Control')
    training_group.add_argument(
        '--early_stopping_patience', type=int,
        default=5,
        help='Early stopping patience'
    )
    training_group.add_argument(
        '--eval_every', type=int,
        default=1,
        help='Evaluate every N epochs'
    )
    training_group.add_argument(
        '--progressive_sparsity', action='store_true',
        help='Use progressive sparsity scheduling'
    )
    training_group.add_argument(
        '--progressive_steps', type=int,
        default=5,
        help='Number of steps for progressive sparsity'
    )
    
    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output_dir', type=str,
        default='./outputs',
        help='Output directory for checkpoints and logs'
    )
    output_group.add_argument(
        '--save_model_dir', type=str,
        default=None,
        help='Directory to save final pruned model'
    )
    output_group.add_argument(
        '--keep_last_n_checkpoints', type=int,
        default=3,
        help='Number of recent checkpoints to keep'
    )
    
    # System arguments
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument(
        '--device', type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device for training'
    )
    system_group.add_argument(
        '--mixed_precision', action='store_true',
        help='Use mixed precision training'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str,
        default='train',
        choices=['train', 'evaluate', 'importance', 'resume'],
        help='Operation mode'
    )
    parser.add_argument(
        '--resume_from', type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def run_importance_scoring(args, device):
    """Run importance scoring only."""
    print("\n" + "=" * 70)
    print("MODE: Importance Scoring")
    print("=" * 70)
    
    # Load model
    print("\n[1/3] Loading model...")
    prunable_model, tokenizer, moe_layer_indices = create_prunable_qwen_model(
        args.model_name, device=device
    )
    
    # Prepare data
    print("\n[2/3] Preparing calibration data...")
    train_dataloader, _ = prepare_calibration_dataloaders(
        tokenizer,
        dataset_name=args.dataset,
        num_train_samples=args.importance_samples,
        num_val_samples=0,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Compute importance
    print("\n[3/3] Computing importance scores...")
    scorer = ExpertImportanceScorer(prunable_model, device=device)
    
    if args.importance_method == 'combined':
        importance_scores = scorer.compute_combined_importance(
            train_dataloader,
            num_batches=min(100, len(train_dataloader)),
            weights={'router': 0.3, 'gradient': 0.4, 'activation': 0.3}
        )
    else:
        method_map = {
            'router': scorer.compute_router_based_importance,
            'gradient': scorer.compute_gradient_based_importance,
            'activation': scorer.compute_activation_magnitude_importance,
            'fisher': scorer.compute_fisher_information_importance
        }
        importance_scores = method_map[args.importance_method](
            train_dataloader,
            num_batches=min(100, len(train_dataloader))
        )
    
    # Visualize
    visualize_importance_scores(
        importance_scores, 
        f"Expert Importance ({args.importance_method})"
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    importance_file = output_dir / f"importance_scores_{args.importance_method}.pt"
    torch.save(importance_scores, importance_file)
    print(f"\nImportance scores saved to: {importance_file}")
    
    return importance_scores


def run_training(args, device):
    """Run full training pipeline."""
    print("\n" + "=" * 70)
    print("MODE: Training")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    import json
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_file}")
    
    # Load model
    print("\n[1/6] Loading model...")
    prunable_model, tokenizer, moe_layer_indices = create_prunable_qwen_model(
        args.model_name, device=device
    )
    
    experts_per_layer = [
        prunable_model.prunable_layers[idx].num_experts
        for idx in moe_layer_indices
    ]
    
    print(f"  MoE layers: {len(moe_layer_indices)}")
    print(f"  Total experts: {sum(experts_per_layer)}")
    
    # Create mask controller
    print("\n[2/6] Creating mask controller...")
    mask_controller = create_mask_controller(
        moe_layer_indices=moe_layer_indices,
        experts_per_layer=experts_per_layer,
        target_sparsity=args.target_sparsity,
        use_layerwise=args.use_layerwise,
        initial_temp=args.temperature_init,
        min_temp=args.temperature_min
    )
    mask_controller = mask_controller.to(device)
    
    # Initialize with importance (optional)
    if args.importance_init:
        print("\n[3/6] Computing importance scores for initialization...")
        calib_dataloader, _ = prepare_calibration_dataloaders(
            tokenizer,
            dataset_name=args.dataset,
            num_train_samples=args.importance_samples,
            num_val_samples=0,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers
        )
        
        scorer = ExpertImportanceScorer(prunable_model, device=device)
        
        if args.importance_method == 'combined':
            importance_scores = scorer.compute_combined_importance(
                calib_dataloader, num_batches=50
            )
        else:
            method_map = {
                'router': scorer.compute_router_based_importance,
                'gradient': scorer.compute_gradient_based_importance,
                'activation': scorer.compute_activation_magnitude_importance,
                'fisher': scorer.compute_fisher_information_importance
            }
            importance_scores = method_map[args.importance_method](
                calib_dataloader, num_batches=50
            )
        
        mask_controller.initialize_from_importance(importance_scores)
        visualize_importance_scores(importance_scores, "Initialization Importance")
    else:
        print("\n[3/6] Skipping importance initialization")
    
    # Prepare training data
    print("\n[4/6] Preparing training data...")
    train_dataloader, val_dataloader = prepare_calibration_dataloaders(
        tokenizer,
        dataset_name=args.dataset,
        num_train_samples=args.num_train_samples,
        num_val_samples=args.num_val_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        keep_last_n=args.keep_last_n_checkpoints
    )
    
    # Create optimizer
    print("\n[5/6] Creating DiEP optimizer...")
    diep_optimizer = DiEPOptimizer(
        prunable_model=prunable_model,
        mask_controller=mask_controller,
        device=device
    )
    # === BẮT ĐẦU CODE MỚI ĐỂ RESUME ===
    if args.resume_from:
        print(f"\n[!!!] Đang tải checkpoint Giai đoạn 1 từ: {args.resume_from}")
        try:
            # Tải trạng thái của mask_controller từ file .pt
            checkpoint = diep_optimizer.load_checkpoint(args.resume_from)
            print("  Tải checkpoint thành công. Sẽ bỏ qua Giai đoạn 1.")
            
            # Đồng bộ target_sparsity từ checkpoint (quan trọng)
            if 'mask_statistics' in checkpoint and 'target_sparsity' in checkpoint['mask_statistics']:
                 mask_controller.target_sparsity = checkpoint['mask_statistics']['target_sparsity']
                 
        except Exception as e:
            print(f"Lỗi khi tải checkpoint: {e}")
            print("Tiếp tục chạy Giai đoạn 1 từ đầu...")
            args.resume_from = None # Xóa cờ resume nếu lỗi
    # === KẾT THÚC CODE MỚI ===
    # Progressive sparsity scheduler (optional)
    sparsity_scheduler = None
    if args.progressive_sparsity:
        print("\n  Using progressive sparsity scheduling")
        sparsity_scheduler = ProgressiveSparsityScheduler(
            mask_controller=mask_controller,
            initial_sparsity=args.target_sparsity / 2,
            final_sparsity=args.target_sparsity,
            num_steps=args.progressive_steps
        )
    
    # Stage 1: Learn pruning configuration
    if not args.resume_from:
        # Stage 1: Learn pruning configuration
        print("\n[6/6] Stage 1: Learning Pruning Configuration...")
        stage1_stats = diep_optimizer.stage1_learn_pruning_config(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.stage1_epochs,
            learning_rate=args.stage1_lr,
            lambda_sparsity=args.lambda_sparsity,
            gradient_clip=args.gradient_clip,
            anneal_every=1,
            save_dir=str(output_dir / "checkpoints"),
            eval_every=args.eval_every,
            early_stopping_patience=args.early_stopping_patience
        )
    else:
        print("\n[6/6] Bỏ qua Giai đoạn 1 (đã tải từ checkpoint).")
    
    # Get final mask statistics
    print("\n" + "=" * 70)
    print("Stage 1 Complete - Final Mask Statistics")
    print("=" * 70)
    final_stats = mask_controller.get_statistics()
    print(f"Overall Sparsity: {final_stats['overall_sparsity']:.2%}")
    print(f"Active Experts: {final_stats['total_active_experts']}/{final_stats['total_experts']}")
    
    for layer_stat in final_stats['layer_stats']:
        print(f"  Layer {layer_stat['layer_idx']}: "
              f"{layer_stat['num_active']}/{layer_stat['num_experts']} active "
              f"(sparsity: {layer_stat['sparsity']:.2%})")
    
    # Stage 2: Fine-tune pruned model (optional)
    if args.stage2_epochs > 0:
        print("\n" + "=" * 70)
        print("Stage 2: Fine-tuning Pruned Model")
        print("=" * 70)
        
        stage2_stats = diep_optimizer.stage2_finetune_pruned_model(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.stage2_epochs,
            learning_rate=args.stage2_lr,
            gradient_clip=args.gradient_clip,
            save_dir=str(output_dir / "checkpoints"),
            eval_every=args.eval_every
        )
        
        print(f"\nFinal Training Loss: {stage2_stats['loss'][-1]:.4f}")
        print(f"Final Perplexity: {stage2_stats['perplexity'][-1]:.2f}")
    
    # Finalize and save model
    print("\n" + "=" * 70)
    print("Finalizing and Saving Pruned Model")
    print("=" * 70)
    
    prunable_model.finalize_pruning(threshold=0.5)
    
    compression_ratio = prunable_model.get_compression_ratio()
    total_sparsity = prunable_model.get_total_sparsity()
    
    print(f"Compression Ratio: {compression_ratio:.2%}")
    print(f"Total Sparsity: {total_sparsity:.2%}")
    print(f"Estimated Speedup: {1 / compression_ratio:.2f}x")
    
    # Save pruned model
    if args.save_model_dir:
        save_pruned_model(
            prunable_model=prunable_model,
            tokenizer=tokenizer,
            mask_controller=mask_controller,
            output_dir=args.save_model_dir,
            save_format="huggingface"
        )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return prunable_model, mask_controller


def run_evaluation(args, device):
    """Run evaluation on pruned model."""
    print("\n" + "=" * 70)
    print("MODE: Evaluation")
    print("=" * 70)
    
    if not args.save_model_dir:
        print("Error: --save_model_dir required for evaluation mode")
        return
    
    print(f"\nLoading pruned model from: {args.save_model_dir}")
    model, tokenizer, pruning_config = load_pruned_model(
        args.save_model_dir, device=device
    )
    
    if pruning_config:
        print("\nPruning Statistics:")
        print(f"  Compression Ratio: {pruning_config['compression_ratio']:.2%}")
        print(f"  Total Sparsity: {pruning_config['total_sparsity']:.2%}")
    
    # TODO: Add evaluation benchmarks here
    print("\nEvaluation benchmarks would run here...")
    print("(Implement in evaluation module)")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup environment
    if args.device == 'auto':
        device = setup_environment()
    else:
        device = args.device
    
    # Print configuration
    print("\n" + "=" * 70)
    print("DiEP: Differentiable Expert Pruning for Qwen MoE")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"Target Sparsity: {args.target_sparsity * 100}%")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Run appropriate mode
    try:
        if args.mode == 'train':
            run_training(args, device)
        elif args.mode == 'importance':
            run_importance_scoring(args, device)
        elif args.mode == 'evaluate':
            run_evaluation(args, device)
        elif args.mode == 'resume':
            # TODO: Implement resume functionality
            print("Resume mode not yet implemented")
        else:
            print(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()