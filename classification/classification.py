#classification.py
import argparse
import gc
import signal
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import tqdm
from datetime import datetime
from pathlib import Path

# Import your organized models
from models import (
    create_wrn_16_2, create_wrn_16_4,
    create_qwrn_16_2, create_qwrn_16_4, create_qrn_34, create_qrn34_imagenet,
    create_qwrn_50_2_imagenet, create_qrn18_imagenet, create_qrn_18, create_qwrn16_4_imagenet
)
from utils.experiment_manager import ExperimentManager
from utils.data_loading import get_data_loaders, get_dataset_info
from utils.training import train_epoch, evaluate, count_parameters, evaluate_with_top5
 
device = torch.device('cuda')
 
def handle_keyboard_interrupt(signum, frame):
    """Custom handler for keyboard interrupt to ensure clean exit"""
    print("\n\n🛑 Training interrupted by user. Cleaning up...")
   
    try:
        if 'exp_manager' in globals():
            exp_manager.save_interrupt_checkpoint(model, optimizer, scheduler, epoch)
            exp_manager.finalize()
    except Exception as e:
        print(f"Error during interrupt cleanup: {e}")
   
    sys.exit(0)
 
# Register the keyboard interrupt handler
signal.signal(signal.SIGINT, handle_keyboard_interrupt)
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='Train models on various datasets')
    parser.add_argument('--model', choices=['wrn16_2', 'wrn16_4', 'qwrn16_2', 'qwrn16_4', 'qrn34', 'qrn34_imagenet',
                                            'qwrn50_2', 'qrn18', 'qrn18_i', 'qwrn16_4i'],
                        default='qwrn16_4', help='Model architecture')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn', 'imagenet'], default='imagenet',
                        help='Dataset to use')
    parser.add_argument('--mapping', choices=['poincare', 'hamilton', 'raw_normalized', 'mean_brightness'],
                        default='poincare', help='Mapping strategy for quaternion models')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--bs', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--num_augs', type=int, default=1, help='Number of augmentations')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of data loading workers')
    parser.add_argument('--save_freq', type=int, default=10, help='Save plots every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., experiments/exp_name/checkpoints/checkpoint_epoch_050.pth)')
    parser.add_argument('--resume_best', action='store_true',
                        help='Resume from best_model.pth in the experiment directory')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='Experiment directory to resume from (finds latest checkpoint automatically)')
    return parser.parse_args()
 
 
def create_model(model_name, num_classes, mapping_type=None):
    """Factory function to create models"""
    model_factory = {
        'wrn16_2': lambda: create_wrn_16_2(num_classes=num_classes),
        'wrn16_4': lambda: create_wrn_16_4(num_classes=num_classes),
        'qwrn16_2': lambda: create_qwrn_16_2(num_classes=num_classes, mapping_type=mapping_type),
        'qwrn16_4': lambda: create_qwrn_16_4(num_classes=num_classes, mapping_type=mapping_type),
        'qrn34': lambda: create_qrn_34(num_classes=num_classes, mapping_type=mapping_type),
       
        # IMAGENET:
        'qrn34_imagenet': lambda: create_qrn34_imagenet(num_classes=num_classes, mapping_type=mapping_type),
        'qwrn50_2': lambda: create_qwrn_50_2_imagenet(num_classes=num_classes, mapping_type=mapping_type),
        'qrn18_i': lambda: create_qrn18_imagenet(num_classes=num_classes, mapping_type=mapping_type),
        'qrn18': lambda: create_qrn_18(num_classes=num_classes, mapping_type=mapping_type),
        'qwrn16_4i': lambda: create_qwrn16_4_imagenet(num_classes=num_classes, mapping_type=mapping_type)
    }
   
    if model_name not in model_factory:
        raise ValueError(f"Unknown model: {model_name}")
   
    return model_factory[model_name]()
 
 
def main():
    global exp_manager, model, optimizer, scheduler, epoch   # For interrupt handler
   
    args = parse_args()
   
    # Check if resuming from checkpoint
    start_epoch = 0
    checkpoint_data = None
    
    if args.resume or args.resume_best or args.exp_dir:
        checkpoint_path = None
        
        if args.resume:
            # Direct path to checkpoint
            checkpoint_path = Path(args.resume)
        elif args.exp_dir:
            # Find latest checkpoint in experiment directory
            exp_path = Path(args.exp_dir)
            if args.resume_best:
                checkpoint_path = exp_path / 'checkpoints' / 'best_model.pth'
            else:
                # Find latest checkpoint
                checkpoints = list((exp_path / 'checkpoints').glob('checkpoint_epoch_*.pth'))
                if checkpoints:
                    checkpoint_path = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                else:
                    # Check for interrupt checkpoint
                    interrupt_path = exp_path / 'checkpoints' / 'interrupt_checkpoint.pth'
                    if interrupt_path.exists():
                        checkpoint_path = interrupt_path
        
        if checkpoint_path and checkpoint_path.exists():
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            start_epoch = checkpoint_data['epoch'] + 1
            
            # Use the same experiment directory
            if args.exp_dir:
                exp_manager = ExperimentManager(
                    base_dir=str(Path(args.exp_dir).parent),
                    exp_name=Path(args.exp_dir).name,
                    model_type=args.model,
                    dataset=args.dataset,
                    mapping=args.mapping
                )
                # Load existing metrics
                if 'metrics' in checkpoint_data:
                    exp_manager.metrics = checkpoint_data['metrics']
            else:
                # Create new experiment but note it's a continuation
                exp_manager = ExperimentManager(
                    exp_name=f"{args.exp_name or args.model}_resumed",
                    model_type=args.model,
                    dataset=args.dataset,
                    mapping=args.mapping
                )
        else:
            print(f"⚠️  Checkpoint not found at {checkpoint_path}")
            return
    else:
        # Normal new experiment
        exp_manager = ExperimentManager(
            exp_name=args.exp_name,
            model_type=args.model,
            dataset=args.dataset,
            mapping=args.mapping
        )

    # Save configuration
    config = {
        'model': args.model,
        'dataset': args.dataset,
        'mapping': args.mapping,
        'data_root': args.data_root,
        'batch_size': args.bs,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'num_augmentations': args.num_augs,
        'timestamp': datetime.now().isoformat()
    }
    exp_manager.save_config(config)
   
    # Setup data and model
    train_loader, test_loader, NUM_CLASSES = get_data_loaders(
        batch_size=args.bs,
        augmentations_per_image=args.num_augs,
        num_workers=args.num_workers,
        dataset_type=args.dataset,
        data_root=args.data_root
    )
   
    # Create model
    model = create_model(args.model, NUM_CLASSES, args.mapping)
    model = model.to(device)
    
    # Load checkpoint weights if resuming
    if checkpoint_data is not None:
        print(f"🔄 Restoring model weights from epoch {checkpoint_data['epoch']}")
        model.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"✅ Model weights restored successfully")
   
    # Count and save parameters
    num_params = count_parameters(model)
    exp_manager.total_parameters = num_params
    print(f'📊 Total trainable parameters: {num_params:,}')
   
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                               weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    
    # Restore optimizer and scheduler states if resuming
    if checkpoint_data is not None:
        print(f"🔄 Restoring optimizer and scheduler states")
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        if checkpoint_data['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        print(f"✅ Optimizer and scheduler states restored successfully")
 
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    # )
    # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.epochs - 5, eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[5])
    scaler = GradScaler
    # Training loop
    best_acc = 0
    
    # Initialize best_acc from checkpoint if resuming
    if checkpoint_data is not None and 'test_acc_top1' in checkpoint_data:
        best_acc = checkpoint_data['test_acc_top1']
        print(f"🎯 Resuming with best accuracy so far: {best_acc:.2f}%")
    pbar = tqdm.tqdm(total=args.epochs, desc='Training Progress', position=0)
   
    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device)
       
        # Validation
        # test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        test_loss, test_acc_top1, test_acc_top5 = evaluate_with_top5(model, test_loader, criterion, device)
 
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
       
        # Update metrics
        # exp_manager.update_metrics(epoch, train_acc, test_acc, train_loss, test_loss, current_lr)
        exp_manager.update_metrics(epoch, train_acc, test_acc_top1, test_acc_top5, train_loss, test_loss, current_lr)
 
        # Save checkpoint
        # is_best = test_acc > best_acc
        # if is_best:
        #     best_acc = test_acc
       
        # Track best Top-1 accuracy
        is_best = test_acc_top1 > best_acc
        if is_best:
            best_acc = test_acc_top1
 
        exp_manager.save_checkpoint(model, optimizer, scheduler, epoch, test_acc_top1, is_best)
        # exp_manager.save_checkpoint(model, optimizer, scheduler, epoch, test_acc)
 
        # Save plots periodically
        if (epoch + 1) % args.save_freq == 0:
            exp_manager.save_plots()
       
        # Update progress bar
        pbar.update(1)
        # pbar.set_postfix({
        #     'Train Acc': f'{train_acc:.2f}%',
        #     'Test Acc': f'{test_acc:.2f}%',
        #     'Best': f'{best_acc:.2f}%',
        #     'LR': f'{current_lr:.6f}'
        # })
        pbar.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Top-1': f'{test_acc_top1:.2f}%',
            'Top-5': f'{test_acc_top5:.2f}%',
            'Best': f'{best_acc:.2f}%',
            'LR': f'{current_lr:.6f}'
        })
       
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
   
    pbar.close()
   
    # Finalize experiment
    exp_manager.finalize()
    print(f'🎯 Best test accuracy: {best_acc:.2f}%')
 
 
if __name__ == '__main__':
    main()