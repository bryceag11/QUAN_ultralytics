#!/usr/bin/env python
"""
Simplified configuration handler for training experiments.
"""
import os
import yaml
import datetime
from pathlib import Path


def expand_config(config):
    """Expand simplified config to full format with smart defaults."""
    full_config = {
        'experiment': {
            'name': config.get('name', 'default'),
            'description': config.get('description', 'Experiment with ' + config.get('model_type', 'model')),
            'seed': config.get('seed', 42)
        },
        'dataset': {
            'name': config.get('dataset', 'cifar10'),
            'batch_size': config.get('batch', 128),
            'num_workers': config.get('workers', 4),
            'augmentations_per_image': config.get('augmentations', 1),
            'data_dir': config.get('data_dir', './data'),
            'cutout': config.get('cutout', False),
            'cutout_length': config.get('cutout_length', 16),
            'mixup': config.get('mixup', False),
            'mixup_alpha': config.get('mixup_alpha', 0.2)
        },
        'model': {
            'type': config.get('model_type', 'resnet'),
            'name': config.get('backbone', 'resnet18'),
            'num_classes': config.get('nc', 10),
            'in_channels': config.get('in_channels', 3)
        },
        'training': {
            'epochs': config.get('epochs', 100),
            'learning_rate': config.get('lr', 0.001),
            'optimizer': config.get('optimizer', 'adamw'),
            'optimizer_params': {
                'weight_decay': config.get('weight_decay', 0.01),
                'beta1': config.get('beta1', 0.9),
                'beta2': config.get('beta2', 0.999),
                'momentum': config.get('momentum', 0.9)
            },
            'scheduler': config.get('scheduler', 'cosine'),
            'scheduler_params': {
                'eta_min': config.get('min_lr', 0.000001),
                'T_0': config.get('t0', 10),
                'T_mult': config.get('t_mult', 2),
                'warmup_epochs': config.get('warmup_epochs', 5)
            },
            'gradient_clip': config.get('grad_clip', 1.0),
            'label_smoothing': config.get('label_smoothing', 0.1)
        },
        'logging': {
            'save_freq': config.get('save_freq', 10),
            'log_freq': config.get('log_freq', 100),
            'use_tensorboard': config.get('use_tensorboard', True),
            'visualize_features': config.get('visualize_features', False)
        }
    }
    
    # Auto-detect quaternion components if model_type starts with 'q'
    if config.get('model_type', '').startswith('q'):
        full_config['model']['mapping_type'] = config.get('mapping', 'poincare')
        full_config['components'] = {
            'conv_type': 'QConv2D',
            'norm_type': 'IQBN',
            'activation_type': 'QSiLU',
            'pooling_type': 'QuaternionAvgPool'
        }
    else:
        full_config['components'] = {
            'conv_type': 'nn.Conv2d',
            'norm_type': 'nn.BatchNorm2d',
            'activation_type': 'nn.SiLU',
            'pooling_type': 'nn.AvgPool2d'
        }
    
    # For detection tasks, add detection-specific settings
    if config.get('task', 'classify') == 'detect':
        full_config['task_type'] = 'detection'
        # Add detection-specific config parameters
        
    return full_config


def load_simplified_config(config_path):
    """Load and expand simplified configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand the simplified config to full format
    return expand_config(config)


def setup_experiment_dir(config):
    """Create experiment directory structure based on config."""
    # Get current date
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Find latest experiment number for today
    exp_num = 1
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        existing_exps = [d for d in experiments_dir.iterdir() 
                        if d.is_dir() and d.name.startswith(date_str)]
        exp_num = len(existing_exps) + 1
    
    # Create experiment name
    model_type = config['model']['type']
    model_name = config['model']['name']
    exp_name = f"{date_str}_{exp_num:02d}_{model_type}_{model_name}"
    
    # Create experiment directory
    exp_dir = experiments_dir / exp_name
    checkpoints_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    visualizations_dir = exp_dir / "visualizations"
    
    # Create directories
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    
    # Save full configuration
    with open(exp_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return {
        'exp_dir': exp_dir,
        'checkpoints_dir': checkpoints_dir,
        'logs_dir': logs_dir,
        'visualizations_dir': visualizations_dir
    }