#!/usr/bin/env python
"""
Model builder for creating neural network models from YAML configurations.
Supports both classification and detection models with quaternion and standard components.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Type

# Import registries
from models.registry import (
    ARCHITECTURE_REGISTRY,
    CONV_REGISTRY,
    NORM_REGISTRY,
    ACTIVATION_REGISTRY,
    POOLING_REGISTRY,
    BLOCK_REGISTRY
)


def get_component_class(registry, component_name: str, default_class=None):
    """
    Get a component class from registry by name with fallback.
    
    Args:
        registry: Component registry to use
        component_name: Name of registered component
        default_class: Default class to use if not found
        
    Returns:
        Component class
    """
    if component_name is None:
        return default_class
    
    try:
        return registry.get(component_name)
    except KeyError:
        if default_class:
            print(f"Warning: {component_name} not found in registry. Using default: {default_class.__name__}")
            return default_class
        else:
            available = ', '.join(registry.list_registered())
            raise ValueError(f"Component {component_name} not found in registry. Available: {available}")


def create_backbone(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model backbone based on configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated backbone model
    """
    model_type = config['model']['type']
    model_name = config['model']['name']
    
    # Get model class from registry
    try:
        # First try to get model by name (more specific)
        model_class = ARCHITECTURE_REGISTRY.get(model_name)
    except KeyError:
        # Fallback to model type (more general)
        try:
            model_class = ARCHITECTURE_REGISTRY.get(model_type)
        except KeyError:
            available = ', '.join(ARCHITECTURE_REGISTRY.list_registered())
            raise ValueError(f"Model {model_name} ({model_type}) not found in registry. Available: {available}")
    
    # Extract parameters for model constructor
    model_params = {
        'num_classes': config['model']['num_classes'],
        'in_channels': config['model'].get('in_channels', 3)
    }
    
    # Add architecture-specific parameters from config
    if 'params' in config['model']:
        model_params.update(config['model']['params'])
    
    # Add quaternion mapping type if it's a quaternion model
    if model_type.startswith('q') and 'mapping_type' in config['model']:
        model_params['mapping_type'] = config['model']['mapping_type']
    
    # Get component classes if specified
    components = config.get('components', {})
    
    # Convolution class
    if 'conv_type' in components:
        model_params['conv_class'] = get_component_class(
            CONV_REGISTRY, 
            components['conv_type'], 
            default_class=nn.Conv2d
        )
    
    # Normalization class
    if 'norm_type' in components:
        model_params['norm_class'] = get_component_class(
            NORM_REGISTRY, 
            components['norm_type'], 
            default_class=nn.BatchNorm2d
        )
    
    # Activation class
    if 'activation_type' in components:
        model_params['activation_class'] = get_component_class(
            ACTIVATION_REGISTRY, 
            components['activation_type'], 
            default_class=nn.ReLU
        )
    
    # Pooling class
    if 'pooling_type' in components:
        model_params['pooling_class'] = get_component_class(
            POOLING_REGISTRY, 
            components['pooling_type'], 
            default_class=nn.AvgPool2d
        )
    
    # If blocks are specified in the config, we need to build a custom model
    if config['model'].get('blocks'):
        return build_model_from_blocks(config)
    
    # Instantiate the model with parameters
    try:
        return model_class(**model_params)
    except TypeError as e:
        # Print available parameters for debugging
        print(f"Error creating model {model_name}: {e}")
        print(f"Available parameters: {model_params}")
        raise


def build_model_from_blocks(config: Dict[str, Any]) -> nn.Module:
    """
    Build a custom model from block specifications.
    
    Args:
        config: Model configuration dictionary with block specifications
        
    Returns:
        Instantiated custom model
    """
    model_type = config['model']['type']
    blocks_config = config['model']['blocks']

    # Get component classes
    components = config.get('components', {})
    
    conv_class = get_component_class(
        CONV_REGISTRY, 
        components.get('conv_type'), 
        default_class=nn.Conv2d
    )
    
    norm_class = get_component_class(
        NORM_REGISTRY, 
        components.get('norm_type'), 
        default_class=nn.BatchNorm2d
    )
    
    activation_class = get_component_class(
        ACTIVATION_REGISTRY, 
        components.get('activation_type'), 
        default_class=nn.ReLU
    )
    

def build_detection_head(config: Dict[str, Any], backbone_channels: List[int]) -> nn.Module:
    """
    Build a detection head based on configuration.
    
    Args:
        config: Model configuration dictionary
        backbone_channels: List of channel dimensions from backbone feature maps
        
    Returns:
        Instantiated detection head
    """
    model_type = config['model']['type']
    head_type = config['model'].get('head_type', 'default')
    is_quaternion = model_type.startswith('q')
    
    # Get head class from registry
    try:
        head_class = ARCHITECTURE_REGISTRY.get(f"{head_type}_head")
    except KeyError:
        if is_quaternion:
            # Try quaternion-specific head
            try:
                head_class = ARCHITECTURE_REGISTRY.get(f"q{head_type}_head")
            except KeyError:
                raise ValueError(f"Detection head {head_type} (quaternion={is_quaternion}) not found in registry.")
        else:
            raise ValueError(f"Detection head {head_type} not found in registry.")
    
    # Extract parameters for head constructor
    head_params = {
        'num_classes': config['model']['num_classes'],
        'backbone_channels': backbone_channels
    }
    
    # Add head-specific parameters from config
    if 'head_params' in config['model']:
        head_params.update(config['model']['head_params'])
    
    # Get component classes if specified
    components = config.get('components', {})
    
    # Convolution class
    if 'conv_type' in components:
        head_params['conv_class'] = get_component_class(
            CONV_REGISTRY, 
            components['conv_type'], 
            default_class=nn.Conv2d
        )
    
    # Normalization class
    if 'norm_type' in components:
        head_params['norm_class'] = get_component_class(
            NORM_REGISTRY, 
            components['norm_type'], 
            default_class=nn.BatchNorm2d
        )
    
    # Activation class
    if 'activation_type' in components:
        head_params['activation_class'] = get_component_class(
            ACTIVATION_REGISTRY, 
            components['activation_type'], 
            default_class=nn.ReLU
        )
    
    # Add quaternion mapping type if it's a quaternion model
    if is_quaternion and 'mapping_type' in config['model']:
        head_params['mapping_type'] = config['model']['mapping_type']
    
    # Instantiate the head with parameters
    return head_class(**head_params)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a complete model based on configuration.
    This is the main entry point for model creation.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_type = config['model']['type']
    task_type = config.get('task_type', 'classification')
    
    # Build backbone
    backbone = create_backbone(config)
    
    # For classification models, the backbone is the full model
    if task_type == 'classification':
        return backbone
    
    # For detection models, we need to add a detection head
    elif task_type == 'detection':
        # Get backbone output channels
        if hasattr(backbone, 'get_output_channels'):
            backbone_channels = backbone.get_output_channels()
        else:
            # Default assumption for most common backbones
            backbone_channels = [256, 512, 1024]  # P3, P4, P5
            print(f"Warning: Could not determine backbone output channels. Using default: {backbone_channels}")
        
        # Build detection head
        head = build_detection_head(config, backbone_channels)
        
