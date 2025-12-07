#!/usr/bin/env python
"""
Utility functions for saving and loading model checkpoints.
"""
import os
import torch
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = "checkpoint.pth"
) -> None:
    """
    Save training checkpoint.
    
    Args:
        state: State dictionary containing model state, optimizer state, etc.
        is_best: Whether this is the best model so far.
        checkpoint_dir: Directory to save checkpoints.
        filename: Name of the checkpoint file.
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # If this is the best model, create a copy
    if is_best:
        best_path = checkpoint_dir / "model_best.pth"
        torch.save(state, best_path)
        logger.info(f"Best model saved to {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Tuple[int, float, float]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Learning rate scheduler to load state into.
        
    Returns:
        Tuple of (epoch, best_accuracy, current_learning_rate)
    """
    if not checkpoint_path.exists():
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0, 0.0
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get training state
    epoch = checkpoint.get('epoch', 0)
    best_accuracy = checkpoint.get('best_accuracy', 0.0)
    current_lr = checkpoint.get('learning_rate', 0.0)
    
    logger.info(f"Loaded checkpoint from epoch {epoch} with accuracy {best_accuracy:.2f}%")
    
    return epoch, best_accuracy, current_lr


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints.
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    # Check if checkpoint directory exists
    if not checkpoint_dir.exists():
        return None
    
    # First check if there's a regular checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint.pth"
    if checkpoint_path.exists():
        return checkpoint_path
    
    # If not, look for any .pth files and sort by modification time
    checkpoint_files = list(checkpoint_dir.glob("*.pth"))
    if not checkpoint_files:
        return None
    
    # Sort by modification time, newest first
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]