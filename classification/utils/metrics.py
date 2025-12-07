#!/usr/bin/env python
"""
Utilities for tracking and visualizing training metrics.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import time
import torch
logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Logger for tracking and visualizing training and evaluation metrics.
    
    Args:
        save_dir: Directory to save metrics files and visualizations.
        experiment_name: Name of the experiment.
        task_type: Type of task (classification, detection, etc.).
    """
    def __init__(
        self, 
        save_dir: Union[str, Path], 
        experiment_name: str = "experiment",
        task_type: str = "classification"
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.task_type = task_type.lower()
        
        # Initialize metrics dictionary
        self.metrics = self._init_metrics()
        
        # Track experiment timing
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
    def _init_metrics(self) -> Dict[str, List[float]]:
        """Initialize metrics dictionary based on task type."""
        # Common metrics for all tasks
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'time_elapsed': [],
        }
        
        # Task-specific metrics
        if self.task_type == "classification":
            metrics.update({
                'train_acc': [],
                'val_acc': [],
                'train_acc_top5': [],
                'val_acc_top5': [],
            })
        elif self.task_type == "detection":
            metrics.update({
                'train_box_loss': [],
                'train_obj_loss': [],
                'train_cls_loss': [],
                'val_precision': [],
                'val_recall': [],
                'val_mAP50': [],
                'val_mAP': [],
            })
        
        return metrics
    
    def update(self, metrics_dict: Dict[str, float], epoch: int) -> None:
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric values to update.
            epoch: Current epoch number.
        """
        # Add epoch if not already present
        if epoch not in self.metrics['epochs']:
            self.metrics['epochs'].append(epoch)
        
        # Add elapsed time
        current_time = time.time()
        elapsed = current_time - self.start_time
        elapsed_since_last = current_time - self.last_update_time
        self.metrics['time_elapsed'].append(elapsed)
        self.last_update_time = current_time
        
        # Log time per epoch
        logger.info(f"Epoch {epoch} completed in {elapsed_since_last:.2f}s. "
                   f"Total time: {elapsed/60:.2f} min")
        
        # Update metrics
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
            else:
                # Allow adding new metrics dynamically
                self.metrics[key] = [value]
                
        # Save after each update
        self.save()
    
    def save(self, filename: Optional[str] = None) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            filename: Filename to save metrics to. If None, uses default name.
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        metrics_path = self.save_dir / filename
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        logger.debug(f"Metrics saved to {metrics_path}")
    
    def load(self, filename: Optional[str] = None) -> None:
        """
        Load metrics from JSON file.
        
        Args:
            filename: Filename to load metrics from. If None, uses default name.
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.json"
        
        metrics_path = self.save_dir / filename
        try:
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Loaded metrics from {metrics_path}")
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {metrics_path}")
    
    def plot(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Create and save visualization plots for metrics.
        
        Args:
            save_path: Path to save the plots. If None, uses default name.
        """
        if save_path is None:
            save_path = self.save_dir / f"{self.experiment_name}_plots.png"
        else:
            save_path = Path(save_path)
        
        # Determine plot layout based on task type
        if self.task_type == "classification":
            self._plot_classification_metrics(save_path)
        elif self.task_type == "detection":
            self._plot_detection_metrics(save_path)
        else:
            # Generic plotting for custom tasks
            self._plot_generic_metrics(save_path)
    
    def _plot_classification_metrics(self, save_path: Path) -> None:
        """
        Plot classification-specific metrics.
        
        Args:
            save_path: Path to save the plot.
        """
        epochs = self.metrics['epochs']
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Accuracy plot
        if 'train_acc' in self.metrics and 'val_acc' in self.metrics:
            ax1.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy')
            ax1.plot(epochs, self.metrics['val_acc'], 'r-', label='Validation Accuracy')
            ax1.set_title('Training and Validation Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy (%)')
            ax1.legend()
            ax1.grid(True)
            
            # Add top-5 accuracy if available
            if 'train_acc_top5' in self.metrics and 'val_acc_top5' in self.metrics:
                ax1.plot(epochs, self.metrics['train_acc_top5'], 'b--', 
                         label='Training Top-5 Accuracy')
                ax1.plot(epochs, self.metrics['val_acc_top5'], 'r--', 
                         label='Validation Top-5 Accuracy')
                ax1.legend()
        
        # Loss plot
        if 'train_loss' in self.metrics and 'val_loss' in self.metrics:
            ax2.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss')
            ax2.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss')
            ax2.set_title('Training and Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
        
        # Learning rate plot
        if 'learning_rate' in self.metrics:
            ax3.plot(epochs, self.metrics['learning_rate'], 'g-')
            ax3.set_title('Learning Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
            # Use log scale for learning rate
            ax3.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Plots saved to {save_path}")
    
    def _plot_detection_metrics(self, save_path: Path) -> None:
        """
        Plot detection-specific metrics.
        
        Args:
            save_path: Path to save the plot.
        """
        epochs = self.metrics['epochs']
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss components plot
        losses = ['train_loss', 'train_box_loss', 'train_obj_loss', 'train_cls_loss']
        loss_labels = ['Total Loss', 'Box Loss', 'Objectness Loss', 'Class Loss']
        loss_colors = ['b', 'g', 'r', 'c']
        
        for loss, label, color in zip(losses, loss_labels, loss_colors):
            if loss in self.metrics:
                ax1.plot(epochs, self.metrics[loss], f'{color}-', label=label)
        
        ax1.set_title('Training Loss Components')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation loss
        if 'val_loss' in self.metrics:
            ax2.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss')
            ax2.set_title('Validation Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # Precision/Recall plot
        if 'val_precision' in self.metrics and 'val_recall' in self.metrics:
            ax3.plot(epochs, self.metrics['val_precision'], 'b-', label='Precision')
            ax3.plot(epochs, self.metrics['val_recall'], 'r-', label='Recall')
            ax3.set_title('Precision and Recall')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True)
        
        # mAP plot
        if 'val_mAP50' in self.metrics and 'val_mAP' in self.metrics:
            ax4.plot(epochs, self.metrics['val_mAP50'], 'g-', label='mAP@0.5')
            ax4.plot(epochs, self.metrics['val_mAP'], 'b-', label='mAP@0.5:0.95')
            ax4.set_title('Mean Average Precision (mAP)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('mAP')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Plots saved to {save_path}")
    
    def _plot_generic_metrics(self, save_path: Path) -> None:
        """
        Plot generic metrics for custom tasks.
        
        Args:
            save_path: Path to save the plot.
        """
        epochs = self.metrics['epochs']
        
        # Create a figure with flexible number of subplots
        metric_groups = [
            # Group 1: Loss metrics
            {
                'title': 'Loss Metrics',
                'keys': [key for key in self.metrics.keys() if 'loss' in key.lower()],
                'ylabel': 'Loss'
            },
            # Group 2: Accuracy/performance metrics
            {
                'title': 'Performance Metrics',
                'keys': [key for key in self.metrics.keys() 
                         if any(term in key.lower() for term in ['acc', 'map', 'precision', 'recall', 'f1'])],
                'ylabel': 'Value'
            },
            # Group 3: Learning rate
            {
                'title': 'Learning Rate',
                'keys': ['learning_rate'],
                'ylabel': 'Learning Rate',
                'yscale': 'log'
            }
        ]
        
        # Filter out empty groups
        metric_groups = [g for g in metric_groups if any(k in self.metrics for k in g['keys'])]
        
        # Create figure
        fig, axes = plt.subplots(len(metric_groups), 1, figsize=(12, 5 * len(metric_groups)))
        if len(metric_groups) == 1:
            axes = [axes]
        
        # Plot each group
        for ax, group in zip(axes, metric_groups):
            for key in group['keys']:
                if key in self.metrics and key != 'epochs':
                    ax.plot(epochs, self.metrics[key], label=key)
            
            ax.set_title(group['title'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(group['ylabel'])
            if 'yscale' in group:
                ax.set_yscale(group['yscale'])
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Plots saved to {save_path}")
    
    def get_best_metric(self, metric_name: str, mode: str = 'max') -> Tuple[float, int]:
        """
        Get the best value and epoch for a specific metric.
        
        Args:
            metric_name: Name of the metric to find best value for.
            mode: 'max' for metrics where higher is better (e.g., accuracy),
                  'min' for metrics where lower is better (e.g., loss).
                  
        Returns:
            Tuple of (best_value, best_epoch)
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0, 0
        
        if mode == 'max':
            best_idx = np.argmax(self.metrics[metric_name])
        else:
            best_idx = np.argmin(self.metrics[metric_name])
        
        best_value = self.metrics[metric_name][best_idx]
        best_epoch = self.metrics['epochs'][best_idx]
        
        return best_value, best_epoch
    
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the latest metrics.
        
        Returns:
            Dictionary with the latest values of all metrics.
        """
        summary = {}
        
        for key, values in self.metrics.items():
            if values:  # If the list isn't empty
                summary[key] = values[-1]
        
        # Add best values for common metrics
        if 'val_acc' in self.metrics:
            best_acc, best_acc_epoch = self.get_best_metric('val_acc', 'max')
            summary['best_val_acc'] = best_acc
            summary['best_val_acc_epoch'] = best_acc_epoch
            
        if 'val_loss' in self.metrics:
            best_loss, best_loss_epoch = self.get_best_metric('val_loss', 'min')
            summary['best_val_loss'] = best_loss
            summary['best_val_loss_epoch'] = best_loss_epoch
            
        if 'val_mAP' in self.metrics:
            best_map, best_map_epoch = self.get_best_metric('val_mAP', 'max')
            summary['best_val_mAP'] = best_map
            summary['best_val_mAP_epoch'] = best_map_epoch
        
        return summary
    
    def print_summary(self) -> None:
        """Print a summary of the current metrics."""
        summary = self.get_current_metrics_summary()
        
        logger.info("=" * 50)
        logger.info(f"Metrics summary for experiment: {self.experiment_name}")
        logger.info("=" * 50)
        
        # Print current epoch
        if 'epochs' in summary:
            logger.info(f"Current epoch: {summary['epochs']}")
        
        # Print task-specific metrics
        if self.task_type == "classification":
            if 'val_acc' in summary:
                logger.info(f"Validation accuracy: {summary['val_acc']:.2f}%")
            if 'best_val_acc' in summary:
                logger.info(f"Best validation accuracy: {summary['best_val_acc']:.2f}% "
                           f"(epoch {summary['best_val_acc_epoch']})")
        elif self.task_type == "detection":
            if 'val_mAP' in summary:
                logger.info(f"Validation mAP: {summary['val_mAP']:.4f}")
            if 'val_mAP50' in summary:
                logger.info(f"Validation mAP@0.5: {summary['val_mAP50']:.4f}")
            if 'best_val_mAP' in summary:
                logger.info(f"Best validation mAP: {summary['best_val_mAP']:.4f} "
                           f"(epoch {summary['best_val_mAP_epoch']})")
        
        # Print common metrics
        if 'val_loss' in summary:
            logger.info(f"Validation loss: {summary['val_loss']:.4f}")
        if 'best_val_loss' in summary:
            logger.info(f"Best validation loss: {summary['best_val_loss']:.4f} "
                       f"(epoch {summary['best_val_loss_epoch']})")
            
        # Print timing information
        if 'time_elapsed' in summary:
            total_time = summary['time_elapsed'] / 60  # Convert to minutes
            logger.info(f"Total training time: {total_time:.2f} minutes")
            
        logger.info("=" * 50)


def compute_accuracy(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    topk: Tuple[int] = (1,)
) -> List[float]:
    """
    Compute accuracy at specified top-k levels.
    
    Args:
        outputs: Model output logits (B, C).
        targets: Target class indices (B,).
        topk: Tuple of k values for top-k accuracy.
        
    Returns:
        List of top-k accuracies (in %).
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()  # Transpose to (k, B)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Adapted from PyTorch ImageNet example.
    """
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)