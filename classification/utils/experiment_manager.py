import json
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class ExperimentManager:
    """Manages experiment directories, saving, and logging"""
    
    def __init__(self, base_dir='experiments', exp_name=None, model_type=None, dataset=None, mapping=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Generate experiment name if not provided
        if exp_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{model_type}_{dataset}_{mapping}_{timestamp}"
        
        # Find available experiment directory
        self.exp_dir = self._get_next_exp_dir(exp_name)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_dir = self.exp_dir / 'checkpoints'
        self.plots_dir = self.exp_dir / 'plots'
        self.logs_dir = self.exp_dir / 'logs'
        
        for dir_path in [self.checkpoints_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            'train_acc': [],
            'test_acc_top1': [],  # Rename for clarity
            'test_acc_top5': [],  # Add Top-5
            'train_loss': [],
            'test_loss': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Setup TensorBoard
        self.writer = SummaryWriter(str(self.logs_dir / 'tensorboard'))
        
        print(f"📁 Experiment directory: {self.exp_dir}")
        
    def _get_next_exp_dir(self, exp_name):
        """Find next available experiment directory"""
        base_path = self.base_dir / exp_name
        if not base_path.exists():
            return base_path
        
        # Find next available number
        counter = 1
        while True:
            new_path = self.base_dir / f"{exp_name}_{counter:03d}"
            if not new_path.exists():
                return new_path
            counter += 1
    
    def save_config(self, config):
        """Save experiment configuration"""
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config saved to {config_path}")
    
    def update_metrics(self, epoch, train_acc, test_acc_top1, test_acc_top5, train_loss, test_loss, lr):
        """Update metrics for current epoch"""
        self.metrics['epochs'].append(epoch)
        self.metrics['train_acc'].append(train_acc)
        # self.metrics['test_acc'].append(test_acc)
        self.metrics['test_acc_top1'].append(test_acc_top1)
        self.metrics['test_acc_top5'].append(test_acc_top5)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['learning_rates'].append(lr)
        
        # Log to TensorBoard
        self.writer.add_scalar('accuracy/train', train_acc, epoch)
        # self.writer.add_scalar('accuracy/test', test_acc, epoch)
        self.writer.add_scalar('accuracy/test_top1', test_acc_top1, epoch)
        self.writer.add_scalar('accuracy/test_top5', test_acc_top5, epoch)
        self.writer.add_scalar('loss/train', train_loss, epoch)
        self.writer.add_scalar('loss/test', test_loss, epoch)
        self.writer.add_scalar('learning_rate', lr, epoch)
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_path = self.exp_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_plots(self, filename='training_curves.png'):
        """Create and save training plots"""
        if not self.metrics['epochs']:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs = self.metrics['epochs']
        
        # Accuracy plot
        ax1.plot(epochs, self.metrics['train_acc'], 'b-', label='Train Accuracy', alpha=0.8)
        ax1.plot(epochs, self.metrics['test_acc_top1'], 'r-', label='Test Accuracy', alpha=0.8)
        ax1.set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Loss plot
        ax2.plot(epochs, self.metrics['train_loss'], 'b-', label='Train Loss', alpha=0.8)
        ax2.plot(epochs, self.metrics['test_loss'], 'r-', label='Test Loss', alpha=0.8)
        ax2.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Learning rate plot
        ax3.plot(epochs, self.metrics['learning_rates'], 'g-', alpha=0.8)
        ax3.set_title('Learning Rate vs Epoch', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Accuracy difference plot
        if len(epochs) > 1:
            acc_diff = [t - tr for t, tr in zip(self.metrics['test_acc_top1'], self.metrics['train_acc'])]
            ax4.plot(epochs, acc_diff, 'm-', label='Test - Train', alpha=0.8)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax4.set_title('Generalization Gap', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Difference (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save a quick summary plot with just accuracy
        self._save_summary_plot()
    
    def _save_summary_plot(self):
        """Save a simple accuracy summary plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = self.metrics['epochs']
        
        ax.plot(epochs, self.metrics['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, self.metrics['test_acc_top1'], 'r-', label='Test Accuracy', linewidth=2)
        
        # Add best accuracy annotation
        if self.metrics['test_acc_top1']:
            best_test_acc = max(self.metrics['test_acc_top1'])
            best_epoch = epochs[self.metrics['test_acc_top1'].index(best_test_acc)]
            ax.annotate(f'Best: {best_test_acc:.2f}% (Epoch {best_epoch})', 
                       xy=(best_epoch, best_test_acc), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_title('Training Progress', fontsize=16, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'accuracy_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, test_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'test_accuracy': test_acc,
            'train_accuracy': self.metrics['train_acc'][-1] if self.metrics['train_acc'] else 0,
            'metrics': self.metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoints_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best model saved: {test_acc:.2f}% accuracy")
        
        # Keep only last 5 regular checkpoints to save space
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self, keep_last=5):
        """Remove old checkpoint files, keeping only the most recent ones"""
        checkpoint_files = list(self.checkpoints_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoint_files) > keep_last:
            # Sort by epoch number and remove oldest
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for old_checkpoint in checkpoint_files[:-keep_last]:
                old_checkpoint.unlink()
    
    def save_interrupt_checkpoint(self, model, optimizer, scheduler, epoch):
        """Save checkpoint when training is interrupted"""
        checkpoint_path = self.checkpoints_dir / 'interrupt_checkpoint.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': self.metrics,
            'interrupted': True
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Interrupt checkpoint saved to {checkpoint_path}")
    
    def finalize(self):
        """Clean up and finalize experiment"""
        self.save_metrics()
        self.save_plots()
        self.writer.close()
        
        # Save final summary
        summary = {
            'experiment_name': self.exp_dir.name,
            'total_epochs': len(self.metrics['epochs']),
            'best_test_accuracy': max(self.metrics['test_acc_top1']) if self.metrics['test_acc_top1'] else 0,
            'final_test_accuracy': self.metrics['test_acc_top1'][-1] if self.metrics['test_acc_top1'] else 0,
            'total_parameters': getattr(self, 'total_parameters', 'unknown')
        }
        
        with open(self.exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Experiment completed. Results saved in: {self.exp_dir}")
