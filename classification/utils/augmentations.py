# utils/augmentations.py
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

class MixUp:
    """MixUp augmentation
    Reference: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, inputs, targets):
        if np.random.random() > self.prob:
            return inputs, targets, targets, 1.0
        
        batch_size = inputs.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size).to(inputs.device)
        
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam


class CutMix:
    """CutMix augmentation
    Reference: https://arxiv.org/abs/1905.04899
    """
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, inputs, targets):
        if np.random.random() > self.prob:
            return inputs, targets, targets, 1.0
        
        batch_size = inputs.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size).to(inputs.device)
        
        # Generate random box
        H, W = inputs.size(2), inputs.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        targets_a, targets_b = targets, targets[index]
        
        return mixed_inputs, targets_a, targets_b, lam


class MixUpCutMix:
    """Randomly applies either MixUp or CutMix"""
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup = MixUp(alpha=mixup_alpha, prob=1.0)
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=1.0)
        self.prob = prob
        self.switch_prob = switch_prob
    
    def __call__(self, inputs, targets):
        if np.random.random() > self.prob:
            return inputs, targets, targets, 1.0
        
        if np.random.random() < self.switch_prob:
            return self.mixup(inputs, targets)
        else:
            return self.cutmix(inputs, targets)


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for MixUp/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Updated training function with modern augmentations
def train_epoch_modern(model, train_loader, criterion, optimizer, scaler, epoch, device, 
                       gradient_accumulation_steps=1, use_mixup_cutmix=True):
    """
    Train for one epoch with modern augmentations and optimizations
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize MixUp/CutMix augmentation
    if use_mixup_cutmix:
        mixup_cutmix = MixUpCutMix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,  # Always apply one of them
            switch_prob=0.5  # 50/50 chance of MixUp vs CutMix
        )
    
    train_pbar = torch.cuda.nvtx.range("train_epoch") if torch.cuda.is_available() else None
    train_pbar = tqdm.tqdm(train_loader, desc='Training', position=1, leave=False)
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(train_pbar):
        # Move data to GPU efficiently
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        
        # Apply MixUp/CutMix augmentation
        if use_mixup_cutmix and model.training:
            inputs, targets_a, targets_b, lam = mixup_cutmix(inputs, targets)
        else:
            targets_a = targets_b = targets
            lam = 1.0
        
        # Forward pass with automatic mixed precision
        with autocast():
            outputs = model(inputs)
            if lam == 1.0:
                loss = criterion(outputs, targets)
            else:
                loss = mixup_cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        running_loss += loss.item() * gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        # For MixUp/CutMix, calculate accuracy differently
        if lam == 1.0:
            correct += predicted.eq(targets).sum().item()
        else:
            correct_a = predicted.eq(targets_a).sum().item()
            correct_b = predicted.eq(targets_b).sum().item()
            correct += lam * correct_a + (1 - lam) * correct_b
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Periodic memory cleanup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    train_pbar.close()
    return running_loss / len(train_loader), 100. * correct / total