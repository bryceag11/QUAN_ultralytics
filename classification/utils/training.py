import torch
import torch.nn as nn
import tqdm
import numpy as np

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, test_loader, criterion, device):
    """Evaluate with optimized GPU handling"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            del outputs, loss
    
    return test_loss / len(test_loader), 100. * correct / total


def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device):
    """
    Train for one epoch with optimized GPU handling and MixUp
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm.tqdm(train_loader, desc='Training', position=1, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(train_pbar):
        # Move data to GPU efficiently
        inputs = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
        
        # Apply MixUp augmentation
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.05)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)  # More efficient than standard zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # # MixUp loss calculation
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # loss = criterion(outputs, targets)
        # with torch.amp.autocast('cuda'):
        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        
        # AMP backward pass
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.step(optimizer)
        # scaler.update()
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics - use weighted accuracy for MixUp
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # Calculate accuracy for both target sets and weight by lambda
        correct_a = predicted.eq(targets_a).sum().item()
        correct_b = predicted.eq(targets_b).sum().item()
        correct += lam * correct_a + (1 - lam) * correct_b
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # Clean up GPU memory
        del outputs, loss
        
    train_pbar.close()
    return running_loss / len(train_loader), 100. * correct / total

def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate_with_top5(model, test_loader, criterion, device):
    """Evaluate with Top-1 and Top-5 accuracy for ImageNet"""
    model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(device, non_blocking=True)
            targets = targets.cuda(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            
            # Top-1 accuracy
            _, pred_top1 = outputs.topk(1, 1, True, True)
            pred_top1 = pred_top1.t()
            correct_top1 += pred_top1.eq(targets.view(1, -1).expand_as(pred_top1)).sum().item()
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).sum().item()
            
            total += targets.size(0)
            del outputs, loss
    
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, top1_acc, top5_acc