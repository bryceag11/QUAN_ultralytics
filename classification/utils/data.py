#!/usr/bin/env python
"""
Data loading and preprocessing utilities.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Tuple, Optional, List, Any


class Cutout:
    """
    Randomly mask out a square patch from an image.
    
    Args:
        n_holes: Number of square patches to cut out.
        length: Length of the square side.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of shape (C, H, W).
        
        Returns:
            Tensor: Image with n_holes of specified length cut out.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), device=img.device)

        for n in range(self.n_holes):
            y = torch.randint(0, h, (1,))
            x = torch.randint(0, w, (1,))

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img


class MultiAugmentDataset(Dataset):
    """
    Dataset wrapper that applies multiple augmentations per image.
    
    Args:
        dataset: Base dataset to augment.
        augmentations_per_image: Number of augmented versions per image.
        train: Whether this is for training or evaluation.
        dataset_type: Type of dataset (e.g., 'cifar10', 'cifar100').
        cutout: Whether to apply cutout augmentation.
        cutout_length: Length of cutout square.
    """
    def __init__(
        self, 
        dataset: Dataset, 
        augmentations_per_image: int = 1, 
        train: bool = True, 
        dataset_type: str = 'cifar10',
        cutout: bool = False,
        cutout_length: int = 16
    ):
        self.dataset = dataset
        self.augmentations_per_image = augmentations_per_image
        self.train = train
        self.cutout = cutout
        self.cutout_length = cutout_length
        
        # Set the correct normalization values based on dataset
        if dataset_type.lower() == 'cifar10':
            # CIFAR-10 normalization values
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        elif dataset_type.lower() == 'cifar100':
            # CIFAR-100 normalization values
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
        elif dataset_type.lower().startswith('imagenet'):
            # ImageNet normalization values
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Import AutoAugment for CIFAR
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy
        
        # Build transform pipeline
        primary_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.RandomRotation((-15, 15)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        
        if self.cutout:
            primary_transforms.append(Cutout(n_holes=1, length=self.cutout_length))
        
        # Primary transform with AutoAugment (for most instances)
        self.primary_transform = transforms.Compose(primary_transforms)
        
        # Secondary transform for adding diversity (slightly different)
        secondary_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        
        if self.cutout:
            secondary_transforms.append(Cutout(n_holes=1, length=self.cutout_length))
            
        self.secondary_transform = transforms.Compose(secondary_transforms)
        
        # Test transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        # Pre-compute augmentation indices for efficiency
        if self.train:
            self.indices = []
            for idx in range(len(dataset)):
                self.indices.extend([idx] * augmentations_per_image)

    def __getitem__(self, index):
        if self.train:
            real_idx = self.indices[index]
            image, label = self.dataset[real_idx]
            
            # Apply appropriate transform based on augmentation index
            # This creates diversity in augmentations
            if index % self.augmentations_per_image == 0:
                transformed = self.primary_transform(image)
            elif index % self.augmentations_per_image == 1:
                transformed = self.primary_transform(image)  # Second copy also uses AutoAugment
            else:
                transformed = self.secondary_transform(image)  # Add more diversity
                
            return transformed, label
        else:
            image, label = self.dataset[index]
            return self.test_transform(image), label

    def __len__(self):
        if self.train:
            return len(self.dataset) * self.augmentations_per_image
        return len(self.dataset)


def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup augmentation to the batch.
    
    Args:
        x: Input batch.
        y: Target batch.
        alpha: Mixup interpolation coefficient.
        
    Returns:
        Tuple of (mixed inputs, target a, target b, lambda)
    """
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
    """
    Mixup loss function.
    
    Args:
        criterion: Base loss function.
        pred: Model predictions.
        y_a: First target.
        y_b: Second target.
        lam: Mixup interpolation coefficient.
        
    Returns:
        Mixed loss value.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_dataset(
    name: str,
    root: str = './data',
    train: bool = True,
    download: bool = True,
    transform: Optional[Any] = None
) -> Dataset:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name (cifar10, cifar100, imagenet, etc.)
        root: Root directory for dataset.
        train: Whether to get training or test set.
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply.
        
    Returns:
        Dataset instance.
    """
    if name.lower() == 'cifar10':
        return torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform)
    elif name.lower() == 'cifar100':
        return torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=transform)
    elif name.lower() == 'imagenet':
        # ImageNet has a different directory structure
        split = 'train' if train else 'val'
        return torchvision.datasets.ImageNet(
            root=root, split=split, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test data loaders based on configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (train_loader, test_loader).
    """
    dataset_config = config['dataset']
    
    # Get dataset name and parameters
    dataset_name = dataset_config['name']
    batch_size = dataset_config['batch_size']
    num_workers = dataset_config['num_workers']
    augmentations_per_image = dataset_config['augmentations_per_image']
    data_dir = dataset_config['data_dir']
    cutout = dataset_config.get('cutout', False)
    cutout_length = dataset_config.get('cutout_length', 16)
    
    # Get raw datasets (without transforms)
    train_dataset = get_dataset(
        name=dataset_name,
        root=data_dir,
        train=True,
        download=True,
        transform=None
    )
    
    test_dataset = get_dataset(
        name=dataset_name,
        root=data_dir,
        train=False,
        download=True,
        transform=None
    )
    
    # Wrap datasets with multi-augmentation
    train_dataset = MultiAugmentDataset(
        dataset=train_dataset,
        augmentations_per_image=augmentations_per_image,
        train=True,
        dataset_type=dataset_name,
        cutout=cutout,
        cutout_length=cutout_length
    )
    
    test_dataset = MultiAugmentDataset(
        dataset=test_dataset,
        augmentations_per_image=1,  # No augmentation for test
        train=False,
        dataset_type=dataset_name
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=3 if num_workers > 0 else None,
        drop_last=False
    )
    
    return train_loader, test_loader