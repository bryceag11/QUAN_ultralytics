import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torch.utils.data import DataLoader
from pathlib import Path

class Cutout:
    """Randomly mask out a square patch from an image."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
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


class MultiAugmentDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies AutoAugment with multiple augmentations per image"""
    def __init__(self, dataset, augmentations_per_image=3, train=True, dataset_type='cifar10'):
        self.dataset = dataset
        self.augmentations_per_image = augmentations_per_image
        self.train = train
        
        # Dataset-specific configurations
        dataset_configs = {
            'cifar10': {
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2023, 0.1994, 0.2010),
                'size': 32,
                'policy': AutoAugmentPolicy.CIFAR10
            },
            'cifar100': {
                'mean': (0.5071, 0.4867, 0.4408),
                'std': (0.2675, 0.2565, 0.2761),
                'size': 32,
                'policy': AutoAugmentPolicy.CIFAR10
            },
            'svhn': {
                'mean': (0.4377, 0.4438, 0.4728),
                'std': (0.1980, 0.2010, 0.1970),
                'size': 32,
                'policy': AutoAugmentPolicy.SVHN
            },
            'imagenet': {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225),
                'size': 224,
                'policy': AutoAugmentPolicy.IMAGENET
            }
        }
        
        if dataset_type.lower() not in dataset_configs:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(dataset_configs.keys())}")
        
        config = dataset_configs[dataset_type.lower()]
        self.mean = config['mean']
        self.std = config['std']
        self.size = config['size']
        self.policy = config['policy']
      
        # Primary transform with AutoAugment
        if train:
            if dataset_type.lower() == 'imagenet':
                self.primary_transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.size, scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    AutoAugment(self.policy),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
                self.secondary_transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.size, scale=(0.1, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            else:  # CIFAR-10, CIFAR-100, SVHN
                self.primary_transform = transforms.Compose([
                    transforms.RandomCrop(self.size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    AutoAugment(self.policy),
                    transforms.RandomRotation((-15, 15)),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
                self.secondary_transform = transforms.Compose([
                    transforms.RandomCrop(self.size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
        
        # Test transform
        if dataset_type.lower() == 'imagenet':
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:  # CIFAR-10, CIFAR-100, SVHN
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
            if index % self.augmentations_per_image < (self.augmentations_per_image // 2) + 1:
                transformed = self.primary_transform(image)
            else:
                transformed = self.secondary_transform(image)
                
            return transformed, label
        else:
            image, label = self.dataset[index]
            return self.test_transform(image), label

    def __len__(self):
        if self.train:
            return len(self.dataset) * self.augmentations_per_image
        return len(self.dataset)


def get_data_loaders(batch_size=256, augmentations_per_image=1, num_workers=4, dataset_type='cifar10', data_root='./data'):
    """Get train and test data loaders with multiple augmentations"""
    
    data_root = Path(data_root)
    data_root.mkdir(exist_ok=True)
    
    # Dataset configurations
    dataset_info = {
        'cifar10': {
            'train_dataset': lambda: torchvision.datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=None),
            'test_dataset': lambda: torchvision.datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=None),
            'num_classes': 10
        },
        'cifar100': {
            'train_dataset': lambda: torchvision.datasets.CIFAR100(
                root=data_root, train=True, download=True, transform=None),
            'test_dataset': lambda: torchvision.datasets.CIFAR100(
                root=data_root, train=False, download=True, transform=None),
            'num_classes': 100
        },
        'svhn': {
            'train_dataset': lambda: torchvision.datasets.SVHN(
                root=data_root, split='train', download=True, transform=None),
            'test_dataset': lambda: torchvision.datasets.SVHN(
                root=data_root, split='test', download=True, transform=None),
            'num_classes': 10
        },
        'imagenet': {
            'train_dataset': lambda: torchvision.datasets.ImageFolder(
                root= data_root / "ILSVRC/train", transform=None),
            'test_dataset': lambda: torchvision.datasets.ImageFolder(
                root=data_root / "ILSVRC/val", transform=None),
            'num_classes': 1000
        }
    }
    
    dataset_key = dataset_type.lower()
    if dataset_key not in dataset_info:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(dataset_info.keys())}")
    
    # Create datasets
    info = dataset_info[dataset_key]
    try:
        train_dataset = info['train_dataset']()
        test_dataset = info['test_dataset']()
    except Exception as e:
        if dataset_key == 'imagenet':
            raise RuntimeError(
                f"ImageNet dataset not found at {data_root}. "
                "Please download ImageNet manually and organize it as:\n"
                f"{data_root}/train/[class_folders]/[images]\n"
                f"{data_root}/val/[class_folders]/[images]\n"
                "Original error: " + str(e)
            )
        else:
            raise e
    
    # Wrap datasets with multi-augmentation
    train_dataset = MultiAugmentDataset(
        train_dataset, 
        augmentations_per_image=augmentations_per_image,
        train=True,
        dataset_type=dataset_type
    )
    test_dataset = MultiAugmentDataset(
        test_dataset,
        augmentations_per_image=1, 
        train=False,
        dataset_type=dataset_type
    )
    
    # Adjust batch size for ImageNet if needed
    if dataset_key == 'imagenet' and batch_size > 128:
        print(f"⚠️  Large batch size ({batch_size}) for ImageNet. Consider reducing to avoid OOM.")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  
        persistent_workers=True if num_workers > 0 else False,  
        prefetch_factor=3 if num_workers > 0 else 2,
        drop_last=True
    ) 
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  
        persistent_workers=True if num_workers > 0 else False, 
        prefetch_factor=3 if num_workers > 0 else 2,
        drop_last=True 
    )
    
    print(f"📊 Dataset: {dataset_type.upper()}")
    print(f"   - Classes: {info['num_classes']}")
    print(f"   - Train samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    print(f"   - Batch size: {batch_size}")
    
    return train_loader, test_loader, info['num_classes']


def get_dataset_info(dataset_type):
    """Get dataset information without loading the data"""
    dataset_info = {
        'cifar10': {'num_classes': 10, 'input_size': 32},
        'cifar100': {'num_classes': 100, 'input_size': 32},
        'svhn': {'num_classes': 10, 'input_size': 32},
        'imagenet': {'num_classes': 1000, 'input_size': 224}
    }
    
    dataset_key = dataset_type.lower()
    if dataset_key not in dataset_info:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(dataset_info.keys())}")
    
    return dataset_info[dataset_key]