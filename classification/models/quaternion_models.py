# models/quaternion_models.py
import torch
import torch.nn as nn
from quaternion.qconv import QConv2D, IQBN, QDense
from quaternion.qactivation import QSiLU
from models.blocks.quaternion_blocks import QuaternionAvgPool, QuaternionMaxPool
from .quaternion_blocks import QWideBasicBlock, QWideResNetBlock, QuaternionBasicBlock
from .qextract import QNormExtract, QExtract



class QWideResNet(nn.Module):
    """
    Quaternion-based Wide ResNet
    Depth should be 6n+4 for WRNs. Width factor controls width.
    """
    def __init__(self, depth=16, width_factor=4, drop_rate=0.0, num_classes=10, mapping_type='poincare'):
        super(QWideResNet, self).__init__()
        
        n = (depth - 4) // 6  # For WRN-16-4, n=2
        k = width_factor       # Width factor (4 for WRN-16-4)
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        # Initial convolution
        self.conv1 = QConv2D(3, nStages[0], kernel_size=3, stride=1, padding=1, mapping_type=mapping_type)
        
        # Three stages of wide blocks
        self.stage1 = QWideResNetBlock(n, nStages[0], nStages[1], QWideBasicBlock, 1, drop_rate, mapping_type)
        self.stage2 = QWideResNetBlock(n, nStages[1], nStages[2], QWideBasicBlock, 2, drop_rate, mapping_type)
        self.stage3 = QWideResNetBlock(n, nStages[2], nStages[3], QWideBasicBlock, 2, drop_rate, mapping_type)
        
        # Final classification
        self.bn = IQBN(nStages[3])
        self.silu = nn.SiLU()
        self.avgpool = QuaternionAvgPool()
        
        # Classifier with adequate capacity for CIFAR-100
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(nStages[3], num_classes * 4, mapping_type=mapping_type)
        )
        
    def forward(self, x):
        # Initial convolution
        out = self.conv1(x)
        
        # Three stages
        out = self.stage1(out)  # 32x32
        out = self.stage2(out)  # 16x16
        out = self.stage3(out)  # 8x8
        
        # Final activation and pooling
        out = self.bn(out)
        out = self.silu(out)
        out = self.avgpool(out)  # 1x1
        
        # Classification
        out = self.classifier(out)
        
        # Extract quaternion norm for final classification
        batch_size = out.size(0)
        out = out.view(batch_size, -1, 4)
        quaternion_norm = torch.norm(out, dim=2)
        
        return quaternion_norm

def create_qwrn_16_4(num_classes=10, drop_rate=0.3, mapping_type='poincare'):
    """
    Creates a Quaternion WRN-16-4 model (16 layers, width factor 4)
    """
    return QWideResNet(
        depth=16,
        width_factor=4,
        drop_rate=drop_rate,
        num_classes=num_classes,
        mapping_type=mapping_type
    )

def create_qwrn_16_2(num_classes=10, drop_rate=0.0, mapping_type='poincare'):
    """
    Creates a super-lightweight Quaternion WRN-16-2 model (16 layers, width factor 2)
    """
    return QWideResNet(
        depth=16,
        width_factor=2,
        drop_rate=drop_rate,
        num_classes=num_classes,
        mapping_type=mapping_type
    )

class QResNet34(nn.Module):
    """QResNet34 with mixed block types and increased width"""
    def __init__(self, num_classes=10, drop_rate = 0.0, mapping_type='poincare'):
        super().__init__()
        
        self.drop_rate = drop_rate
        # Start with narrow base width (16 channels)
        base_width = 16
        
        # Initial 3×3 convolution (no 7×7 and no maxpool for CIFAR)
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(base_width),
            nn.SiLU()
        )
        
        # 3 stages with standard ResNet34 block counts [3, 4, 6]
        self.stage1 = self._make_layer(base_width, base_width, blocks=3, stride=1)
        self.stage2 = self._make_layer(base_width, base_width*2, blocks=4, stride=2)
        self.stage3 = self._make_layer(base_width*2, base_width*4, blocks=6, stride=2)
        
        # Global Average Pooling and Classification
        self.avg_pool = QuaternionAvgPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(base_width*4, 256, mapping_type=mapping_type),
            nn.SiLU(),
            QDense(256, num_classes * 4, mapping_type=mapping_type)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer of residual blocks"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_rate=self.drop_rate
        ))
        
        # Remaining blocks with stride=1
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                drop_rate=self.drop_rate
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global average pooling and classification
        x = self.avg_pool(x)
        x = self.classifier(x)

        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm

def create_qrn_34(num_classes=10, drop_rate=0.1, mapping_type='poincare'):
    """Creates a Quaternion ResNet 34 layer model"""
    return QResNet34(
        num_classes=num_classes,
        drop_rate=drop_rate,
        mapping_type=mapping_type
    )


class QResNet34_ImageNet(nn.Module):
    """
    QResNet34 optimized for ImageNet (224x224 input)
    """
    def __init__(self, num_classes=1000, drop_rate=0.1, mapping_type='poincare'):
        super().__init__()
        
        self.drop_rate = drop_rate
        
        # Larger base width for ImageNet complexity
        base_width = 64  # Standard ImageNet width
        
        # ImageNet-style initial convolution with 7x7 kernel and stride 2
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=7, stride=2, padding=3, mapping_type=mapping_type),
            IQBN(base_width),
            nn.SiLU()
        )
        
        # Max pooling to reduce spatial dimensions early
        self.maxpool = QuaternionMaxPool(kernel_size=3, stride=2, padding=1)
        
        # 4 stages with proper ResNet34 block counts [3, 4, 6, 3]
        self.stage1 = self._make_layer(base_width, base_width, blocks=3, stride=1)      # 56x56
        self.stage2 = self._make_layer(base_width, base_width*2, blocks=4, stride=2)    # 28x28
        self.stage3 = self._make_layer(base_width*2, base_width*4, blocks=6, stride=2)  # 14x14
        self.stage4 = self._make_layer(base_width*4, base_width*8, blocks=3, stride=2)  # 7x7
        
        # Global Average Pooling and Classification
        self.avg_pool = QuaternionAvgPool()
        
        # ImageNet classifier with proper capacity
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            QDense(base_width*8, num_classes * 4, mapping_type=mapping_type)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_rate=self.drop_rate
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                drop_rate=self.drop_rate
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv and pooling
        x = self.conv1(x)      # 224x224 -> 112x112
        x = self.maxpool(x)    # 112x112 -> 56x56
        
        # 4 residual stages
        x = self.stage1(x)     # 56x56
        x = self.stage2(x)     # 28x28
        x = self.stage3(x)     # 14x14
        x = self.stage4(x)     # 7x7
        
        # Global average pooling and classification
        x = self.avg_pool(x)   # 7x7 -> 1x1
        x = self.classifier(x)

        # Extract quaternion norm
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm


class QWideResNet_ImageNet(nn.Module):
    """
    QWideResNet optimized for ImageNet
    WRN-50-2 equivalent (50 layers, width factor 2)
    """
    def __init__(self, depth=50, width_factor=2, num_classes=1000, drop_rate=0.2, mapping_type='poincare'):
        super().__init__()
        
        # Calculate blocks per stage for WRN-50
        # (50 - 4) / 6 = 7.67, so we use [3, 4, 6, 3] like ResNet-50
        blocks = [3, 4, 6, 3]
        
        base_width = 64
        widths = [base_width * width_factor * (2**i) for i in range(4)]
        
        # ImageNet-style initial convolution
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=7, stride=2, padding=3, mapping_type=mapping_type),
            IQBN(base_width),
            nn.SiLU()
        )
        
        self.maxpool = QuaternionMaxPool(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        self.stage1 = QWideResNetBlock(blocks[0], base_width, widths[0], QWideBasicBlock, 1, drop_rate, mapping_type)
        self.stage2 = QWideResNetBlock(blocks[1], widths[0], widths[1], QWideBasicBlock, 2, drop_rate, mapping_type)
        self.stage3 = QWideResNetBlock(blocks[2], widths[1], widths[2], QWideBasicBlock, 2, drop_rate, mapping_type)
        self.stage4 = QWideResNetBlock(blocks[3], widths[2], widths[3], QWideBasicBlock, 2, drop_rate, mapping_type)
        
        # Final layers
        self.avg_pool = QuaternionAvgPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            # nn.Linear(widths[3], num_classes)
            QDense(widths[3], num_classes * 4, mapping_type=mapping_type)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avg_pool(x)
        x = self.classifier(x)
        
        # Extract quaternion norm
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm
    
def create_qrn34_imagenet(num_classes=1000, mapping_type='poincare'):
    """Create QResNet34 for ImageNet"""
    return QResNet34_ImageNet(
        num_classes=num_classes,
        drop_rate=0.1,
        mapping_type=mapping_type
    )


def create_qwrn_50_2_imagenet(num_classes=1000, mapping_type='poincare'):
    """Create QWideResNet-50-2 for ImageNet"""
    return QWideResNet_ImageNet(
        depth=50,
        width_factor=2,
        num_classes=num_classes,
        drop_rate=0.2,
        mapping_type=mapping_type
    )


# Add this to models/quaternion_models.py

class QResNet18(nn.Module):
    """
    Fixed QResNet18 - Properly scaled for CIFAR
    """
    def __init__(self, num_classes=10, drop_rate=0.0, mapping_type='poincare'):
        super().__init__()
        
        self.drop_rate = drop_rate
        base_width = 16  # Same as QResNet34 for consistency
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(base_width),
            QSiLU()
        )
        
        # 3 stages only (like your QResNet34) - NO stage4!
        # ResNet18 uses [2, 2, 2, 2] but we adapt for CIFAR
        self.stage1 = self._make_layer(base_width, base_width, blocks=2, stride=1)      # 32x32
        self.stage2 = self._make_layer(base_width, base_width*2, blocks=2, stride=2)    # 16x16
        self.stage3 = self._make_layer(base_width*2, base_width*4, blocks=2, stride=2)  # 8x8
        # NO stage4 - that was the problem!
        
        # Global Average Pooling and Classification
        self.avg_pool = QuaternionAvgPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(base_width*4, 256, mapping_type=mapping_type),  # Match QResNet34
            nn.SiLU(),
            QDense(256, num_classes * 4, mapping_type=mapping_type)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a layer of residual blocks"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_rate=self.drop_rate
        ))
        
        # Remaining blocks with stride=1
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                drop_rate=self.drop_rate
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual stages
        x = self.stage1(x)  # 32x32
        x = self.stage2(x)  # 16x16
        x = self.stage3(x)  # 8x8
        
        # Global average pooling and classification
        x = self.avg_pool(x)
        x = self.classifier(x)

        # Extract quaternion norm
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm


class QResNet18_ImageNet(nn.Module):
    """
    Fixed QResNet18 for ImageNet - matching the efficiency of QResNet34
    """
    def __init__(self, num_classes=1000, drop_rate=0.1, mapping_type='poincare'):
        super().__init__()
        
        self.drop_rate = drop_rate
        
        # SAME base width as QResNet34_ImageNet - this was the issue!
        base_width = 64
        
        # ImageNet-style initial convolution
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=7, stride=2, padding=3, mapping_type=mapping_type),
            IQBN(base_width),
            QSiLU()
        )
        
        self.maxpool = QuaternionMaxPool(kernel_size=3, stride=2, padding=1)
        
        # 4 stages with ResNet18 block counts [2, 2, 2, 2]
        # But same channel progression as QResNet34
        self.stage1 = self._make_layer(base_width, base_width, blocks=2, stride=1)      # 56x56
        self.stage2 = self._make_layer(base_width, base_width*2, blocks=2, stride=2)    # 28x28
        self.stage3 = self._make_layer(base_width*2, base_width*4, blocks=2, stride=2)  # 14x14
        self.stage4 = self._make_layer(base_width*4, base_width*8, blocks=2, stride=2)  # 7x7
        
        # Global Average Pooling and Classification
        self.avg_pool = QuaternionAvgPool()
        
        # SAME classifier structure as QResNet34
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            QDense(base_width*8, num_classes * 4, mapping_type=mapping_type)
            # nn.Linear(base_width*8, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_rate=self.drop_rate
        ))
        
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                drop_rate=self.drop_rate
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv and pooling
        x = self.conv1(x)      # 224x224 -> 112x112
        x = self.maxpool(x)    # 112x112 -> 56x56
        
        # 4 residual stages
        x = self.stage1(x)     # 56x56
        x = self.stage2(x)     # 28x28
        x = self.stage3(x)     # 14x14
        x = self.stage4(x)     # 7x7
        
        # Global average pooling and classification
        x = self.avg_pool(x)   # 7x7 -> 1x1
        x = self.classifier(x)

        # Extract quaternion norm
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm


def create_qrn_18(num_classes=10, drop_rate=0.1, mapping_type='poincare'):
    """Creates a Quaternion ResNet 18 layer model for CIFAR"""
    return QResNet18(
        num_classes=num_classes,
        drop_rate=drop_rate,
        mapping_type=mapping_type
    )


def create_qrn18_imagenet(num_classes=1000, mapping_type='poincare'):
    """Create QResNet18 for ImageNet"""
    return QResNet18_ImageNet(
        num_classes=num_classes,
        drop_rate=0.1,
        mapping_type=mapping_type
    )

class QWRN16_4I(nn.Module):
    """
    QWideResNet-16-4 optimized for ImageNet
    """
    def __init__(self, num_classes=1000, drop_rate=0.2, mapping_type='poincare'):
        super().__init__()
        
        depth = 16
        width_factor = 2
        n = (depth - 4) // 6 # n=2 for WRN-16-4
        k = width_factor
        
        base_width = 64 # Standard for ImageNet
        nStages = [base_width, base_width*k, base_width*2*k, base_width*4*k]

        # ImageNet-style initial convolution
        self.conv1 = nn.Sequential(
            QConv2D(3, nStages[0], kernel_size=7, stride=2, padding=3, mapping_type=mapping_type),
            IQBN(nStages[0]),
            nn.SiLU()
        )
        
        self.maxpool = QuaternionMaxPool(kernel_size=3, stride=2, padding=1)
        
        # Three stages of wide blocks for ImageNet
        self.stage1 = QWideResNetBlock(n, nStages[0], nStages[1], QWideBasicBlock, 1, drop_rate, mapping_type)
        self.stage2 = QWideResNetBlock(n, nStages[1], nStages[2], QWideBasicBlock, 2, drop_rate, mapping_type)
        self.stage3 = QWideResNetBlock(n, nStages[2], nStages[3], QWideBasicBlock, 2, drop_rate, mapping_type)

        # Final layers
        self.avg_pool = QuaternionAvgPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            QDense(nStages[3], num_classes * 4, mapping_type=mapping_type)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        x = self.avg_pool(x)
        x = self.classifier(x)
        
        # Extract quaternion norm
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm

def create_qwrn16_4_imagenet(num_classes=1000, mapping_type='poincare'):
    """Create QWideResNet-16-4 for ImageNet"""
    return QWRN16_4I(
        num_classes=num_classes,
        drop_rate=0.2,
        mapping_type=mapping_type
    )
