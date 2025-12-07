# models/standard_models.py
import torch
import torch.nn as nn
from .resnet_blocks import RBasicBlock, WideResNetBlock

class WideResNet(nn.Module):
    """
    Standard Wide ResNet
    Depth should be 6n+4 for WRNs. Width factor controls width.
    """
    def __init__(self, depth=16, width_factor=4, drop_rate=0.0, num_classes=10):
        super(WideResNet, self).__init__()
        
        n = (depth - 4) // 6  # For WRN-16-4, n=2
        k = width_factor       # Width factor (4 for WRN-16-4)
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, 
                              stride=1, padding=1, bias=False)
        
        # Three stages of wide blocks
        self.stage1 = WideResNetBlock(n, nStages[0], nStages[1], RBasicBlock, 1, drop_rate)
        self.stage2 = WideResNetBlock(n, nStages[1], nStages[2], RBasicBlock, 2, drop_rate)
        self.stage3 = WideResNetBlock(n, nStages[2], nStages[3], RBasicBlock, 2, drop_rate)
        
        # Final classification
        self.bn = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        out = self.conv1(x)
        
        # Three stages
        out = self.stage1(out)  # 32x32
        out = self.stage2(out)  # 16x16
        out = self.stage3(out)  # 8x8
        
        # Final activation and pooling
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)  # 1x1
        out = out.view(out.size(0), -1)
        
        # Classification
        out = self.fc(out)
        
        return out


def create_wrn_16_4(num_classes=10, drop_rate=0.3):
    """Creates a WRN-16-4 model (16 layers, width factor 4)"""
    return WideResNet(
        depth=16,
        width_factor=4,
        drop_rate=drop_rate,
        num_classes=num_classes
    )


def create_wrn_16_2(num_classes=100, drop_rate=0.3):
    """Creates a lighter WRN-16-2 model (16 layers, width factor 2)"""
    return WideResNet(
        depth=16,
        width_factor=2,
        drop_rate=drop_rate,
        num_classes=num_classes
    )