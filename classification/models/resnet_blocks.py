import torch
import torch.nn as nn

class RBasicBlock(nn.Module):
    """Wide ResNet basic block with pre-activation"""
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(RBasicBlock, self).__init__()
        
        # First convolution block with pre-activation (BN->ReLU->Conv)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        
        # Second convolution block
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        
        # Dropout if needed
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                     stride=stride, bias=False)
    
    def forward(self, x):
        # First block
        out = self.bn1(x)
        out = self.relu1(out)
        
        # Shortcut from pre-activated input
        residual = self.shortcut(out)
        
        # Continue with convolutions
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Add residual connection
        out = out + residual
        
        return out


class WideResNetBlock(nn.Module):
    """Block containing multiple wide basic blocks"""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(WideResNetBlock, self).__init__()
        
        layers = []
        # Only the first block might have a stride != 1
        layers.append(block(in_planes, out_planes, stride, drop_rate))
        
        # Rest of the blocks maintain dimensions
        for i in range(1, nb_layers):
            layers.append(block(out_planes, out_planes, 1, drop_rate))
        
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)