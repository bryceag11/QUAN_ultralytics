import torch
import torch.nn as nn
from quaternion.qconv import QConv2D, IQBN
from quaternion.qactivation import QSiLU
from models.blocks.quaternion_blocks import QuaternionDropout

class QWideBasicBlock(nn.Module):
    """Wide ResNet basic block with quaternion operations"""
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, mapping_type='poincare'):
        super(QWideBasicBlock, self).__init__()
        
        # First convolution block with pre-activation (BN->SiLU->Conv)
        self.bn1 = IQBN(in_planes)
        self.silu1 = nn.SiLU()
        self.conv1 = QConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, mapping_type=mapping_type)
        
        # Second convolution block
        self.bn2 = IQBN(out_planes)
        self.silu2 = nn.SiLU()
        self.conv2 = QConv2D(out_planes, out_planes, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type)
        
        # Dropout if needed
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = QConv2D(in_planes, out_planes, kernel_size=1, stride=stride, mapping_type=mapping_type)
    
    def forward(self, x):
        # First block
        out = self.bn1(x)
        out = self.silu1(out)
        
        # Shortcut from pre-activated input
        residual = self.shortcut(out)
        
        # Continue with convolutions
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.silu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Add residual connection
        out = out + residual
        
        return out


class QWideResNetBlock(nn.Module):
    """Block containing multiple wide basic blocks"""
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, mapping_type='poincare'):
        super(QWideResNetBlock, self).__init__()
        
        layers = []
        # Only the first block might have a stride != 1
        layers.append(block(in_planes, out_planes, stride, drop_rate, mapping_type))
        
        # Rest of the blocks maintain dimensions
        for i in range(1, nb_layers):
            layers.append(block(out_planes, out_planes, 1, drop_rate, mapping_type))
        
        self.layer = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)


class QuaternionBasicBlock(nn.Module):
    """
    Basic quaternion residual block with pre-activation
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Defaults to 1.
        drop_rate: Dropout rate. Defaults to 0.0.
        norm_class: Batch normalization class to use.
        activation_class: Activation function class to use.
        conv_class: Convolution class to use.
        mapping_type: Quaternion mapping type for convolutions.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        drop_rate: float = 0.0,
        norm_class: nn.Module = IQBN,
        activation_class: nn.Module = QSiLU,
        conv_class: nn.Module = QConv2D,
        mapping_type: str = 'poincare'
    ):
        super().__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # First convolution block
        self.bn1 = norm_class(in_channels)
        self.relu1 = activation_class()
        self.conv1 = conv_class(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            mapping_type=mapping_type
        )
        
        # Second convolution block
        self.bn2 = norm_class(out_channels)
        self.relu2 = activation_class()
        self.conv2 = conv_class(
            out_channels, 
            out_channels, 
            kernel_size=3,
            stride=1, 
            padding=1,
            mapping_type=mapping_type
        )
        
        # Dropout if specified
        self.dropout = QuaternionDropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv_class(
                in_channels, 
                out_channels, 
                kernel_size=1,
                stride=stride,
                mapping_type=mapping_type
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # First block with pre-activation
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        # Second block
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Add residual
        out = out + identity
        
        return out


