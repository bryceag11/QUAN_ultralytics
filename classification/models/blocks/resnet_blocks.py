#!/usr/bin/env python
"""
Standard neural network building blocks (non-quaternion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Type


class BasicBlock(nn.Module):
    """
    Standard ResNet basic block with two 3x3 convolutions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Defaults to 1.
        expansion: Channel expansion factor. Defaults to 1.
        dilation: Dilation rate. Defaults to 1.
        downsample: Optional downsampling layer. Defaults to None.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU.
    """
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        expansion: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(BasicBlock, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # Store parameters
        self.stride = stride
        self.dilation = dilation
        
        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=dilation, 
            dilation=dilation, 
            bias=False
        )
        self.bn1 = norm_layer(out_channels)
        self.act1 = activation(inplace=True)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=dilation, 
            dilation=dilation, 
            bias=False
        )
        self.bn2 = norm_layer(out_channels)
        self.act2 = activation(inplace=True)
        
        # Shortcut connection
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        out = self.act2(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Standard ResNet bottleneck block with 1x1, 3x3, 1x1 convolutions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Defaults to 1.
        expansion: Channel expansion factor. Defaults to 4.
        dilation: Dilation rate. Defaults to 1.
        downsample: Optional downsampling layer. Defaults to None.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU.
    """
    expansion = 4
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        expansion: int = 4,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(Bottleneck, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # Store parameters
        self.stride = stride
        self.dilation = dilation
        self.expansion = expansion
        width = out_channels  # In standard ResNet, width = out_channels
        
        # First 1x1 convolution (reduction)
        self.conv1 = nn.Conv2d(
            in_channels, 
            width, 
            kernel_size=1, 
            bias=False
        )
        self.bn1 = norm_layer(width)
        self.act1 = activation(inplace=True)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(
            width, 
            width, 
            kernel_size=3, 
            stride=stride, 
            padding=dilation, 
            dilation=dilation, 
            bias=False
        )
        self.bn2 = norm_layer(width)
        self.act2 = activation(inplace=True)
        
        # Second 1x1 convolution (expansion)
        self.conv3 = nn.Conv2d(
            width, 
            out_channels * self.expansion, 
            kernel_size=1, 
            bias=False
        )
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.act3 = activation(inplace=True)
        
        # Shortcut connection
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First 1x1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        
        # 3x3 convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        # Second 1x1 convolution
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        out = self.act3(out)
        
        return out

class ResNet34(nn.Module):
    """
    Standard ResNet34 implementation following the original paper structure
    """
    def __init__(self, num_classes=10, small_input=True):
        super().__init__()
        
        # Initial layers - adapted for CIFAR-10 (small_input=True) or ImageNet (small_input=False)
        if small_input:  # For CIFAR-10
            # Single 3x3 conv for CIFAR sized images (32x32)
            self.initial_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        else:  # For ImageNet
            # Standard 7x7 conv followed by max pooling for ImageNet sized images
            self.initial_layer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        # ResNet blocks (layers)
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final FC layer
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create a ResNet layer composed of multiple BasicBlocks"""
        layers = []
        
        # First block with possible downsampling
        layers.append(BasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights (Kaiming initialization)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolutional layer
        x = self.initial_layer(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification layer
        x = self.fc(x)
        
        return x


class PreActBasicBlock(nn.Module):
    """
    Pre-activation ResNet basic block.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Defaults to 1.
        drop_rate: Dropout rate. Defaults to 0.0.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU.
    """
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        drop_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(PreActBasicBlock, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # First convolution block with pre-activation
        self.bn1 = norm_layer(in_channels)
        self.act1 = activation(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        # Second convolution block
        self.bn2 = norm_layer(out_channels)
        self.act2 = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        # Dropout if specified
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=1, 
                stride=stride, 
                bias=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First block with pre-activation
        preact = self.bn1(x)
        preact = self.act1(preact)
        
        # Shortcut from pre-activated input
        identity = self.shortcut(preact)
        
        # Continue with convolutions
        out = self.conv1(preact)
        
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        
        # Add residual connection
        out = out + identity
        
        return out


class PreActBottleneck(nn.Module):
    """
    Pre-activation ResNet bottleneck block with 1x1, 3x3, 1x1 convolutions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for convolution. Defaults to 1.
        expansion: Channel expansion factor. Defaults to 4.
        drop_rate: Dropout rate. Defaults to 0.0.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU.
    """
    expansion = 4
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        expansion: int = 4,
        drop_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(PreActBottleneck, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # Store parameters
        self.expansion = expansion
        width = out_channels  # In standard ResNet, width = out_channels
        
        # Pre-activation for the first block
        self.bn1 = norm_layer(in_channels)
        self.act1 = activation(inplace=True)
        # 1x1 projection
        self.conv1 = nn.Conv2d(
            in_channels, 
            width, 
            kernel_size=1, 
            bias=False
        )
        
        # 3x3 convolution
        self.bn2 = norm_layer(width)
        self.act2 = activation(inplace=True)
        self.conv2 = nn.Conv2d(
            width, 
            width, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        
        # 1x1 expansion
        self.bn3 = norm_layer(width)
        self.act3 = activation(inplace=True)
        self.conv3 = nn.Conv2d(
            width, 
            out_channels * self.expansion, 
            kernel_size=1, 
            bias=False
        )
        
        # Dropout if specified
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        
        # Skip connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Conv2d(
                in_channels, 
                out_channels * self.expansion, 
                kernel_size=1, 
                stride=stride, 
                bias=False
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First get pre-activation for both main branch and shortcut
        preact = self.bn1(x)
        preact = self.act1(preact)
        
        # Shortcut from pre-activated input
        identity = self.shortcut(preact)
        
        # First 1x1 projection
        out = self.conv1(preact)
        
        # 3x3 processing
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        
        # 1x1 expansion
        out = self.bn3(out)
        out = self.act3(out)
        out = self.dropout(out)
        out = self.conv3(out)
        
        # Add identity
        out += identity
        
        return out


class ConvBnAct(nn.Module):
    """
    Basic convolution-normalization-activation block.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size. Defaults to 3.
        stride: Convolution stride. Defaults to 1.
        padding: Convolution padding. Defaults to None (auto-calculated).
        groups: Convolution groups. Defaults to 1.
        dilation: Convolution dilation. Defaults to 1.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU.
        apply_act: Whether to apply activation. Defaults to True.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU,
        apply_act: bool = True
    ):
        super(ConvBnAct, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        # Auto-calculate padding if not specified
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=False
        )
        
        # Normalization layer
        self.bn = norm_layer(out_channels)
        
        # Activation layer
        self.act = activation(inplace=True) if apply_act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck. Defaults to 16.
        activation: Activation function. Defaults to nn.ReLU.
    """
    def __init__(
        self, 
        channels: int, 
        reduction: int = 16,
        activation: Type[nn.Module] = nn.ReLU
    ):
        super(SEBlock, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, 
            channels // reduction, 
            kernel_size=1, 
            padding=0
        )
        self.act = activation(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, 
            channels, 
            kernel_size=1, 
            padding=0
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling
        y = self.avg_pool(x)
        
        # First FC layer
        y = self.fc1(y)
        y = self.act(y)
        
        # Second FC layer
        y = self.fc2(y)
        y = self.sigmoid(y)
        
        # Channel-wise multiplication
        return x * y


class InvertedResidual(nn.Module):
    """
    Inverted Residual block from MobileNetV2/EfficientNet.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the depthwise convolution. Defaults to 1.
        expansion: Expansion factor for the input channels. Defaults to 6.
        norm_layer: Normalization layer. Defaults to nn.BatchNorm2d.
        activation: Activation function. Defaults to nn.ReLU6.
        use_se: Whether to use Squeeze-and-Excitation block. Defaults to False.
        se_reduction: Reduction ratio for SE block. Defaults to 4.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        expansion: int = 6,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        activation: Type[nn.Module] = nn.ReLU6,
        use_se: bool = False,
        se_reduction: int = 4
    ):
        super(InvertedResidual, self).__init__()
        # Register this class in the registry
        self.registry_type = "block"
        
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expansion
        
        layers = []
        
        # Expand
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                norm_layer(hidden_dim),
                activation(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            # Depthwise conv
            nn.Conv2d(
                hidden_dim, 
                hidden_dim, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                groups=hidden_dim, 
                bias=False
            ),
            norm_layer(hidden_dim),
            activation(inplace=True)
        ])
        
        # SE block
        if use_se:
            layers.append(SEBlock(hidden_dim, reduction=se_reduction))
        
        # Project back
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)
        

class MaxSigmoidAttnBlock(nn.Module):
    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        super().__init__()
        # Convert input channels to quaternion channels
        quat_c1 = c1 // 4
        quat_c2 = c2 // 4
        quat_ec = ec // 4
        
        self.nh = nh
        self.hc = quat_c2 // nh  # Already in quaternion channels
        
        # Quaternion embedding conv if needed
        self.ec = QConv2D(quat_c1, quat_ec, kernel_size=1) if quat_c1 != quat_ec else None
        
        # Guide linear layer remains unchanged as it processes regular features
        self.gl = nn.Linear(gc, quat_ec * 4)  # Multiply by 4 to match quaternion channels
        
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = QConv2D(quat_c1, quat_c2, kernel_size=3, stride=1, padding=1)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0
    
    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion input"
        
        # Process guide
        guide = self.gl(guide)  # [B, ec]
        guide = guide.view(B, -1, self.nh, self.hc)
        
        # Process input
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(B, Q, self.nh, self.hc, H, W)
        
        # Compute attention with quaternion structure preservation
        aw = torch.einsum('bqnchw,bnmc->bmhwn', embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc ** 0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale
        
        # Apply attention
        x = self.proj_conv(x)  # [B, C2, 4, H, W]
        x = x.view(B, self.nh, -1, Q, H, W)
        x = x * aw.unsqueeze(2).unsqueeze(2)
        
        return x.view(B, -1, Q, H, W)

class C2PSA(nn.Module):
    """C2PSA module with proper quaternion handling."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super(C2PSA, self).__init__()
        

        
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)

        
        # PSA blocks
        self.m = nn.Sequential(*[
            PSABlock(
                c=self.c,  # This is already in quaternion channels
                attn_ratio=0.5,
                num_heads=max(1, self.c // 16)  # Adjust head count appropriately
            ) for _ in range(n)
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C//4, 4, H, W] 
               e.g., for c1=1024: [B, 256, 4, H, W]
        """
        # Split while preserving quaternion structure
        features = self.cv1(x)  # [B, 2*c, 4, H, W]
        a, b = features.chunk(2, dim=1)  # Each [B, c, 4, H, W]
        
        # Process through attention blocks
        b = self.m(b)
        
        # Combine and project
        out = self.cv2(torch.cat([a, b], dim=1))  # Back to [B, C//4, 4, H, W]
        return out
