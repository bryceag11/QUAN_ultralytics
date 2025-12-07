# quaternion_blocks
"""
Quaternion building blocks for deeper networks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List
from quaternion.qconv import QConv2D, QDense, QConv, IQBN
from quaternion.qactivation import QPReLU, QREReLU, QSiLU


class QuaternionDropout(nn.Module):
    """Quaternion-aware dropout that maintains quaternion structure"""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        """
        Input: [B, C, H, W, 4] - BCHWQ format
        Output: [B, C, H, W, 4] 
        """
        if not self.training:
            return x
            
        B, C, H, W, Q = x.shape
        # Apply same dropout mask to entire quaternion (all 4 components)
        # Mask shape: [B, C, H, W, 1] -> broadcast to [B, C, H, W, 4]
        mask = torch.bernoulli(torch.full((B, C, H, W, 1), 1-self.p, device=x.device))
        return x * mask  # Broadcasting handles the quaternion dimension




class QuaternionAvgPool(nn.Module):
    def __init__(self, kernel_size=None, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        # x shape: [B, C, H, W, Q=4]
        B, C, H, W, Q = x.shape
        
        if self.kernel_size is None:
            # Global average pooling
            return x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1, Q]
        else:
            # Regular pooling - need to handle each quaternion component
            x_reshaped = x.view(B, C * Q, H, W)  # [B, C*Q, H, W]
            pooled = F.avg_pool2d(x_reshaped, self.kernel_size, self.stride, self.padding)
            
            # Reshape back to BCHWQ
            B, CQ, H_out, W_out = pooled.shape
            return pooled.view(B, C, Q, H_out, W_out).permute(0, 1, 3, 4, 2)


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

class QBottleneck(nn.Module):
    """
    Quaternion-aware bottleneck block used in C3k.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k =(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        
        # First 1x1 conv to reduce channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quaternion bottleneck.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """
        identity = x if self.add else None
        
        # First conv + bn + relu
        x = self.cv1(x)

        
        # Second conv + bn + relu
        x = self.cv2(x)
        
        # Add shortcut if enabled
        if self.add:
            x = x + identity
                
        return x

class QuaternionPolarPool(nn.Module):
    """
    Novel pooling layer that operates in quaternion polar form to preserve 
    rotational relationships while reducing spatial dimensions.
    """
    def __init__(self, kernel_size: int, stride: int = None):
        super(QuaternionPolarPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, 4, H, W]
        B, C, Q, H, W = x.shape
        assert Q == 4, "Quaternion dimension must be 4."
        
        # Reshape to (B, C, H, W)
        x_flat = x.view(B, C, H, W)
        
        # Compute magnitudes and phases for each quaternion
        # Assuming quaternions are normalized; if not, adjust accordingly
        magnitudes = torch.norm(x_flat, dim=1, keepdim=True)  # [B, 1, H, W]
        phases = torch.atan2(x_flat[:, 1:, :, :], x_flat[:, :1, :, :])  # [B, 3, H, W]
        
        # Pool magnitudes using max pooling
        pooled_magnitudes = F.max_pool2d(
            magnitudes, 
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2
        )  # [B, 1, H', W']
        
        # Pool phases using circular mean
        # Unwrap phases for proper averaging
        cos_phases = torch.cos(phases)
        sin_phases = torch.sin(phases)
        
        pooled_cos = F.avg_pool2d(cos_phases, self.kernel_size, self.stride, padding=self.kernel_size // 2)
        pooled_sin = F.avg_pool2d(sin_phases, self.kernel_size, self.stride, padding=self.kernel_size // 2)
        pooled_phases = torch.atan2(pooled_sin, pooled_cos)  # [B, 3, H', W']
        
        # Reconstruct quaternion
        pooled_real = pooled_magnitudes * torch.cos(pooled_phases[:, 0:1, :, :])
        pooled_i = pooled_magnitudes * torch.sin(pooled_phases[:, 0:1, :, :])
        pooled_j = pooled_magnitudes * torch.sin(pooled_phases[:, 1:2, :, :])
        pooled_k = pooled_magnitudes * torch.sin(pooled_phases[:, 2:3, :, :])
        
        # Concatenate quaternion components
        pooled = torch.cat([pooled_real, pooled_i, pooled_j, pooled_k], dim=1)  # [B, 4, H', W']
        
        return pooled.view(B, C, Q, pooled.shape[2], pooled.shape[3])  # [B, C, 4, H', W']

class QuaternionMaxPool(nn.Module):
    """Quaternion-aware max pooling for BCHWQ layout"""
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [B, C, H, W, 4] - BCHWQ format
        Output: [B, C, H_out, W_out, 4]
        """
        B, C, H, W, Q = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Apply pooling to each quaternion component separately
        pooled_components = []
        for q_idx in range(4):
            # Extract component: [B, C, H, W]
            component = x[..., q_idx]
            # Apply pooling: [B, C, H_out, W_out]
            pooled = self.pool(component)
            pooled_components.append(pooled)
        
        # Stack components back: [B, C, H_out, W_out, 4]
        return torch.stack(pooled_components, dim=-1)
    
class InformationTheoreticQuaternionPool(nn.Module):
    """
    Information-Theoretic Quaternion Pooling (ITQPP) layer.
    Emphasizes interchannel relationships by selecting quaternions that maximize mutual information within pooling regions.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        Initializes the Information-Theoretic Quaternion Pooling layer.

        Args:
            kernel_size (int or tuple): Size of the pooling window.
            stride (int or tuple, optional): Stride of the pooling window. Defaults to kernel_size.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to 0.
        """
        super(InformationTheoreticQuaternionPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding

    def forward(self, x):
        """
        Forward pass for Information-Theoretic Quaternion Pooling.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, 4, H, W).

        Returns:
            torch.Tensor: Pooled tensor with preserved quaternion structure.
        """
        # Ensure channel dimension is a multiple of 4
        batch_size, channels, quat_dim, H, W = x.shape
        assert quat_dim == 4, "Quaternion dimension must be 4."
        assert channels % 4 == 0, "Number of channels must be a multiple of 4."

        # Reshape to separate quaternion components
        x = x.view(batch_size, channels // 4, 4, H, W)  # Shape: (B, C_q, 4, H, W)

        # Apply adaptive pooling to obtain windows
        x_unfold = F.unfold(x.view(batch_size, -1, H, W), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # x_unfold shape: (B, C_q*4*kernel_size*kernel_size, L)

        # Reshape to (B, C_q, 4, kernel_size*kernel_size, L)
        kernel_area = self.kernel_size * self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0] * self.kernel_size[1]
        x_unfold = x_unfold.view(batch_size, channels // 4, quat_dim, kernel_area, -1)

        # Compute entropy for each quaternion across the window
        # Simplified entropy: -sum(p * log(p)), where p is normalized magnitude
        magnitudes = torch.norm(x_unfold, dim=2)  # Shape: (B, C_q, K, L)
        p = magnitudes / (magnitudes.sum(dim=3, keepdim=True) + 1e-8)  # Shape: (B, C_q, K, L)
        entropy = - (p * torch.log(p + 1e-8)).sum(dim=2)  # Shape: (B, C_q, L)

        # Select the quaternion with the highest entropy within each window
        _, indices = entropy.max(dim=1)  # Shape: (B, L)

        # Gather the selected quaternions
        # Create index tensors
        batch_indices = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, indices.size(1))
        channel_indices = indices  # Shape: (B, L)

        # Extract quaternions
        pooled_quaternions = x_unfold[batch_indices, channel_indices, :, :, torch.arange(indices.size(1), device=x.device)]

        # Reshape back to (B, C_q*4, H_out, W_out)
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        pooled_quaternions = pooled_quaternions.view(batch_size, -1, H_out, W_out)

        return pooled_quaternions


class QuaternionAdaptiveAvgPool2d(nn.Module):
    """
    Quaternion-aware Adaptive Average Pooling 2D layer.

    This layer applies 2D adaptive average pooling to each of the 4 components
    of the quaternion-valued input feature maps independently.

    Args:
        output_size (Union[int, None, Tuple[int, None], Tuple[None, int], Tuple[int, int]]):
            The target output size of the image of the form H x W.
            Can be a tuple (H, W) or a single H for a square image H x H.
            H and W can be None, which means the size will be the same as that of the input.
    """
    def __init__(self, output_size: Union[int, None, Tuple[Optional[int], Optional[int]]]):
        super().__init__()
        self.output_size = output_size
        # We will use the standard nn.AdaptiveAvgPool2d internally
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C_q, Q, H, W]
                              where B is batch size,
                                    C_q is the number of quaternion channels,
                                    Q is the quaternion dimension (must be 4),
                                    H, W are spatial dimensions.
        Returns:
            torch.Tensor: Output tensor of shape [B, C_q, Q, H_out, W_out]
                          where H_out, W_out are the specified output_size.
        """
        B, C_q, Q, H, W = x.shape
        if Q != 4:
            raise ValueError(f"Input quaternion dimension Q must be 4, but got {Q}")

        # To use nn.AdaptiveAvgPool2d, we need to treat the quaternion components
        # as if they are part of the channel dimension temporarily.
        # Reshape x from [B, C_q, Q, H, W] to [B, C_q * Q, H, W]
        # This flattens C_q and Q together.
        x_reshaped = x.reshape(B, C_q * Q, H, W)

        # Apply adaptive average pooling
        # Output will be [B, C_q * Q, H_out, W_out]
        pooled_reshaped = self.pool(x_reshaped)

        # Get the new spatial dimensions
        _ , _, H_out, W_out = pooled_reshaped.shape

        # Reshape back to the quaternion format [B, C_q, Q, H_out, W_out]
        output = pooled_reshaped.reshape(B, C_q, Q, H_out, W_out)

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(output_size={self.output_size})"


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class QShallowResNet(nn.Module):
    def __init__(self, num_classes=10, mapping_type='poincare'):
        super().__init__()
        
        # Initial layer to learn quaternion components from RGB
        self.conv1 = nn.Sequential(
            QConv2D(3, 16, kernel_size=3, stride=1, padding=1),
            IQBN(16),
            QSiLU()
        )
        
        # Stage 1: 2 residual blocks (16 quaternion filters)
        self.stage1 = self._make_stage(16, 32, blocks=2, stride=1)
        
        # Stage 2: 1 residual block (32 quaternion filters)
        self.stage2 = self._make_stage(32, 64, blocks=1, stride=2)
        
        # Stage 3: 1 residual block (64 quaternion filters)
        self.stage3 = self._make_stage(64, 128, blocks=1, stride=2)
        
        # Global Average Pooling
        self.avg_pool = QuaternionAvgPool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(128, num_classes * 4, mapping_type=mapping_type)
        )
    
    def _make_stage(self, in_channels, out_channels, blocks, stride):
        layers = []
        
        # First block may have stride for downsampling
        layers.append(QuaternionBasicBlock(in_channels, out_channels, stride))
        
        # Remaining blocks have stride=1
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(out_channels, out_channels, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        x = self.classifier(x)
        
        # Extract only real component for final classification
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        quaternion_norm = torch.norm(x, dim=2)

        return quaternion_norm


class QResNet110(nn.Module):
    """
    ResNet110 with quaternion operations
    - 3 stages with 18 blocks each (total 110 layers)
    - Narrow base width (16 channels)
    - Separate heads for CIFAR-10 and CIFAR-100
    """
    def __init__(self, num_classes=10, mapping_type='poincare', width_multiplier=1.0):
        super().__init__()
        
        # Base width - use multiplier to adjust model capacity
        # For true ResNet110, width_multiplier=1.0 is appropriate
        base_width = max(int(16 * 1), 8)  # Minimum 8 channels
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(base_width),
            QSiLU()
        )
        
        # 3 stages with 18 blocks each (36 convs per stage)
        # Stage 1: No downsampling (retain spatial dimensions)
        self.stage1 = self._make_layer(base_width, base_width*2, blocks=10, stride=1, mapping_type=mapping_type)
        
        # Stage 2: Downsample by 2
        self.stage2 = self._make_layer(base_width*2, base_width*4, blocks=9, stride=2, mapping_type=mapping_type)
        
        # Stage 3: Downsample by 2 again
        self.stage3 = self._make_layer(base_width*4, base_width*8, blocks=8, stride=2, mapping_type=mapping_type)
        
        # Global Average Pooling
        self.avg_pool = QuaternionAvgPool()
        
        # Choose classifier based on dataset
        if num_classes == 10:  # CIFAR-10
            self.classifier = self._make_cifar10_head(base_width*8, mapping_type)
        else:  # CIFAR-100
            self.classifier = self._make_cifar100_head(base_width*8, mapping_type)
            
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, mapping_type):
        """Create a stage with specified number of residual blocks"""
        layers = []
        
        # First block may have stride for downsampling
        layers.append(QuaternionBasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride
        ))
        
        # Remaining blocks have stride=1 (no downsampling)
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1
            ))
        
        return nn.Sequential(*layers)
    
    def _make_cifar10_head(self, in_features, mapping_type):
        """Create a compact classifier head for CIFAR-10"""
        return nn.Sequential(
            nn.Flatten(),
            QDense(in_features, 256, mapping_type=mapping_type),
            nn.SiLU(),
            QDense(256, 10 * 4, mapping_type=mapping_type)  # *4 for quaternion components
        )
    
    def _make_cifar100_head(self, in_features, mapping_type):
        """Create a classifier head for CIFAR-100 with more capacity"""
        return nn.Sequential(
            nn.Flatten(),
            QDense(in_features, 1024, mapping_type=mapping_type),
            nn.SiLU(),
            QDense(1024, 512, mapping_type=mapping_type),
            nn.SiLU(),
            QDense(512, 100 * 4, mapping_type=mapping_type)  # *4 for quaternion components
        )
    
    def _initialize_weights(self):
        """Optional custom weight initialization"""
        for m in self.modules():
            if isinstance(m, QConv2D) or isinstance(m, QDense):
                # Weight initialization is already handled in these layers
                pass
            elif isinstance(m, IQBN):
                # Optional additional initialization for batch norm
                pass
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        
        # Residual stages
        x = self.stage1(x)  # 32×32
        x = self.stage2(x)  # 16×16
        x = self.stage3(x)  # 8×8
        
        # Global average pooling
        x = self.avg_pool(x)  # 1×1
        
        # Classification
        x = self.classifier(x)
        
        # Use quaternion norm for classification
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)  # Reshape to separate quaternion components
        quaternion_norm = torch.norm(x, dim=2)  # Calculate magnitude of each quaternion
        
        return quaternion_norm



class QDenseLayer(nn.Module):
    """
    Quaternion-based Dense Layer for DenseNet:
    BN -> SiLU -> QConv (1x1) -> BN -> SiLU -> QConv (3x3)
    """
    def __init__(self, in_channels, growth_rate, mapping_type='poincare'):
        super().__init__()
        
        # Bottleneck reduces computational cost
        bottleneck_channels = 4 * growth_rate
        
        # First part: Bottleneck using 1x1 conv
        self.bn1 = IQBN(in_channels)
        self.silu1 = QSiLU()
        self.conv1 = QConv2D(in_channels, bottleneck_channels, kernel_size=1, 
                          stride=1, padding=0, mapping_type=mapping_type)
        
        # Second part: 3x3 conv to produce k feature maps
        self.bn2 = IQBN(bottleneck_channels)
        self.silu2 = QSiLU()
        self.conv2 = QConv2D(bottleneck_channels, growth_rate, kernel_size=3, 
                          stride=1, padding=1, mapping_type=mapping_type)
        
    def forward(self, x):
        # Bottleneck part
        out = self.bn1(x)
        out = self.silu1(out)
        out = self.conv1(out)
        
        # 3x3 conv part
        out = self.bn2(out)
        out = self.silu2(out)
        out = self.conv2(out)
        
        # Concatenate input with output (dense connection)
        return torch.cat([x, out], dim=1)

class QDenseBlock(nn.Module):
    """
    Quaternion-based Dense Block for DenseNet:
    Stack of dense layers with dense connections
    """
    def __init__(self, in_channels, num_layers, growth_rate, mapping_type='poincare'):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(
                QDenseLayer(in_channels + i * growth_rate, growth_rate, mapping_type)
            )
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class QTransitionLayer(nn.Module):
    """
    Quaternion-based Transition Layer for DenseNet:
    BN -> SiLU -> QConv (1x1) -> AvgPool (2x2)
    Used to reduce feature map size between dense blocks
    """
    def __init__(self, in_channels, out_channels, mapping_type='poincare'):
        super().__init__()
        
        self.bn = IQBN(in_channels)
        self.silu = QSiLU()
        self.conv = QConv2D(in_channels, out_channels, kernel_size=1, 
                         stride=1, padding=0, mapping_type=mapping_type)
        self.pool = QuaternionAvgPool(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.silu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

class QDenseNet(nn.Module):
    """
    Quaternion-based DenseNet for CIFAR datasets
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), 
                 compression=0.5, num_classes=100, mapping_type='poincare'):
        super().__init__()
        
        self.mapping_type = mapping_type
        
        # Initial convolution (3×3 instead of 7×7 for CIFAR)
        # Note: Input is RGB, so in_channels=3
        self.features = nn.Sequential(
            QConv2D(3, 2 * growth_rate, kernel_size=3, stride=1, 
                   padding=1, mapping_type=mapping_type),
            IQBN(2 * growth_rate),
            QSiLU()
        )
        
        # Current number of channels after initial conv
        num_channels = 2 * growth_rate
        
        # Add dense blocks with transition layers
        for i, num_layers in enumerate(block_config):
            # Add dense block
            self.features.add_module(
                f'dense_block_{i+1}',
                QDenseBlock(num_channels, num_layers, growth_rate, mapping_type)
            )
            
            # Update number of channels
            num_channels += num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i < len(block_config) - 1:
                # Calculate number of output channels with compression
                out_channels = int(num_channels * compression)
                
                self.features.add_module(
                    f'transition_{i+1}',
                    QTransitionLayer(num_channels, out_channels, mapping_type)
                )
                
                # Update number of channels
                num_channels = out_channels
        
        # Final batch norm
        self.features.add_module('norm_final', IQBN(num_channels))
        self.features.add_module('silu_final', QSiLU())
        
        # Global average pooling and classifier
        self.avg_pool = QuaternionAvgPool()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(num_channels, 512, mapping_type=mapping_type),
            nn.SiLU(),
            QDense(512, num_classes * 4, mapping_type=mapping_type)
        )
        
    def forward(self, x):
        # Feature extraction
        features = self.features(x)
        
        # Global average pooling
        out = self.avg_pool(features)
        
        # Classification
        out = self.classifier(out)
        
        # Extract quaternion norm for final classification
        batch_size = out.size(0)
        out = out.view(batch_size, -1, 4)
        quaternion_norm = torch.norm(out, dim=2)
        
        return quaternion_norm

def create_qdensenet121(num_classes=100, mapping_type='poincare'):
    """
    Creates a Quaternion DenseNet-121 model
    DenseNet-121 has block configuration of [6, 12, 24, 16] but we adapt for CIFAR
    """
    return QDenseNet(
        growth_rate=12,
        block_config=(6, 12, 24),  # Reduced configuration for CIFAR
        compression=0.5,
        num_classes=num_classes,
        mapping_type=mapping_type
    )

def create_qdensenet_cifar(num_classes=100, mapping_type='poincare'):
    """
    Creates a Quaternion DenseNet optimized for CIFAR datasets (BC version with k=12)
    This uses the configuration from the original paper for CIFAR
    """
    return QDenseNet(
        growth_rate=12,  # k=12 as suggested
        block_config=(16, 16, 16),  # 3 blocks of 16 layers each
        compression=0.5,  # Standard BC compression factor
        num_classes=num_classes,
        mapping_type=mapping_type
    )


class QuaternionBottleneckBlock(nn.Module):
    """Bottleneck block with pre-activation but no SE or dropout"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        bottleneck_channels = out_channels // 2
        
        # Pre-activation for the first block
        self.bn1 = IQBN(in_channels)
        self.relu1 = QSiLU()
        # 1x1 projection
        self.conv1 = QConv2D(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        
        # 3x3 convolution
        self.bn2 = IQBN(bottleneck_channels)
        self.relu2 = QSiLU()
        self.conv2 = QConv2D(bottleneck_channels, bottleneck_channels, kernel_size=3, 
                           stride=stride, padding=1)
        
        # 1x1 expansion
        self.bn3 = IQBN(bottleneck_channels)
        self.relu3 = QSiLU()
        self.conv3 = QConv2D(bottleneck_channels, out_channels, kernel_size=1)
        
        # Skip connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                QConv2D(in_channels, out_channels, kernel_size=1, stride=stride),
                IQBN(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        # First 1x1 projection with pre-activation
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        # 3x3 processing
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        # 1x1 expansion
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        
        # Add identity
        out += identity
        
        return out

class QResNet34(nn.Module):
    """
    QResNet34 with mixed block types and increased width,
    without SE attention or dropout
    """
    def __init__(self, num_classes=10, mapping_type='poincare'):
        super().__init__()
        
        # Start with narrow base width (16 channels)
        # This is critical for CIFAR models
        base_width = 16
        
        # Initial 3×3 convolution (no 7×7 and no maxpool for CIFAR)
        self.conv1 = nn.Sequential(
            QConv2D(3, base_width, kernel_size=3, stride=1, padding=1, mapping_type=mapping_type),
            IQBN(base_width),
            QSiLU()
        )
        
        # 3 stages with standard ResNet34 block counts [3, 4, 6]
        # But channel counts follow CIFAR pattern [16, 32, 64]
        self.stage1 = self._make_layer(base_width, base_width, blocks=3, stride=1)
        self.stage2 = self._make_layer(base_width, base_width*2, blocks=4, stride=2)
        self.stage3 = self._make_layer(base_width*2, base_width*4, blocks=6, stride=2)
        
        # Global Average Pooling
        self.avg_pool = QuaternionAvgPool()
        
        # Simple classifier for CIFAR-10
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
            stride=stride
        ))
        
        # Remaining blocks with stride=1
        for _ in range(1, blocks):
            layers.append(QuaternionBasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        # Use quaternion norm for classification
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 4)
        q_norm = torch.norm(x, dim=2)
        
        return q_norm
