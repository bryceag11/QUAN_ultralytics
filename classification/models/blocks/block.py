# block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quaternion.conv import QConv, QConv1D, QConv2D, QConv3D, Conv
from quaternion.qactivation import QHardTanh, QLeakyReLU, QuaternionActivation, QReLU, QPReLU, QREReLU, QSigmoid, QTanh
from quaternion.qbatch_norm import QBN, IQBN, VQBN
from typing import List
import math
from quaternion_blocks import Qua


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

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
    """Quaternion-aware max pooling"""
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        pooled = self.pool(x_reshaped)
        
        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

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


class QSPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for quaternion neural networks.
    Maintains quaternion structure throughout the pooling pyramid.
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        
        # Set up intermediate channels (half of in_channels), ensuring multiple of 4
        c_ = c1 // 2
        assert c_ % 4 == 0, "Hidden channels must be a multiple of 4"

        # First convolution to reduce channel dimensionality
        self.cv1 = Conv(c1, c_, 1, 1)
        
        # Max pooling with kernel size and stride of 1
        self.m = QuaternionMaxPool(kernel_size=k, stride=1, padding = k//2)
        
        # Final convolution after concatenation to project back to out_channels
        self.cv2 = Conv(c_ * 4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SPPF layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, 4, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, 4, H, W)
        """
        # Initial convolution
        y = self.cv1(x)  # Shape: (B, c_, 4, H, W)
        # Apply three max pooling layers and concatenate outputs
        pooled_outputs = [y]  # Initialize with first convolution output
        for _ in range(3):
            pooled = self.m(pooled_outputs[-1])  # Apply pooling while maintaining quaternion structure
            pooled_outputs.append(pooled)  # Append pooled output
        
        # Concatenate along channel dimension and apply final convolution
        y = torch.cat(pooled_outputs, dim=1)  # Shape: (B, c_ * 4, 4, H, W)
        y = self.cv2(y)  # Project to out_channels: (B, out_channels, 4, H, W)
        return y


class QC3k2(nn.Module):
    """
    Enhanced C3k2 with parallel processing paths for quaternion features.
    Maintains cross-stage partial design while preserving quaternion structure.
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=False):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)

        self.m = nn.Sequential(*[
            QC3k(self.c, self.c, 2, shortcut, g) if c3k else 
            QBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        ])
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with parallel processing.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """

        # Split processing
        y = list(self.cv1(x).chunk(2, 1))
        
        # Extend with outputs from each module in self.m
        y.extend(m(y[-1]) for m in self.m)
     
        # # Combine all paths
        return self.cv2(torch.cat(y, 1))
    

class QC3k(nn.Module):
    """
    C3k module for quaternion data - CSP bottleneck with customizable kernel sizes.
    This version is designed to work with the parallel C3k2 implementation.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2*e)        
        # Calculate hidden channels (ensure multiple of 4)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        
        # Bottleneck block sequence
        self.m = nn.Sequential(*(QBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass maintaining quaternion structure.
        
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Output tensor [B, C, 4, H, W]
        """

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


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


class QPSABlock(nn.Module):
    """
    Position Sensitive Attention Block with quaternion support.
    """
    def __init__(self, c: int, attn_ratio: float = 1.0, num_heads: int = 8, shortcut: bool = True):
        super().__init__()
        assert c % 4 == 0, "Channels must be multiple of 4"
        self.Q = 4  # Quaternion dimension

        # Attention: [B, C, 4, H, W] -> [B, C, 4, H, W]
        # Splits into Q,K,V while maintaining quaternion structure
        self.attn = QAttention(dim=c, num_heads=num_heads, attn_ratio=attn_ratio)
        
        # FFN with quaternion-aware operations
        # Shape: [B, C, 4, H, W] -> [B, 2C, 4, H, W] -> [B, C, 4, H, W]
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

        self.shortcut = shortcut

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W] where C is already quaternion-adjusted
        """
        x = x + self.attn(x) if self.shortcut else self.attn(x)
        x = x + self.ffn(x) if self.shortcut else self.ffn(x)
        return x


class QPSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in quaternion neural networks.
    
    Combines quaternion-specific convolutions with attention mechanisms for enhanced feature extraction.
    
    Args:
        c1 (int): Number of input channels (must be a multiple of 4).
        c2 (int): Number of output channels (must be a multiple of 4).
        e (float): Expansion ratio for hidden channels.
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super(QPSA, self).__init__()
        assert c1 % 4 == 0 and c2 % 4 == 0, "Input and output channels must be multiples of 4."
        
        self.c = int(c1 * e)
        self.c = (self.c // 4) * 4  # Ensure hidden channels are multiples of 4
        assert self.c > 0, "Hidden channels must be positive and a multiple of 4."
        
        # Quaternion-aware convolution to reduce channels
        self.cv1 = QConv2D(c1, 2 * self.c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = IQBN(2 * self.c)
        self.act1 = QReLU()
        
        # Split into two parts: 'a' and 'b'
        # 'a' will go through attention and FFN
        # 'b' will remain unchanged
        self.attn = QAttention(dim=self.c, num_heads=self.c // (4 * 4), attn_ratio=1.0)
        
        # Feed-Forward Network for 'a'
        self.ffn = nn.Sequential(
            QConv2D(self.c, self.c * 2, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c * 2),
            QReLU(),
            QConv2D(self.c * 2, self.c, kernel_size=1, stride=1, padding=0, bias=False),
            IQBN(self.c)
        )
        
        # Final convolution to restore channels
        self.cv2 = QConv2D(2 * self.c, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = IQBN(c2)
        self.act2 = QReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PSA module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C1, H, W).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C2, H, W).
        """
        # Initial convolution, normalization, and activation
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Split into 'a' and 'b'
        a, b = x.chunk(2, dim=1)  # Each has channels = self.c
        
        # Apply attention to 'a'
        a = self.attn(a)
        a = self.ffn(a)
        
        # Concatenate 'a' and 'b'
        out = torch.cat((a, b), dim=1)  # Shape: (B, 2 * self.c, H, W)
        
        # Final convolution, normalization, and activation
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        
        return out

class QAttention(nn.Module):
    """
    Quaternion-aware attention module that performs self-attention while preserving quaternion structure.
    
    Args:
        dim (int): The input quaternion channels
        num_heads (int): Number of attention heads
        attn_ratio (float): Ratio to determine key dimension
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_ratio=0.5):
        super().__init__()
        # `dim` here is the *total* dimension (C_in_quat * 4)
        if dim % (num_heads * 4) != 0:
             raise ValueError(f"dim ({dim}) must be divisible by num_heads*4 ({num_heads*4})")

        self.num_heads = num_heads
        # Original QAttention calculates head_dim based on C_in_quat
        self.head_dim_orig = (dim // 4) // num_heads # Features per head per quaternion component
        self.key_dim_orig = max(1, int(self.head_dim_orig * attn_ratio)) # Key features per head per quat component

        # Dimensions for the scaled_dot_product_attention function per head
        # Embed dim per head = features_per_head * 4 (combining quat dim)
        self.attn_head_dim = self.head_dim_orig * 4
        self.attn_key_dim = self.key_dim_orig * 4

        # Total dimensions for the linear projection layer
        self.q_dim_total = self.attn_head_dim * num_heads # Should be equal to `dim`
        self.k_dim_total = self.attn_key_dim * num_heads
        self.v_dim_total = self.attn_head_dim * num_heads # Should be equal to `dim`
        qkv_out_dim = self.q_dim_total + self.k_dim_total + self.v_dim_total

        self.scale = self.attn_key_dim ** -0.5 # Scale based on K dim per head

        self.qkv = Conv(dim, qkv_out_dim, 1, act=False, bias=qkv_bias) # Use Conv wrapper
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Conv(dim, dim, 1, act=False) # Use Conv wrapper
        self.proj_drop = nn.Dropout(proj_drop)

        pe_groups = max(1, dim // 4)
        self.pe = Conv(dim, dim, 3, 1, g=pe_groups, act=False) # Use Conv wrapper

    def forward(self, x):
        # Input x shape: [B, C_in_quat, 4, H, W], where C_in_quat = dim // 4
        B, C_in_quat, Q_in, H, W = x.shape
        N = H * W
        dim_total_in = C_in_quat * Q_in

        # 1. QKV Projection
        qkv = self.qkv(x) # Expected output: [B, qkv_out_dim//4, 4, H, W]

        # 2. Reshape for Attention
        # Target format for SDPA: [B, num_heads, N, attn_head_dim]
        # qkv shape: [B, qkv_out_dim//4, 4, H, W]
        # Flatten spatial dims: [B, qkv_out_dim//4, 4, N]
        # Permute and view to combine C_quat and Q: [B, N, qkv_out_dim]
        qkv = qkv.view(B, -1, Q_in, N).permute(0, 3, 1, 2).reshape(B, N, -1)

        # Split into Q, K, V based on their total calculated dimensions
        q, k, v = torch.split(qkv, [self.q_dim_total, self.k_dim_total, self.v_dim_total], dim=-1)
        # q: [B, N, q_dim_total], k: [B, N, k_dim_total], v: [B, N, v_dim_total]

        # Reshape Q, K, V to introduce the head dimension
        # Q: [B, N, num_heads, attn_head_dim] -> [B, num_heads, N, attn_head_dim]
        q = q.view(B, N, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3)
        # K: [B, N, num_heads, attn_key_dim] -> [B, num_heads, N, attn_key_dim]
        k = k.view(B, N, self.num_heads, self.attn_key_dim).permute(0, 2, 1, 3)
        # V: [B, N, num_heads, attn_head_dim] -> [B, num_heads, N, attn_head_dim]
        v = v.view(B, N, self.num_heads, self.attn_head_dim).permute(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        # Uses self.scale implicitly if not provided, but docs say it's based on Q dim. Let's check.
        # Pytorch SDPA scales by sqrt(Q.shape[-1]), which is self.attn_head_dim here.
        # Our desired scale is based on self.attn_key_dim.
        # We can apply our custom scale *after* SDPA if needed, or scale Q beforehand.
        # Let's try scaling Q beforehand.
        # q = q * self.scale # Apply manual scaling based on K dim

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        # attn_output shape: [B, num_heads, N, attn_head_dim]

        # 4. Reshape output back to spatial quaternion format
        # attn_output: [B, num_heads, N, attn_head_dim] -> [B, N, num_heads, attn_head_dim]
        # -> [B, N, dim_total_out] -> [B, dim_total_out, N] -> [B, C_out_quat, 4, N]
        # -> [B, C_out_quat, 4, H, W]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, -1) # [B, N, dim_total_out]
        dim_total_out = attn_output.shape[-1] # Should be same as input dim_total_in
        attn_output = attn_output.permute(0, 2, 1) # [B, dim_total_out, N]

        # Reshape N back to H, W and separate quaternion dimension
        C_out_quat = dim_total_out // 4
        attn_output = attn_output.view(B, C_out_quat, 4, H, W) # [B, C_out_quat, 4, H, W]

        # 5. Add Positional Encoding
        # Pass the correctly shaped attn_output
        attn_output = attn_output + self.pe(attn_output)

        # 6. Final Projection and Dropout
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x # Shape: [B, C_out_quat, 4, H, W]


class QC2PSA(nn.Module):
    """C2PSA module with proper quaternion handling."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super(QC2PSA, self).__init__()
        

        
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)

        
        # PSA blocks
        self.m = nn.Sequential(*[
            QPSABlock(
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
