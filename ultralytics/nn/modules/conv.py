# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""
# ultralytics/nn/modules/conv.py

import math

import numpy as np
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
import torch.nn.functional as F
from .activation import *
from torch.jit import script
from torch.utils import cpp_extension
# import quaternion_ops
from .quaternion_autograd_cuda import qconv2d_function
# print('Sucessfully imported compiled quaternion_ops')
# CUDA_EXT = True

__all__ = (
    "Conv",
    "Conv2",
    "QConv",
    "QConv2D",
    "QConcat",
    "QUpsample",
    "IQBN",
    "IQLN"
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "QConcat",
    "Concat",
    "RepConv",
    "Index",
    "QUpsample"
)



try:
    import quaternion_ops
    print('Sucessfully imported compiled quaternion_ops')
    CUDA_EXT = True
except:
    print("Failed to import compiled quaternion_ops. Falling back to Pytorch implementation")
    CUDA_EXT = False

def autopad(k, p=None, d=1): 
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class QConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros', 
                 dtype=None,
                 mapping_type: str='poincare'):
        super().__init__()

        # Parameter validation
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        # Handle autopad for padding
        if padding == 'same':
            # For same: padding = (kernel_size - 1) // 2 when stride=1
            if stride == (1, 1):
                padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            else:
                # For stride > 1, use the autopad function
                padding = tuple(autopad(k, None, d) for k, d in zip(kernel_size, dilation))
        elif isinstance(padding, int): 
            padding = (padding, padding)
        elif isinstance(padding, (list, tuple)) and len(padding) == 2:
            padding = tuple(padding)
        else:
            raise ValueError(f"Invalid padding: {padding}")

        # if padding == 'same':
        #      padding = tuple(autopad(k, None, d) for k, d in zip(kernel_size, dilation))
        # elif isinstance(padding, int): padding = (padding, padding)

        self.in_channels_total = in_channels
        self.out_channels_total = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.mapping_type = mapping_type

        self.is_first_layer = (in_channels == 3)
        if self.is_first_layer:
            self.in_channels_per_comp = 1 
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            self.in_channels_per_comp = in_channels // 4

        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        self.out_channels_per_comp = out_channels // 4

        # Input channels per component per group
        assert self.in_channels_per_comp % groups == 0, "Input channels per component must be divisible by groups"
        self.in_channels_per_comp_grp = self.in_channels_per_comp // groups
        
        self.bias_flag_overall = bias
        
        weight_shape = (
            self.out_channels_per_comp,
            self.in_channels_per_comp_grp,
            *self.kernel_size
        )
        self.weight_r = nn.Parameter(torch.zeros(weight_shape))
        self.weight_i = nn.Parameter(torch.zeros(weight_shape))
        self.weight_j = nn.Parameter(torch.zeros(weight_shape))
        self.weight_k = nn.Parameter(torch.zeros(weight_shape))

        if bias:
            bias_shape = (self.out_channels_per_comp,)
            self.bias_r = nn.Parameter(torch.zeros(bias_shape))
        else:
            self.register_parameter('bias_r', None)
        self.register_parameter('bias_i', None)
        self.register_parameter('bias_j', None)
        self.register_parameter('bias_k', None)

        self._initialize_weights() 


    # def _initialize_weights(self):
    #         """
    #         Initializes weights using a manual Lecun Normal implementation for the real part
    #         and zeros for the imaginary parts, providing a stable starting point for training.
    #         """
    #         # Calculate fan-in, which is essential for the initialization math
    #         kernel_prod = np.prod(self.kernel_size)
    #         fan_in_component = self.in_channels_per_comp_grp * kernel_prod
    #         if fan_in_component > 0:
    #             # Manually implement Lecun Normal initialization for backward compatibility.
    #             # Lecun Normal uses a standard deviation of sqrt(1 / fan_in).
    #             std_dev = math.sqrt(5.0 / fan_in_component)
    #             nn.init.normal(self.weight_r, mean=0.0, std=std_dev)
    #         # Initialize imaginary components to zero for a stable start.
    #         with torch.no_grad():
    #             self.weight_i.zero_()
    #             self.weight_j.zero_()
    #             self.weight_k.zero_()
    #         # Initialize the real bias to zero.
    #         if self.bias_flag_overall and self.biasr is not None:
    #             nn.init.zeros(self.bias_r)

    # def _initialize_weights(self):
    #     """
    #     Fixed initialization that matches your CUDA kernel's Hamilton mixing.
    #     This is THE MOST IMPORTANT fix for your 0.12 mAP issue.
    #     """
    #     kernel_prod = np.prod(self.kernel_size)
    #     fan_in_component = self.in_channels_per_comp_grp * kernel_prod
        
    #     if fan_in_component > 0:
    #         # He initialization for ReLU activation
    #         std_real = math.sqrt(2.0 / fan_in_component)
            
    #         # Initialize real component with He init
    #         nn.init.normal_(self.weight_r, mean=0.0, std=std_real)
            
    #         # CRITICAL FIX: Initialize imaginary components properly!
    #         # Your CUDA kernel SUBTRACTS these, so use smaller magnitude
    #         std_imag = std_real * 0.3  # 30% of real component
            
    #         nn.init.normal_(self.weight_i, mean=0.0, std=std_imag)
    #         nn.init.normal_(self.weight_j, mean=0.0, std=std_imag)
    #         nn.init.normal_(self.weight_k, mean=0.0, std=std_imag)
            
    #         # CRITICAL: Add noise to break symmetry
    #         # Without this, network can't learn properly
    #         with torch.no_grad():
    #             noise_scale = 0.02 * std_imag
    #             self.weight_i += torch.randn_like(self.weight_i) * noise_scale
    #             self.weight_j += torch.randn_like(self.weight_j) * noise_scale
    #             self.weight_k += torch.randn_like(self.weight_k) * noise_scale
                
    #             # # Ensure no exact zeros
    #             # min_val = 1e-4
    #             # self.weight_i = torch.where(
    #             #     torch.abs(self.weight_i) < min_val,
    #             #     torch.sign(self.weight_i) * min_val,
    #             #     self.weight_i
    #             # )
    #             # self.weight_j = torch.where(
    #             #     torch.abs(self.weight_j) < min_val,
    #             #     torch.sign(self.weight_j) * min_val,
    #             #     self.weight_j
    #             # )
    #             # self.weight_k = torch.where(
    #             #     torch.abs(self.weight_k) < min_val,
    #             #     torch.sign(self.weight_k) * min_val,
    #             #     self.weight_k
    #             # )
        
    #     # Initialize bias with small positive value
    #     if self.bias_flag_overall and self.bias_r is not None:
    #         nn.init.constant_(self.bias_r, 0.01)


    def _initialize_weights(self):
        kernel_prod = np.prod(self.kernel_size)
        # For one of the r, i, j, or k weight matrices
        fan_in_component = self.in_channels_per_comp_grp * kernel_prod

        scale_factors_map = {
            'luminance': [1.0, 1.0, 1.0, 1.0],
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],
            'raw_normalized': [1.0, 1.0, 1.0, 1.0],
            'hamilton': [1.0, 1.0, 1.0, 1.0], 
            'poincare': [1.0, 1.0, 1.0, 1.0]
        }
        # Default scales if self.mapping_type is not in the map
        scales = scale_factors_map.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])

        weights_to_init = [self.weight_r, self.weight_i, self.weight_j, self.weight_k]

        for i, weight_param in enumerate(weights_to_init):
            # Initialize weights using scaled 'a' for kaiming_uniform_
            nn.init.kaiming_uniform_(weight_param, a=math.sqrt(5.0) * scales[i])

        # Only initialize self.bias_r if it was created 
        if self.bias_flag_overall and self.bias_r is not None:
            bound_r = (1 / math.sqrt(fan_in_component)) * scales[0] if fan_in_component > 0 else 0
            nn.init.uniform_(self.bias_r, -bound_r, bound_r)

    # def _initialize_weights(self):

    #     if hasattr(self, 'weight_r'):
    #         with torch.no_grad():
    #             fan_in = self.weight_r[0].numel()
    #             std = math.sqrt(2.0 / fan_in)
                
    #             # Real component - normal He init
    #             nn.init.normal_(self.weight_r, 0, std)
                
    #             # Imaginary - smaller because they're SUBTRACTED
    #             nn.init.normal_(self.weight_i, 0, std)
    #             nn.init.normal_(self.weight_j, 0, std)
    #             nn.init.normal_(self.weight_k, 0, std)
                
    #             # Critical: Add noise to break symmetry
    #             self.weight_i += torch.randn_like(self.weight_i) * 0.01
    #             self.weight_j += torch.randn_like(self.weight_j) * 0.01
    #             self.weight_k += torch.randn_like(self.weight_k) * 0.01

    # def _initialize_weights(self):
    #     """
    #     This is the REAL fix. Your initialization is broken.
    #     Add this to your QConv2D.__init__ method.
    #     """
    #     # Calculate proper scale
    #     kernel_prod = np.prod(self.kernel_size)
    #     fan_in = self.in_channels_per_comp_grp * kernel_prod
        
    #     # He initialization scale
    #     scale = np.sqrt(2.0 / fan_in)
        
    #     # CRITICAL: All components need proper initialization
    #     nn.init.normal_(self.weight_r, mean=0, std=scale)
    #     nn.init.normal_(self.weight_i, mean=0, std=scale)  # Half scale
    #     nn.init.normal_(self.weight_j, mean=0, std=scale)
    #     nn.init.normal_(self.weight_k, mean=0, std=scale)
        
    #     # Optional: Add structured initialization for better quaternion properties
    #     with torch.no_grad():
    #         # Ensure quaternion properties
    #         norm = torch.sqrt(
    #             self.weight_r**2 + self.weight_i**2 + 
    #             self.weight_j**2 + self.weight_k**2
    #         )
    #         # Normalize to unit quaternions
    #         self.weight_r /= (norm + 1e-8)
    #         self.weight_i /= (norm + 1e-8)
    #         self.weight_j /= (norm + 1e-8)
    #         self.weight_k /= (norm + 1e-8)
            
    #         # Scale back up
    #         self.weight_r *= scale
    #         self.weight_i *= scale
    #         self.weight_j *= scale
    #         self.weight_k *= scale

    # def _initialize_weights(self):
    #     """
    #     Initialize quaternion weights with proper scaling for COCO/large datasets.
    #     Key insight: Don't zero out imaginary components - just reduce them!
    #     """
    #     kernel_prod = np.prod(self.kernel_size)
    #     fan_in_component = self.in_channels_per_comp_grp * kernel_prod
        
    #     if fan_in_component > 0:
    #         # Use Kaiming initialization but with different scaling for components
    #         # This maintains quaternion properties while being stable
            
    #         # Real component - normal Kaiming
    #         nn.init.kaiming_normal_(self.weight_r, a=0.1, mode='fan_out', nonlinearity='relu')
            
    #         # Imaginary components - reduced but NOT zero
    #         # This is critical for quaternion networks to work properly!
    #         nn.init.kaiming_normal_(self.weight_i, a=0.1, mode='fan_out', nonlinearity='relu')
    #         nn.init.kaiming_normal_(self.weight_j, a=0.1, mode='fan_out', nonlinearity='relu')
    #         nn.init.kaiming_normal_(self.weight_k, a=0.1, mode='fan_out', nonlinearity='relu')
            
    #         with torch.no_grad():
    #             # Scale down imaginary components but keep them non-zero
    #             self.weight_i *= 0.75  # 10% of real component
    #             self.weight_j *= 0.75
    #             self.weight_k *= 0.75
                
    #             # Add small noise to break symmetry
    #             self.weight_i += torch.randn_like(self.weight_i) * 0.01
    #             self.weight_j += torch.randn_like(self.weight_j) * 0.01
    #             self.weight_k += torch.randn_like(self.weight_k) * 0.01
        
    #     # Initialize bias (if exists)
    #     if self.bias_flag_overall and self.bias_r is not None:
    #         # Small bias initialization
    #         bound = 1 / math.sqrt(fan_in_component) if fan_in_component > 0 else 0
    #         nn.init.uniform_(self.bias_r, -bound * 0.1, bound * 0.1)

    # def _initialize_weights(self):
    #         """
    #         Initializes weights using a manual Lecun Normal implementation for the real part
    #         and zeros for the imaginary parts, providing a stable starting point for training.
    #         """
    #         # Calculate fan-in, which is essential for the initialization math
    #         kernel_prod = np.prod(self.kernel_size)
    #         fan_in_component = self.in_channels_per_comp_grp * kernel_prod

    #         if fan_in_component > 0:
    #             # Manually implement Lecun Normal initialization for backward compatibility.
    #             # Lecun Normal uses a standard deviation of sqrt(1 / fan_in).
    #             std_dev = math.sqrt(1.0 / fan_in_component)
    #             nn.init.normal_(self.weight_r, mean=0.0, std=std_dev)

    #         # Initialize imaginary components to zero for a stable start.
    #         with torch.no_grad():
    #             self.weight_i.zero_()
    #             self.weight_j.zero_()
    #             self.weight_k.zero_()

    #         # Initialize the real bias to zero.
    #         if self.bias_flag_overall and self.bias_r is not None:
    #             nn.init.zeros_(self.bias_r)

    def _rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2])
        mean_brightness = rgb_input.mean(dim=1)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min()))
        
        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0])
            return torch.stack([real, x[:, 0], x[:, 1], x[:, 2]], dim=-1)
        
        def poincare_mapping(x):
            # norm = torch.norm(x, dim=1)
            # x_normalized = x / (norm.unsqueeze(1) + 1)
            # real = torch.sqrt(1 - torch.sum(x_normalized**2, dim=1))
            # return torch.stack([real, x_normalized[:, 0], x_normalized[:, 1], x_normalized[:, 2]], dim=-1)
            norm_sq = torch.sum(x**2, dim=1, keepdim=True)
            denominator = 1 + norm_sq
            real = (1 - norm_sq.squeeze(1)) / denominator.squeeze(1)
            vector_components = 2 * x / denominator
            return torch.stack([real, vector_components[:, 0], vector_components[:, 1], vector_components[:, 2]], dim=-1)
        mappings = {
            'luminance': torch.stack([luminance, rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]], dim=-1),
            'mean_brightness': torch.stack([mean_brightness, rgb_input[:, 0], rgb_input[:, 1], rgb_input[:, 2]], dim=-1),
            'raw_normalized': torch.stack([rgb_normalized.mean(dim=1), 
                                        rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]], dim=-1),
            'hamilton': hamilton_mapping(rgb_input),
            'poincare': poincare_mapping(rgb_input)
        }
        
        # Output shape: [B, 1, H, W, 4]
        return mappings[self.mapping_type].unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of quaternion convolution
        
        Args:
            x: Input tensor
            - [B, C, H, W] for RGB input (first layer)
            - [B, C*4, H, W] for standard input (will be reshaped to BCHWQ)
            - [B, C, H, W, 4] for quaternion input (BCHWQ format)
        Returns:
            torch.Tensor: Output in BCHWQ format [B, C_out, H_out, W_out, 4]
        """    
        original_shape = None
        if self.is_first_layer:
            B, C_rgb, H, W = x.shape
            assert C_rgb == 3, f"Expected 3 RGB channels, got {C_rgb}"
            x = self._rgb_to_quaternion(x) # Output shape [B, 1, H, W, 4]
        elif x.dim() == 4: # Standard tensor B, C, H, W -> B, C/4, 4, H, W
            B, C, H, W = x.shape
            original_shape = x.shape
            assert C == self.in_channels_total, f"Expected {self.in_channels_total} channels, got {C}"
            assert C % 4 == 0, f"Input channels {C} must be multiple of 4"
            # Reshape to BCHWQ: [B, C//4, H, W, 4]
            x = x.view(B, C // 4, 4, H, W).permute(0, 1, 3, 4, 2)
        elif x.dim() == 5: # Already in quaternion format
            assert x.size(1) == self.in_channels_per_comp, f"Input C_per_q mismatch {x.size(1)} vs {self.in_channels_per_comp}"
            assert x.size(4) == 4, "Input quaternion dim must be 4"
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # Ensure tensor is contiguous
        x = x.contiguous()
        
        input_dtype = x.dtype
        weight_r = self.weight_r.to(dtype=input_dtype).contiguous()
        weight_i = self.weight_i.to(dtype=input_dtype).contiguous()
        weight_j = self.weight_j.to(dtype=input_dtype).contiguous()
        weight_k = self.weight_k.to(dtype=input_dtype).contiguous()
        bias_r = self.bias_r.to(dtype=input_dtype).contiguous() if self.bias_r is not None else None
        bias_i = None
        bias_j = None
        bias_k = None

        if CUDA_EXT and x.is_cuda:
            # Use the autograd function wrapper with CUDA backward
            output = qconv2d_function(
                x,
                weight_r, weight_i, weight_j, weight_k,
                bias_r,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                original_shape

            )
            return output
        else:
            # PyTorch fallback for BCHWQ layout
            # Input x shape: [B, C, H, W, 4]
            xr, xi, xj, xk = torch.split(x, 1, dim=4)
            xr = xr.squeeze(4)
            xi = xi.squeeze(4)
            xj = xj.squeeze(4)
            xk = xk.squeeze(4)

            conv_params = dict(stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

            # Separable conv
            r_conv = F.conv2d(xr, self.weight_r, self.bias_r, **conv_params)
            i_conv = F.conv2d(xi, self.weight_i, self.bias_i, **conv_params)
            j_conv = F.conv2d(xj, self.weight_j, self.bias_j, **conv_params)
            k_conv = F.conv2d(xk, self.weight_k, self.bias_k, **conv_params)

            out_r = r_conv - i_conv - j_conv - k_conv
            out_i = -r_conv + i_conv + j_conv - k_conv
            out_j = -r_conv - i_conv + j_conv + k_conv
            out_k = -r_conv + i_conv - j_conv + k_conv

            # Stack back into [B, C_out, H_out, W_out, 4]
            return torch.stack([out_r, out_i, out_j, out_k], dim=4)

class IQBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        actual_features = num_features // 4
        assert num_features % 4 == 0, "num_features must be a multiple of 4 for IQBN"
        self.num_features = actual_features # Store the C part

        self.eps = eps
        self.momentum = momentum


        self.gamma = nn.Parameter(torch.ones(self.num_features, 4))
        self.beta = nn.Parameter(torch.zeros(self.num_features, 4))

        # Running stats with shapes [C, Q]
        self.register_buffer('running_mean', torch.zeros(self.num_features, 4))
        self.register_buffer('running_var', torch.ones(self.num_features, 4))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BCHWQ layout.
        
        Args:
            x: Input tensor of shape [B, C, H, W, 4]
        Returns:
            Normalized tensor of shape [B, C, H, W, 4]
        """
        assert x.dim() == 5 and x.size(4) == 4, "Input must be [B, C, H, W, 4]"
        B, C, H, W, Q = x.shape

        input_dtype = x.dtype
        gamma = self.gamma.to(dtype=input_dtype)
        beta = self.beta.to(dtype=input_dtype)
        running_mean = self.running_mean.to(dtype=input_dtype)
        running_var = self.running_var.to(dtype=input_dtype)
        if not self.training or not CUDA_EXT:
            if CUDA_EXT and not self.training and x.is_cuda:
                 x = x.contiguous()
                 gamma = gamma.contiguous()
                 beta = beta.contiguous()
                 running_mean = running_mean.contiguous()
                 running_var = running_var.contiguous()
                 return quaternion_ops.iqbn_forward(x, gamma, beta, running_mean, running_var, self.eps)
            else:
                # Pytorch fallback
                mean = running_mean.view(1, C, 1, 1, 4)
                var = running_var.view(1, C, 1, 1, 4)
                gamma_view = gamma.view(1, C, 1, 1, 4)
                beta_view = beta.view(1, C, 1, 1, 4)
                x_norm = (x - mean) / torch.sqrt(var + self.eps)
                return x_norm * gamma_view + beta_view
        else:
            # Training mode - calculate batch statistics
            # For BCHWQ: average over B, H, W dimensions, keep C and Q separate
            mean_batch = x.mean(dim=[0, 2, 3])  # [C, 4]
            var_batch = x.var(dim=[0, 2, 3], unbiased=False) + 1e-8  # [C, 4]

            # Update running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_batch
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
                self.num_batches_tracked += 1

            # Normalize w/batch statistics
            x_norm = (x - mean_batch.view(1, C, 1, 1, 4)) / torch.sqrt(var_batch.view(1, C, 1, 1, 4) + self.eps)

            # Apply affine parameters
            gamma_view = gamma.view(1, C, 1, 1, 4)
            beta_view = beta.view(1, C, 1, 1, 4)
            return x_norm * gamma_view + beta_view

# class IQLN(nn.Module):
#     def __init__(self, num_features, eps=1e-5, elementwise_affine=True):
#         super().__init__()
#         self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        
#     def forward(self, x):
#         # x shape: [B, C, H, W, 4]
#         B, C, H, W, Q = x.shape
        
#         x_reshaped = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C * Q)
#         x_ln = self.ln(x_reshaped)
        
#         # Reshape back to [B, C, H, W, 4]
#         return x_ln.view(B, H, W, C, Q).permute(0, 3, 1, 2, 4)
    
class IQLN(nn.Module):
    """Layer normalization for quaternion features"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Separate parameters for each quaternion component
        self.weight = nn.Parameter(torch.ones(normalized_shape, 4))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, 4))
        
    def forward(self, x):
        # x shape: [B, C, H, W, 4]
        # Normalize over C, H, W dimensions (keep B and quaternion separate)
        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        var = x.var(dim=[1, 2, 3], keepdim=True, unbiased=False)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transform
        weight = self.weight.view(1, -1, 1, 1, 4)
        bias = self.bias.view(1, -1, 1, 1, 4)
        
        return x_norm * weight + bias
# Deprecated class
class QConv(nn.Module):
    """
    Base Quaternion Convolution class.
    """
    def __init__(self, 
                 rank: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[str, int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 dtype=None,
                 mapping_type: str = 'poincare') -> None:
        super(QConv, self).__init__()
        

        assert rank in [1, 2, 3], "rank must be 1, 2, or 3"
        
        valid_mappings = ['luminance', 'mean_brightness', 'raw_normalized', 'hamilton', 'poincare']
        assert mapping_type in valid_mappings, f"Invalid mapping type. Choose from {valid_mappings}"
        
        self.mapping_type = mapping_type

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
        self.mapping_type = mapping_type
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        self.is_first_layer = (in_channels == 3) 
        if self.is_first_layer:
            # RGB input maps to 4 channels (1 quat channel)
            actual_in_channels = 1  
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        
        out_channels_quat = out_channels // 4
        
        self.conv_r = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode)
        
        self.conv_i = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode)
        
        self.conv_j = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode)
        
        self.conv_k = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, False, 
                          padding_mode)

        self._initialize_weights()




    def _initialize_weights(self):
        
        kernel_prod = np.prod(self.kernel_size)
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Scale factors 
        scale_factors = {
            'luminance': [1.0, 1.0, 1.0, 1.0], 
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],  
            'raw_normalized': [1.0, 1.0, 1.0, 1.0],  
            'poincare': [1.0, 1.0, 1.0, 1.0]  

        }
        scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
        # All convolution layers
        convs = [self.conv_r, self.conv_i, self.conv_j, self.conv_k]
        
        for i, conv in enumerate(convs):
            # Weight initialization with scaled Kaiming
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scales[i])
            
            # Bias initialization 
            if conv.bias is not None:
                bound = 1 / math.sqrt(fan_in) * scales[i] 
                nn.init.uniform_(conv.bias, -bound, bound)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle RGB input
        if x.size(1) == 3:
            x = self.rgb_to_quaternion(x)
            
        if self.is_first_layer:
            B, Q, H, W = x.shape
            # Stack components for single batch processing
            x_stacked = x.reshape(B*Q, 1, H, W)
            r_conv = self.conv_r(x_stacked.view(B, Q, H, W)[:, 0:1])
            i_conv = self.conv_i(x_stacked.view(B, Q, H, W)[:, 1:2])
            j_conv = self.conv_j(x_stacked.view(B, Q, H, W)[:, 2:3])
            k_conv = self.conv_k(x_stacked.view(B, Q, H, W)[:, 3:4])
        else:
            # Channel-wise
            x_r = x[:, :, 0, :, :]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]
            
            r_conv = self.conv_r(x_r)
            i_conv = self.conv_i(x_i)
            j_conv = self.conv_j(x_j)
            k_conv = self.conv_k(x_k)
        
        # In-place operations
        out_r = r_conv
        out_r.add_(i_conv).add_(j_conv).add_(k_conv)
        
        out_i = - r_conv.clone()
        out_i.add_(i_conv).add_(j_conv).sub_(k_conv)
        
        out_j = - r_conv.clone()
        out_j.sub_(i_conv).add_(j_conv).add_(k_conv)
        
        out_k = - r_conv.clone()
        out_k.add_(i_conv).sub_(j_conv).add_(k_conv)
        
        # Stack outputs
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        # Clean up intermediate tensors
        del r_conv, i_conv, j_conv, k_conv
        
        return out
        

    def rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2]).unsqueeze(1).to(rgb_input.device)
        mean_brightness = rgb_input.mean(dim=1, keepdim=True).to(rgb_input.device)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())).to(rgb_input.device)
        
        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0:1])
            return torch.cat([real, x[:, 0:1], x[:, 1:2], x[:, 2:3]], dim=1)
        
        def poincare_mapping(x):
            norm = torch.norm(x, dim=1, keepdim=True)
            x_normalized = (x / (norm + 1)).to(x.device)
            return torch.cat([torch.sqrt(1 - torch.sum(x_normalized**2, dim=1, keepdim=True)), 
                            x_normalized[:, 0:1], x_normalized[:, 1:2], x_normalized[:, 2:3]], dim=1)
        
        mappings = {
            'luminance': torch.cat([luminance, rgb_normalized[:, 0:1], rgb_normalized[:, 1:2], rgb_normalized[:, 2:3]], dim=1),
            'mean_brightness': torch.cat([mean_brightness, rgb_input[:, 0:1], rgb_input[:, 1:2], rgb_input[:, 2:3]], dim=1),
            'raw_normalized': torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
                                        rgb_normalized[:, 0:1], rgb_normalized[:, 1:2], rgb_normalized[:, 2:3]], dim=1),
            'hamilton': hamilton_mapping(rgb_input),
            'poincare': poincare_mapping(rgb_input)
        }
        return mappings[self.mapping_type]


# # Conv using quaternion conv
class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        padding = autopad(k, p, d)

        self.conv = QConv2D(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
        self.bn = IQBN(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # self.conv.stride = s if isinstance(s, tuple) else (s, s)
        # self.conv.padding = padding if isinstance(padding, tuple) else (padding, padding)

        # self.conv.dilation = d if isinstance(d, tuple) else (d, d)
        # self.conv.groups = g

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # print(f"X type: {x.dtype}")

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


# # Conv using real-valued
# class Conv(nn.Module):
#     """
#     Standard convolution module with batch normalization and activation.

#     Attributes:
#         conv (nn.Conv2d): Convolutional layer.
#         bn (nn.BatchNorm2d): Batch normalization layer.
#         act (nn.Module): Activation function layer.
#         default_act (nn.Module): Default activation function (SiLU).
#     """

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """
#         Initialize Conv layer with given parameters.

#         Args:
#             c1 (int): Number of input channels.
#             c2 (int): Number of output channels.
#             k (int): Kernel size.
#             s (int): Stride.
#             p (int, optional): Padding.
#             g (int): Groups.
#             d (int): Dilation.
#             act (bool | nn.Module): Activation function.
#         """
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """
#         Apply convolution, batch normalization and activation to input tensor.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """
#         Apply convolution and activation without batch normalization.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

# Use base ultralytics DW-Conv
class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1//4, c2//4), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]


# Unused Class
class QConcat(nn.Module):
    def __init__(self, dim=1, reduce=False, target_channels=None):
        super().__init__()
        self.dim = dim
        self.reduce = reduce
        self.target_channels = target_channels
        
        if reduce:
            assert target_channels is not None, "target_channels must be specified when reduce=True"
            assert target_channels % 4 == 0, "target_channels must be multiple of 4"
            # Create single quaternion convolution to reduce channels
            self.reduce_conv = QConv2D(target_channels * 4, target_channels, kernel_size=1)

    def forward(self, x: list) -> torch.Tensor:
        """
        Args:
            x: List of quaternion tensors each of shape [B, C, 4, H, W]
        Returns:
            torch.Tensor: Concatenated tensor [B, C', 4, H, W]
        """
        # Verify all inputs have quaternion structure
        assert all(tensor.size(2) == 4 for tensor in x), "All inputs must have quaternion dimension"
        
        # Concatenate along channel dimension while preserving quaternion structure
        concat = torch.cat(x, dim=1)  # [B, sum(C), 4, H, W]
        
        # Reduce channels if needed
        if self.reduce:
            concat = self.reduce_conv(concat)
            
        return concat

# class QUpsample(nn.Module):
#     def __init__(self, scale_factor=2, mode='nearest'):
#         super().__init__()
#         self.scale_factor = scale_factor
#         self.mode = mode

#     def forward(self, x):
#         B, C, Q, H, W = x.shape
#         # Flatten Q into channel dim
#         x = x.view(B, C * Q, H, W)
#         # Apply upsampling once across all channels
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
#         # Reshape back to quaternion format
#         x = x.view(B, C, Q, x.shape[-2], x.shape[-1])
#         return x

class QUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        B, C, H, W, Q = x.shape
        assert Q == 4, "Expected quaternion format"
        
        # Apply upsampling to each quaternion component
        upsampled_components = []
        for q_idx in range(4):
            component = x[..., q_idx]  # [B, C, H, W]
            upsampled = F.interpolate(
                component, 
                scale_factor=self.scale_factor, 
                mode=self.mode
            )
            upsampled_components.append(upsampled)
        
        return torch.stack(upsampled_components, dim=-1)
