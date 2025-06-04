# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

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



    def _rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        device = rgb_input.device
        dtype = rgb_input.dtype  
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2]).unsqueeze(1).to(rgb_input.device, dtype=dtype)
        mean_brightness = rgb_input.mean(dim=1, keepdim=True).to(rgb_input.device, dtype=dtype)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())).to(rgb_input.device, dtype=dtype)

        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0:1])
            return torch.cat([real, x[:, 0:1], x[:, 1:2], x[:, 2:3]], dim=1)

        def poincare_mapping(x):
            norm = torch.norm(x, dim=1, keepdim=True)
            x_normalized = (x / (norm + 1)).to(x.device, dtype=dtype)
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
        # Map RGB to B, 1, 4, H, W for the first layer input
        # The CUDA kernel expects input [B, C_in_per_q, Q=4, H, W]
        mapped = mappings[self.mapping_type] # Shape [B, 4, H, W]
        # Add C_in_per_q=1 dimension
        mapped_final = mapped.unsqueeze(1) # Shape [B, 1, 4, H, W]
        return mapped_final.to(dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = None
        if self.is_first_layer:
            x = self._rgb_to_quaternion(x) # Output shape [B, 1, 4, H, W]
        elif x.dim() == 4: # Standard tensor B, C, H, W -> B, C/4, 4, H, W
            B, C, H, W = x.shape
            original_shape = x.shape
            if C != self.in_channels_total:
                print(f"Warning: Input channel mismatch! Expected {self.in_channels_total}, got {C}")
            assert C % 4 == 0, f"Input channels {C} must be multiple of 4 for non-first layer standard input"
            x = x.view(B, C // 4, 4, H, W)
        elif x.dim() == 5: # Already in quaternion format
            assert x.size(1) == self.in_channels_per_comp, f"Input C_per_q mismatch {x.size(1)} vs {self.in_channels_per_comp}"
            assert x.size(2) == 4, "Input quaternion dim must be 4"
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



        # if CUDA_EXT and x.is_cuda:
        #      # Call the updated function with 4 weights and 4 biases
        #      output = quaternion_ops.qconv_forward(
        #          x,
        #          weight_r, weight_i, weight_j, weight_k,
        #          bias_r, bias_i, bias_j, bias_k,
        #          list(self.stride),
        #          list(self.padding),
        #          list(self.dilation),
        #          self.groups
        #      )
        #      return output
        # else:
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
            # Pytorch fallback
            xr, xi, xj, xk = torch.split(x, 1, dim=2) 
            xr = xr.squeeze(2) 
            xi = xi.squeeze(2)
            xj = xj.squeeze(2)
            xk = xk.squeeze(2)

            conv_params = dict(stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

            r_conv = F.conv2d(xr, self.weight_r, self.bias_r, **conv_params)
            i_conv = F.conv2d(xi, self.weight_i, self.bias_i, **conv_params)
            j_conv = F.conv2d(xj, self.weight_j, self.bias_j, **conv_params)
            k_conv = F.conv2d(xk, self.weight_k, self.bias_k, **conv_params)

            out_r = r_conv - i_conv - j_conv - k_conv
            out_i = -r_conv + i_conv + j_conv - k_conv
            out_j = -r_conv - i_conv + j_conv + k_conv
            out_k = -r_conv + i_conv - j_conv + k_conv

            # Stack back into [B, C_out_per_q, Q=4, H_out, W_out]
            return torch.stack([out_r, out_i, out_j, out_k], dim=2)


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
        # x shape: [B, C, Q, H, W]
        B, C, Q, H, W = x.shape
        assert C == self.num_features, f"Input channels {C} != registered features {self.num_features}"
        assert Q == 4, "Input quaternion dimension must be 4"


        input_dtype = x.dtype
        gamma = self.gamma.to(dtype=input_dtype)
        beta = self.beta.to(dtype=input_dtype)
        running_mean = self.running_mean.to(dtype=input_dtype)
        running_var = self.running_var.to(dtype=input_dtype)
        if not self.training or not CUDA_EXT:
            # Use CUDA extension for inference OR if fallback 
            if CUDA_EXT and not self.training and x.is_cuda:
                 x = x.contiguous()
                 gamma = self.gamma.contiguous()
                 beta = self.beta.contiguous()
                 running_mean = self.running_mean.contiguous()
                 running_var = self.running_var.contiguous()
                 return quaternion_ops.iqbn_forward(x, gamma, beta, running_mean, running_var, self.eps)
            else:
                # Pytorch fallback
                mean = self.running_mean.view(1, self.num_features, 4, 1, 1)
                var = self.running_var.view(1, self.num_features, 4, 1, 1)
                gamma_view = self.gamma.view(1, self.num_features, 4, 1, 1)
                beta_view = self.beta.view(1, self.num_features, 4, 1, 1)
                x_norm = (x - mean) / torch.sqrt(var + self.eps)
                return x_norm * gamma_view + beta_view
        else:
            # Training path

            x_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B * Q, C, H * W) 
            mean = x_flat.mean(dim=[0, 2]) 

            # Calculate batch stats per (C, Q)
            mean_batch = x.mean(dim=[0, 3, 4]) # [C, Q]
            var_batch = x.var(dim=[0, 3, 4], unbiased=False) + 1e-8 # [C, Q]

            # Update running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_batch
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
                self.num_batches_tracked += 1

            # Normalize w/batch statistics
            x_norm = (x - mean_batch.view(1, C, Q, 1, 1)) / torch.sqrt(var_batch.view(1, C, Q, 1, 1) + self.eps)

            # Apply affine parameters
            gamma_view = self.gamma.view(1, C, Q, 1, 1)
            beta_view = self.beta.view(1, C, Q, 1, 1)
            return x_norm * gamma_view + beta_view

class IQLN(nn.Module):
    def __init__(self, num_features, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
        
    def forward(self, x):
        # x shape: [B, C, 4, H, W]
        B, C, Q, H, W = x.shape
        
        x_reshaped = x.permute(0, 3, 4, 1, 2).reshape(B, H*W, C*Q)
        x_ln = self.ln(x_reshaped)
        
        # Reshape back to [B, C, 4, H, W]
        return x_ln.reshape(B, H, W, C, Q).permute(0, 3, 4, 1, 2)

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




# Conv using quaternion conv
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
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        out = self.conv(x)
        out = self.act(out)
        return out

# Conv using real-valued
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

class QUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        B, C, Q, H, W = x.shape
        # Flatten Q into channel dim
        x = x.view(B, C * Q, H, W)
        # Apply upsampling once across all channels
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        # Reshape back to quaternion format
        x = x.view(B, C, Q, x.shape[-2], x.shape[-1])
        return x

