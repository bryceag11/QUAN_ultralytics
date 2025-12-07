# quaternion/conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional
import math
from .qactivation import QPReLU
from torch.jit import script
from torch.utils import cpp_extension

# from .quaternion_autograd_cuda import qconv2d_function

__all__ = ['Conv', 'DWConv', 'QConv', 'QConv1D', 'QConv2D',
           'QConv3D', 'QDense', 'QInit']



# try:
#     import quaternion_ops
#     print('Sucessfully imported compiled quaternion_ops')
#     CUDA_EXT = True
# except:
#     print("Failed to import compiled quaternion_ops. Falling back to Pytorch implementation")
#     CUDA_EXT = False

def autopad(k, p=None, d=1): 
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# class QConv2D(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Union[int, Tuple[int, int]],
#                  stride: Union[int, Tuple[int, int]] = 1,
#                  padding: Union[str, int, Tuple[int, int]] = 0,
#                  dilation: Union[int, Tuple[int, int]] = 1,
#                  groups: int = 1,
#                  bias: bool = True,
#                  padding_mode: str = 'zeros', 
#                  dtype=None,
#                  mapping_type: str='poincare'):
#         super().__init__()

#         # Parameter validation
#         if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
#         if isinstance(stride, int): stride = (stride, stride)
#         if isinstance(dilation, int): dilation = (dilation, dilation)

#         # Handle autopad for padding
#         if padding == 'same':
#             # For same: padding = (kernel_size - 1) // 2 when stride=1
#             if stride == (1, 1):
#                 padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
#             else:
#                 # For stride > 1, use the autopad function
#                 padding = tuple(autopad(k, None, d) for k, d in zip(kernel_size, dilation))
#         elif isinstance(padding, int): 
#             padding = (padding, padding)
#         elif isinstance(padding, (list, tuple)) and len(padding) == 2:
#             padding = tuple(padding)
#         else:
#             raise ValueError(f"Invalid padding: {padding}")

#         # if padding == 'same':
#         #      padding = tuple(autopad(k, None, d) for k, d in zip(kernel_size, dilation))
#         # elif isinstance(padding, int): padding = (padding, padding)

#         self.in_channels_total = in_channels
#         self.out_channels_total = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.mapping_type = mapping_type

#         self.is_first_layer = (in_channels == 3)
#         if self.is_first_layer:
#             self.in_channels_per_comp = 1 
#         else:
#             assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
#             self.in_channels_per_comp = in_channels // 4

#         assert out_channels % 4 == 0, "out_channels must be multiple of 4"
#         self.out_channels_per_comp = out_channels // 4

#         # Input channels per component per group
#         assert self.in_channels_per_comp % groups == 0, "Input channels per component must be divisible by groups"
#         self.in_channels_per_comp_grp = self.in_channels_per_comp // groups
        
#         self.bias_flag_overall = bias
        
#         weight_shape = (
#             self.out_channels_per_comp,
#             self.in_channels_per_comp_grp,
#             *self.kernel_size
#         )
#         self.weight_r = nn.Parameter(torch.zeros(weight_shape))
#         self.weight_i = nn.Parameter(torch.zeros(weight_shape))
#         self.weight_j = nn.Parameter(torch.zeros(weight_shape))
#         self.weight_k = nn.Parameter(torch.zeros(weight_shape))

#         if bias:
#             bias_shape = (self.out_channels_per_comp,)
#             self.bias_r = nn.Parameter(torch.zeros(bias_shape))
#         else:
#             self.register_parameter('bias_r', None)
#         self.register_parameter('bias_i', None)
#         self.register_parameter('bias_j', None)
#         self.register_parameter('bias_k', None)

#         self._initialize_weights() 



#     def _initialize_weights(self):
#         kernel_prod = np.prod(self.kernel_size)
#         # For one of the r, i, j, or k weight matrices
#         fan_in_component = self.in_channels_per_comp_grp * kernel_prod

#         scale_factors_map = {
#             'luminance': [1.0, 1.0, 1.0, 1.0],
#             'mean_brightness': [1.0, 0.75, 0.75, 0.75],
#             'raw_normalized': [1.0, 1.0, 1.0, 1.0],
#             'hamilton': [1.0, 1.0, 1.0, 1.0], 
#             'poincare': [1.0, 1.0, 1.0, 1.0]
#         }
#         # Default scales if self.mapping_type is not in the map
#         scales = scale_factors_map.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])

#         weights_to_init = [self.weight_r, self.weight_i, self.weight_j, self.weight_k]

#         for i, weight_param in enumerate(weights_to_init):
#             # Initialize weights using scaled 'a' for kaiming_uniform_
#             nn.init.kaiming_uniform_(weight_param, a=math.sqrt(5.0) * scales[i])

#         # Only initialize self.bias_r if it was created 
#         if self.bias_flag_overall and self.bias_r is not None:
#             bound_r = (1 / math.sqrt(fan_in_component)) * scales[0] if fan_in_component > 0 else 0
#             nn.init.uniform_(self.bias_r, -bound_r, bound_r)



#     def _rgb_to_quaternion(self, rgb_input):
#         B, C, H, W = rgb_input.shape
#         luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2])
#         mean_brightness = rgb_input.mean(dim=1)
#         rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min()))
        
#         def hamilton_mapping(x):
#             real = torch.zeros_like(x[:, 0])
#             return torch.stack([real, x[:, 0], x[:, 1], x[:, 2]], dim=-1)
        
#         def poincare_mapping(x):
#             # norm = torch.norm(x, dim=1)
#             # x_normalized = x / (norm.unsqueeze(1) + 1)
#             # real = torch.sqrt(1 - torch.sum(x_normalized**2, dim=1))
#             # return torch.stack([real, x_normalized[:, 0], x_normalized[:, 1], x_normalized[:, 2]], dim=-1)
#             norm_sq = torch.sum(x**2, dim=1, keepdim=True)
#             denominator = 1 + norm_sq
#             real = (1 - norm_sq.squeeze(1)) / denominator.squeeze(1)
#             vector_components = 2 * x / denominator
#             return torch.stack([real, vector_components[:, 0], vector_components[:, 1], vector_components[:, 2]], dim=-1)
#         mappings = {
#             'luminance': torch.stack([luminance, rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]], dim=-1),
#             'mean_brightness': torch.stack([mean_brightness, rgb_input[:, 0], rgb_input[:, 1], rgb_input[:, 2]], dim=-1),
#             'raw_normalized': torch.stack([rgb_normalized.mean(dim=1), 
#                                         rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2]], dim=-1),
#             'hamilton': hamilton_mapping(rgb_input),
#             'poincare': poincare_mapping(rgb_input)
#         }
        
#         # Output shape: [B, 1, H, W, 4]
#         return mappings[self.mapping_type].unsqueeze(1)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of quaternion convolution
        
#         Args:
#             x: Input tensor
#             - [B, C, H, W] for RGB input (first layer)
#             - [B, C*4, H, W] for standard input (will be reshaped to BCHWQ)
#             - [B, C, H, W, 4] for quaternion input (BCHWQ format)
#         Returns:
#             torch.Tensor: Output in BCHWQ format [B, C_out, H_out, W_out, 4]
#         """    
#         original_shape = None
#         if self.is_first_layer:
#             B, C_rgb, H, W = x.shape
#             assert C_rgb == 3, f"Expected 3 RGB channels, got {C_rgb}"
#             x = self._rgb_to_quaternion(x) # Output shape [B, 1, H, W, 4]
#         elif x.dim() == 4: # Standard tensor B, C, H, W -> B, C/4, 4, H, W
#             B, C, H, W = x.shape
#             original_shape = x.shape
#             assert C == self.in_channels_total, f"Expected {self.in_channels_total} channels, got {C}"
#             assert C % 4 == 0, f"Input channels {C} must be multiple of 4"
#             # Reshape to BCHWQ: [B, C//4, H, W, 4]
#             x = x.view(B, C // 4, 4, H, W).permute(0, 1, 3, 4, 2)
#         elif x.dim() == 5: # Already in quaternion format
#             assert x.size(1) == self.in_channels_per_comp, f"Input C_per_q mismatch {x.size(1)} vs {self.in_channels_per_comp}"
#             assert x.size(4) == 4, "Input quaternion dim must be 4"
#         else:
#             raise ValueError(f"Unsupported input shape: {x.shape}")

#         # Ensure tensor is contiguous
#         x = x.contiguous()
        
#         input_dtype = x.dtype
#         weight_r = self.weight_r.to(dtype=input_dtype).contiguous()
#         weight_i = self.weight_i.to(dtype=input_dtype).contiguous()
#         weight_j = self.weight_j.to(dtype=input_dtype).contiguous()
#         weight_k = self.weight_k.to(dtype=input_dtype).contiguous()
#         bias_r = self.bias_r.to(dtype=input_dtype).contiguous() if self.bias_r is not None else None
#         bias_i = None
#         bias_j = None
#         bias_k = None

#         if CUDA_EXT and x.is_cuda:
#             # Use the autograd function wrapper with CUDA backward
#             output = qconv2d_function(
#                 x,
#                 weight_r, weight_i, weight_j, weight_k,
#                 bias_r,
#                 self.stride,
#                 self.padding,
#                 self.dilation,
#                 self.groups,
#                 original_shape

#             )
#             return output
#         else:
#             # PyTorch fallback for BCHWQ layout
#             # Input x shape: [B, C, H, W, 4]
#             xr, xi, xj, xk = torch.split(x, 1, dim=4)
#             xr = xr.squeeze(4)
#             xi = xi.squeeze(4)
#             xj = xj.squeeze(4)
#             xk = xk.squeeze(4)

#             conv_params = dict(stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

#             # Separable conv
#             r_conv = F.conv2d(xr, self.weight_r, self.bias_r, **conv_params)
#             i_conv = F.conv2d(xi, self.weight_i, self.bias_i, **conv_params)
#             j_conv = F.conv2d(xj, self.weight_j, self.bias_j, **conv_params)
#             k_conv = F.conv2d(xk, self.weight_k, self.bias_k, **conv_params)

#             out_r = r_conv - i_conv - j_conv - k_conv
#             out_i = -r_conv + i_conv + j_conv - k_conv
#             out_j = -r_conv - i_conv + j_conv + k_conv
#             out_k = -r_conv + i_conv - j_conv + k_conv

#             # Stack back into [B, C_out, H_out, W_out, 4]
#             return torch.stack([out_r, out_i, out_j, out_k], dim=4)

# class IQBN(nn.Module):
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__()
#         actual_features = num_features // 4
#         assert num_features % 4 == 0, "num_features must be a multiple of 4 for IQBN"
#         self.num_features = actual_features # Store the C part

#         self.eps = eps
#         self.momentum = momentum


#         self.gamma = nn.Parameter(torch.ones(self.num_features, 4))
#         self.beta = nn.Parameter(torch.zeros(self.num_features, 4))

#         # Running stats with shapes [C, Q]
#         self.register_buffer('running_mean', torch.zeros(self.num_features, 4))
#         self.register_buffer('running_var', torch.ones(self.num_features, 4))
#         self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for BCHWQ layout.
        
#         Args:
#             x: Input tensor of shape [B, C, H, W, 4]
#         Returns:
#             Normalized tensor of shape [B, C, H, W, 4]
#         """
#         assert x.dim() == 5 and x.size(4) == 4, "Input must be [B, C, H, W, 4]"
#         B, C, H, W, Q = x.shape

#         input_dtype = x.dtype
#         gamma = self.gamma.to(dtype=input_dtype)
#         beta = self.beta.to(dtype=input_dtype)
#         running_mean = self.running_mean.to(dtype=input_dtype)
#         running_var = self.running_var.to(dtype=input_dtype)
#         if not self.training or not CUDA_EXT:
#             if CUDA_EXT and not self.training and x.is_cuda:
#                  x = x.contiguous()
#                  gamma = gamma.contiguous()
#                  beta = beta.contiguous()
#                  running_mean = running_mean.contiguous()
#                  running_var = running_var.contiguous()
#                  return quaternion_ops.iqbn_forward(x, gamma, beta, running_mean, running_var, self.eps)
#             else:
#                 # Pytorch fallback
#                 mean = running_mean.view(1, C, 1, 1, 4)
#                 var = running_var.view(1, C, 1, 1, 4)
#                 gamma_view = gamma.view(1, C, 1, 1, 4)
#                 beta_view = beta.view(1, C, 1, 1, 4)
#                 x_norm = (x - mean) / torch.sqrt(var + self.eps)
#                 return x_norm * gamma_view + beta_view
#         else:
#             # Training mode - calculate batch statistics
#             # For BCHWQ: average over B, H, W dimensions, keep C and Q separate
#             mean_batch = x.mean(dim=[0, 2, 3])  # [C, 4]
#             var_batch = x.var(dim=[0, 2, 3], unbiased=False) + 1e-8  # [C, 4]

#             # Update running stats
#             with torch.no_grad():
#                 self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_batch
#                 self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_batch
#                 self.num_batches_tracked += 1

#             # Normalize w/batch statistics
#             x_norm = (x - mean_batch.view(1, C, 1, 1, 4)) / torch.sqrt(var_batch.view(1, C, 1, 1, 4) + self.eps)

#             # Apply affine parameters
#             gamma_view = gamma.view(1, C, 1, 1, 4)
#             beta_view = beta.view(1, C, 1, 1, 4)
#             return x_norm * gamma_view + beta_view

class IQBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features // 4
        self.eps = eps
        self.momentum = momentum
        
        # Parameters for correct broadcasting
        self.gamma = nn.Parameter(torch.ones(self.num_features, 4))
        self.beta = nn.Parameter(torch.zeros(self.num_features, 4))
        
        # Running stats with correct shapes
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
        if not self.training:
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

        out_r = r_conv + i_conv + j_conv + k_conv
        out_i = r_conv - i_conv - j_conv + k_conv
        out_j = r_conv + i_conv - j_conv - k_conv
        out_k = r_conv - i_conv + j_conv - k_conv

        # Stack back into [B, C_out, H_out, W_out, 4]
        return torch.stack([out_r, out_i, out_j, out_k], dim=4)


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
        # Special handling for first layer

        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * rank
        
        self.mapping_type = mapping_type
        # Define the underlying real-valued convolution for each quaternion component
        if rank == 1:
            Conv = nn.Conv1d
        elif rank == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            
        self.is_first_layer = (in_channels == 3)  # Changed from 4 to 3
        if self.is_first_layer:
            # For RGB input, map to 4 channels
            actual_in_channels = 1  # Use this for the convolution
        else:
            assert in_channels % 4 == 0, "in_channels must be multiple of 4 for non-first layers"
            actual_in_channels = in_channels // 4
        assert out_channels % 4 == 0, "out_channels must be multiple of 4"
        
        # For first layer, use in_channels=1, for others use in_channels//4
        out_channels_quat = out_channels // 4
        
        self.conv_r = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode)
        
        self.conv_i = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode)
        
        self.conv_j = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode)
        
        self.conv_k = Conv(actual_in_channels, out_channels_quat, kernel_size,
                          stride, padding, dilation, groups, bias, 
                          padding_mode)
                      
        self._initialize_weights()

    # Bias for all layers weight init
    def _initialize_weights(self):
        
        kernel_prod = np.prod(self.kernel_size)
        fan_in = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Scale factors for quaternion components
        scale_factors = {
            'luminance': [1.0, 1.0, 1.0, 1.0],      # Emphasize real component
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],  # Slightly more balanced
            'raw_normalized': [1.0, 1.0, 1.0, 1.0],  # Equal emphasis
            'poincare': [1.0, 1.0, 1.0, 1.0]  # Equal emphasis

        }
        scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
        # All convolution layers
        convs = [self.conv_r, self.conv_i, self.conv_j, self.conv_k]
        
        for i, conv in enumerate(convs):
            # Weight initialization with scaled Kaiming
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5) * scales[i])
            
            # Bias initialization (if present)
            if conv.bias is not None:
                bound = 1 / math.sqrt(fan_in) * scales[i]  # Scale bias bound by component weight
                nn.init.uniform_(conv.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle RGB input
        if x.size(1) == 3:  # RGB input
            x = self.rgb_to_quaternion(x)
            
        if self.is_first_layer:
            # Process first layer more efficiently
            B, Q, H, W = x.shape
            # Stack components for single batch processing
            x_stacked = x.reshape(B*Q, 1, H, W)
            r_conv = self.conv_r(x_stacked.view(B, Q, H, W)[:, 0:1])
            i_conv = self.conv_i(x_stacked.view(B, Q, H, W)[:, 1:2])
            j_conv = self.conv_j(x_stacked.view(B, Q, H, W)[:, 2:3])
            k_conv = self.conv_k(x_stacked.view(B, Q, H, W)[:, 3:4])
        else:
            # For subsequent layers, use channel-wise processing
            x_r = x[:, :, 0, :, :]
            x_i = x[:, :, 1, :, :]
            x_j = x[:, :, 2, :, :]
            x_k = x[:, :, 3, :, :]
            
            # Process in parallel if possible
            r_conv = self.conv_r(x_r)
            i_conv = self.conv_i(x_i)
            j_conv = self.conv_j(x_j)
            k_conv = self.conv_k(x_k)
        
        # Use in-place operations and fuse calculations where possible
        out_r = r_conv
        out_r.add_(i_conv).add_(j_conv).add_(k_conv)
        
        out_i = r_conv.clone()
        out_i.sub_(i_conv).sub_(j_conv).add_(k_conv)
        
        out_j = r_conv.clone()
        out_j.add_(i_conv).sub_(j_conv).sub_(k_conv)
        
        out_k = r_conv.clone()
        out_k.sub_(i_conv).add_(j_conv).sub_(k_conv)
        
        # Stack outputs efficiently
        out = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        
        # Clean up intermediate tensors to save memory
        del r_conv, i_conv, j_conv, k_conv
        
        return out

    def rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2]).unsqueeze(1).to(rgb_input.device)
        mean_brightness = rgb_input.mean(dim=1, keepdim=True).to(rgb_input.device)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())).to(rgb_input.device)
        
        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0:1]).to(x.device)
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

class QConv1D(QConv):
    """1D Quaternion Convolution layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[str, int, Tuple[int]] = 0,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv1D, self).__init__(
            rank=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

# class QConv2D(QConv):
#     """2D Quaternion Convolution layer."""
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: Union[int, Tuple[int, int]],
#                  stride: Union[int, Tuple[int, int]] = 1,
#                  padding: Union[str, int, Tuple[int, int]] = 0,
#                  dilation: Union[int, Tuple[int, int]] = 1,
#                  groups: int = 1,
#                  bias: bool = True,
#                  padding_mode: str = 'zeros',
#                  dtype=None,
#                  mapping_type: str='poincare') -> None:
#         super().__init__(
#             rank=2,  # Fixed for 2D convolution
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#             dtype=dtype,
#             mapping_type=mapping_type
#         )

class QConv3D(QConv):
    """3D Quaternion Convolution layer."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[str, int, Tuple[int, int, int]] = 0,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        super(QConv3D, self).__init__(
            rank=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )


class QDense(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 mapping_type: str = 'poincare',
                 device=None,
                 dtype=None):
        super(QDense, self).__init__()

                # Add mapping strategy
        self.mapping_type = mapping_type
        
        # Ensure input features are handled correctly
        if in_features == 3:  # If input is RGB
            # Adjust input features after mapping
            in_features = 4
        else:
            # Ensure input features are a multiple of 4
            assert in_features % 4 == 0, "in_features must be a multiple of 4"
        
        assert out_features % 4 == 0, "out_features must be a multiple of 4"
        # Compute feature dimensions
        in_features_quat = in_features // 4
        out_features_quat = out_features // 4
        
        # Create separate linear layers for each quaternion component
        self.linear_rr = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        self.linear_ri = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        self.linear_rj = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        self.linear_rk = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2]).unsqueeze(1).to(rgb_input.device)
        mean_brightness = rgb_input.mean(dim=1, keepdim=True).to(rgb_input.device)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())).to(rgb_input.device)
        
        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0:1]).to(x.device)
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
    
    def _initialize_weights(self):

            
            scale_factors = {
                'luminance': [1.0, 1.0, 1.0, 1.0],
                'mean_brightness': [1.0, 0.75, 0.75, 0.75],
                'raw_normalized': [1.0, 0.5, 0.5, 0.5],
                'poincare': [1.0, 1.0, 1.0, 1.0]
            }
            scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
            
            linears = [self.linear_rr, self.linear_ri, self.linear_rj, self.linear_rk]
            for i, linear in enumerate(linears):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
                nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5) * scales[i])
                if linear.bias is not None:
                    bound = 1 / math.sqrt(fan_in) * scales[i]
                    nn.init.uniform_(linear.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.size(1) == 3:
        #     x = self.rgb_to_quaternion(x)
        # Separate input into quaternion components
        x_r = x[:, :x.size(1)//4]
        x_i = x[:, x.size(1)//4:x.size(1)//2]
        x_j = x[:, x.size(1)//2:3*x.size(1)//4]
        x_k = x[:, 3*x.size(1)//4:]
        
        # Apply linear transformations with Hamilton product rules
        r_r = self.linear_rr(x_r)
        r_i = self.linear_ri(x_r)
        r_j = self.linear_rj(x_r)
        r_k = self.linear_rk(x_r)
        
        i_r = self.linear_rr(x_i)
        i_i = self.linear_ri(x_i)
        i_j = self.linear_rj(x_i)
        i_k = self.linear_rk(x_i)
        
        j_r = self.linear_rr(x_j)
        j_i = self.linear_ri(x_j)
        j_j = self.linear_rj(x_j)
        j_k = self.linear_rk(x_j)
        
        k_r = self.linear_rr(x_k)
        k_i = self.linear_ri(x_k)
        k_j = self.linear_rj(x_k)
        k_k = self.linear_rk(x_k)
        
        # Hamilton product rules for output
        out_r = r_r - i_i - j_j - k_k
        out_i = r_i + i_r + j_k - k_j
        out_j = r_j - i_k + j_r + k_i
        out_k = r_k + i_j - j_i + k_r
        
        # Stack back into quaternion format
        out = torch.stack([out_r, out_i, out_j, out_k], dim=1)
        out = out.view(x.size(0), -1)
        
        return out
    