# quaternion/conv.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List, Optional
import math
from .qbatch_norm import QBN, IQBN
from .qactivation import QPReLU
import geoopt
__all__ = ['Conv', 'DWConv', 'QConv', 'QConv1D', 'QConv2D',
           'QConv3D', 'QDense', 'QInit']



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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
                 mapping_type: str = 'poincare',
                 manifold_name: Optional[str] = None, 
                 manifold_k: float = -1.0 ) -> None:
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
        

        self.manifold = None
        if manifold_name:
            if manifold_name.lower() == 'poincare':
                self.manifold = geoopt.manifolds.PoincareBall(c=-manifold_k)
            elif manifold_name.lower() == 'lorentz':
                self.manifold = geoopt.manifolds.Lorentz(k=manifold_k)
            else:
                print(f"Warning: Manifold '{manifold_name}' not recognized for QConv. Weights will be Euclidean.")


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
                      
        self._component_convs = nn.ModuleDict({
            'r': self.conv_r, 'i': self.conv_i, 'j': self.conv_j, 'k': self.conv_k
        })

        if self.manifold:
            for conv_comp in self._component_convs.values():
                original_weight_data = conv_comp.weight.data.clone()
                conv_comp.weight = geoopt.ManifoldParameter(original_weight_data, manifold=self.manifold)
        
        self._initialize_weights()

    # Bias for all layers weight init
    def _initialize_weights(self):
        
        kernel_prod = np.prod(self.kernel_size)
        fan_in_factor = (self.in_channels // 4 if not self.is_first_layer else 1) * kernel_prod
        
        # Scale factors for quaternion components
        scale_factors = {
            'luminance': [1.0, 1.0, 1.0, 1.0],      # Emphasize real component
            'mean_brightness': [1.0, 0.75, 0.75, 0.75],  # Slightly more balanced
            'raw_normalized': [1.0, 1.0, 1.0, 1.0],  # Equal emphasis
            'poincare': [1.2, 0.8, 0.8, 0.8]  # Equal emphasis

        }
        scales = scale_factors.get(self.mapping_type, [0.5, 0.5, 0.5, 0.5])
        
        for i, conv_comp in enumerate(self._component_convs.values()):
            current_scale = scales[i % 4] # Cycle through scales for r, i, j, k components
            
            # Initialize data for the parameter
            # For ManifoldParameter, this initializes the .data attribute
            temp_init_tensor = torch.empty_like(conv_comp.weight.data)
            nn.init.kaiming_uniform_(temp_init_tensor, a=math.sqrt(5) * current_scale)

            if isinstance(conv_comp.weight, geoopt.ManifoldParameter):
                # Project the initialized data onto the manifold
                # Reshape for projection if manifold expects [..., D] and conv_comp.weight is higher dim
                original_shape = temp_init_tensor.shape
                if len(original_shape) > self.manifold.ndim and self.manifold.ndim > 0 : # Check if reshaping is needed
                     # e.g. conv weights [out_ch, in_ch/groups, kH, kW]
                    temp_init_tensor_reshaped = temp_init_tensor.reshape(original_shape[0], -1)
                     # Ensure the last dim matches manifold. Rework if manifold.ndim is not the target feature dim.
                     # This simple reshape might not always be correct depending on manifold expectations.
                     # For PoincareBall, it expects points to be vectors.
                    projected_data = self.manifold.projx(temp_init_tensor_reshaped)
                    conv_comp.weight.data.copy_(projected_data.reshape(original_shape))
                else:
                    conv_comp.weight.data.copy_(self.manifold.projx(temp_init_tensor))
            else: # Standard nn.Parameter
                conv_comp.weight.data.copy_(temp_init_tensor)

            if conv_comp.bias is not None:
                # Bias initialization (assuming bias is Euclidean)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv_comp.weight) # Use fan_in of the weight
                bound = 1 / math.sqrt(fan_in if fan_in > 0 else fan_in_factor) * current_scale
                nn.init.uniform_(conv_comp.bias, -bound, bound)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3: # RGB input
            x = self.rgb_to_quaternion(x) # Output: [B, 4, H, W]

        # Determine input components based on layer type
        if self.is_first_layer:
            # x is [B, 4, H, W]. Each component conv takes [B, 1, H, W]
            xr_in, xi_in, xj_in, xk_in = x.split(1, dim=1) # Splits into four [B, 1, H, W] tensors
        else:
            # x is [B, C_quat_in, 4, H, W] where C_quat_in = in_channels // 4
            # Each component conv takes [B, C_quat_in, H, W]
            xr_in = x[:, :, 0, :, :]
            xi_in = x[:, :, 1, :, :]
            xj_in = x[:, :, 2, :, :]
            xk_in = x[:, :, 3, :, :]

        # Hamilton Product: Y = WX
        # Y_r = W_r X_r - W_i X_i - W_j X_j - W_k X_k
        out_r = self.conv_r(xr_in) - self.conv_i(xi_in) - self.conv_j(xj_in) - self.conv_k(xk_in)
        # Y_i = W_r X_i + W_i X_r + W_j X_k - W_k X_j
        out_i = self.conv_r(xi_in) + self.conv_i(xr_in) + self.conv_j(xk_in) - self.conv_k(xj_in)
        # Y_j = W_r X_j - W_i X_k + W_j X_r + W_k X_i
        out_j = self.conv_r(xj_in) - self.conv_i(xk_in) + self.conv_j(xr_in) + self.conv_k(xi_in)
        # Y_k = W_r X_k + W_i X_j - W_j X_i + K_k X_r
        out_k = self.conv_r(xk_in) + self.conv_i(xj_in) - self.conv_j(xi_in) + self.conv_k(xr_in)

        # Stack to form output: [B, C_quat_out, 4, H, W]
        output = torch.stack([out_r, out_i, out_j, out_k], dim=2)
        return output
        


    def rgb_to_quaternion(self, rgb_input):
        B, C, H, W = rgb_input.shape
        luminance = (0.299 * rgb_input[:, 0] + 0.587 * rgb_input[:, 1] + 0.114 * rgb_input[:, 2]).unsqueeze(1).to(rgb_input.device)
        mean_brightness = rgb_input.mean(dim=1, keepdim=True).to(rgb_input.device)
        rgb_normalized = ((rgb_input - rgb_input.min()) / (rgb_input.max() - rgb_input.min())).to(rgb_input.device)
        
        def hamilton_mapping(x):
            real = torch.zeros_like(x[:, 0:1]).to(x.device)
            return torch.cat([real, x[:, 0:1], x[:, 1:2], x[:, 2:3]], dim=1)
        
        def poincare_mapping(x):
            norm = torch.norm(rgb_input, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            x_normalized = (rgb_input / (norm + 1e-8)) 
            s_squared = torch.sum(x_normalized**2, dim=1, keepdim=True)
            s_squared_clamped = torch.clamp(s_squared, max=0.99999)
            first_component = torch.sqrt(1 - s_squared_clamped)

            return torch.cat([first_component,
                              x_normalized[:, 0:1, :, :],
                              x_normalized[:, 1:2, :, :],
                              x_normalized[:, 2:3, :, :]], dim=1)
        
        mappings = {
            'luminance': torch.cat([luminance, rgb_normalized[:, 0:1], rgb_normalized[:, 1:2], rgb_normalized[:, 2:3]], dim=1),
            'mean_brightness': torch.cat([mean_brightness, rgb_input[:, 0:1], rgb_input[:, 1:2], rgb_input[:, 2:3]], dim=1),
            'raw_normalized': torch.cat([rgb_normalized.mean(dim=1, keepdim=True), 
                                        rgb_normalized[:, 0:1], rgb_normalized[:, 1:2], rgb_normalized[:, 2:3]], dim=1),
            'hamilton': hamilton_mapping(rgb_input),
            'poincare': poincare_mapping(rgb_input)
        }
        return mappings[self.mapping_type]
    
class QConv2D(QConv):
    """2D Quaternion Convolution layer."""
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
                 mapping_type: str='poincare',
                 manifold_name: Optional[str] = None, 
                 manifold_k: float = -1.0 ) -> None:
        super().__init__(
            rank=2,  # Fixed for 2D convolution
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            dtype=dtype,
            mapping_type=mapping_type,
            manifold_name=manifold_name, 
            manifold_k=manifold_k 
        )


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
                 manifold_name: Optional[str] = None,
                 manifold_k: float = -1.0,
                 device=None,
                 dtype=None):
        super(QDense, self).__init__()

        
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
        

        self.manifold = None
        if manifold_name:
            if manifold_name.lower() == 'poincare':
                self.manifold = geoopt.manifolds.PoincareBall(c=-manifold_k)
            elif manifold_name.lower() == 'lorentz':
                self.manifold = geoopt.manifolds.Lorentz(k=manifold_k)
            else:
                print(f"Warning: Manifold '{manifold_name}' not recognized for QDense.")

        # Create separate linear layers for each quaternion component
        self.linear_r = nn.Linear(in_features_quat, out_features_quat, bias=bias)
        self.linear_i = nn.Linear(in_features_quat, out_features_quat, bias=False)
        self.linear_j = nn.Linear(in_features_quat, out_features_quat, bias=False)
        self.linear_k = nn.Linear(in_features_quat, out_features_quat, bias=False)

        self._component_linears = nn.ModuleDict({
            'r': self.linear_r, 'i': self.linear_i, 'j': self.linear_j, 'k': self.linear_k
        })

        if self.manifold:
            for linear_comp in self._component_linears.values():
                original_weight_data = linear_comp.weight.data.clone()
                linear_comp.weight = geoopt.ManifoldParameter(original_weight_data, manifold=self.manifold)

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        scales = [1.2, 0.8, 0.8, 0.8]

        for i, linear_comp in enumerate(self._component_linears.values()):
            current_scale = scales[i % 4]
            temp_init_tensor = torch.empty_like(linear_comp.weight.data)
            # For linear layers, fan_out might be more appropriate for Kaiming with SiLU/ReLU
            nn.init.kaiming_uniform_(temp_init_tensor, a=math.sqrt(5) * current_scale)

            if isinstance(linear_comp.weight, geoopt.ManifoldParameter):
                linear_comp.weight.data.copy_(self.manifold.projx(temp_init_tensor))
            else:
                linear_comp.weight.data.copy_(temp_init_tensor)

            if linear_comp.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_comp.weight)
                bound = 1 / math.sqrt(fan_in if fan_in > 0 else 1) * current_scale
                nn.init.uniform_(linear_comp.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be [B, total_in_features]
        # total_in_features = in_features_quat * 4
        in_features_quat = self.linear_r.in_features

        # Split flattened input into quaternion components
        xr_in = x[:, :in_features_quat]
        xi_in = x[:, in_features_quat : 2 * in_features_quat]
        xj_in = x[:, 2 * in_features_quat : 3 * in_features_quat]
        xk_in = x[:, 3 * in_features_quat:]

        # Hamilton Product for Dense Layer: Y = WX
        out_r = self.linear_r(xr_in) - self.linear_i(xi_in) - self.linear_j(xj_in) - self.linear_k(xk_in)
        out_i = self.linear_r(xi_in) + self.linear_i(xr_in) + self.linear_j(xk_in) - self.linear_k(xj_in)
        out_j = self.linear_r(xj_in) - self.linear_i(xk_in) + self.linear_j(xr_in) + self.linear_k(xi_in)
        out_k = self.linear_r(xk_in) + self.linear_i(xj_in) - self.linear_j(xi_in) + self.linear_k(xr_in)
        
        # Concatenate to form output: [B, total_out_features]
        return torch.cat([out_r, out_i, out_j, out_k], dim=1)
    
    


