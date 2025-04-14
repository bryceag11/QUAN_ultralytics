# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Activation modules."""

import torch
import torch.nn as nn


class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(torch.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        lam = torch.clamp(self.lambd, min=0.0001)
        return torch.exp((1 / lam) * self.act((self.kappa * x) - torch.log(lam)))


class QReLU(nn.Module):
    """
    Quaternion ReLU activation function that applies ReLU separately to each quaternion component
    following the paper's split-type activation approach.
    
    The input tensor should be in the format (batch_size, channels, 4, height, width) where:
    - The 4 represents quaternion components in order: real, i, j, k
    """
    def __init__(self, inplace: bool = False):
        super(QReLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a new tensor instead of modifying in place
        out = torch.zeros_like(x)
        
        # Copy the real component unchanged
        out[:,:,0,:,:] = x[:,:,0,:,:]
        
        # Apply ReLU to imaginary components
        out[:,:,1:,:,:] = F.relu(x[:,:,1:,:,:])
        
        return out
    
    def extra_repr(self) -> str:
        return ''


class QSiLU(nn.Module):
    """
    Split Quaternion SiLU (Swish) Activation Function.
    Applies SiLU to each quaternion component separately, following QPReLU's approach.
    """
    def __init__(self):
        super(QSiLU, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create output tensor
        out = torch.zeros_like(x)
        
        # Copy real component unchanged
        # out[:,:,0,:,:] = x[:,:,0,:,:]
        
        # Apply SiLU to imaginary components
        imag_parts = x[:,:,0:,:,:]
        out[:,:,0:,:,:] = imag_parts * self.sigmoid(imag_parts)
        
        return out

  
class QReLU(nn.Module):
    """
    Quaternion ReLU activation function that applies ReLU separately to each quaternion component
    following the paper's split-type activation approach.
    
    The input tensor should be in the format (batch_size, channels, 4, height, width) where:
    - The 4 represents quaternion components in order: real, i, j, k
    """
    def __init__(self, inplace: bool = False):
        super(QReLU, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Create a new tensor instead of modifying in place
        out = torch.zeros_like(x)
        
        # Copy the real component unchanged
        out[:,:,0,:,:] = x[:,:,0,:,:]
        
        # Apply ReLU to imaginary components
        out[:,:,1:,:,:] = F.relu(x[:,:,1:,:,:])
        
        return out
    
    def extra_repr(self) -> str:
        return ''



class QPReLU(nn.Module):
    """
    Split Quaternion Parametric ReLU Activation Function.
    Applies PReLU to each quaternion component separately with learnable parameters.
    """
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super(QPReLU, self).__init__()
        self.num_parameters = num_parameters
        
        # Create 4 learnable parameters, one for each quaternion component
        self.weight = nn.Parameter(torch.Tensor(4 * num_parameters).fill_(init))
        
    def forward(self, x):
        # x shape: (batch_size, channels, 4, height, width)
        # Get parameters for each component
        w_r, w_i, w_j, w_k = self.weight.chunk(4)
        
        # Apply component-wise PReLU
        x_r = F.prelu(x[:,:,0,:,:], w_r)
        x_i = F.prelu(x[:,:,1,:,:], w_i)
        x_j = F.prelu(x[:,:,2,:,:], w_j) 
        x_k = F.prelu(x[:,:,3,:,:], w_k)
        
        # Stack components back together
        return torch.stack([x_r, x_i, x_j, x_k], dim=2)
