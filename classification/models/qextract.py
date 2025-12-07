# models/qextract.py
import torch
import torch.nn as nn

class QExtract(nn.Module):
    """
    Quaternion extraction module for converting quaternion features to real-valued output.
    This is more efficient than a full quaternion dense layer for classification tasks.
    """
    def __init__(self, in_channels, out_channels=None, extraction_method='concat', use_norm=True):
        super().__init__()
        
        self.extraction_method = extraction_method
        self.use_norm = use_norm
        
        if extraction_method == 'concat':
            # Concatenate all quaternion components
            proj_in_channels = in_channels * 4
        elif extraction_method == 'norm':
            # Use quaternion norm (magnitude)
            proj_in_channels = in_channels
        elif extraction_method == 'real':
            # Use only real component
            proj_in_channels = in_channels
        elif extraction_method == 'weighted_sum':
            # Learnable weighted combination
            proj_in_channels = in_channels
            self.component_weights = nn.Parameter(torch.tensor([1.0, 0.25, 0.25, 0.25]))
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        # Output channels default to input if not specified
        if out_channels is None:
            out_channels = proj_in_channels
            
        # Optional batch norm before projection
        if use_norm and extraction_method == 'concat':
            self.bn = nn.BatchNorm2d(proj_in_channels)
        else:
            self.bn = nn.Identity()
            
        # Activation
        self.act = nn.SiLU()
        
        # Final projection
        self.output_proj = nn.Conv2d(proj_in_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: Quaternion tensor of shape [B, C, Q=4, H, W]
        Returns:
            Real-valued tensor of shape [B, out_channels, H, W]
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, f"Expected quaternion dimension Q=4, got {Q}"
        
        if self.extraction_method == 'concat':
            # Reshape to [B, C*4, H, W] by moving Q dimension
            x = x.permute(0, 1, 3, 4, 2).reshape(B, C * 4, H, W)
            # print(f"RAHHH {x.shape}")
            
        elif self.extraction_method == 'norm':
            # Compute quaternion norm: sqrt(r² + i² + j² + k²)
            x = torch.norm(x, dim=2)  # [B, C, H, W]
            
        elif self.extraction_method == 'real':
            # Use only real component
            x = x[:, :, 0, :, :]  # [B, C, H, W]
            
        elif self.extraction_method == 'weighted_sum':
            # Learnable weighted combination of components
            weights = self.component_weights.view(1, 1, 4, 1, 1)
            x = (x * weights).sum(dim=2)  # [B, C, H, W]
            
        # Apply batch norm, activation, and projection
        x = self.bn(x)
        x = self.act(x)
        x = self.output_proj(x)
        
        return x


class QNormExtract(nn.Module):
    """
    Simple quaternion norm extraction for classification.
    Most efficient option when you just need the quaternion magnitudes.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: Quaternion tensor of shape [B, C, Q=4, H, W]
        Returns:
            Classification logits of shape [B, num_classes]
        """
        # Compute quaternion norm
        x = torch.norm(x, dim=2)  # [B, C, H, W]
        
        # Global average pooling
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        
        # Classification
        x = self.classifier(x)
        
        return x