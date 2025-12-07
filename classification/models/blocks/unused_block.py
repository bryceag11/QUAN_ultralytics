class QuaternionUpsample(nn.Module):
    """
    Custom upsampling module for quaternion tensors.
    Upsamples only the spatial dimensions (H, W), keeping Q intact.
    """
    def __init__(self, scale_factor=2, mode='nearest'):
        super(QuaternionUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, Q, H, W]
        
        Returns:
            torch.Tensor: Upsampled tensor of shape [B, C, Q, H*scale_factor, W*scale_factor]
        """

        B, C, Q, H, W = x.shape
        # Reshape to [B * Q, C, H, W] to apply upsampling on spatial dimensions

        # Permute to [B, Q, C, H, W] and make contiguous
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # Reshape to [B * Q, C, H, W] to apply upsampling on spatial dimensions
        x = x.view(B * Q, C, H, W)

        # Apply upsampling
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        # Reshape back to [B, Q, C, H_new, W_new]
        H_new, W_new = x.shape[-2], x.shape[-1]
        x = x.view(B, Q, C, H_new, W_new).permute(0, 2, 1, 3, 4).contiguous()

        return x

class QuaternionPyramidAttention(nn.Module):
    """
    Novel block: Multi-scale quaternion attention with rotation invariance
    - Processes features at multiple scales
    - Maintains quaternion structure
    - Computationally efficient
    """
    def __init__(self, channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.qatt_blocks = nn.ModuleList([
            QAttention(channels//4) for _ in scales
        ])
        
    def forward(self, x):
        results = []
        for scale, qatt in zip(self.scales, self.qatt_blocks):
            # Pool, attend, upsample while preserving quaternion structure
            pooled = F.avg_pool2d(x, scale)
            attended = qatt(pooled)
            upsampled = F.interpolate(attended, size=x.shape[-2:])
            results.append(upsampled)
        return torch.cat(results, dim=1)

class QuaternionFeatureFusion(nn.Module):
    """
    Novel block: Quaternion-aware feature fusion
    - Dynamically weights feature combinations
    - Preserves rotational equivariance
    """
    def __init__(self, channels):
        super().__init__()
        self.qconv = QConv2D(channels, channels//4, 1)
        self.qatt = QAttention(channels//4)
        
    def forward(self, x1, x2):
        # Fusion while maintaining quaternion properties
        fused = self.qconv(torch.cat([x1, x2], dim=1))
        weighted = self.qatt(fused)
        return weighted

class QRotationAttention(nn.Module):
    """
    Rotation-aware attention block specifically for OBB detection.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Project features while preserving quaternion structure
        self.q = QConv2D(channels, channels, 1)
        self.k = QConv2D(channels, channels, 1)
        self.v = QConv2D(channels, channels, 1)
        
        # Output projection
        self.proj = QConv2D(channels, channels, 1)
        self.norm = IQBN(channels)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features with rotation attention [B, C, 4, H, W]
        """
        B, C, Q, H, W = x.shape
        
        # Project to Q,K,V while keeping quaternion structure
        q = self.q(x)  # [B, C, 4, H, W]
        k = self.k(x)  # [B, C, 4, H, W]
        v = self.v(x)  # [B, C, 4, H, W]
        
        # Reshape for attention computation
        q = q.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        k = k.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        v = v.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)  # [B, 4, C/4, H*W]
        
        # Compute rotation-aware attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C//4)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention and reshape
        out = torch.matmul(attn, v)  # [B, 4, C/4, H*W]
        out = out.permute(0, 2, 1, 3).reshape(B, C, 4, H, W)
        
        # Project and normalize
        out = self.proj(out)
        out = self.norm(out)
        
        return out

class QOBBFeatureFusion(nn.Module):
    """
    Feature fusion block specifically designed for OBB detection.
    Preserves and enhances orientation information during feature fusion.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0
        
        # Dimension reduction
        self.conv1 = QConv2D(in_channels, out_channels, 1)
        self.norm1 = IQBN(out_channels)
        self.act = QReLU()
        
        # Rotation-specific attention
        self.rot_attn = QRotationAttention(out_channels)
        
        # Channel attention with quaternion structure preservation
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(out_channels, out_channels//4, 1),
            QReLU(),
            QConv2D(out_channels//4, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features [B, C, 4, H, W]
        Returns:
            Fused features with enhanced orientation information [B, C_out, 4, H, W]
        """
        # Initial convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        
        # Enhance rotation features
        out = self.rot_attn(out)
        
        # Apply channel attention while preserving quaternion structure
        w = self.ca(out.view(out.shape[0], -1, *out.shape[-2:]))
        w = w.view_as(out)
        out = out * w
        
        return out

class QRotationCrossAttention(nn.Module):
    """
    Cross-attention module specifically for OBB detection.
    Enhances feature interaction while preserving orientation information.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Projections for cross attention
        self.q_proj = QConv2D(channels, channels, kernel_size=1)
        self.k_proj = QConv2D(channels, channels, kernel_size=1)
        self.v_proj = QConv2D(channels, channels, kernel_size=1)
        
        # Output projection
        self.out_proj = QConv2D(channels, channels, kernel_size=1)
        self.norm = IQBN(channels)
        
        # Quaternion-specific angle attention
        self.angle_attn = nn.Sequential(
            QConv2D(channels, channels//4, 1),
            QReLU(),
            QConv2D(channels//4, 4, 1)  # 4 for quaternion components
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1: Current level features [B, C, 4, H, W]
            x2: Cross level features [B, C, 4, H, W]
        Returns:
            Enhanced features with rotation-aware cross attention [B, C, 4, H, W]
        """
        B, C, Q, H, W = x1.shape
        
        # Project to Q,K,V
        q = self.q_proj(x1)
        k = self.k_proj(x2)
        v = self.v_proj(x2)
        
        # Compute quaternion-specific angle attention
        angle_weights = self.angle_attn(x1)  # [B, 4, 4, H, W]
        angle_weights = angle_weights.softmax(dim=1)
        
        # Apply cross attention with angle weighting
        q = q * angle_weights
        k = k * angle_weights
        
        # Reshape and compute attention
        q = q.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        k = k.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        v = v.view(B, C//4, 4, H*W).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C//4)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, C, 4, H, W)
        
        # Project and normalize
        out = self.out_proj(out)
        out = self.norm(out)
        
        return out

class QAdaptiveFeatureExtraction(nn.Module):
    """
    Enhanced feature extraction with multi-scale processing and channel attention.
    Shape: [B, C, 4, H, W] -> [B, C, 4, H, W]
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        self.channels = channels
        mid_channels = max(channels // reduction_ratio, 32)
        
        # Multi-scale branches
        self.local_branch = nn.Sequential(
            QConv2D(channels, channels//2, kernel_size=3, padding=1),
            IQBN(channels//2),
            QReLU()
        )
        
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//2, kernel_size=1),
            QReLU()
        )
        
        # Channel attention
        self.ca = nn.Sequential(
            QConv2D(channels, mid_channels, 1),
            QReLU(),
            QConv2D(mid_channels, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = QConv2D(channels, channels, 3, padding=1)
        self.norm = IQBN(channels)
        self.act = QReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features [B, C, 4, H, W]
        """
        # Process branches
        local_feat = self.local_branch(x)  # [B, C/2, 4, H, W]
        
        # Global context
        global_feat = self.global_branch(x)  # [B, C/2, 4, 1, 1]
        global_feat = global_feat.expand(-1, -1, -1, x.shape[-2], x.shape[-1])
        
        # Combine features
        combined = torch.cat([local_feat, global_feat], dim=1)  # [B, C, 4, H, W]
        
        # Apply channel attention
        attn = self.ca(combined)
        out = combined * attn
        
        # Final refinement
        out = self.refine(out)
        out = self.norm(out)
        out = self.act(out)
        
        return out


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


class QDualAttention(nn.Module):
    """
    Dual path attention combining spatial and channel attention
    with quaternion structure preservation.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Spatial attention branch - ensure output is multiple of 4
        self.spatial = nn.Sequential(
            QConv2D(channels, channels//8, 1),
            QReLU(),
            QConv2D(channels//8, 4, 1),  # Changed from 1 to 4 channels
            nn.Sigmoid()
        )
        
        # Channel attention branch
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//8, 1),
            QReLU(),
            QConv2D(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine = QConv2D(channels, channels, 3, padding=1)
        self.norm = IQBN(channels)
        self.act = QReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, 4, H, W]
        Returns:
            Enhanced features [B, C, 4, H, W]
        """
        # Spatial attention - repeat attention map for all quaternion components
        spatial_attn = self.spatial(x)
        spatial_attn = spatial_attn.repeat(1, x.size(1)//4, 1, 1)  # Repeat to match input channels
        spatial_out = x * spatial_attn
        
        # Channel attention
        channel_attn = self.channel(x)
        channel_out = x * channel_attn
        
        # Combine and refine
        out = spatial_out + channel_out
        out = self.refine(out)
        out = self.norm(out)
        out = self.act(out)
        
        return out

class QAdaptiveFusion(nn.Module):
    """
    Adaptive feature fusion with dynamic weighting
    and enhanced quaternion feature interaction.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "Channels must be multiple of 4"
        
        # Feature transformation
        self.transform1 = QConv2D(channels, channels//2, 1)
        self.transform2 = QConv2D(channels, channels//2, 1)
        
        # Dynamic weight prediction
        self.weight_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            QConv2D(channels, channels//4, 1),
            QReLU(),
            QConv2D(channels//4, 2, 1),  # 2 weights for 2 paths
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.refine = nn.Sequential(
            QConv2D(channels//2, channels//2, 3, padding=1),
            IQBN(channels//2),
            QReLU(),
            QConv2D(channels//2, channels//2, 3, padding=1)
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: Input tensors [B, C, 4, H, W]
        Returns:
            Fused features [B, C//2, 4, H, W]
        """
        # Transform features
        f1 = self.transform1(x1)
        f2 = self.transform2(x2)
        
        # Predict fusion weights
        weights = self.weight_pred(torch.cat([f1, f2], dim=1))
        
        # Weighted fusion
        fused = f1 * weights[:, 0:1, :, :] + f2 * weights[:, 1:2, :, :]
        
        # Refine fused features
        out = self.refine(fused)
        
        return out
