# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.ConvBlock import ConvBlock
from modules.ConvSelfAttn import ConvSelfAttn

class AdaptivePatching(nn.Module):
    """
    Adaptive patching module for content-aware patch extraction in image processing.

    This module implements a specialized Spatial Transformer Network (STN) to dynamically
    select and transform patches from an input image. It generates a set of transformation
    parameters for each patch, including translation and optionally scaling and rotation.
    These parameters are constrained to ensure all extracted patches remain within the
    bounds of the input image.

    Key features:
    1. Content-aware patch selection through learned transformations
    2. Compatible with patch-based transformer architectures (e.g., Vision Transformer)
    3. Flexible configuration for isotropic/anisotropic scaling and rotation
    4. Bounded transformation parameters to maintain valid patch sampling

    The module outputs both the extracted patches and their corresponding translation
    parameters. These outputs can be directly used with transformer architectures:
    - Patches can be fed into a patch embedding layer
    - Translation parameters can be used to interpolate positional embeddings

    Transformation parameters:
    - Translation: Normalized to [-1, 1] and scaled based on patch extent
    - Scaling (optional): Bounded to [0, max_scale], where max_scale ≤ 0.7071 if rotation is enabled, else max_scale ≤ 1
    - Rotation (optional): Normalized to [-π, π] for full 360° rotation

    Args:
        in_channels (int): Number of input image channels
        hidden_channels (int): Number of hidden channels in the STN
        channel_height (int): Height of the input image
        channel_width (int): Width of the input image
        embed_dim (int): Embedding dimension for self-attention layers
        num_patches (int): Number of patches to extract
        patch_size (int): Size of each extracted patch (height and width)
        scaling (str, optional): Scaling type: 'isotropic', 'anisotropic', or None. Default: 'isotropic'
        max_scale (float, optional): Maximum scaling factor. Default: 0.3
        rotating (bool, optional): Enable rotation transformations. Default: True

    Shape:
        - Input: (batch_size, in_channels, height, width)
        - Output:
            - patches: (batch_size, num_patches, in_channels, patch_size, patch_size)
            - translate_params: (batch_size, num_patches, 2)

    Note:
        This module assumes the input image is normalized. The output patches maintain
        the same normalization as the input.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        channel_height,
        channel_width,
        embed_dim,
        num_patches,
        patch_size,
        scaling='isotropic', # 'isotropic', 'anisotropic', None
        max_scale=0.3, # max 0.7071 if rotating=True, else max 1
        rotating=True
        ):
        super(AdaptivePatching, self).__init__()
        assert scaling in ['isotropic', 'anisotropic', None], 'Scaling must be one of "isotropic", "anisotropic", or None'
        if rotating and scaling:
            assert max_scale <= 0.7071, 'Max scale greater than 0.7071 will cause some rotations to exceed bounds'
        elif scaling:
            assert max_scale <= 1, 'Max scale greater than 1 will cause some patches to exceed bounds'

        self.in_channels = in_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.scaling = scaling
        self.max_scale = max_scale
        self.rotating = rotating

        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bn = False
        )
        self.attn1 = ConvSelfAttn(
            channel_height=channel_height,
            channel_width=channel_width,
            embed_dim=patch_size * patch_size * in_channels,
            num_heads=4,
            dropout=0.0
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(
            in_channels=hidden_channels,
            out_channels=num_patches,
            kernel_size=3,
            stride=1,
            padding=1,
            bn = False
        )
        self.attn2 = ConvSelfAttn(
            channel_height=channel_height // 2,
            channel_width=channel_width // 2,
            embed_dim=patch_size * patch_size * in_channels,
            num_heads=4,
            dropout=0.0
        )

        half_channel = channel_height // 2 * channel_width // 2
        self.fc1 = nn.Linear(half_channel, half_channel // 2)
        self.relu = nn.ReLU()

        num_transform_params = 2
        if scaling == 'anisotropic': num_transform_params += 2
        if scaling == 'isotropic': num_transform_params += 1
        if rotating: num_transform_params += 1
        self.fc2 = nn.Linear(half_channel // 2, num_transform_params)

    def forward(self, x): # (B, C, H, W)
        b, c, h, w = x.size()
        features = self.conv1(x) # (B, hidden_channels, H, W)
        features = self.attn1(features) # (B, hidden_channels, H, W)
        features = self.maxpool(features) # (B, hidden_channels, H/2, W/2)
        features = self.conv2(features) # (B, N, H/2, W/2)
        features = self.attn2(features) # (B, N, H/2, W/2)
        features = features.view(b, self.num_patches, -1) # (B, N, H/2*W/2)

        transform_params = self.fc1(features) # (B, N, C*P*P/2)
        transform_params = self.relu(transform_params) # (B, N, C*P*P/2)
        transform_params = self.fc2(transform_params) # (B, N, num_transform_params)

        param_num = 2
        translate_params = transform_params[:, :, :param_num] # (B, N, 2)
        if self.scaling:
            if self.scaling == 'anisotropic':
                scale_params = transform_params[:, :, param_num:param_num+3] # (B, N, 2)
                param_num += 2
            elif self.scaling == 'isotropic':
                scale_params = transform_params[:, :, param_num:param_num+1] # (B, N, 1)
                param_num += 1
        if self.rotating:
            rotate_params = transform_params[:, :, param_num:param_num+1] # (B, N, 1)

        # bound translation to [-1, 1]
        translate_params = torch.tanh(translate_params) # (B, N, 2)

        # bound scaling to [0, max_scale] or assign (patch_size / image_size) if not scaling
        # if isotropic scaling, the single scale param is repeated for both spatial dimensions
        if self.scaling:
            scale_params = torch.sigmoid(scale_params) * self.max_scale
            if self.scaling == 'isotropic':
                scale_params = scale_params.repeat(1, 1, 2) # (B, N, 2)
        else:
            scale_params = torch.tensor([self.patch_size / w, self.patch_size / h], device=x.device)
            scale_params = scale_params.view(1, 1, 2).expand(b, self.num_patches, 2) # (B, N, 2)

        # bound rotation to [-pi, pi] or assign 0s if not rotating
        if self.rotating:
            rotate_params = torch.tanh(rotate_params) * torch.pi
        else:
            rotate_params = torch.zeros(b, self.num_patches, 1, device=x.device) # (B, N, 1)

        cos_theta = torch.cos(rotate_params).squeeze(-1) # (B, N)
        sin_theta = torch.sin(rotate_params).squeeze(-1) # (B, N)

        # calculate patch extents based on scaling and rotation and clamp to [0, 1]
        x_extent = scale_params[:, :, 0] * torch.abs(cos_theta) + scale_params[:, :, 1] * torch.abs(sin_theta)
        y_extent = scale_params[:, :, 0] * torch.abs(sin_theta) + scale_params[:, :, 1] * torch.abs(cos_theta)
        x_extent = torch.clamp(x_extent, max=1)
        y_extent = torch.clamp(y_extent, max=1)

        # scale translation parameters by patch extents
        translation_scale = 1 - torch.stack([x_extent, y_extent], dim=-1)
        translate_params = translate_params * translation_scale

        # calculate affine transformation matrices
        ta = scale_params[:, :, 0] * cos_theta
        tb = -scale_params[:, :, 0] * sin_theta
        tc = scale_params[:, :, 1] * sin_theta
        td = scale_params[:, :, 1] * cos_theta
        tx = translate_params[:, :, 0]
        ty = translate_params[:, :, 1]

        affine_transforms = torch.stack([
            torch.stack([ta, tb, tx], dim=-1),
            torch.stack([tc, td, ty], dim=-1)
        ], dim=-2) # (B, N, 2, 3)

        grid = nn.functional.affine_grid(
            affine_transforms.view(-1, 2, 3), # (B*N, 2, 3)
            [b * self.num_patches, 1, self.patch_size, self.patch_size],
            align_corners=False
        ) # (B*N, P, P, 2)

        patches = nn.functional.grid_sample(
            x.unsqueeze(1).expand(-1, self.num_patches, -1, -1, -1).reshape(-1, c, h, w), # (B*N, C, H, W)
            grid,
            align_corners=False
        ).view(b, self.num_patches, c, self.patch_size, self.patch_size) # (B, N, C, P, P)

        return patches, translate_params # (B, N, C, P, P), (B, N, 2)
