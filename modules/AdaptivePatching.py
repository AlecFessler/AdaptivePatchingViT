# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn

from modules.ConvBlock import ConvBlock
from modules.ConvSelfAttn import ConvSelfAttn

class AdaptivePatching(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        channel_height,
        channel_width,
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

    def sample_patches(self, x, transform_params):
        b, c, h, w = x.size()

        translate_params = transform_params[:, :, :2] # (B, N, 2)
        scale_params = transform_params[:, :, 2:4] # (B, N, 2)
        rotate_params = transform_params[:, :, 4] # (B, N, 1)

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

        return patches

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

        transform_params = torch.cat([translate_params, scale_params, rotate_params], dim=-1) # (B, N, 5)

        return transform_params # (B, N, 5)
