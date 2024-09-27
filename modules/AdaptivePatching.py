# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import numpy as np

from modules.ConvBlock import ConvBlock
from modules.ConvSelfAttn import ConvSelfAttn

class AdaptivePatching(nn.Module):
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
            dropout=0.1
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
            dropout=0.1
        )

        half_channel = channel_height // 2 * channel_width // 2
        self.fc1 = nn.Linear(half_channel, half_channel // 2)
        self.relu = nn.ReLU()

        num_transform_params = 2
        if scaling == 'anisotropic': num_transform_params += 2
        if scaling == 'isotropic': num_transform_params += 1
        if rotating: num_transform_params += 1
        self.fc2 = nn.Linear(half_channel // 2, num_transform_params)

        self.translate_activation = nn.Tanh()
        self.scale_activation = nn.Sigmoid()
        self.rotate_activation = nn.Tanh()

    def forward(self, x):
        b, c, h, w = x.size()
        features = self.conv1(x)
        features = self.attn1(features)
        features = self.maxpool(features)
        features = self.conv2(features)
        features = self.attn2(features)
        features = features.view(b, self.num_patches, -1)

        transform_params = self.fc1(features)
        transform_params = self.relu(transform_params)
        transform_params = self.fc2(transform_params)

        param_num = 2
        translate_params = transform_params[:, :, :param_num]
        if self.scaling == 'anisotropic':
            scale_params = transform_params[:, :, param_num:param_num+2]
            param_num += 2
        elif self.scaling == 'isotropic':
            scale_params = transform_params[:, :, param_num:param_num+1]
            param_num += 1
        if self.rotating:
            rotate_params = transform_params[:, :, param_num:param_num+1]

        translate_params = self.translate_activation(translate_params)
        if self.scaling:
            scale_params = self.scale_activation(scale_params) * self.max_scale
            if self.scaling == 'isotropic':
                scale_params = scale_params.repeat(1, 1, 2)
        else:
            scale_params = torch.tensor([self.patch_size / w, self.patch_size / h], device=x.device)
            scale_params = scale_params.view(1, 1, 2).expand(b, self.num_patches, 2)

        if self.rotating:
            rotate_params = self.rotate_activation(rotate_params) * np.pi
        else:
            rotate_params = torch.zeros(b, self.num_patches, 1, device=x.device)

        cos_theta = torch.cos(rotate_params).squeeze(-1)
        sin_theta = torch.sin(rotate_params).squeeze(-1)

        x_extent = scale_params[:, :, 0] * torch.abs(cos_theta) + scale_params[:, :, 1] * torch.abs(sin_theta)
        y_extent = scale_params[:, :, 0] * torch.abs(sin_theta) + scale_params[:, :, 1] * torch.abs(cos_theta)
        x_extent = torch.clamp(x_extent, max=1)
        y_extent = torch.clamp(y_extent, max=1)

        translation_scale = 1 - torch.stack([x_extent, y_extent], dim=-1)
        translate_params = translate_params * translation_scale

        ta = scale_params[:, :, 0] * cos_theta
        tb = -scale_params[:, :, 0] * sin_theta
        tc = scale_params[:, :, 1] * sin_theta
        td = scale_params[:, :, 1] * cos_theta
        tx = translate_params[:, :, 0]
        ty = translate_params[:, :, 1]

        affine_transforms = torch.stack([
            torch.stack([ta, tb, tx], dim=-1),
            torch.stack([tc, td, ty], dim=-1)
        ], dim=-2)

        grid = nn.functional.affine_grid(
            affine_transforms.view(-1, 2, 3),
            [b * self.num_patches, 1, self.patch_size, self.patch_size],
            align_corners=False
        )

        patches = nn.functional.grid_sample(
            x.unsqueeze(1).expand(-1, self.num_patches, -1, -1, -1).reshape(-1, c, h, w),
            grid,
            align_corners=False
        ).view(-1, c, self.patch_size, self.patch_size)

        return patches, translate_params
