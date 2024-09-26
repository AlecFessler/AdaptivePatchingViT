# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
import torch.nn.init as init
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
        pos_embed_dim,
        num_patches,
        patch_size,
        max_scale=0.3
    ):
        super(AdaptivePatching, self).__init__()
        assert max_scale <= 0.7071, 'Max scale greater than 0.7071 will cause some rotations to exceed bounds'
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.max_scale = max_scale
        self.embed_dim = embed_dim

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
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(half_channel // 2, 5)

        self.translate_activation = nn.Tanh()
        self.scale_activation = nn.Sigmoid()
        self.rotate_activation = nn.Tanh()
        self.pos_embed = nn.Linear(2, pos_embed_dim)

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        init.xavier_uniform_(self.pos_embed.weight)
        init.zeros_(self.pos_embed.bias)
        init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            init.zeros_(self.patch_embed.bias)

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

        translate_params = self.translate_activation(transform_params[:, :, :2])
        scale_params = self.scale_activation(transform_params[:, :, 2:4]) * self.max_scale
        rotate_params = self.rotate_activation(transform_params[:, :, 4]) * np.pi

        cos_theta = torch.cos(rotate_params)
        sin_theta = torch.sin(rotate_params)

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

        patch_embeds = self.patch_embed(patches).view(b, self.num_patches, self.embed_dim)
        pos_embeds = self.pos_embed(translate_params)

        return patch_embeds, pos_embeds
