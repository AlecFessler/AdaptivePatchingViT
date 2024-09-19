# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torchvision.utils as vutils

from ConvBlock import ConvBlock
from ConvSelfAttn import ConvSelfAttn
from save_patch_grid import save_patch_grid
from draw_bounding_boxes import draw_bounding_boxes

class DynamicPatchSelection(nn.Module):
    """
    Dynamic Patch Selection is a mechanism for a learnable crop function
    which enables a Vision Transformer to focus on specific regions of an image.

    The core idea is to use a Spatial Transformer Network to generate a set of
    translation parameters for an affine transformation matrix. The translation
    parameters are bounded between -1 and 1 by a tanh activation. This ensures
    that all ouputs from the STN, even upon initialization, are valid and reasonable
    patch selections. This also slightly biases the patches away from the edges.

    The scaling parameters of the transform matrix are fixed based on the ratio of the
    desired patch size to the input image size. The rotation parameters are fixed to 0.

    The set of transform matrices are then used to generate a set of grid coordinates
    to sample patches from. The patches are then flattened and embedded.

    The output of the STN is also a convenient source of positional encodings.
    So, the translation parameters are also embedded, and the final output 
    includes both the patch and position embeddings.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        channel_height,
        channel_width,
        attn_embed_dim,
        attn_heads,
        pos_embed_dim,
        total_patches,
        patch_size,
        dropout=0.1
    ):
        super(DynamicPatchSelection, self).__init__()
        self.in_channels = in_channels
        self.total_patches = total_patches
        self.patch_size = patch_size

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
            embed_dim=attn_embed_dim,
            num_heads=attn_heads,
            dropout=dropout
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(
            in_channels=hidden_channels,
            out_channels=total_patches,
            kernel_size=3,
            stride=1,
            padding=1,
            bn = False
        )
        self.attn2 = ConvSelfAttn(
            channel_height=channel_height // 2,
            channel_width=channel_width // 2,
            embed_dim=attn_embed_dim,
            num_heads=attn_heads,
            dropout=dropout
        )
        self.fc = nn.Linear(channel_height // 2 * channel_width // 2, 2)
        self.activation = nn.Tanh()

        self.pos_embed = nn.Linear(2, pos_embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.save_data = False
        self.save_path = "0"
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc.weight)
        init.xavier_uniform_(self.pos_embed.weight)

    def forward(self, x):
        b, c, h, w = x.size()
        features = self.conv1(x)
        features = self.attn1(features)
        features = self.maxpool(features)
        features = self.conv2(features)
        features = self.attn2(features)
        features = features.view(b, self.total_patches, -1)

        translation_params = self.fc(features).view(b, self.total_patches, 2)
        translation_params = self.activation(translation_params)

        affine_transforms = torch.zeros(b, self.total_patches, 2, 3, device=x.device)
        affine_transforms[:, :, 0, 0] = self.patch_size / w
        affine_transforms[:, :, 1, 1] = self.patch_size / h
        affine_transforms[:, :, :, 2] = translation_params

        grid = nn.functional.affine_grid(
            affine_transforms.view(-1, 2, 3),
            [b * self.total_patches, 1, self.patch_size, self.patch_size],
            align_corners=False
        )

        patches = x.repeat(1, self.total_patches, 1, 1).view(b * self.total_patches, c, h, w)
        patches = nn.functional.grid_sample(
            patches,
            grid,
            padding_mode='zeros',
            align_corners=False
        )

        if self.save_data:
            # save a random image and its patches for debugging
            image_idx = np.random.randint(0, x.size(0))
            image_tensor = x[image_idx].clone().detach()
            save_patches = patches.view(
                b,
                self.total_patches,
                c,
                self.patch_size,
                self.patch_size
            )[image_idx]
            save_params = translation_params[image_idx]
            save_patch_grid(
                save_patches,
                save_params,
                "saved_data/patches_" + self.save_path + ".png",
            )
            #draw_bounding_boxes(
            #    image_tensor,
            #    save_params,
            #    "saved_data/image_" + self.save_path + ".png",
            #    self.patch_size
            #)
            resize_transform = transforms.Resize((512, 512))
            image_tensor = resize_transform(image_tensor)
            vutils.save_image(
                image_tensor,
                "saved_data/image_" + self.save_path + ".png"
            )

            self.save_data = False
            self.save_path = str(int(self.save_path) + 1)

        patches = patches.view(
            b,
            self.total_patches,
            c * self.patch_size * self.patch_size,
        )

        patches = self.dropout(patches)

        pos_embeds = self.pos_embed(translation_params)

        return patches, pos_embeds
