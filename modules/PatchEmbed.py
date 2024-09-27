# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
