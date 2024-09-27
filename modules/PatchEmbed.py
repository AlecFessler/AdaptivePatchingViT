# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
