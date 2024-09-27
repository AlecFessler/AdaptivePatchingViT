# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
from modules.AdaptivePatching import AdaptivePatching
from modules.SelfAttn import SelfAttn
from modules.PatchEmbed import PatchEmbed
from modules.InterpolatePosEmbeds import interpolate_pos_embeds

class APViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        channel_height=32,
        channel_width=32,
        attn_embed_dim=256,
        attn_heads=4,
        num_transformer_layers=6,
        pos_embed_dim=256,
        num_patches=16,
        patch_size=8,
        stochastic_depth=0.1,
        scaling=None,
        max_scale=0.3,
        rotating=False
    ):
        super(APViT, self).__init__()
        self.num_patches = num_patches

        self.adaptive_patches = AdaptivePatching(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            channel_height=channel_height,
            channel_width=channel_width,
            embed_dim=attn_embed_dim,
            num_patches=num_patches,
            patch_size=patch_size,
            scaling=scaling,
            max_scale=max_scale,
            rotating=rotating
        )
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=attn_embed_dim
        )

        self.pos_embeds = nn.Parameter(torch.randn(num_patches, pos_embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_embed_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttn(embed_dim=attn_embed_dim, num_heads=attn_heads, stochastic_depth=stochastic_depth)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.LayerNorm(attn_embed_dim)
        self.fc = nn.Linear(attn_embed_dim, 10)

    def forward(self, x):
        batch_size = x.size(0)
        patches, patch_coords = self.adaptive_patches(x)
        x = self.patch_embed(patches).squeeze(1).view(batch_size, self.num_patches, -1)
        pos_embeds = interpolate_pos_embeds(self.pos_embeds, patch_coords).view(batch_size, self.num_patches, -1)
        x = x + pos_embeds
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2).contiguous()
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[0]
        x = self.norm(x)
        x = self.fc(x)
        return x
