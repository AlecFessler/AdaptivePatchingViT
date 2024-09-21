# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
from modules.DynamicPatchSelection import DynamicPatchSelection
from modules.SelfAttn import SelfAttn

class DpsViT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        channel_height=32,
        channel_width=32,
        attn_embed_dim=256,
        attn_heads=4,
        num_transformer_layers=6,
        pos_embed_dim=32,
        total_patches=16,
        patch_size=8,
        stn_dropout=0.1,
        patch_dropout=0,
        transformer_dropout=0.1
    ):
        super(DpsViT, self).__init__()

        self.dynamic_patch = DynamicPatchSelection(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            channel_height=channel_height,
            channel_width=channel_width,
            attn_embed_dim=attn_embed_dim,
            attn_heads=attn_heads,
            pos_embed_dim=pos_embed_dim,
            total_patches=total_patches,
            patch_size=patch_size,
            stn_dropout=stn_dropout,
            patch_dropout=patch_dropout
        )
        self.embedding_layer = nn.Linear(in_channels * patch_size * patch_size, attn_embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_embed_dim + pos_embed_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttn(embed_dim=attn_embed_dim + pos_embed_dim, num_heads=attn_heads, dropout=transformer_dropout)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.LayerNorm(attn_embed_dim + pos_embed_dim)
        self.fc = nn.Linear(attn_embed_dim + pos_embed_dim, 10)

    def forward(self, x):
        x, pos_embeds = self.dynamic_patch(x)
        x = self.embedding_layer(x)
        x = torch.cat((x, pos_embeds), dim=-1)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2).contiguous()
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[0]
        x = self.norm(x)
        x = self.fc(x)
        return x
