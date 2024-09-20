# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
from torchvision import utils as vutils
from modules.DynamicPatchSelection import DynamicPatchSelection
from modules.SelfAttn import SelfAttn
from utils.save_patch_grid import save_patch_grid

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
        dropout=0.1
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
            dropout=dropout
        )
        self.embedding_layer = nn.Linear(in_channels * patch_size * patch_size, attn_embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_embed_dim + pos_embed_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttn(embed_dim=attn_embed_dim + pos_embed_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.LayerNorm(attn_embed_dim + pos_embed_dim)
        self.fc = nn.Linear(attn_embed_dim + pos_embed_dim, 10)

    def forward(self, x):
        imgs = x
        x, pos_embeds, translation_params = self.dynamic_patch(x)

        # save a random patch grid during eval for debugging
        if not self.training:
            random_idx = torch.randint(0, x.size(0), (1,)).item()
            save_patch_grid(
                x[random_idx],
                translation_params[random_idx],
                f"assets/patches_{random_idx}.png",
                channels=self.dynamic_patch.in_channels,
                patch_size=self.dynamic_patch.patch_size
            )
            vutils.save_image(imgs[random_idx], f"assets/img_{random_idx}.png")

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
