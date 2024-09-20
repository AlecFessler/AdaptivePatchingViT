# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
from modules.PatchEmbed import PatchEmbed
from modules.SelfAttn import SelfAttn

import torch.nn.functional as F
from torchvision import transforms
import torchvision
import numpy as np

class StandardViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        attn_embed_dim=256,
        pos_embed_dim=32,
        attn_heads=4,
        num_transformer_layers=6,
        dropout=0.1
    ):
        super(StandardViT, self).__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=attn_embed_dim + pos_embed_dim
        )

        self.pos_embeds = nn.Parameter(
            torch.randn(1, (img_size // patch_size) ** 2, attn_embed_dim + pos_embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_embed_dim + pos_embed_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttn(embed_dim=attn_embed_dim + pos_embed_dim, num_heads=attn_heads, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.LayerNorm(attn_embed_dim + pos_embed_dim)
        self.fc = nn.Linear(attn_embed_dim + pos_embed_dim, 10)

    def forward(self, x):
        # Save the original input images
        x_input = x.clone()  # Shape: [batch_size, channels, height, width]
        
        # Existing forward operations
        x = self.patch_embed(x)
        x = x + self.pos_embeds
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2).contiguous()
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[0]
        x = self.norm(x)
        x = self.fc(x)

        """
        if not self.training:
            for i in range(x_input.size(0)):
                image = x_input[i]  # Shape: (channels, height, width)
                
                # Move image to CPU and ensure it's a float tensor
                image = image.detach().cpu().float()
                
                # Resize image to 512x512
                resize_transform = transforms.Resize((512, 512))
                image_resized = resize_transform(image.unsqueeze(0))  # Add batch dimension
                image_resized = image_resized.squeeze(0)  # Remove batch dimension
                
                # Save the resized image
                torchvision.utils.save_image(image_resized, f"image_{i}_resized.png")
                
                # Extract patches from the image
                patch_size = self.patch_embed.patch_size  # Should be an int or tuple
                if isinstance(patch_size, int):
                    patch_size = (patch_size, patch_size)
                else:
                    patch_size = tuple(patch_size)
                
                # Extract patches using unfold
                patches = image.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1])
                # patches shape: [channels, num_patches_h, num_patches_w, patch_size_h, patch_size_w]
                patches = patches.permute(1, 2, 0, 3, 4)  # Shape: [num_patches_h, num_patches_w, channels, patch_size_h, patch_size_w]
                patches = patches.contiguous().view(-1, image.size(0), patch_size[0], patch_size[1])
                
                # Resize patches to fit into a 512x512 grid
                num_patches = patches.size(0)
                grid_size = int(np.sqrt(num_patches))
                patch_resize_dim = 512 // grid_size
                resize_transform = transforms.Resize((patch_resize_dim, patch_resize_dim))
                patches_resized = torch.stack([resize_transform(patch) for patch in patches])
                
                # Create a grid of patches and save
                grid = torchvision.utils.make_grid(patches_resized, nrow=grid_size, padding=2)
                torchvision.utils.save_image(grid, f"image_{i}_patches.png")
            """
        return x
