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
    """
    Adaptive Patching Vision Transformer (APViT)

    This module extends the standard Vision Transformer (ViT) architecture by incorporating
    an adaptive patching mechanism. It maintains the core ViT structure while introducing
    two key modifications:

    1. Adaptive Patching: Uses the AdaptivePatching module to dynamically select patches
       based on image content before passing them to the standard patch embedding layer.
    2. Interpolated Positional Embeddings: Applies the interpolate_pos_embeds function
       to compute positional encodings for the adaptively selected patches.

    Key features:
    1. Content-aware patch selection, enabling dynamic receptive fields and orientations
       at the individual patch level.
    2. Flexible patch configuration: The number and size of patches can be modified
       independently of the input image size, as they no longer need to evenly divide
       the input grid.
    3. Decoupled positional embedding resolution: The number of positional embeddings
       is not tied to the number of patches, allowing for higher or lower resolution
       in interpolation as a tunable hyperparameter.

    These adaptations are seamlessly integrated, requiring no other changes to the
    standard ViT architecture. The module maintains the original ViT's simplicity
    while adding the benefits of adaptive, content-aware patch selection.

    Args:
        in_channels (int): Number of input image channels.
        hidden_channels (int): Number of hidden channels in the adaptive patching module.
        channel_height (int): Height of the input image.
        channel_width (int): Width of the input image.
        attn_embed_dim (int): Embedding dimension for self-attention layers.
        attn_heads (int): Number of attention heads in self-attention layers.
        num_transformer_layers (int): Number of transformer layers.
        pos_embed_size (int): Size of positional embeddings (this value is squared for the total number of embeddings).
        pos_embed_dim (int): Dimension of positional embeddings.
        num_patches (int): Number of patches to extract.
        patch_size (int): Size of each patch.
        stochastic_depth (float): Probability of dropping a layer in stochastic depth.
        scaling (str, optional): Type of scaling in adaptive patching. Options: 'isotropic', 'anisotropic', or None.
        max_scale (float): Maximum scale factor for adaptive patching.
        rotating (bool): Whether to apply rotation in adaptive patching.

    Shape:
        - Input: (batch_size, in_channels, height, width)
        - Output: (batch_size, num_classes), (num_transformer_layers, batch_size, num_patches + 1, num_patches + 1)

    Note:
        This implementation assumes a classification task with 10 classes. Adjust
        the final fully connected layer for different numbers of classes or tasks.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_channels=32,
        channel_height=32,
        channel_width=32,
        attn_embed_dim=256,
        attn_heads=4,
        num_transformer_layers=6,
        pos_embed_size=4, # squared
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

        self.pos_embeds = nn.Parameter(torch.randn(pos_embed_size ** 2, pos_embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, attn_embed_dim))

        self.transformer_layers = nn.ModuleList([
            SelfAttn(embed_dim=attn_embed_dim, num_heads=attn_heads, stochastic_depth=stochastic_depth)
            for _ in range(num_transformer_layers)
        ])

        self.norm = nn.LayerNorm(attn_embed_dim)
        self.fc = nn.Linear(attn_embed_dim, 10)

    def hook_fn(self, module, input, output):
        if isinstance(module, AdaptivePatching):
            patches, translate_params, scale_params, rotate_params = output
            self.selected_patches = patches
            self.translate_params = translate_params
            self.scale_params = scale_params
            self.rotate_params = rotate_params

    def setup_hooks(self):
        self.hooks = []
        self.hooks.append(self.adaptive_patches.register_forward_hook(self.hook_fn))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def forward(self, x):
        transform_params = self.adaptive_patches(x)
        patches = self.adaptive_patches.sample_patches(x, transform_params)
        b, n, c, p, _ = patches.size()
        patches = patches.view(b, c, p, p * n)
        x = self.patch_embed(patches)

        translate_params = transform_params[:, :, :2]
        pos_embeds = interpolate_pos_embeds(self.pos_embeds, translate_params)
        x = x + pos_embeds

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x.permute(1, 0, 2).contiguous()

        for layer in self.transformer_layers:
            x, _ = layer(x)

        x = x[0]
        x = self.norm(x)
        x = self.fc(x)
        return x
