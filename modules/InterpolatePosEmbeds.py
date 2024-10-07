# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch

def interpolate_pos_embeds(pos_embeds, coords):
    """
    Interpolate positional embeddings for use with the AdaptivePatching module.

    This function performs bilinear interpolation on a grid of positional embeddings
    using coordinates provided by the AdaptivePatching module. It enables accurate
    positional encoding for dynamically selected patches.

    Assumptions:
    - Input positional embeddings form a square grid
    - Coordinates are normalized to the range [-1, 1], where (-1, -1) is the
      top-left corner and (1, 1) is the bottom-right corner of the grid

    Args:
        pos_embeds (torch.Tensor): Grid of positional embeddings
        coords (torch.Tensor): Normalized coordinates for interpolation

    Shape:
        - pos_embeds: (num_embeds, embedding_dim)
        - coords: (batch_size, num_patches, 2)
        - Output: (batch_size, num_patches, embedding_dim)

    Returns:
        torch.Tensor: Interpolated embeddings for the given coordinates

    Note:
        This function is typically used after an adaptive patching module to
        compute appropriate positional embeddings for the selected patches.
        It ensures that the positional information accurately reflects the
        patches' locations, even when they don't align with the original grid.
    """
    num_embeds, embedding_dim = pos_embeds.size()
    grid_size = int(num_embeds ** 0.5)
    pos_embeds = pos_embeds.view(grid_size, grid_size, embedding_dim)

    coords = (coords + 1) * (grid_size - 1) / 2
    coords_floor = torch.floor(coords)
    delta = coords - coords_floor

    x0y0 = coords_floor.long().clamp(0, grid_size - 1)
    x1y1 = (x0y0 + 1).clamp(0, grid_size - 1)

    embed00 = pos_embeds[x0y0[..., 1], x0y0[..., 0]]
    embed01 = pos_embeds[x1y1[..., 1], x0y0[..., 0]]
    embed10 = pos_embeds[x0y0[..., 1], x1y1[..., 0]]
    embed11 = pos_embeds[x1y1[..., 1], x1y1[..., 0]]

    delta_x, delta_y = delta[..., 0], delta[..., 1]
    w00 = (1 - delta_x) * (1 - delta_y)
    w01 = (1 - delta_x) * delta_y
    w10 = delta_x * (1 - delta_y)
    w11 = delta_x * delta_y

    w00 = w00.unsqueeze(-1)
    w01 = w01.unsqueeze(-1)
    w10 = w10.unsqueeze(-1)
    w11 = w11.unsqueeze(-1)

    interpolated = (w00 * embed00) + (w01 * embed01) + (w10 * embed10) + (w11 * embed11)

    return interpolated
