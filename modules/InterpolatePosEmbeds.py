# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch

def interpolate_pos_embeds(pos_embeds, coords):
    num_patches, embedding_dim = pos_embeds.size()
    grid_size = int(num_patches ** 0.5)
    assert grid_size * grid_size == num_patches, f"pos_embeds must be a square grid. Got {num_patches} patches."
    pos_embeds = pos_embeds.view(grid_size, grid_size, embedding_dim)

    coords = coords.view(-1, 2)
    coords = (coords + 1) * (grid_size - 1) / 2
    coords_floor = torch.floor(coords)
    delta = coords - coords_floor

    x0y0 = coords_floor.long().clamp(0, grid_size - 1)
    x1y1 = (x0y0 + 1).clamp(0, grid_size - 1)

    w00 = (1 - delta[:, 0:1]) * (1 - delta[:, 1:2])
    w01 = (1 - delta[:, 0:1]) * delta[:, 1:2]
    w10 = delta[:, 0:1] * (1 - delta[:, 1:2])
    w11 = delta[:, 0:1] * delta[:, 1:2]

    embed00 = pos_embeds[x0y0[:, 1], x0y0[:, 0]]
    embed01 = pos_embeds[x1y1[:, 1], x0y0[:, 0]]
    embed10 = pos_embeds[x0y0[:, 1], x1y1[:, 0]]
    embed11 = pos_embeds[x1y1[:, 1], x1y1[:, 0]]

    w00 = w00.view(-1, 1)
    w01 = w01.view(-1, 1)
    w10 = w10.view(-1, 1)
    w11 = w11.view(-1, 1)

    interpolated = w00 * embed00 + w01 * embed01 + w10 * embed10 + w11 * embed11
    return interpolated
