# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import torch

def perturb_transform_params(transform_params, min_perturb=0.01, max_perturb=0.05, perturb_scale=False, perturb_rotate=False):
    # scale params to 0, 1 (scale already is)
    transform_params[:, :, :2] = (transform_params[:, :, :2] + 1) / 2
    transform_params[:, :, 4:5] = (transform_params[:, :, 4:5] / torch.pi + 1) / 2

    perturbs = torch.rand_like(transform_params) * (max_perturb - min_perturb) + min_perturb
    if not perturb_scale:
        perturbs[:, :, 2:4] = 0
    if not perturb_rotate:
        perturbs[:, :, 4:5] = 0

    # if param is more positive, liklihood of negative perturb is higher, and vice versa
    sign_flip_rolls = torch.rand_like(transform_params)
    signs = torch.where(sign_flip_rolls > transform_params, 1, -1)
    perturbs = perturbs * signs

    transform_params = transform_params + perturbs
    transform_params = torch.clamp(transform_params, 0, 1)

    # scale params back to original range
    transform_params[:, :, :2] = transform_params[:, :, :2] * 2 - 1
    transform_params[:, :, 4:5] = (transform_params[:, :, 4:5] * 2 - 1) * torch.pi

    return transform_params
