import torch
import torchvision
from torchvision import transforms
import numpy as np

def save_patch_grid(
    patches,
    translation_params,
    output_path,
    resize_dim=(512, 512)
):
    sorted_indices = torch.argsort(translation_params[:, 1] * 2 + translation_params[:, 0])
    patches = patches[sorted_indices]
    resize = transforms.Resize(resize_dim)
    patches = torch.stack([resize(patch) for patch in patches])
    grid = torchvision.utils.make_grid(patches, nrow=int(np.sqrt(patches.size(0))))
    torchvision.utils.save_image(grid, output_path)
