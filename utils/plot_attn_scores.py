import torch
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_attention_scores(
    attn_weights,
    translation_params,
    output_path=None
):
    """
    Plots the attention scores for a single image's patches using matplotlib.
    Args:
        attn_weights (torch.Tensor): Tensor of attention weights. Could be from a single transformer layer.
                                     Expected shapes: (num_patches,), (num_heads, num_patches, num_patches), or similar.
        translation_params (torch.Tensor): Tensor of translation parameters for patches. Shape: (num_patches, 2).
        output_path (str, optional): Path to save the output plot. If None, the plot will be shown directly.
    """
    # Determine how to handle attn_weights based on its shape
    if attn_weights.dim() == 1:
        # If attn_weights is 1D, it represents attention scores for the patches
        attn_map = attn_weights
    elif attn_weights.dim() == 2:
        # If attn_weights is 2D, assume it's already averaged
        attn_map = attn_weights[0]
    elif attn_weights.dim() == 3:
        # If attn_weights is 3D, average over heads
        attn_map = attn_weights.mean(dim=0)[0]
    else:
        raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

    attn_map = attn_map.cpu().detach().numpy()

    # Sort patches using the same heuristic as in save_patch_grid
    sorted_indices = torch.argsort(translation_params[:, 1] * 2 + translation_params[:, 0]).cpu().numpy()

    # Make sure that sorted_indices does not exceed the number of available patches
    if len(sorted_indices) > attn_map.shape[0]:
        sorted_indices = sorted_indices[:attn_map.shape[0]]

    attn_map = attn_map[sorted_indices]

    num_patches = attn_map.shape[0]
    grid_size = math.ceil(math.sqrt(num_patches))

    # Pad and reshape attention map
    padded_attention = np.pad(attn_map, (0, grid_size**2 - num_patches), mode='constant', constant_values=np.nan)
    reshaped_attention = padded_attention.reshape(grid_size, grid_size)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot attention scores as a heatmap
    im = ax.imshow(reshaped_attention, cmap='viridis')

    # Add patch labels
    for idx in range(num_patches):
        y, x = divmod(idx, grid_size)
        patch_idx = sorted_indices[idx]
        label = f"{patch_idx}"
        ax.text(x, y, label, ha='center', va='center', color='white', fontsize=6)

    ax.set_title("Attention Scores")
    ax.axis('off')

    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
