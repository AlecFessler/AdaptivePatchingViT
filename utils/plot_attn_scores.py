import torch
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_attention_scores(
    attn_weights,
    translation_params,
    rollout=True,
    output_path=None
):
    """
    Plots the attention scores for a single image's patches using matplotlib.
    Args:
        attn_weights (list): List of attention weights for each transformer layer. Each element should be (1, num_heads, num_patches + 1, num_patches + 1).
        translation_params (torch.Tensor): Tensor of translation parameters for patches. Shape: (num_patches, 2).
        rollout (bool): Whether to perform attention rollout for aggregated attention visualization.
        output_path (str, optional): Path to save the output plot. If None, the plot will be shown directly.
    """
    # Perform attention rollout if requested
    if rollout:
        attn_weights_sum = torch.eye(attn_weights[0].size(-1)).to(attn_weights[0].device)
        for layer_attn in attn_weights:
            attn_layer_mean = layer_attn.mean(dim=1)
            attn_with_residual = attn_layer_mean + torch.eye(attn_layer_mean.size(-1)).to(attn_layer_mean.device)
            attn_with_residual /= attn_with_residual.sum(dim=-1, keepdim=True)
            attn_weights_sum = torch.matmul(attn_weights_sum, attn_with_residual)
        attn_map = attn_weights_sum[0, 1:]  # Get attention from the class token to all patches
    else:
        attn_map = attn_weights[-1].mean(dim=1)[0, 1:]  # Use the last layer's average head attention

    attn_map = attn_map.cpu().detach().numpy()

    # Sort patches using the same heuristic as in save_patch_grid
    sorted_indices = torch.argsort(translation_params[:, 1] * 2 + translation_params[:, 0]).cpu().numpy()
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
