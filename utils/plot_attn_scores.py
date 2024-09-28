import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

def plot_attention_scores(
    attn_weights,
    patches,
    translation_params,
    patch_size,
    channels,
    rollout=True,
    output_path=None,
    resize_dim=(64, 64)
):
    """
    Plots the attention scores across patches using matplotlib.

    Args:
        attn_weights (list): List of attention weights for each transformer layer. Each element should be (batch_size, num_heads, num_patches + 1, num_patches + 1).
        patches (torch.Tensor): Tensor of selected patches. Shape: (batch_size, num_patches, channels, patch_size, patch_size).
        translation_params (torch.Tensor): Tensor of translation parameters for patches. Shape: (batch_size, num_patches, 2).
        patch_size (int): Size of each patch.
        channels (int): Number of channels in each patch.
        rollout (bool): Whether to perform attention rollout for aggregated attention visualization.
        output_path (str, optional): Path to save the output plot. If None, the plot will be shown directly.
        resize_dim (tuple): Resize dimensions for displaying patches.
    """
    # Perform attention rollout if requested
    if rollout:
        attn_weights_sum = torch.eye(attn_weights[0].size(-1)).to(attn_weights[0].device)  # Start with identity
        for layer_attn in attn_weights:
            # Average over heads and add identity to simulate residual connections
            attn_layer_mean = layer_attn.mean(dim=1)
            attn_with_residual = attn_layer_mean + torch.eye(attn_layer_mean.size(-1)).to(attn_layer_mean.device)
            # Normalize rows
            attn_with_residual /= attn_with_residual.sum(dim=-1, keepdim=True)
            attn_weights_sum = torch.matmul(attn_weights_sum, attn_with_residual)
        attn_map = attn_weights_sum[:, 0, 1:]  # Get attention from the class token to all patches
    else:
        attn_map = attn_weights[-1].mean(dim=1)[:, 0, 1:]  # Use the last layer's average head attention

    attn_map = attn_map.cpu().detach().numpy()  # Convert to numpy array for plotting

    # Sort patches using the same heuristic as in save_patch_grid (based on translation_params)
    sorted_indices = torch.argsort(translation_params[:, 1] * 2 + translation_params[:, 0]).cpu().numpy()
    attn_map = attn_map[:, sorted_indices]

    # Set up the plot
    batch_size = patches.size(0)
    fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))
    if batch_size == 1:
        axs = [axs]  # Ensure axs is always iterable

    # Resize patches for better visualization
    resize = transforms.Resize(resize_dim)
    resized_patches = torch.stack([resize(patch) for patch in patches.view(-1, channels, patch_size, patch_size)])

    for img_num in range(batch_size):
        ax = axs[img_num]
        attention = attn_map[img_num]
        num_patches = attention.shape[0]

        # Plot attention scores as a heatmap
        im = ax.imshow(attention.reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches))), cmap='viridis')

        # Add patch labels (either patch images or their indices)
        for idx, patch_idx in enumerate(sorted_indices):
            y, x = divmod(idx, int(np.sqrt(num_patches)))
            label = f"{patch_idx}"  # Use the flattened index as label by default
            ax.text(x, y, label, ha='center', va='center', color='white', fontsize=6)

        ax.set_title(f"Image {img_num + 1}")
        ax.axis('off')

    fig.colorbar(im, ax=axs, orientation='horizontal', fraction=0.02, pad=0.05)

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
