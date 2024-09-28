import torch
import torch.nn as nn
import math

class ScaledAttnEntropyLoss(nn.Module):
    def __init__(self, target_entropy=0.7, temperature=0.1, below_mean_penalty_weight=2.0, diversity_threshold=0.95):
        super().__init__()
        self.target_entropy = target_entropy
        self.temperature = temperature
        self.below_mean_penalty_weight = below_mean_penalty_weight
        self.diversity_threshold = diversity_threshold

    def forward(self, attention_weights):
        attn_weights_sum = torch.eye(attention_weights[0].size(-1), device=attention_weights[0].device)
        for layer_attn in attention_weights:
            attn_layer_mean = layer_attn.mean(dim=1)
            attn_with_residual = attn_layer_mean + torch.eye(attn_layer_mean.size(-1), device=attn_layer_mean.device)
            attn_with_residual /= attn_with_residual.sum(dim=-1, keepdim=True)
            attn_weights_sum = torch.matmul(attn_weights_sum, attn_with_residual)

        cls_to_patches = attn_weights_sum[:, 0, 1:]
        num_patches = cls_to_patches.size(-1)

        scaled_attention = cls_to_patches ** (1 / self.temperature)
        scaled_attention /= scaled_attention.sum(dim=-1, keepdim=True)

        entropy = -(scaled_attention * torch.log(scaled_attention + 1e-9)).sum(dim=-1).mean()
        max_entropy = math.log(num_patches)
        normalized_entropy = entropy / max_entropy

        entropy_loss = torch.abs(normalized_entropy - self.target_entropy)

        mean_attention = scaled_attention.mean(dim=-1, keepdim=True)
        below_mean_mask = (scaled_attention < mean_attention).float()
        below_mean_diff = (mean_attention - scaled_attention) * below_mean_mask
        below_mean_penalty = torch.sqrt(below_mean_diff.mean())

        sorted_attention, _ = torch.sort(scaled_attention, descending=True)
        cumulative_attention = torch.cumsum(sorted_attention, dim=-1)
        attention_diversity = (cumulative_attention < self.diversity_threshold).float().sum(dim=-1) / num_patches

        diversity_factor = 1 - attention_diversity

        unscaled_loss = entropy_loss + self.below_mean_penalty_weight * below_mean_penalty

        loss = unscaled_loss * diversity_factor

        return loss