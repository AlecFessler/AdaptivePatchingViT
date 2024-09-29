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
        num_layers, batch_size, num_tokens, _ = attention_weights.shape
        num_patches = num_tokens - 1  # Subtract 1 for CLS token

        # Initialize identity matrix for rollout
        attention_rollout = torch.eye(num_tokens, device=attention_weights.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Perform attention rollout
        for layer in range(num_layers):
            layer_attn = attention_weights[layer]
            attention_rollout = torch.bmm(attention_rollout, layer_attn)
            attention_rollout = attention_rollout / attention_rollout.sum(dim=-1, keepdim=True)

        # Extract CLS token attention to patches
        cls_to_patches = attention_rollout[:, 0, 1:]

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
        
        sorted_attention, _ = torch.sort(scaled_attention, descending=True, dim=-1)
        cumulative_attention = torch.cumsum(sorted_attention, dim=-1)
        attention_diversity = (cumulative_attention < self.diversity_threshold).float().sum(dim=-1) / num_patches
        diversity_factor = 1 - attention_diversity
        
        unscaled_loss = entropy_loss + self.below_mean_penalty_weight * below_mean_penalty
        loss = (unscaled_loss * diversity_factor).mean()
        
        return loss
