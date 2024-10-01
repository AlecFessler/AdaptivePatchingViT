import torch
import torch.nn as nn

class AdaptivePatchLoss(nn.Module):
    def __init__(
        self,
        lower_quantile=0.25,
    ):
        super().__init__()
        self.lower_quantile = lower_quantile

    def forward(self, attention_weights):
        threshold = torch.quantile(attention_weights, self.lower_quantile, dim=-1, keepdim=True)
        penalty = torch.relu(threshold - attention_weights) * 1000
        penalty_sum = penalty.sum(dim=-1) / attention_weights.size(-1)
        attn_loss = penalty_sum.mean()
        return attn_loss
