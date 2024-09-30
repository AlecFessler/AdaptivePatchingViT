import torch
import torch.nn as nn

class AdaptivePatchLoss(nn.Module):
    def __init__(
        self,
        attn_temperature=1.0,
        rand_samples=5,
        lower_quantile=0.25,
        attn_loss_weight=1.0,
        diversity_loss_weight=1.0,
    ):
        super().__init__()
        self.temperature = attn_temperature
        self.rand_samples = rand_samples
        self.lower_quantile = lower_quantile
        self.attn_loss_weight = attn_loss_weight
        self.diversity_loss_weight = diversity_loss_weight

    def forward(self, attention_weights, fixed_pos_embeds, interpolated_pos_embeds):
        scaled_attention = torch.pow(attention_weights, 1.0 / self.temperature)
        scaled_attention /= scaled_attention.sum(dim=-1, keepdim=True)

        attn_loss = self.compute_attention_loss(scaled_attention)

        rand_pos_embeds = self.random_sample_pos_embeds(interpolated_pos_embeds, self.rand_samples)
        diversity_loss = self.compute_diversity_loss(fixed_pos_embeds, rand_pos_embeds)

        return attn_loss * self.attn_loss_weight + diversity_loss * self.diversity_loss_weight

    def compute_attention_loss(self, scaled_attention):
        threshold = torch.quantile(scaled_attention, self.lower_quantile)
        penalty = torch.relu(threshold - scaled_attention)
        normalized_penalty = penalty / (threshold + 1e-9)
        return normalized_penalty.mean()

    def compute_diversity_loss(self, fixed_pos_embeds, interpolated_pos_embeds):
        fixed_dist = self.euclidean_distance(fixed_pos_embeds)
        interp_dist = self.euclidean_distance(interpolated_pos_embeds)
        fixed_mean_dist = self.compute_mean_distance(fixed_dist)
        interp_mean_dist = self.compute_mean_distance(interp_dist)
        return torch.abs(fixed_mean_dist - interp_mean_dist)

    @staticmethod
    def random_sample_pos_embeds(pos_embeds, num_samples):
        num_patches = pos_embeds.shape[1]
        indices = torch.randperm(num_patches, device=pos_embeds.device)[:num_samples]
        return pos_embeds[:, indices]

    @staticmethod
    def euclidean_distance(embeddings):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 3:
            embeddings = embeddings.view(-1, *embeddings.shape[-2:])
        diff = embeddings.unsqueeze(2) - embeddings.unsqueeze(1)
        distances = torch.norm(diff, p=2, dim=-1)
        return distances

    @staticmethod
    def compute_mean_distance(distance_matrix):
        if distance_matrix.dim() == 2:
            distance_matrix = distance_matrix.unsqueeze(0)
        batch_size, num_items, _ = distance_matrix.shape
        mask = ~torch.eye(num_items, dtype=torch.bool, device=distance_matrix.device).unsqueeze(0)
        mean_dist = distance_matrix.masked_select(mask).view(batch_size, -1).mean(dim=1)
        return mean_dist.mean()
