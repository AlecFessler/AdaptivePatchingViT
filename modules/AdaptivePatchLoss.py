import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptivePatchLoss(nn.Module):
    def __init__(
        self,
        attn_temperature=1.0,
        top_k_focus=5,
        rand_samples=5,
        attn_loss_weight=1.0,
        diversity_loss_weight=1.0,
    ):
        super().__init__()
        self.temperature = attn_temperature
        self.top_k = top_k_focus
        self.rand_samples = rand_samples
        self.attn_loss_weight = attn_loss_weight
        self.diversity_loss_weight = diversity_loss_weight

    def forward(self, attention_weights, fixed_pos_embeds, interpolated_pos_embeds):
        scaled_attention = torch.pow(attention_weights, 1.0 / self.temperature)
        scaled_attention /= scaled_attention.sum(dim=-1, keepdim=True) + 1e-9

        attn_loss = self.compute_attention_loss(scaled_attention)

        rand_pos_embeds = self.random_sample_pos_embeds(interpolated_pos_embeds, self.rand_samples)
        diversity_loss = self.compute_diversity_loss(fixed_pos_embeds, rand_pos_embeds)

        return attn_loss * self.attn_loss_weight + diversity_loss * self.diversity_loss_weight

    def compute_attention_loss(self, scaled_attention):
        top_k_attention, _ = torch.topk(scaled_attention, k=self.top_k, dim=-1)
        focus_score = torch.mean(top_k_attention, dim=-1)
        mean_attention = scaled_attention.mean(dim=-1, keepdim=True)
        below_mean_attention = F.relu(mean_attention - scaled_attention)
        usage_score = torch.mean(below_mean_attention, dim=-1)
        attn_loss_score = usage_score * (1 - focus_score)
        return torch.mean(attn_loss_score)

    def compute_diversity_loss(self, fixed_pos_embeds, interpolated_pos_embeds):
        fixed_sim = self.cosine_similarity(fixed_pos_embeds)
        interp_sim = self.cosine_similarity(interpolated_pos_embeds)
        fixed_mean = self.compute_mean_similarity(fixed_sim)
        interp_mean = self.compute_mean_similarity(interp_sim)
        return torch.abs(fixed_mean - interp_mean)

    @staticmethod
    def random_sample_pos_embeds(pos_embeds, num_samples):
        num_patches = pos_embeds.shape[1]
        indices = torch.randperm(num_patches, device=pos_embeds.device)[:num_samples]
        return pos_embeds[:, indices]

    @staticmethod
    def cosine_similarity(embeddings):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        elif embeddings.dim() > 3:
            embeddings = embeddings.view(-1, *embeddings.shape[-2:])
        normalized_embeds = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.bmm(normalized_embeds, normalized_embeds.transpose(1, 2))
        return similarity

    @staticmethod
    def compute_mean_similarity(similarity_matrix):
        if similarity_matrix.dim() == 2:
            similarity_matrix = similarity_matrix.unsqueeze(0)
        batch_size, num_items, _ = similarity_matrix.shape
        mask = ~torch.eye(num_items, dtype=torch.bool, device=similarity_matrix.device).unsqueeze(0)
        mean_sim = similarity_matrix.masked_select(mask).view(batch_size, -1).mean(dim=1)
        return mean_sim.mean()
