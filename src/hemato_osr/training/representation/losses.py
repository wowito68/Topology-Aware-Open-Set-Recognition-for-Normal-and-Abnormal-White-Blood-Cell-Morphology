"""Losses for metric representation learning."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss over L2-normalized projections."""

    def __init__(self, temperature: float = 0.10) -> None:
        super().__init__()
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        self.temperature = float(temperature)

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return mean SupCon loss over anchors with at least one positive."""

        if projections.ndim != 2:
            msg = "projections must be a 2D tensor"
            raise ValueError(msg)
        if labels.ndim != 1 or labels.shape[0] != projections.shape[0]:
            msg = "labels must be a 1D tensor aligned with projections"
            raise ValueError(msg)
        z = F.normalize(projections, dim=1)
        batch_size = z.shape[0]
        logits = z @ z.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~eye
        denominator_mask = ~eye
        exp_logits = torch.exp(logits) * denominator_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positive_count = positive_mask.sum(dim=1)
        valid_anchor = positive_count > 0
        if not torch.any(valid_anchor):
            return z.sum() * 0.0
        mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
        return -mean_log_prob_pos[valid_anchor].mean()
