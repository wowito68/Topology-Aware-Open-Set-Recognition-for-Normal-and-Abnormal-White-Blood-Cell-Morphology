from __future__ import annotations

import torch

from hemato_osr.models.fusion import DeepTDAFusionMLP, FusionConfig


def test_deep_tda_fusion_shapes() -> None:
    model = DeepTDAFusionMLP(FusionConfig(deep_dim=8, tda_dim=3, num_classes=5))
    output = model(torch.randn(4, 8), torch.randn(4, 3))

    assert output["logits"].shape == (4, 5)
    assert output["embedding"].shape == (4, 11)
