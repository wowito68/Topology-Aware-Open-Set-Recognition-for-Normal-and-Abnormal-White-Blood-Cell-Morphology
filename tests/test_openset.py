from __future__ import annotations

import numpy as np

from hemato_osr.openset.scorers import (
    EnergyScore,
    MahalanobisScorer,
    MaximumSoftmaxProbability,
    PredictiveEntropy,
)


def test_openset_scores_have_expected_shape() -> None:
    logits = np.asarray([[6.0, 0.1], [0.0, 0.0], [0.1, 5.0]])

    assert MaximumSoftmaxProbability().score(logits).shape == (3,)
    assert PredictiveEntropy().score(logits).shape == (3,)
    assert EnergyScore().score(logits).shape == (3,)
    msp = MaximumSoftmaxProbability().score(logits)
    assert msp[1] > msp[0]


def test_mahalanobis_scorer() -> None:
    embeddings = np.asarray([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    labels = np.asarray([0, 0, 1, 1])
    scorer = MahalanobisScorer().fit(np.zeros((4, 2)), embeddings, labels)

    scores = scorer.score(np.zeros((2, 2)), np.asarray([[0.0, 0.0], [8.0, 8.0]]))

    assert scores.shape == (2,)
    assert scores[1] > scores[0]
