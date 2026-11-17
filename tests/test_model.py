# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI (see DERIVATION.md).
"""Unit tests that do not require ImageNet weights or a GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import BinaryFocalLoss, DeepfakeDetector  # noqa: E402


def test_forward_shape_resnet18():
    model = DeepfakeDetector(
        backbone="resnet18",
        lstm_hidden=64,
        lstm_layers=1,
        dropout=0.0,
        pretrained=False,
    )
    model.eval()
    x = torch.randn(2, 6, 3, 224, 224)
    with torch.no_grad():
        logits = model(x, chunk_size=2)
    assert logits.shape == (2,), logits.shape
    assert torch.isfinite(logits).all()


def test_predict_proba_range():
    model = DeepfakeDetector(
        backbone="resnet18", lstm_hidden=32, lstm_layers=1, dropout=0.0, pretrained=False
    )
    model.eval()
    x = torch.randn(1, 4, 3, 224, 224)
    with torch.no_grad():
        p = model.predict_proba(x, chunk_size=2)
    assert p.shape == (1,)
    assert 0.0 <= float(p) <= 1.0


def test_focal_loss_easy_vs_hard():
    loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.75, reduction="none")
    targets = torch.tensor([1.0, 1.0])
    easy = torch.tensor([8.0, 8.0])     # very confident fake
    hard = torch.tensor([0.1, 0.1])     # barely fake
    easy_l = loss_fn(easy, targets)
    hard_l = loss_fn(hard, targets)
    assert torch.all(easy_l < hard_l)


def test_focal_loss_matches_bce_at_gamma_zero():
    logits = torch.tensor([0.4, -1.2, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    focal = BinaryFocalLoss(gamma=0.0, alpha=0.5, reduction="mean")
    # α=0.5, γ=0 → 0.5 * BCE
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    got = focal(logits, targets)
    assert torch.allclose(got, 0.5 * bce, atol=1e-6)


def test_focal_defaults():
    fl = BinaryFocalLoss()
    assert fl.gamma == 2.0
    assert fl.alpha == 0.75


if __name__ == "__main__":
    test_forward_shape_resnet18()
    test_predict_proba_range()
    test_focal_loss_easy_vs_hard()
    test_focal_loss_matches_bce_at_gamma_zero()
    test_focal_defaults()
    print("all tests passed")
