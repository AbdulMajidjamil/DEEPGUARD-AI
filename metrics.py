# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""Binary classification metrics for deepfake evaluation (label 1 = FAKE = positive class).

Pure numpy so the numbers do not depend on scikit-learn being installed; ROC-AUC is
the exact rank statistic (ties get average rank) and is cross-checked against
scikit-learn in the test-suite when scikit-learn is available.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic. NaN if one class is absent."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def binary_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, FPR, FNR and the confusion matrix.

    ``confusion_matrix`` is ``[[TN, FP], [FN, TP]]`` (rows = true REAL / FAKE).
    Undefined ratios (no positives predicted, a single class present, ...) are NaN, never 0,
    so an empty denominator cannot masquerade as a perfect or a terrible score.
    """
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("y_true and y_prob must be 1-D and the same length")
    if len(y) == 0:
        raise ValueError("cannot compute metrics on an empty set")
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    if math.isnan(precision) or math.isnan(recall):
        f1 = float("nan")
    else:
        f1 = _div(2 * precision * recall, precision + recall)
    return {
        "accuracy": _div(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(y, p),
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_fake": int((y == 1).sum()),
        "n_real": int((y == 0).sum()),
    }
