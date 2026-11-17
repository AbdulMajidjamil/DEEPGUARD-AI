#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""Evaluate a trained checkpoint on a labelled test split and write ``weights/eval_report.json``.

    python scripts/evaluate.py --data data/datasets --split test

The report is what the dashboard's Model Information page shows. It is only produced from a
checkpoint that passes strict validation, on videos that really are in the split - there is
no fallback that invents numbers. Evaluating on data that overlaps training data inflates
the figures; the leakage check below compares against ``--train-split`` when it exists.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint import CheckpointError, load_checkpoint  # noqa: E402
from config import EVAL_REPORT_PATH, SEQ_LEN, VERSION, WEIGHTS_PATH  # noqa: E402
from dataprep import VideoFaceDataset, check_leakage, class_counts, collate, scan_split, write_json  # noqa: E402
from metrics import binary_metrics  # noqa: E402
from model import DeepfakeDetector  # noqa: E402
from train import collect_predictions  # noqa: E402

logger = logging.getLogger("deepguard.evaluate")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a DEEPGUARD AI checkpoint")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "datasets")
    ap.add_argument("--split", default="test")
    ap.add_argument("--train-split", default="train", help="split used for the leakage check")
    ap.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--backbone", default="resnext50", choices=["resnext50", "efficientnet_b3", "resnet18"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--dataset-name", default=None, help="label shown next to the metrics, e.g. 'Celeb-DF v2 test'")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    ap.add_argument("--out", type=Path, default=EVAL_REPORT_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-leakage", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.weights.is_file():
        logger.error("No checkpoint at %s - nothing to evaluate. Train one first (scripts/train.py).", args.weights.name)
        return 2
    model = DeepfakeDetector(backbone=args.backbone, pretrained=False)
    try:
        info = load_checkpoint(model, args.weights, backbone=args.backbone)
    except CheckpointError as exc:
        logger.error("Checkpoint rejected: %s", exc)
        return 2
    device = torch.device(args.device)
    model.to(device).eval()

    items = scan_split(args.data, args.split)
    counts = class_counts(items)
    if not counts["real"] or not counts["fake"]:
        logger.error("Split %r needs videos in both real/ and fake/ (found %s).", args.split, counts)
        return 2
    train_items = scan_split(args.data, args.train_split)
    if train_items and args.train_split != args.split:
        leak = check_leakage(train_items, items, args.split)
        for e in leak["errors"]:
            logger.error("LEAKAGE: %s", e)
        if leak["errors"] and not args.allow_leakage:
            logger.error("Test data overlaps training data; the metrics would be inflated. Aborting.")
            return 2

    ds = VideoFaceDataset(items, args.cache, args.seq_len).prepare()
    if ds.skipped:
        logger.warning("%d video(s) skipped (unreadable / too few faces)", len(ds.skipped))
    used = class_counts(ds.items)
    if not used["real"] or not used["fake"]:
        logger.error("After face extraction one class is empty (%s).", used)
        return 2
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    pred = collect_predictions(model, loader, device)
    m = binary_metrics(pred["labels"], pred["probs"], args.threshold)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deepguard_version": VERSION,
        "checkpoint": {k: info.get(k) for k in ("sha256", "epoch", "backbone", "trained_by_deepguard")},
        "dataset": args.dataset_name or f"{args.data.name}/{args.split}",
        "split": args.split,
        "n_samples": m["n"],
        "n_skipped": len(ds.skipped),
        "metrics": {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr")},
        "confusion_matrix": m["confusion_matrix"],
        "threshold": m["threshold"],
        "class_counts": used,
    }
    write_json(args.out, report)
    logger.info("accuracy=%.4f f1=%.4f auc=%.4f fpr=%.4f fnr=%.4f (n=%d) -> %s",
                m["accuracy"], m["f1"], m["roc_auc"], m["fpr"], m["fnr"], m["n"], args.out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
