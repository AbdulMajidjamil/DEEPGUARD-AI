#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Original work: SENTINEL by Dev Parth (Apache-2.0).
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: dataset validation and train/val leakage
# checks, safe cache, class-imbalance handling, seeds, run logs, resume, early stopping,
# full metrics, no silent overwrite of existing weights, checkpoint metadata.
"""Train the DEEPGUARD AI detector (ResNeXt-50 + LSTM + Binary Focal Loss).

Expected layout (``scripts/validate_dataset.py`` checks it first)::

    <data>/train/{real,fake}/*.mp4
    <data>/val/{real,fake}/*.mp4

Usage::

    python scripts/train.py --data data/datasets --epochs 20 --backbone resnext50

Nothing is trained until you run this with your own, legitimately licensed dataset.
The checkpoint refuses to overwrite an existing file unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_kwargs):
        return x

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SEQ_LEN, VERSION as APP_VERSION, WEIGHTS_PATH  # noqa: E402
from dataprep import (  # noqa: E402
    VideoFaceDataset, check_leakage, class_counts, collate, scan_split, write_json,
)
from metrics import binary_metrics  # noqa: E402
from model import BinaryFocalLoss, DeepfakeDetector  # noqa: E402

logger = logging.getLogger("deepguard.train")


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_predictions(model, loader, device, loss_fn=None) -> dict:
    """Run ``model`` over ``loader``; return labels, P(fake) and the mean loss."""
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    total, n = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits = model(x, lengths=lengths)
        if loss_fn is not None:
            total += float(loss_fn(logits, y).item()) * x.size(0)
        n += x.size(0)
        probs.extend(torch.sigmoid(logits).view(-1).cpu().tolist())
        labels.extend(int(v) for v in y.view(-1).cpu().tolist())
    return {"labels": labels, "probs": probs, "loss": total / max(n, 1)}


def set_backbone_trainable(model: DeepfakeDetector, trainable: bool) -> None:
    for p in model.spatial.parameters():
        p.requires_grad = trainable


def make_optimizer(model, lr: float, backbone_lr: float, backbone_on: bool):
    backbone_params = [p for p in model.spatial.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("spatial.")]
    groups = [{"params": head_params, "lr": lr}]
    if backbone_on and backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def build_sampler(items) -> WeightedRandomSampler:
    """Draw classes with equal probability so the minority class is not drowned out."""
    counts = class_counts(items)
    weight = {0: 1.0 / max(counts["real"], 1), 1: 1.0 / max(counts["fake"], 1)}
    return WeightedRandomSampler([weight[i.label] for i in items], num_samples=len(items), replacement=True)


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the DEEPGUARD AI deepfake detector")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "datasets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="epochs with the CNN backbone frozen")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--backbone", default="resnext50", choices=["resnext50", "efficientnet_b3", "resnet18"])
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet backbone weights")
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--alpha", type=float, default=0.75, help="focal loss alpha (weight of the FAKE class)")
    ap.add_argument("--balance", choices=["sampler", "none"], default="sampler",
                    help="class-imbalance handling for the training loader")
    ap.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs); 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out checkpoint")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", type=Path, default=None, help="resume from a runs/<name>/last.pt state file")
    ap.add_argument("--cache", type=Path, default=ROOT / "data" / "cache")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--allow-leakage", action="store_true", help="continue despite train/val leakage errors (not recommended)")
    args = ap.parse_args(argv)

    run_dir = args.runs / (args.run_name or time.strftime("%Y%m%d-%H%M%S"))
    if args.resume:
        run_dir = args.resume.resolve().parent
    setup_logging(run_dir)
    seed_all(args.seed)
    device = torch.device(args.device)
    logger.info("DEEPGUARD AI train v%s device=%s backbone=%s seed=%d", APP_VERSION, device, args.backbone, args.seed)

    if args.out.exists() and not args.overwrite and not args.resume:
        logger.error("%s already exists. Use --overwrite to replace it, or --out for a new file.", args.out.name)
        return 2

    train_items = scan_split(args.data, "train")
    val_items = scan_split(args.data, "val")
    for name, items in (("train", train_items), ("val", val_items)):
        counts = class_counts(items)
        logger.info("%s: %d real / %d fake", name, counts["real"], counts["fake"])
        if not counts["real"] or not counts["fake"]:
            logger.error("%s split needs videos in both real/ and fake/ folders under %s", name, args.data)
            return 2

    leak = check_leakage(train_items, val_items)
    for w in leak["warnings"]:
        logger.warning("leakage warning: %s", w)
    for e in leak["errors"]:
        logger.error("LEAKAGE: %s", e)
    if leak["errors"] and not args.allow_leakage:
        logger.error("%d train/val leakage error(s); fix the split (or pass --allow-leakage).", len(leak["errors"]))
        return 2

    train_ds = VideoFaceDataset(train_items, args.cache, args.seq_len).prepare(lambda x: tqdm(x, desc="prepare train", ncols=88))
    val_ds = VideoFaceDataset(val_items, args.cache, args.seq_len).prepare(lambda x: tqdm(x, desc="prepare val", ncols=88))
    skipped = train_ds.skipped + val_ds.skipped
    if skipped:
        logger.warning("%d video(s) skipped (unreadable or too few faces); see skipped.json", len(skipped))
        write_json(run_dir / "skipped.json", {"skipped": skipped})
    for name, ds in (("train", train_ds), ("val", val_ds)):
        counts = class_counts(ds.items)
        if not counts["real"] or not counts["fake"]:
            logger.error("after face extraction the %s split lacks one class (%s); cannot train", name, counts)
            return 2

    sampler = build_sampler(train_ds.items) if args.balance == "sampler" else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate)

    model = DeepfakeDetector(backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    loss_fn = BinaryFocalLoss(gamma=args.gamma, alpha=args.alpha)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best, bad_epochs, history = 1, float("-inf"), 0, []
    optim = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("backbone") != args.backbone:
            logger.error("resume file was written for backbone %r, not %r", state.get("backbone"), args.backbone)
            return 2
        model.load_state_dict(state["model"])
        start_epoch = int(state["epoch"]) + 1
        best, bad_epochs, history = float(state["best"]), int(state["bad_epochs"]), list(state["history"])
        backbone_on = start_epoch > args.freeze_epochs
        set_backbone_trainable(model, backbone_on)
        optim = make_optimizer(model, args.lr, args.backbone_lr, backbone_on)
        optim.load_state_dict(state["optim"])
        logger.info("resumed after epoch %d (best=%.4f)", start_epoch - 1, best)
    else:
        set_backbone_trainable(model, False)
        optim = make_optimizer(model, args.lr, args.backbone_lr, False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            optim = make_optimizer(model, args.lr, args.backbone_lr, True)
            logger.info("unfroze backbone at epoch %d", epoch)

        model.train()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=88)
        for x, y, lengths in bar:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x, lengths=lengths), y)
            if not torch.isfinite(loss):
                logger.error("non-finite loss at epoch %d; stopping so a corrupt checkpoint is not written", epoch)
                return 3
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        pred = collect_predictions(model, val_loader, device, loss_fn)
        m = binary_metrics(pred["labels"], pred["probs"])
        score = m["roc_auc"] if m["roc_auc"] == m["roc_auc"] else m["accuracy"]
        row = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_loss": pred["loss"], "val": m}
        history.append(row)
        logger.info("epoch %02d train_loss=%.4f val_loss=%.4f val_acc=%.3f val_auc=%.3f val_f1=%.3f",
                    epoch, row["train_loss"], row["val_loss"], m["accuracy"], m["roc_auc"], m["f1"])

        if score > best:
            best, bad_epochs = score, 0
            torch.save({
                "state_dict": model.state_dict(), "backbone": args.backbone, "epoch": epoch,
                "val": {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "fpr", "fnr", "n")},
                "seq_len": args.seq_len, "deepguard_version": APP_VERSION,
            }, args.out)
            logger.info("  new best (%.4f) -> %s", best, args.out.name)
        else:
            bad_epochs += 1

        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "best": best,
                    "bad_epochs": bad_epochs, "history": history, "backbone": args.backbone}, run_dir / "last.pt")
        write_json(run_dir / "metrics.json", {"args": {k: str(v) for k, v in vars(args).items()}, "history": history, "best": best})

        if args.patience and bad_epochs >= args.patience:
            logger.info("early stopping: no improvement for %d epochs", bad_epochs)
            break

    logger.info("finished. best validation score %.4f. Run files in %s", best, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
