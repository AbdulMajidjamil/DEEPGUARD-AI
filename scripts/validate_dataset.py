#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
"""Validate a dataset folder before training: structure, class balance, readability, leakage.

    python scripts/validate_dataset.py --data data/datasets [--probe-faces]

Exit code 0 = usable, 1 = problems found. ``--probe-faces`` additionally runs the face
pipeline on every clip (slow) and lists clips that would be skipped during training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataprep import check_leakage, class_counts, scan_split  # noqa: E402


def validate(data: Path, probe_faces: bool = False) -> tuple[list[str], list[str], dict]:
    """Return ``(errors, warnings, summary)`` for ``data``."""
    errors: list[str] = []
    warnings: list[str] = []
    splits = {s: scan_split(data, s) for s in ("train", "val", "test")}
    summary = {s: class_counts(items) for s, items in splits.items()}

    for required in ("train", "val"):
        c = summary[required]
        if not c["real"] or not c["fake"]:
            errors.append(f"split '{required}' needs videos in both real/ and fake/ (found {c['real']} real, {c['fake']} fake)")
    if not splits["test"]:
        warnings.append("no 'test' split - scripts/evaluate.py needs one for honest metrics")

    for name, items in splits.items():
        c = summary[name]
        lo, hi = sorted((c["real"], c["fake"]))
        if lo and hi / lo > 3:
            warnings.append(f"split '{name}' is imbalanced ({c['real']} real / {c['fake']} fake); training uses a balanced sampler")
        if 0 < len(items) < 20:
            warnings.append(f"split '{name}' has only {len(items)} videos - too few for meaningful metrics")

    import cv2

    for items in splits.values():
        for it in items:
            if it.path.stat().st_size == 0:
                errors.append(f"empty file: {it.split}/{it.path.name}")
                continue
            cap = cv2.VideoCapture(str(it.path))
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if not ok:
                errors.append(f"unreadable video: {it.split}/{it.path.name}")

    for other in ("val", "test"):
        if splits["train"] and splits[other]:
            leak = check_leakage(splits["train"], splits[other], other)
            errors += [f"leakage: {e}" for e in leak["errors"]]
            warnings += [f"leakage: {w}" for w in leak["warnings"]]
    if splits["val"] and splits["test"]:
        leak = check_leakage(splits["val"], splits["test"], "test")
        errors += [f"leakage (val/test): {e}" for e in leak["errors"]]

    if probe_faces:
        from config import MIN_FACES
        from pipeline import extract_face_sequence

        for items in splits.values():
            for it in items:
                try:
                    n = extract_face_sequence(it.path).n_faces
                    if n < MIN_FACES:
                        warnings.append(f"too few faces ({n}); would be skipped: {it.split}/{it.path.name}")
                except Exception as exc:
                    warnings.append(f"face extraction failed ({str(exc)[:80]}): {it.split}/{it.path.name}")
    return errors, warnings, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a DEEPGUARD AI dataset folder")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "datasets")
    ap.add_argument("--probe-faces", action="store_true")
    args = ap.parse_args(argv)
    if not args.data.is_dir():
        print(f"ERROR: dataset folder not found: {args.data}")
        return 1
    errors, warnings, summary = validate(args.data, args.probe_faces)
    for split, c in summary.items():
        print(f"{split:5s}: {c['real']:5d} real  {c['fake']:5d} fake")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:  ", e)
    print("RESULT:", "dataset has problems" if errors else "dataset looks usable")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
