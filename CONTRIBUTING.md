# Contributing to DEEPGUARD AI

Thanks for helping. This project is a derivative of SENTINEL by Dev Parth (Apache-2.0); see `DERIVATION.md`.

## Ground rules

- **No fake results.** Do not add random or hard-coded predictions, and do not describe a model
  as trained unless a validated checkpoint exists. Anything heuristic must be labelled as such.
- **No claims of accuracy** without a reproducible evaluation (`scripts/evaluate.py`) on a named dataset.
- **Keep attribution.** Do not remove the `LICENSE`, `NOTICE` or origin headers. New files get
  `# Copyright <year> <your name>` and `# SPDX-License-Identifier: Apache-2.0`; if you change an
  inherited file, add a "Modified <year> by <name>" line to its header.
- **Never commit** datasets, private videos, model weights you may not redistribute, `.env`, keys or caches.
- Contributions are licensed under Apache-2.0.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
BACKBONE=resnet18 PRETRAINED_BACKBONE=0 python -m pytest tests -q
ruff check . --select E9,F
```

The tests use synthetic videos, a tiny backbone and no network, Redis or GPU.

## Pull requests

1. Branch from `main`; keep changes focused.
2. Add or update tests for behaviour you change. Say honestly what you did **not** test.
3. Update `CHANGELOG.md`.
4. For security issues do not open a public issue; follow `SECURITY.md`.
