#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
# DEEPGUARD AI installer for Kali / Debian / Ubuntu. Safe to re-run:
# it never deletes data, never touches weights/, never overwrites an existing .env.
#   ./install.sh            create .venv, install Python packages, prepare folders
#   ./install.sh --apt      also run sudo apt-get install for missing system packages
#   ./install.sh --cpu      install the CPU-only PyTorch wheel (smaller; recommended without a GPU)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

USE_APT=0
CPU_TORCH=0
for arg in "$@"; do
  case "$arg" in
    --apt) USE_APT=1 ;;
    --cpu) CPU_TORCH=1 ;;
    -h|--help) sed -n 5,9p "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '[install] %s\n' "$*"; }
fail() { printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || fail "python3 not found. Install it: sudo apt-get install python3 python3-venv python3-pip"
"$PY" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || fail "Python 3.10 or newer is required."

missing=()
"$PY" -c "import venv, ensurepip" >/dev/null 2>&1 || missing+=(python3-venv)
command -v redis-server >/dev/null 2>&1 || missing+=(redis-server)
command -v curl >/dev/null 2>&1 || missing+=(curl)
if ((${#missing[@]})); then
  if ((USE_APT)); then
    say "installing system packages: ${missing[*]}"
    sudo apt-get update && sudo apt-get install -y "${missing[@]}"
  else
    say "missing system packages: ${missing[*]}"
    say "install with:  sudo apt-get install -y ${missing[*]}   (or re-run ./install.sh --apt)"
    for m in "${missing[@]}"; do
      [[ "$m" == "python3-venv" ]] && fail "python3-venv is required."
    done
    say "redis-server and curl are optional: without Redis the app uses its documented in-process fallback."
  fi
fi

if [[ ! -d .venv ]]; then
  say "creating virtual environment in .venv"
  "$PY" -m venv .venv
else
  say "reusing existing .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

if ((CPU_TORCH)); then
  say "installing CPU-only PyTorch"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
say "installing Python requirements (first run downloads PyTorch: this can take several minutes)"
pip install -r requirements.txt

mkdir -p data/uploads data/results data/thumbs data/run data/logs weights
if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  say "created .env from .env.example (edit it to set API_KEY / ENVIRONMENT)"
fi

if [[ -f weights/deepfake_detector.pth ]]; then
  say "found weights/deepfake_detector.pth (kept as is; it is validated at startup)"
else
  say "NOTE: no trained checkpoint in weights/. The app will run in labelled DEMO mode."
  say "      See README 'Training' to produce one with your own licensed dataset."
fi
say "done. Start with:  ./start.sh"
