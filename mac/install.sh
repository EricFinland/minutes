#!/bin/bash
# Sets up Minutes on macOS: a private Python environment, the dependencies and the
# speech model. Run once from Terminal:  bash mac/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$major" -lt 13 ]; then
  echo "Minutes needs macOS 13 (Ventura) or newer to capture call audio."
  exit 1
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
      PY="$c"
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "Minutes needs Python 3.10 or newer. Install it with:  brew install python@3.12"
  exit 1
fi

echo "Using $($PY --version) at $(command -v "$PY")"
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "Downloading the speech model (about 1.6 GB, one time only)..."
.venv/bin/python -c "from minutes import config; from minutes.engine import get_model; get_model(config.load())"

chmod +x mac/Minutes.command
echo
echo "Done. Double-click mac/Minutes.command to start Minutes."
echo "The first time, macOS asks for Microphone and Screen & System Audio Recording"
echo "access. Allow both, then quit Minutes (... menu > Quit) and start it again."
