#!/bin/bash
# Double-click to start Minutes. It keeps running after this Terminal window closes.
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "Minutes is not set up yet. In Terminal, run:  bash mac/install.sh"
  read -r -p "Press Enter to close."
  exit 1
fi
.venv/bin/python run.pyw
echo "Minutes is running (Ctrl+Option+M shows or hides it). You can close this window."
