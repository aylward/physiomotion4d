#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Clone uniGradICON (feat-add-finetuning branch required for fine-tuning support)
if [ ! -d "uniGradICON" ]; then
    git clone -b feat-add-finetuning https://github.com/uncbiag/uniGradICON.git
else
    echo "uniGradICON/ already exists, skipping clone."
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Detect venv Python path (Windows vs Linux/Mac)
if [ -f "venv/Scripts/python" ]; then
    PYTHON="venv/Scripts/python"
else
    PYTHON="venv/bin/python"
fi

# Install all dependencies (including editable physiomotion4d and uniGradICON)
"$PYTHON" -m pip install uv
"$PYTHON" -m uv pip install -e .
