#!/bin/bash
# Setup venv for gsm8k_custom_reward example
# Run from the repo root: bash examples/gsm8k_custom_reward/setup_venv.sh

set -euxo pipefail

cd "$(git rev-parse --show-toplevel)"

uv venv
source .venv/bin/activate
uv pip install -e ".[test,sglang]"
uv pip install "datasets>=3.0" flash-attn cachetools

# Align rollout-serving runtime versions with verl-playwright.
uv pip install \
  "ray==2.53.0" \
  "fastapi==0.129.0" \
  "uvicorn==0.40.0"
