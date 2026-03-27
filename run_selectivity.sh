#!/bin/bash
# Run selectivity experiment on RunPod
# Usage: bash run_selectivity.sh qwen7b-instruct

set -e

MODEL="${1:?Usage: bash run_selectivity.sh <model-key>}"

# Install if needed
pip install -e . 2>/dev/null || true

# Login to HuggingFace (set HUGGINGFACE_TOKEN in env)
python -c "from huggingface_hub import login; import os; login(token=os.environ['HUGGINGFACE_TOKEN'])" 2>/dev/null

# Run
python compute_selectivity.py "$MODEL"
