#!/bin/bash
# parcae-srpo: one-command setup → test → train
set -euo pipefail

MODEL="google/gemma-4-E4B-it"
PYTHON=${PYTHON:-python3}
NPROC=${NPROC:-2}

echo "=== Installing ==="
pip install -e ".[test]" -q

echo "=== Downloading model ==="
$PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', ignore_patterns=['*.md', '.gitattributes'])
"

echo "=== Identity test ==="
$PYTHON -m pytest tests/test_identity.py -v

echo "=== Context manager tests ==="
$PYTHON -m pytest tests/test_old_policy.py -v

echo "=== Training ==="
if [ "${1:-}" = "ddp" ]; then
    echo "DDP mode: $NPROC GPUs"
    torchrun --nproc_per_node=$NPROC scripts/train_srpo.py
else
    echo "Single-GPU mode. For DDP: bash scripts/launch.sh ddp"
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_srpo.py
fi
