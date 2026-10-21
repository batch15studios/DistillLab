#!/bin/bash
# Headless training script for batchcode
set -e

echo "=== Setting up Unsloth Headless Environment ==="
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --quiet --no-deps trl peft accelerate bitsandbytes

echo "=== Running train_batchcode.py ==="
python train_batchcode.py \
  --dataset data/cs_programming_dataset.jsonl \
  --output-name batchcode \
  --quant-method q4_k_m \
  --max-steps 60

echo "=== Training & GGUF Quantization Finished ==="
ls -lh batchcode/
