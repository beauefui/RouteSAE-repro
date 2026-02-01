#!/bin/bash
# 下载 Llama-3.2-1B-Instruct 模型

set -e
source scripts/env_setup.sh

# 初始化 conda（解决脚本中 conda activate 问题）
eval "$(conda shell.bash hook)"
conda activate routesaerepro

echo "=========================================="
echo "Downloading Llama-3.2-1B-Instruct"
echo "=========================================="

python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("✓ Tokenizer downloaded")

print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Model downloaded")

print("\nModel Config:")
print(f"  hidden_size: {model.config.hidden_size}")
print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
print(f"  vocab_size: {model.config.vocab_size}")
EOF

echo "=========================================="
echo "Download complete!"
echo "=========================================="