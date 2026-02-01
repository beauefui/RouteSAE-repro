#!/bin/bash
# 下载完整 OpenWebText2 数据集

set -e
source scripts/env_setup.sh

# 初始化 conda（解决脚本中 conda activate 问题）
eval "$(conda shell.bash hook)"
conda activate routesaerepro

echo "=========================================="
echo "Downloading OpenWebText2"
echo "=========================================="
echo "注意: 完整数据集约 12GB，需要一些时间"
echo "=========================================="

python << 'EOF'
from datasets import load_dataset
import json
import os

output_dir = os.environ.get('DATA_DIR', '/mnt/nfs/zijie/datasets/openwebtext2')
os.makedirs(output_dir, exist_ok=True)

print("Loading OpenWebText dataset...")
# 使用 Skylion007/openwebtext，这是更易用的版本
ds = load_dataset('Skylion007/openwebtext', split='train')

print(f"Total examples: {len(ds)}")

# 保存为 JSONL 格式
output_file = os.path.join(output_dir, 'train.jsonl')
print(f"Saving to {output_file}...")

with open(output_file, 'w') as f:
    for i, example in enumerate(ds):
        text = example['text']
        if len(text) > 100:  # 过滤太短的文本
            f.write(json.dumps({'text': text}) + '\n')
        
        if i % 100000 == 0:
            print(f"Processed {i:,} examples...")

print(f"✓ Saved to {output_file}")
EOF

echo "=========================================="
echo "Download complete!"
echo "Data saved to: $DATA_DIR/train.jsonl"
echo "=========================================="
