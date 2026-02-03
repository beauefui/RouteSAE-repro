#!/bin/bash
# ============================================================
# 训练 Gated SAE 脚本
# ============================================================
# 注意: 所有环境变量和设置只影响当前脚本，不影响服务器全局设置
# ============================================================

set -e

# ============================================================
# 环境设置 (仅在本脚本内生效)
# ============================================================

# HuggingFace 缓存目录 (使用已下载的模型)
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/nfs/zijie/huggingface_cache/datasets

# 只使用 GPU 2 (不影响其他用户)
export CUDA_VISIBLE_DEVICES=2

# 禁用 wandb 的网络请求
export WANDB_MODE=offline

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
MODEL_PATH=/mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(ls /mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/ | head -1)
DATA_PATH=/mnt/nfs/zijie/datasets/openwebtext2
OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs/SAE_models

# ============================================================
# 激活环境
# ============================================================
cd $PROJECT_ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate routesaerepro

echo "=============================================="
echo "Training Gated SAE"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "=============================================="

# ============================================================
# 训练参数
# ============================================================
# Gated SAE:
#   结合了 Gate (是否激活) 和 Magnitude (激活强度)。
#   同样使用 L1 正则化 (l1_coeff) 来稀疏化 latents。
# ============================================================

python train.py \
    --model Gated \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --layer 16 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --l1_coeff 5e-4 \
    --batch_size 128 \
    --max_length 512 \
    --num_epochs 1 \
    --lr 5e-4 \
    --steps 100 \
    --max_samples 100000 \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --seed 42 \
    --use_wandb 0

echo "=============================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=============================================="
