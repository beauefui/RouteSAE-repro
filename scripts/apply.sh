#!/bin/bash
# ============================================================
# RouteSAE Apply (Extract Contexts) 脚本
# ============================================================

set -e

# ============================================================
# 环境变量
# ============================================================
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/nfs/zijie/huggingface_cache/datasets
export CUDA_VISIBLE_DEVICES=1  # 使用空闲 GPU
export WANDB_MODE=offline

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
# 自动寻找已下载的模型
MODEL_PATH=/mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(ls /mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/ | head -1)
DATA_PATH=/mnt/nfs/zijie/datasets/openwebtext2
OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs

# SAE 模型路径 (请根据实际训练结果修改文件名)
# 假设训练好的模型是 RouteSAE_context.pt，这里需要用户确认具体文件名
# 我这里先使用通配符或指定一个示例文件名
# 用户说 "目前我已经完成了TopK模型和RouteSAE的训练"
# 文件名通常包含参数，例如: sum_hard_K64_Llama-3.2-1B-Instruct_openwebtext2_16384.pt
# 我们假设用户会修改这里，或者我们尝试自动找一个
SAE_NAME="RouteSAE_openwebtext2.pt"
SAE_PATH="${OUTPUT_DIR}/SAE_models/${SAE_NAME}"

# 如果找不到，尝试找任意一个 RouteSAE 开头的 .pt
if [ ! -f "$SAE_PATH" ]; then
    echo "Error: SAE model not found at $SAE_PATH"
    echo "Please check the filename in scripts/apply.sh"
    exit 1
fi

# ============================================================
# 运行
# ============================================================
cd $PROJECT_ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate routesaerepro

echo "=============================================="
echo "Extracting Contexts for RouteSAE"
echo "=============================================="
echo "SAE Path: $SAE_PATH"
echo "Output:   $OUTPUT_DIR/contexts"
echo "=============================================="

# 解析 SAE_PATH 中的参数以传递给脚本 (如果文件名包含配置)
# 或者我们直接硬编码参数，只要与训练一致即可
# RouteSAE 需要 --model RouteSAE --routing hard --aggre sum 等

python apply.py \
    --model RouteSAE \
    --model_path "$MODEL_PATH" \
    --SAE_path "$SAE_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_layers 16 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --aggre sum \
    --routing hard \
    --batch_size 64 \
    --max_length 512 \
    --max_samples 10000 \
    --device cuda:0

echo "=============================================="
echo "Done!"
echo "Contexts saved to $OUTPUT_DIR/contexts"
echo "=============================================="
