#!/bin/bash
# ============================================================
# RouteSAE Interpret 脚本
# ============================================================

set -e

# ============================================================
# 加载环境 (包含 OPENAI_API_KEY)
# ============================================================
source scripts/env_setup.sh

# ============================================================
# 环境变量
# ============================================================
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline

# ============================================================
# 路径
# ============================================================
PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs

# 输入: 之前 apply.sh 生成的上下文文件
# 假设 apply.sh 生成的是 RouteSAE_openwebtext2_contexts.json
# 如果文件名不同，请修改这里
CONTEXT_FILE="${OUTPUT_DIR}/contexts/RouteSAE_openwebtext2_contexts.json"

# SAE 模型 (仅用于命名输出文件)
SAE_NAME="RouteSAE_openwebtext2.pt"
SAE_PATH="${OUTPUT_DIR}/SAE_models/${SAE_NAME}"

# 检查 Context 文件是否存在
if [ ! -f "$CONTEXT_FILE" ]; then
    echo "Error: Context file not found at $CONTEXT_FILE"
    echo "Please run scripts/apply.sh first."
    exit 1
fi

# ============================================================
# 运行
# ============================================================
cd $PROJECT_ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate routesaerepro

echo "=============================================="
echo "Interpreting Features with GPT-4"
echo "=============================================="
echo "Contexts: $CONTEXT_FILE"
echo "Output:   $OUTPUT_DIR/interpret"
echo "=============================================="

# 解析 API 配置
# 如果 env_setup.sh 中没有设置 API 参数，这里需要手动指定
# --api_key $OPENAI_API_KEY
# --engine gpt-4o

python interpret.py \
    --model RouteSAE \
    --model_path "meta-llama/Llama-3.2-1B-Instruct" \
    --SAE_path "$SAE_PATH" \
    --data_path "$CONTEXT_FILE" \
    --output_dir "$OUTPUT_DIR/interpret" \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --engine gpt-4o \
    --device cuda:0

echo "=============================================="
echo "Done!"
echo "Interpretation saved to $OUTPUT_DIR/interpret"
echo "=============================================="
