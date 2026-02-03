#!/bin/bash
# ============================================================
# 评估并对比 TopK, RouteSAE, MLSAE, RandomK 脚本
# ============================================================

set -e

# ============================================================
# 环境设置
# ============================================================
source scripts/env_setup.sh
export WANDB_MODE=offline

# ============================================================
# 路径设置
# ============================================================
MODEL_PATH=/mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(ls /mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/ | head -1)
OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs/SAE_models

# 模型路径
TOPK_MODEL="${OUTPUT_DIR}/TopK_openwebtext2.pt"
ROUTESAE_MODEL="${OUTPUT_DIR}/RouteSAE_openwebtext2.pt"
MLSAE_MODEL="${OUTPUT_DIR}/MLSAE_openwebtext2.pt"
VANILLA_MODEL="${OUTPUT_DIR}/Vanilla_openwebtext2.pt"
GATED_MODEL="${OUTPUT_DIR}/Gated_openwebtext2.pt"
JUMPRELU_MODEL="${OUTPUT_DIR}/JumpReLU_openwebtext2.pt"
CROSSCODER_MODEL="${OUTPUT_DIR}/Crosscoder_openwebtext2.pt"
# RandomK 使用 TopK 的权重
RANDOM_MODEL="${OUTPUT_DIR}/TopK_openwebtext2.pt"

# 评估结果文件
RESULTS_FILE="$PROJECT_ROOT/docs/evaluation_results.md"

# ============================================================
# 激活环境
# ============================================================
cd $PROJECT_ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate routesaerepro

echo "=============================================="
echo "Evaluating SAE Models"
echo "=============================================="

# 通用评估函数
evaluate_model() {
    local MODEL_TYPE=$1
    local SAE_PATH=$2
    local MODEL_LABEL=$3
    local EXTRA_ARGS=$4

    echo ""
    echo ">>> Evaluating $MODEL_LABEL..."

    local NORMMSE=$(python evaluate.py \
        --model $MODEL_TYPE \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_DIR" \
        --SAE_path "$SAE_PATH" \
        --layer 16 \
        --n_layers 16 \
        --k 64 \
        --hidden_size 2048 \
        --latent_size 16384 \
        --batch_size 64 \
        --max_length 512 \
        --max_samples 10000 \
        --metric NormMSE \
        --device $CUDA_VISIBLE_DEVICES \
        --use_wandb 0 $EXTRA_ARGS 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

    echo "$MODEL_LABEL NormMSE: $NORMMSE"
    eval "${MODEL_LABEL}_NORMMSE='$NORMMSE'"

    local KLDIV=$(python evaluate.py \
        --model $MODEL_TYPE \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_DIR" \
        --SAE_path "$SAE_PATH" \
        --layer 16 \
        --n_layers 16 \
        --k 64 \
        --hidden_size 2048 \
        --latent_size 16384 \
        --batch_size 64 \
        --max_length 512 \
        --max_samples 10000 \
        --metric KLDiv \
        --device $CUDA_VISIBLE_DEVICES \
        --use_wandb 0 $EXTRA_ARGS 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

    echo "$MODEL_LABEL KLDiv: $KLDIV"
    eval "${MODEL_LABEL}_KLDIV='$KLDIV'"
}

# 1. TopK
evaluate_model "TopK" "$TOPK_MODEL" "TOPK" ""

# 2. RouteSAE
evaluate_model "RouteSAE" "$ROUTESAE_MODEL" "ROUTESAE" "--aggre sum --routing hard"

# 3. MLSAE
evaluate_model "MLSAE" "$MLSAE_MODEL" "MLSAE" ""

# 4. Vanilla
evaluate_model "Vanilla" "$VANILLA_MODEL" "VANILLA" ""

# 5. Gated
evaluate_model "Gated" "$GATED_MODEL" "GATED" ""

# 6. JumpReLU
evaluate_model "JumpReLU" "$JUMPRELU_MODEL" "JUMPRELU" ""

# 7. Crosscoder
# Note: Crosscoder only supports NormMSE currently (KLDiv requires complex hooking)
echo ""
echo ">>> Evaluating CROSSCODER..."
CROSSCODER_NORMMSE=$(python evaluate.py --model Crosscoder --model_path "$MODEL_PATH" --data_path "$DATA_DIR" --SAE_path "$CROSSCODER_MODEL" --layer 16 --n_layers 16 --hidden_size 2048 --latent_size 16384 --batch_size 64 --max_length 512 --max_samples 10000 --metric NormMSE --device $CUDA_VISIBLE_DEVICES --use_wandb 0 2>&1 | grep "Evaluation complete" | awk '{print $NF}')
echo "CROSSCODER NormMSE: $CROSSCODER_NORMMSE"
CROSSCODER_KLDIV="N/A (Not Impl)"

# 8. RandomK (使用 Random 模式，它会自动选择随机层)
# 注意: Random 模式下评估器会忽略 --layer 参数，改为随机选择
evaluate_model "Random" "$RANDOM_MODEL" "RANDOM" ""

# ============================================================
# 生成对比报告
# ============================================================
echo ""
echo ">>> Generating comparison report..."
echo ""

cat > "$RESULTS_FILE" << EOF
# SAE 模型评估对比报告

> 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 评估配置

| 配置项 | 值 |
|--------|-----|
| 评估样本数 | 10,000 |
| Batch Size | 64 |
| Sequence Length | 512 |
| GPU | cuda:$CUDA_VISIBLE_DEVICES |

## 评估结果

| 指标 | TopK | RouteSAE | MLSAE | Vanilla | Gated | JumpReLU | Crosscoder | RandomK |
|------|------|----------|-------|---------|-------|----------|------------|---------|
| **NormMSE** ↓ | ${TOPK_NORMMSE:-N/A} | ${ROUTESAE_NORMMSE:-N/A} | ${MLSAE_NORMMSE:-N/A} | ${VANILLA_NORMMSE:-N/A} | ${GATED_NORMMSE:-N/A} | ${JUMPRELU_NORMMSE:-N/A} | ${CROSSCODER_NORMMSE:-N/A} | ${RANDOM_NORMMSE:-N/A} |
| **KL Divergence** ↓ | ${TOPK_KLDIV:-N/A} | ${ROUTESAE_KLDIV:-N/A} | ${MLSAE_KLDIV:-N/A} | ${VANILLA_KLDIV:-N/A} | ${GATED_KLDIV:-N/A} | ${JUMPRELU_KLDIV:-N/A} | ${CROSSCODER_KLDIV:-N/A} | ${RANDOM_KLDIV:-N/A} |


> 注: ↓ 表示越低越好

## 模型说明

1.  **TopK**: 标准 SAE，在 Layer 16 训练并评估。
2.  **RouteSAE**: 多层路由 SAE，动态选择 Layer 4-12。
3.  **MLSAE**: 多层 SAE，在所有层上训练统一的特征空间。
4.  **Vanilla**: 传统 SAE，使用 ReLU 和 L1 正则化。
5.  **Gated**: Gated SAE，分离 Gate 和 Magnitude。
6.  **JumpReLU**: JumpReLU SAE，使用与特征相关的可学习阈值。
7.  **Crosscoder**: 跨层稀疏编码器，学习多层共享特征 (NormMSE only)。
8.  **RandomK**: 使用 TopK 模型，但评估时随机选择层 (Baseline)。

## 结论

- **RouteSAE vs RandomK**: 若 RouteSAE 优于 RandomK，证明路由机制有效。
- **MLSAE vs TopK**: 若 MLSAE 表现接近 TopK，说明特征在层间具有很高的通用性。

## 模型文件

- TopK: \`${TOPK_MODEL}\`
- RouteSAE: \`${ROUTESAE_MODEL}\`
- MLSAE: \`${MLSAE_MODEL}\`
- Vanilla: \`${VANILLA_MODEL}\`
- Gated: \`${GATED_MODEL}\`
- JumpReLU: \`${JUMPRELU_MODEL}\`
- Crosscoder: \`${CROSSCODER_MODEL}\`
- RandomK: \`${RANDOM_MODEL}\` (Shared weights with TopK)
EOF

echo "=============================================="
echo "Evaluation complete!"
echo "Report saved to: $RESULTS_FILE"
echo "=============================================="
