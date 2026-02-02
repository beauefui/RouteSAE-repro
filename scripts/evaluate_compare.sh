#!/bin/bash
# ============================================================
# 评估并对比 TopK SAE 和 RouteSAE 脚本
# ============================================================

set -e

# ============================================================
# 环境设置
# ============================================================
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/nfs/zijie/huggingface_cache/datasets
export CUDA_VISIBLE_DEVICES=2
export WANDB_MODE=offline

# ============================================================
# 路径设置
# ============================================================
PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
MODEL_PATH=/mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/$(ls /mnt/nfs/zijie/huggingface_cache/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/ | head -1)
DATA_PATH=/mnt/nfs/zijie/datasets/openwebtext2
OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs/SAE_models

# 模型路径
TOPK_MODEL="${OUTPUT_DIR}/TopK_openwebtext2.pt"
ROUTESAE_MODEL="${OUTPUT_DIR}/RouteSAE_openwebtext2.pt"

# 评估结果文件
RESULTS_FILE="${PROJECT_ROOT}/docs/evaluation_results.md"

# ============================================================
# 激活环境
# ============================================================
cd $PROJECT_ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate routesaerepro

echo "=============================================="
echo "Evaluating SAE Models"
echo "=============================================="

# ============================================================
# 评估 TopK SAE
# ============================================================
echo ""
echo ">>> Evaluating TopK SAE..."
echo ""

TOPK_NORMMSE=$(python evaluate.py \
    --model TopK \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --SAE_path "$TOPK_MODEL" \
    --layer 16 \
    --k 64 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --batch_size 64 \
    --max_length 512 \
    --max_samples 10000 \
    --metric NormMSE \
    --device cuda:0 \
    --use_wandb 0 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

echo "TopK NormMSE: $TOPK_NORMMSE"

TOPK_KLDIV=$(python evaluate.py \
    --model TopK \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --SAE_path "$TOPK_MODEL" \
    --layer 16 \
    --k 64 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --batch_size 64 \
    --max_length 512 \
    --max_samples 10000 \
    --metric KLDiv \
    --device cuda:0 \
    --use_wandb 0 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

echo "TopK KLDiv: $TOPK_KLDIV"

# ============================================================
# 评估 RouteSAE
# ============================================================
echo ""
echo ">>> Evaluating RouteSAE..."
echo ""

ROUTESAE_NORMMSE=$(python evaluate.py \
    --model RouteSAE \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --SAE_path "$ROUTESAE_MODEL" \
    --n_layers 16 \
    --k 64 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --aggre sum \
    --routing hard \
    --batch_size 64 \
    --max_length 512 \
    --max_samples 10000 \
    --metric NormMSE \
    --device cuda:0 \
    --use_wandb 0 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

echo "RouteSAE NormMSE: $ROUTESAE_NORMMSE"

ROUTESAE_KLDIV=$(python evaluate.py \
    --model RouteSAE \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --SAE_path "$ROUTESAE_MODEL" \
    --n_layers 16 \
    --k 64 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --aggre sum \
    --routing hard \
    --batch_size 64 \
    --max_length 512 \
    --max_samples 10000 \
    --metric KLDiv \
    --device cuda:0 \
    --use_wandb 0 2>&1 | grep "Evaluation complete" | awk '{print $NF}')

echo "RouteSAE KLDiv: $ROUTESAE_KLDIV"

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
| GPU | cuda:0 (H200) |

## 模型参数

| 参数 | TopK SAE | RouteSAE |
|------|----------|----------|
| 模型类型 | TopK | RouteSAE |
| Hidden Size | 2048 | 2048 |
| Latent Size | 16384 | 16384 |
| K | 64 | 64 |
| 目标层 | Layer 16 | Layer 4-12 (多层路由) |
| 聚合方式 | - | sum |
| 路由策略 | - | hard |

## 评估结果

| 指标 | TopK SAE | RouteSAE | 更优 |
|------|----------|----------|------|
| **NormMSE** ↓ | ${TOPK_NORMMSE:-N/A} | ${ROUTESAE_NORMMSE:-N/A} | $([ "${TOPK_NORMMSE:-0}" \< "${ROUTESAE_NORMMSE:-0}" ] && echo "TopK" || echo "RouteSAE") |
| **KL Divergence** ↓ | ${TOPK_KLDIV:-N/A} | ${ROUTESAE_KLDIV:-N/A} | $([ "${TOPK_KLDIV:-0}" \< "${ROUTESAE_KLDIV:-0}" ] && echo "TopK" || echo "RouteSAE") |

> 注: ↓ 表示越低越好

## 指标说明

### NormMSE (Normalized Mean Squared Error)
衡量 SAE 重构的质量。计算公式:
\`\`\`
NormMSE = ||x - x_hat||^2 / ||x||^2
\`\`\`
- 值越低表示重构越准确
- 0 = 完美重构

### KL Divergence
衡量使用 SAE 重构后对 LLM 输出分布的影响:
\`\`\`
KL(P_original || P_reconstructed)
\`\`\`
- 值越低表示对 LLM 行为影响越小
- 0 = 完全不影响

## 结论

$(if [ -n "$TOPK_NORMMSE" ] && [ -n "$ROUTESAE_NORMMSE" ]; then
    echo "基于评估结果，两种模型在重构质量和 LLM 行为保持方面的表现如上表所示。"
    echo ""
    echo "**关键发现:**"
    echo "- TopK SAE 只关注单一层 (Layer 16) 的激活"
    echo "- RouteSAE 通过路由机制动态选择最佳层，可以捕获跨层特征"
else
    echo "评估数据获取中，请检查日志。"
fi)

## 模型文件

| 模型 | 路径 |
|------|------|
| TopK SAE | \`${TOPK_MODEL}\` |
| RouteSAE | \`${ROUTESAE_MODEL}\` |
EOF

echo "=============================================="
echo "Evaluation complete!"
echo "Report saved to: $RESULTS_FILE"
echo "=============================================="
