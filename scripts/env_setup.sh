#!/bin/bash
# ============================================================
# RouteSAE-repro 环境配置
# ============================================================
# 注意: 这些环境变量只在本脚本内生效，不会影响其他项目
# ============================================================

# HuggingFace Token (用于下载 Llama 等受限模型)
export HF_TOKEN="your_huggingface_token_here"

# OpenAI API Key (用于 GPT-4 特征解释)
export OPENAI_API_KEY="your_openai_api_key_here"

# HuggingFace 缓存目录 (存到 NFS，不占用 projects 空间)
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/nfs/zijie/huggingface_cache/datasets

# CUDA 设置
export CUDA_VISIBLE_DEVICES=3  # 使用 GPU 3

# 项目路径
export PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
export OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs
export DATA_DIR=/mnt/nfs/zijie/datasets/openwebtext2
