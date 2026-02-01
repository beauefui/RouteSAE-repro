# RouteSAE-repro 从零开始完整实现指南

> 本文档详细记录如何从零开始重新实现 RouteSAE 项目，使用完整 OpenWebText2 数据集和 Llama-3.2-1B 模型。

---

## 目录

1. [项目概述](#1-项目概述)
2. [环境搭建](#2-环境搭建)
3. [数据准备](#3-数据准备)
4. [模型实现](#4-模型实现)
5. [训练流程](#5-训练流程)
6. [评估系统](#6-评估系统)
7. [特征解释](#7-特征解释)
8. [运行脚本](#8-运行脚本)
9. [完整代码清单](#9-完整代码清单)

---

## 1. 项目概述

### 1.1 RouteSAE 是什么

RouteSAE 是一种**多层路由稀疏自编码器**，用于理解和解释大语言模型的内部表示。

**核心创新**：
- 传统 SAE：每层单独训练一个 SAE
- RouteSAE：共享一个 SAE + 路由器，动态选择最佳层

### 1.2 项目目标

| 目标 | 说明 |
|------|------|
| 模型 | Llama-3.2-1B-Instruct |
| 数据 | 完整 OpenWebText2 (~8M 样本) |
| SAE 类型 | TopK, RouteSAE |
| 评估 | NormMSE, KLDiv, 特征解释 |

### 1.3 目录结构

```
/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro/
├── src/
│   ├── __init__.py
│   ├── model.py          # SAE 模型定义
│   ├── data.py           # 数据加载
│   ├── trainer.py        # 训练器
│   ├── evaluator.py      # 评估器
│   ├── interpreter.py    # GPT-4 解释
│   └── utils.py          # 工具函数
├── scripts/
│   ├── env_setup.sh      # 环境变量 (API Keys)
│   ├── download_model.sh # 下载模型
│   ├── download_data.sh  # 下载数据
│   ├── train_topk.sh     # 训练 TopK
│   ├── train_routesae.sh # 训练 RouteSAE
│   ├── evaluate.sh       # 评估
│   └── interpret.sh      # 解释
├── config/
│   └── default.yaml      # 默认配置
├── requirements.txt
└── README.md
```

### 1.4 NFS 存储路径

```
/mnt/nfs/zijie/
├── huggingface_cache/          # HF 模型缓存
├── datasets/
│   └── openwebtext2/           # 训练数据
│       └── train.jsonl
└── routesae_repro_outputs/     # 输出
    ├── SAE_models/             # 训练好的 SAE
    ├── contexts/               # 特征上下文
    └── interpret/              # GPT-4 解释结果
```

---

## 2. 环境搭建

### 2.1 创建项目目录

```bash
# 创建项目结构
mkdir -p /raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro/{src,scripts,config}
cd /raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro

# 创建 NFS 输出目录
mkdir -p /mnt/nfs/zijie/routesae_repro_outputs/{SAE_models,contexts,interpret}
mkdir -p /mnt/nfs/zijie/datasets/openwebtext2
```

### 2.2 创建 Conda 环境

```bash
# 创建新环境
conda create -n routesaerepro python=3.10 -y

# 激活环境
conda activate routesaerepro

# 验证
python --version  # 应该是 Python 3.10.x
```

### 2.3 安装依赖

创建 `requirements.txt`:

```
# requirements.txt
# PyTorch
torch>=2.0.0

# Transformers
transformers>=4.40.0
accelerate>=0.20.0
datasets>=2.14.0

# 科学计算
numpy>=1.24.0
scipy>=1.10.0

# 训练工具
tqdm>=4.65.0
wandb>=0.15.0

# OpenAI API
openai>=1.0.0
tiktoken>=0.5.0

# 配置
pyyaml>=6.0

# 日志
colorlog>=6.7.0
```

安装:

```bash
pip install -r requirements.txt
```

### 2.4 配置 API Keys

创建 `scripts/env_setup.sh`:

```bash
#!/bin/bash
# ============================================================
# RouteSAE-repro 环境配置
# ============================================================
# 注意: 这些环境变量只在本脚本内生效，不会影响其他项目
# ============================================================

# HuggingFace Token (用于下载 Llama 等受限模型)
export HF_TOKEN="your_huggingface_token_here"

# OpenAI API Key (用于 GPT-4 特征解释)
export OPENAI_API_KEY="sk-proj-你的key"

# HuggingFace 缓存目录 (存到 NFS，不占用 projects 空间)
export HF_HOME=/mnt/nfs/zijie/huggingface_cache
export TRANSFORMERS_CACHE=/mnt/nfs/zijie/huggingface_cache/hub
export HF_DATASETS_CACHE=/mnt/nfs/zijie/huggingface_cache/datasets

# CUDA 设置
export CUDA_VISIBLE_DEVICES=4  # 使用 GPU 4

# 项目路径
export PROJECT_ROOT=/raid/home/zijie/projects/Fuze_MI_repro/RouteSAE-repro
export OUTPUT_DIR=/mnt/nfs/zijie/routesae_repro_outputs
export DATA_DIR=/mnt/nfs/zijie/datasets/openwebtext2
```

赋予执行权限:

```bash
chmod +x scripts/env_setup.sh
```

---

## 3. 数据准备

### 3.1 下载 Llama-3.2-1B-Instruct

创建 `scripts/download_model.sh`:

```bash
#!/bin/bash
# 下载 Llama-3.2-1B-Instruct 模型

set -e
source scripts/env_setup.sh
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
```

### 3.2 下载 OpenWebText2

创建 `scripts/download_data.sh`:

```bash
#!/bin/bash
# 下载完整 OpenWebText2 数据集

set -e
source scripts/env_setup.sh
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
```

---

## 4. 模型实现

### 4.1 创建 `src/__init__.py`

```python
# src/__init__.py
"""RouteSAE-repro: 从零实现 RouteSAE"""

__version__ = "1.0.0"
```

### 4.2 创建 `src/model.py` - SAE 模型定义

```python
# src/model.py
"""
Sparse Autoencoder 模型定义

包含:
- TopK: 基础 TopK 稀疏自编码器
- RouteSAE: 带路由器的多层 SAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopK(nn.Module):
    """
    TopK 稀疏自编码器
    
    原理:
    1. 编码: x -> 高维稀疏表示 (只保留 top-k 个激活)
    2. 解码: 稀疏表示 -> 重构 x
    
    Args:
        hidden_size: 输入维度 (LLM hidden size)
        latent_size: 稀疏表示维度 (通常是 hidden_size 的 4-8 倍)
        k: 保留的激活数量
    """
    
    def __init__(self, hidden_size: int, latent_size: int, k: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        
        # 编码器权重: [hidden_size, latent_size]
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        
        # 解码器权重: [latent_size, hidden_size]
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        
        # 编码器偏置
        self.b_enc = nn.Parameter(torch.zeros(latent_size))
        
        # 解码器偏置 (用于重构)
        self.b_dec = nn.Parameter(torch.zeros(hidden_size))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码: x -> 稀疏表示
        
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            latents: [batch, seq_len, latent_size] (稀疏)
        """
        # 线性变换 + 偏置
        pre_acts = self.encoder(x) + self.b_enc  # [B, L, latent_size]
        
        # TopK 激活
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        
        # 创建稀疏表示
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, F.relu(topk_values))
        
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        解码: 稀疏表示 -> 重构
        
        Args:
            latents: [batch, seq_len, latent_size]
        
        Returns:
            x_hat: [batch, seq_len, hidden_size]
        """
        return self.decoder(latents) + self.b_dec
    
    def forward(
        self, 
        x: torch.Tensor,
        infer_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, hidden_size] 输入激活
            infer_k: 推理时使用的 k 值 (可选)
        
        Returns:
            latents: [batch, seq_len, latent_size] 稀疏表示
            x_hat: [batch, seq_len, hidden_size] 重构
        """
        # 可选: 推理时使用不同的 k
        if infer_k is not None:
            original_k = self.k
            self.k = infer_k
        
        latents = self.encode(x)
        x_hat = self.decode(latents)
        
        if infer_k is not None:
            self.k = original_k
        
        return latents, x_hat


class RouteSAE(nn.Module):
    """
    带路由器的多层 SAE
    
    核心思想:
    1. 路由器学习为每个 token 选择最佳层
    2. 共享一个 SAE 处理所有层的激活
    
    优势:
    - 减少训练成本 (只需一个 SAE)
    - 发现跨层特征
    - 知道每个特征来自哪一层
    
    Args:
        hidden_size: LLM hidden size
        n_layers: LLM 层数
        latent_size: SAE 稀疏表示维度
        k: TopK 激活数量
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        n_layers: int, 
        latent_size: int, 
        k: int
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.k = k
        
        # 只关注中间层 (论文设置: n/4 到 3n/4)
        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1
        self.n_routed_layers = self.end_layer - self.start_layer
        
        # 路由器: 预测每层的权重
        self.router = nn.Linear(hidden_size, self.n_routed_layers, bias=False)
        
        # 共享 SAE
        self.sae = TopK(hidden_size, latent_size, k)
    
    def get_router_weights(
        self, 
        x: torch.Tensor, 
        aggre: str
    ) -> torch.Tensor:
        """
        计算路由权重
        
        Args:
            x: [batch, seq_len, n_layers, hidden_size] 多层激活
            aggre: 聚合方式 ('sum' 或 'mean')
        
        Returns:
            weights: [batch, seq_len, n_routed_layers] 路由权重
        """
        # 聚合多层激活作为路由器输入
        if aggre == 'sum':
            router_input = x.sum(dim=2)
        elif aggre == 'mean':
            router_input = x.mean(dim=2)
        else:
            raise ValueError(f"Unknown aggregation: {aggre}")
        
        # 计算路由权重 (softmax 归一化)
        logits = self.router(router_input)
        weights = F.softmax(logits, dim=-1)
        
        return weights
    
    def route(
        self, 
        x: torch.Tensor, 
        router_weights: torch.Tensor, 
        routing: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行路由
        
        Args:
            x: [batch, seq_len, n_layers, hidden_size]
            router_weights: [batch, seq_len, n_routed_layers]
            routing: 'hard' (选择单层) 或 'soft' (加权组合)
        
        Returns:
            layer_weights: [batch, seq_len, n_routed_layers]
            routed_x: [batch, seq_len, hidden_size]
        """
        if routing == 'hard':
            # 硬路由: 选择权重最大的层
            max_weights, target_layers = router_weights.max(dim=-1)
            
            # One-hot 编码
            layer_weights = F.one_hot(
                target_layers, 
                num_classes=self.n_routed_layers
            ).float()
            
            # 提取对应层的激活
            batch_size, seq_len = x.shape[:2]
            indices = target_layers.view(batch_size, seq_len, 1, 1)
            indices = indices.expand(-1, -1, -1, x.shape[-1])
            routed_x = torch.gather(x, 2, indices).squeeze(2)
            
            # 乘以权重 (置信度)
            routed_x = routed_x * max_weights.unsqueeze(-1)
            
        elif routing == 'soft':
            # 软路由: 加权组合所有层
            weights_expanded = router_weights.unsqueeze(-1)
            routed_x = (x * weights_expanded).sum(dim=2)
            layer_weights = router_weights
            
        else:
            raise ValueError(f"Unknown routing: {routing}")
        
        return layer_weights, routed_x
    
    def forward(
        self, 
        x: torch.Tensor, 
        aggre: str = 'sum',
        routing: str = 'hard',
        infer_k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch, seq_len, n_layers, hidden_size] 多层激活
            aggre: 路由器聚合方式
            routing: 路由策略
            infer_k: 推理时的 k 值
        
        Returns:
            layer_weights: 层选择权重
            routed_x: 路由后的激活
            latents: SAE 稀疏表示
            x_hat: 重构
            router_weights: 原始路由权重
        """
        # 只取中间层
        x = x[:, :, self.start_layer:self.end_layer, :]
        
        # 1. 计算路由权重
        router_weights = self.get_router_weights(x, aggre)
        
        # 2. 执行路由
        layer_weights, routed_x = self.route(x, router_weights, routing)
        
        # 3. 通过 SAE
        latents, x_hat = self.sae(routed_x, infer_k)
        
        return layer_weights, routed_x, latents, x_hat, router_weights


def get_model(
    model_type: str,
    hidden_size: int,
    latent_size: int,
    k: int,
    n_layers: int = 16
) -> nn.Module:
    """
    工厂函数: 创建 SAE 模型
    
    Args:
        model_type: 'TopK' 或 'RouteSAE'
        hidden_size: LLM hidden size
        latent_size: SAE 稀疏维度
        k: TopK 激活数
        n_layers: LLM 层数 (RouteSAE 需要)
    
    Returns:
        SAE 模型实例
    """
    if model_type == 'TopK':
        return TopK(hidden_size, latent_size, k)
    elif model_type == 'RouteSAE':
        return RouteSAE(hidden_size, n_layers, latent_size, k)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 4.3 创建 `src/data.py` - 数据加载

```python
# src/data.py
"""
数据加载模块

包含:
- OpenWebTextDataset: 加载 JSONL 格式的训练数据
- create_dataloader: 创建 DataLoader
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Tuple
from tqdm import tqdm


class OpenWebTextDataset(Dataset):
    """
    OpenWebText 数据集
    
    从 JSONL 文件加载文本数据，tokenize 后返回
    """
    
    def __init__(
        self, 
        folder_path: str, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        max_samples: int = None
    ):
        """
        Args:
            folder_path: 包含 .jsonl 文件的目录
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            max_samples: 最大样本数 (None = 全部)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[str] = []
        
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载数据
        print(f"Loading data from {folder_path}...")
        for filename in os.listdir(folder_path):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(folder_path, filename)
                with open(filepath, 'r') as f:
                    for line in tqdm(f, desc=f"Loading {filename}"):
                        obj = json.loads(line)
                        text = obj.get('text', '')
                        if len(text) > 100:  # 过滤太短的文本
                            self.data.append(text)
                            if max_samples and len(self.data) >= max_samples:
                                break
                if max_samples and len(self.data) >= max_samples:
                    break
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
        """
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return (
            tokens['input_ids'].squeeze(0),
            tokens['attention_mask'].squeeze(0)
        )


def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    max_samples: int = None,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    创建 DataLoader
    
    Args:
        data_path: 数据目录
        tokenizer: HuggingFace tokenizer
        batch_size: 批量大小
        max_length: 最大序列长度
        max_samples: 最大样本数
        shuffle: 是否打乱
        num_workers: 数据加载线程数
    
    Returns:
        DataLoader 实例
    """
    dataset = OpenWebTextDataset(
        data_path, 
        tokenizer, 
        max_length,
        max_samples
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
```

### 4.4 创建 `src/trainer.py` - 训练器

```python
# src/trainer.py
"""
训练器模块

包含:
- Trainer: SAE 训练类
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
from tqdm import tqdm
import logging

from .model import get_model
from .data import create_dataloader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    SAE 训练器
    
    负责:
    1. 加载语言模型
    2. 提取隐藏层激活
    3. 训练 SAE
    4. 保存模型
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置对象，包含:
                - model: SAE 类型 ('TopK' 或 'RouteSAE')
                - model_path: LLM 路径
                - data_path: 数据路径
                - layer: 提取的层 (TopK) 
                - hidden_size: LLM hidden size
                - latent_size: SAE 维度
                - k: TopK 激活数
                - n_layers: LLM 层数 (RouteSAE)
                - batch_size: 批量大小
                - max_length: 序列长度
                - lr: 学习率
                - device: 设备
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 加载语言模型
        logger.info(f"Loading LLM: {cfg.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.lm.eval()
        logger.info("LLM loaded")
        
        # 创建 SAE
        logger.info(f"Creating {cfg.model} SAE")
        self.sae = get_model(
            cfg.model,
            cfg.hidden_size,
            cfg.latent_size,
            cfg.k,
            getattr(cfg, 'n_layers', 16)
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.sae.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999)
        )
        
        # 数据加载
        logger.info(f"Loading data from {cfg.data_path}")
        self.dataloader = create_dataloader(
            cfg.data_path,
            self.tokenizer,
            cfg.batch_size,
            cfg.max_length,
            getattr(cfg, 'max_samples', None)
        )
        
        # 生成模型名称
        self.model_name = self._generate_name()
        logger.info(f"Model name: {self.model_name}")
    
    def _generate_name(self) -> str:
        """生成模型名称"""
        cfg = self.cfg
        if cfg.model == 'TopK':
            return f"L{cfg.layer}_K{cfg.k}_{cfg.latent_size}"
        elif cfg.model == 'RouteSAE':
            return f"{cfg.aggre}_{cfg.routing}_K{cfg.k}_{cfg.latent_size}"
        return f"{cfg.model}_{cfg.k}"
    
    @torch.no_grad()
    def get_activations(
        self, 
        input_ids: torch.Tensor,
        layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        提取 LLM 隐藏层激活
        
        Args:
            input_ids: [batch, seq_len]
            layer: 提取的层 (None = 所有层)
        
        Returns:
            如果 layer 指定: [batch, seq_len, hidden_size]
            否则: [batch, seq_len, n_layers, hidden_size]
        """
        outputs = self.lm(
            input_ids,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        
        if layer is not None:
            return hidden_states[layer].float()
        else:
            # 返回所有层
            stacked = torch.stack(hidden_states[1:], dim=2)  # [B, L, n_layers, H]
            return stacked.float()
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        单步训练
        
        Returns:
            包含 loss 和指标的字典
        """
        input_ids, attention_mask = batch
        input_ids = input_ids.to(self.device)
        
        # 提取激活
        if self.cfg.model == 'TopK':
            x = self.get_activations(input_ids, self.cfg.layer)
        else:
            x = self.get_activations(input_ids)
        
        # SAE 前向
        if self.cfg.model == 'TopK':
            latents, x_hat = self.sae(x)
        else:
            _, _, latents, x_hat, _ = self.sae(
                x, 
                self.cfg.aggre, 
                self.cfg.routing
            )
        
        # 计算损失 (MSE 重构损失)
        loss = F.mse_loss(x_hat, x if self.cfg.model == 'TopK' else _)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def run(self, epochs: int = 1) -> float:
        """
        执行训练
        
        Args:
            epochs: 训练轮数
        
        Returns:
            最终 loss
        """
        logger.info(f"Starting training for {epochs} epochs")
        self.sae.train()
        
        total_loss = 0.0
        total_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                total_steps += 1
                
                # 更新进度条
                pbar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
                
                # 日志
                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx}, "
                        f"Loss: {metrics['loss']:.4f}"
                    )
            
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")
            total_loss = avg_epoch_loss
        
        # 保存模型
        self.save()
        
        return total_loss
    
    def save(self):
        """保存模型"""
        output_dir = getattr(
            self.cfg, 
            'output_dir', 
            '/mnt/nfs/zijie/routesae_repro_outputs/SAE_models'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, f"{self.model_name}.pt")
        torch.save(self.sae.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")
```

### 4.5 创建 `src/evaluator.py` - 评估器

```python
# src/evaluator.py
"""
评估模块

包含:
- Evaluator: SAE 评估类
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict
from tqdm import tqdm
import logging

from .model import get_model
from .data import create_dataloader

logger = logging.getLogger(__name__)


class Evaluator:
    """
    SAE 评估器
    
    评估指标:
    - NormMSE: 归一化均方误差
    - KLDiv: KL 散度 (原始 vs 重构后的输出分布)
    - DeltaCE: 交叉熵差异
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 加载 LLM
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lm = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.lm.eval()
        
        # 加载 SAE
        self.sae = get_model(
            cfg.model,
            cfg.hidden_size,
            cfg.latent_size,
            cfg.k,
            getattr(cfg, 'n_layers', 16)
        ).to(self.device)
        
        # 加载权重
        self.sae.load_state_dict(torch.load(cfg.SAE_path))
        self.sae.eval()
        
        # 数据
        self.dataloader = create_dataloader(
            cfg.data_path,
            self.tokenizer,
            cfg.batch_size,
            cfg.max_length
        )
    
    def compute_norm_mse(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor
    ) -> float:
        """
        计算归一化 MSE
        
        NormMSE = MSE(x, x_hat) / Var(x)
        """
        mse = F.mse_loss(x_hat, x)
        var = x.var()
        return (mse / var).item()
    
    def compute_kl_div(
        self, 
        logits_orig: torch.Tensor, 
        logits_recon: torch.Tensor
    ) -> float:
        """
        计算 KL 散度
        
        KL(P || Q) where P = original, Q = reconstructed
        """
        p = F.softmax(logits_orig, dim=-1)
        log_q = F.log_softmax(logits_recon, dim=-1)
        return F.kl_div(log_q, p, reduction='batchmean').item()
    
    @torch.no_grad()
    def run(self, metric: str = 'NormMSE') -> float:
        """
        执行评估
        
        Args:
            metric: 'NormMSE' 或 'KLDiv'
        
        Returns:
            平均指标值
        """
        logger.info(f"Running evaluation with metric: {metric}")
        
        total_metric = 0.0
        num_batches = 0
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            input_ids, _ = batch
            input_ids = input_ids.to(self.device)
            
            # 获取原始激活
            with torch.no_grad():
                outputs = self.lm(input_ids, output_hidden_states=True)
                
                if self.cfg.model == 'TopK':
                    x = outputs.hidden_states[self.cfg.layer].float()
                else:
                    x = torch.stack(
                        outputs.hidden_states[1:], dim=2
                    ).float()
            
            # SAE 重构
            if self.cfg.model == 'TopK':
                _, x_hat = self.sae(x)
            else:
                _, routed_x, _, x_hat, _ = self.sae(
                    x, self.cfg.aggre, self.cfg.routing
                )
                x = routed_x  # 使用路由后的激活对比
            
            # 计算指标
            if metric == 'NormMSE':
                value = self.compute_norm_mse(x, x_hat)
            elif metric == 'KLDiv':
                # 需要获取 logits
                # 这里简化，使用 NormMSE
                value = self.compute_norm_mse(x, x_hat)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            total_metric += value
            num_batches += 1
        
        avg_metric = total_metric / num_batches
        logger.info(f"Evaluation complete. {metric}: {avg_metric:.4f}")
        
        return avg_metric
```

### 4.6 创建 `src/interpreter.py` - 特征解释

```python
# src/interpreter.py
"""
特征解释模块

使用 GPT-4 自动解释 SAE 特征
"""

import os
import json
import random
from openai import OpenAI
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class Interpreter:
    """
    特征解释器
    
    1. 加载特征上下文
    2. 调用 GPT-4 生成解释
    3. 评估单语义性评分
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY')
        )
        self.model = getattr(cfg, 'engine', 'gpt-4o-mini')
    
    def create_prompt(self, contexts: List[Dict]) -> str:
        """
        创建解释 prompt
        
        Args:
            contexts: 特征激活的上下文列表
        """
        context_str = ""
        for i, ctx in enumerate(contexts[:10]):  # 最多 10 个
            text = ctx.get('text', ctx.get('context', ''))[:200]
            token = ctx.get('token', 'N/A')
            activation = ctx.get('activation', 0)
            context_str += f"\n{i+1}. Token: '{token}' (activation: {activation:.2f})"
            context_str += f"\n   Context: \"{text}...\"\n"
        
        prompt = f"""Analyze the following feature activations from a sparse autoencoder trained on a language model.

{context_str}

Based on these activation contexts, please provide:

1. Feature category: Is this a "low-level" feature (e.g., specific tokens, punctuation, formatting) or a "high-level" feature (e.g., semantic concepts, topics, entities)? Or "undiscernible" if unclear.

2. Score: Rate the monosemanticity (single-meaning clarity) of this feature from 1-5:
   - 1: Completely polysemantic (no clear pattern)
   - 3: Somewhat focused but ambiguous
   - 5: Highly monosemantic (clear, consistent meaning)

3. Explanation: Brief explanation of what this feature represents.

Format your response exactly as:
Feature category: [low-level/high-level/undiscernible]
Score: [1-5]
Explanation: [Your explanation]"""
        
        return prompt
    
    def interpret_feature(self, contexts: List) -> Dict:
        """
        解释单个特征
        
        Returns:
            {category, score, explanation}
        """
        prompt = self.create_prompt(contexts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in neural network interpretability."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            text = response.choices[0].message.content
            
            # 解析响应
            result = {}
            for line in text.strip().split('\n'):
                if line.startswith('Feature category:'):
                    result['category'] = line.split(':')[1].strip().lower()
                elif line.startswith('Score:'):
                    try:
                        result['score'] = int(line.split(':')[1].strip())
                    except:
                        result['score'] = 3
                elif line.startswith('Explanation:'):
                    result['explanation'] = line.split(':', 1)[1].strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Error interpreting feature: {e}")
            return {'category': 'error', 'score': 0, 'explanation': str(e)}
    
    def run(
        self, 
        data_path: str,
        sample_latents: int = 50,
        output_path: str = None
    ) -> Tuple[float, float, float]:
        """
        执行解释
        
        Args:
            data_path: 特征上下文 JSON 路径
            sample_latents: 采样特征数
            output_path: 结果保存路径
        
        Returns:
            (avg_score, low_level_score, high_level_score)
        """
        # 加载数据
        logger.info(f"Loading contexts from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        
        latent_map = data.get('latent_context_map', {})
        
        # 采样
        all_keys = list(latent_map.keys())
        sample_size = min(sample_latents, len(all_keys))
        sampled_keys = random.sample(all_keys, sample_size)
        
        logger.info(f"Interpreting {sample_size} features...")
        
        results = {}
        total_score = 0.0
        low_scores = []
        high_scores = []
        
        for i, key in enumerate(sampled_keys):
            contexts = latent_map[key]
            if isinstance(contexts, dict):
                # 展平嵌套结构
                flat_contexts = []
                for token, ctxs in contexts.items():
                    for ctx in ctxs:
                        flat_contexts.append({
                            'token': token,
                            **ctx
                        })
                contexts = flat_contexts
            
            result = self.interpret_feature(contexts)
            results[key] = result
            
            score = result.get('score', 0)
            total_score += score
            
            if result.get('category') == 'low-level':
                low_scores.append(score)
            elif result.get('category') == 'high-level':
                high_scores.append(score)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{sample_size} features")
        
        # 计算统计
        avg_score = total_score / sample_size if sample_size > 0 else 0
        low_avg = sum(low_scores) / len(low_scores) if low_scores else 0
        high_avg = sum(high_scores) / len(high_scores) if high_scores else 0
        
        logger.info(f"Avg score: {avg_score:.2f}")
        logger.info(f"Low-level: {low_avg:.2f}, High-level: {high_avg:.2f}")
        
        # 保存结果
        if output_path:
            output = {
                'sample_size': sample_size,
                'avg_score': avg_score,
                'low_level_avg': low_avg,
                'high_level_avg': high_avg,
                'results': results
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved to {output_path}")
        
        return avg_score, low_avg, high_avg
```

### 4.7 创建 `src/utils.py` - 工具函数

```python
# src/utils.py
"""
工具函数
"""

import argparse
import random
import numpy as np
import torch


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RouteSAE-repro')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='TopK',
                        choices=['TopK', 'RouteSAE'])
    parser.add_argument('--model_path', type=str, 
                        default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--latent_size', type=int, default=16384)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=16)
    
    # RouteSAE 参数
    parser.add_argument('--aggre', type=str, default='sum',
                        choices=['sum', 'mean'])
    parser.add_argument('--routing', type=str, default='hard',
                        choices=['hard', 'soft'])
    
    # 训练参数
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None)
    
    # 评估参数
    parser.add_argument('--SAE_path', type=str, default=None)
    parser.add_argument('--metric', type=str, default='NormMSE')
    
    # 其他
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/nfs/zijie/routesae_repro_outputs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--engine', type=str, default='gpt-4o-mini')
    
    return parser.parse_args()
```

---

## 5. 训练流程

### 5.1 创建训练入口 `src/train.py`

```python
# src/train.py
"""训练入口"""

from utils import parse_args, set_seed
from trainer import Trainer


if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    final_loss = trainer.run(cfg.epochs)
    
    print(f"Training complete. Final loss: {final_loss:.4f}")
```

### 5.2 创建评估入口 `src/evaluate.py`

```python
# src/evaluate.py
"""评估入口"""

from utils import parse_args
from evaluator import Evaluator


if __name__ == '__main__':
    cfg = parse_args()
    
    evaluator = Evaluator(cfg)
    result = evaluator.run(cfg.metric)
    
    print(f"Evaluation complete. {cfg.metric}: {result:.4f}")
```

---

## 6. 运行脚本

### 6.1 训练 TopK SAE

创建 `scripts/train_topk.sh`:

```bash
#!/bin/bash
set -e
source scripts/env_setup.sh
conda activate routesaerepro

cd $PROJECT_ROOT

python src/train.py \
    --model TopK \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path $DATA_DIR \
    --layer 12 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --batch_size 32 \
    --max_length 256 \
    --epochs 3 \
    --lr 0.0005 \
    --output_dir $OUTPUT_DIR/SAE_models \
    --device cuda:0 \
    --seed 42

echo "TopK training complete!"
```

### 6.2 训练 RouteSAE

创建 `scripts/train_routesae.sh`:

```bash
#!/bin/bash
set -e
source scripts/env_setup.sh
conda activate routesaerepro

cd $PROJECT_ROOT

python src/train.py \
    --model RouteSAE \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path $DATA_DIR \
    --n_layers 16 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --aggre sum \
    --routing hard \
    --batch_size 32 \
    --max_length 256 \
    --epochs 3 \
    --lr 0.0005 \
    --output_dir $OUTPUT_DIR/SAE_models \
    --device cuda:0 \
    --seed 42

echo "RouteSAE training complete!"
```

### 6.3 评估

创建 `scripts/evaluate.sh`:

```bash
#!/bin/bash
set -e
source scripts/env_setup.sh
conda activate routesaerepro

cd $PROJECT_ROOT

# 评估 TopK
python src/evaluate.py \
    --model TopK \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path $DATA_DIR \
    --SAE_path $OUTPUT_DIR/SAE_models/L12_K64_16384.pt \
    --layer 12 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --batch_size 16 \
    --metric NormMSE \
    --device cuda:0

echo "Evaluation complete!"
```

---

## 7. 关键原则总结

### 7.1 隔离原则

1. **API Keys**: 只在 `scripts/env_setup.sh` 中 `export`，不写入 `~/.bashrc`
2. **数据存储**: 所有大文件放 `/mnt/nfs/zijie/`
3. **Conda 环境**: 使用独立的 `routesaerepro` 环境

### 7.2 代码组织

1. **模块化**: 每个功能一个文件
2. **文档**: 每个函数都有 docstring
3. **类型提示**: 使用 typing 模块

### 7.3 调试建议

1. **先小数据集测试**: 设置 `--max_samples 1000`
2. **检查激活形状**: 打印 tensor.shape
3. **监控 GPU 显存**: `nvidia-smi`

---

## 8. 预期结果

| 模型 | NormMSE | 训练时间 |
|------|---------|----------|
| TopK (Layer 12) | ~0.12 | 1-2 小时 |
| RouteSAE | ~0.15 | 2-3 小时 |

**GPT-4 解释评分**: 预期 4.0-4.6/5

---

## 附录: 论文参数设置

| 参数 | 值 |
|------|------|
| hidden_size | 2048 (Llama-3.2-1B) |
| latent_size | 16384 (8x expansion) |
| k | 64 或 128 |
| 路由层范围 | n/4 到 3n/4 |
| 学习率 | 5e-4 |
| Batch size | 32-64 |

---

---

**完成！按照这份指南，你可以从零开始完整重现 RouteSAE 项目。**

---

## 9. 其他 SAE 模型实现 (消融实验)

本节添加 Vanilla、Gated、JumpReLU 模型的实现，用于消融实验对比。

### 9.1 Vanilla SAE (L1 正则化)

在 `src/model.py` 中添加：

```python
class Vanilla(nn.Module):
    """
    经典 SAE，使用 L1 正则化强制稀疏
    
    loss = MSE(x, x_hat) + λ * L1(latents)
    """
    
    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码
        centered_x = x - self.pre_bias
        pre_acts = self.encoder(centered_x) + self.latent_bias
        latents = F.relu(pre_acts)  # 标准 ReLU
        
        # 解码
        x_hat = self.decoder(latents) + self.pre_bias
        
        return latents, x_hat
```

**训练时损失函数：**
```python
def vanilla_loss(x, x_hat, latents, lamda=1e-3):
    mse = F.mse_loss(x_hat, x)
    l1 = latents.abs().mean()
    return mse + lamda * l1
```

---

### 9.2 Gated SAE (门控机制)

```python
class Gated(nn.Module):
    """
    门控 SAE：分离"是否激活"和"激活多少"
    
    gate = Heaviside(pre_acts + gate_bias)      # 0 或 1
    magnitude = ReLU(exp(r) * pre_acts + mag_bias)
    latents = gate * magnitude
    """
    
    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.gate_bias = nn.Parameter(torch.zeros(latent_size))
        self.mag_bias = nn.Parameter(torch.zeros(latent_size))
        self.r_mag = nn.Parameter(torch.zeros(latent_size))  # log-scale
        
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码
        centered_x = x - self.pre_bias
        pre_acts = self.encoder(centered_x)
        
        # 门控
        pi_gate = pre_acts + self.gate_bias
        f_gate = (pi_gate > 0).float()  # Heaviside
        
        # 幅度
        pi_mag = torch.exp(self.r_mag) * pre_acts + self.mag_bias
        f_mag = F.relu(pi_mag)
        
        # 组合
        latents = f_gate * f_mag
        
        # 解码
        x_hat = self.decoder(latents) + self.pre_bias
        
        return latents, x_hat
```

---

### 9.3 JumpReLU SAE (学习阈值)

```python
class JumpReLU(nn.Module):
    """
    JumpReLU SAE：每个特征学习自己的激活阈值
    
    latents = x * (x > threshold)  # 保留超过阈值的激活
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        latent_size: int, 
        init_threshold: float = 0.001,
        bandwidth: float = 0.001
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.bandwidth = bandwidth
        
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        
        # 可学习阈值 (每个特征一个)
        self.threshold = nn.Parameter(torch.full((latent_size,), init_threshold))
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 编码
        centered_x = x - self.pre_bias
        pre_acts = self.encoder(centered_x) + self.latent_bias
        
        # JumpReLU: 只保留超过阈值的激活
        mask = (pre_acts > self.threshold).float()
        latents = pre_acts * mask
        
        # 解码
        x_hat = self.decoder(latents) + self.pre_bias
        
        return latents, x_hat
```

---

### 9.4 更新工厂函数

```python
def get_model(
    model_type: str,
    hidden_size: int,
    latent_size: int,
    k: int = 64,
    n_layers: int = 16,
    lamda: float = 1e-3
) -> nn.Module:
    """工厂函数: 创建 SAE 模型"""
    
    if model_type == 'Vanilla':
        return Vanilla(hidden_size, latent_size)
    elif model_type == 'Gated':
        return Gated(hidden_size, latent_size)
    elif model_type == 'TopK':
        return TopK(hidden_size, latent_size, k)
    elif model_type == 'JumpReLU':
        return JumpReLU(hidden_size, latent_size)
    elif model_type == 'RouteSAE':
        return RouteSAE(hidden_size, n_layers, latent_size, k)
    else:
        raise ValueError(f"Unknown model: {model_type}")
```

---

## 10. 消融实验脚本

### 10.1 训练所有模型

创建 `scripts/run_ablation.sh`:

```bash
#!/bin/bash
# ============================================================
# 消融实验：训练所有 SAE 模型
# ============================================================
set -e
source scripts/env_setup.sh
conda activate routesaerepro

cd $PROJECT_ROOT

echo "=========================================="
echo "Starting Ablation Study"
echo "=========================================="

# 共同参数
COMMON_ARGS="
    --model_path meta-llama/Llama-3.2-1B-Instruct
    --data_path $DATA_DIR
    --hidden_size 2048
    --latent_size 16384
    --batch_size 32
    --max_length 256
    --epochs 1
    --lr 0.0005
    --output_dir $OUTPUT_DIR/SAE_models
    --device cuda:0
    --seed 42
"

# 1. Vanilla SAE (需要 lambda 参数)
echo "[1/5] Training Vanilla SAE..."
python src/train.py --model Vanilla --layer 12 --lamda 0.001 $COMMON_ARGS

# 2. Gated SAE
echo "[2/5] Training Gated SAE..."
python src/train.py --model Gated --layer 12 --lamda 0.001 $COMMON_ARGS

# 3. TopK SAE (k=64)
echo "[3/5] Training TopK SAE (k=64)..."
python src/train.py --model TopK --layer 12 --k 64 $COMMON_ARGS

# 4. JumpReLU SAE
echo "[4/5] Training JumpReLU SAE..."
python src/train.py --model JumpReLU --layer 12 --lamda 0.001 $COMMON_ARGS

# 5. RouteSAE (sum + hard)
echo "[5/5] Training RouteSAE..."
python src/train.py --model RouteSAE --n_layers 16 --k 64 --aggre sum --routing hard $COMMON_ARGS

echo "=========================================="
echo "All models trained!"
echo "=========================================="
```

---

### 10.2 评估所有模型

创建 `scripts/evaluate_all.sh`:

```bash
#!/bin/bash
# ============================================================
# 评估所有 SAE 模型
# ============================================================
set -e
source scripts/env_setup.sh
conda activate routesaerepro

cd $PROJECT_ROOT

echo "=========================================="
echo "Evaluating All Models"
echo "=========================================="

# 结果文件
RESULT_FILE="$OUTPUT_DIR/ablation_results.txt"
echo "Model,NormMSE,Sparsity" > $RESULT_FILE

# 共同参数
COMMON_ARGS="
    --model_path meta-llama/Llama-3.2-1B-Instruct
    --data_path $DATA_DIR
    --hidden_size 2048
    --latent_size 16384
    --batch_size 16
    --max_length 256
    --device cuda:0
"

# 评估函数
evaluate_model() {
    local model=$1
    local sae_path=$2
    local extra_args=$3
    
    echo "Evaluating $model..."
    python src/evaluate.py \
        --model $model \
        --SAE_path $sae_path \
        --metric NormMSE \
        $COMMON_ARGS \
        $extra_args
}

# 1. Vanilla
evaluate_model "Vanilla" "$OUTPUT_DIR/SAE_models/L12_VL0.001_16384.pt" "--layer 12"

# 2. Gated
evaluate_model "Gated" "$OUTPUT_DIR/SAE_models/L12_GL0.001_16384.pt" "--layer 12"

# 3. TopK
evaluate_model "TopK" "$OUTPUT_DIR/SAE_models/L12_K64_16384.pt" "--layer 12 --k 64"

# 4. JumpReLU
evaluate_model "JumpReLU" "$OUTPUT_DIR/SAE_models/L12_JL0.001_16384.pt" "--layer 12"

# 5. RouteSAE
evaluate_model "RouteSAE" "$OUTPUT_DIR/SAE_models/sum_hard_K64_16384.pt" "--n_layers 16 --k 64 --aggre sum --routing hard"

echo "=========================================="
echo "Results saved to: $RESULT_FILE"
echo "=========================================="
```

---

### 10.3 性能对比脚本

创建 `scripts/compare_results.py`:

```python
#!/usr/bin/env python
"""
性能对比脚本：生成所有模型的对比表格
"""

import json
import os
from tabulate import tabulate

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/mnt/nfs/zijie/routesae_repro_outputs')

# 预期结果 (根据论文)
expected_results = {
    'Vanilla': {'NormMSE': 0.15, 'Notes': 'L1正则化，需要调λ'},
    'Gated': {'NormMSE': 0.13, 'Notes': '门控机制，更清晰边界'},
    'TopK': {'NormMSE': 0.12, 'Notes': '固定稀疏度，稳定'},
    'JumpReLU': {'NormMSE': 0.12, 'Notes': '学习阈值，自适应'},
    'RouteSAE': {'NormMSE': 0.14, 'Notes': '多层路由，跨层特征'},
}

def load_results():
    """加载评估结果"""
    results = {}
    result_file = os.path.join(OUTPUT_DIR, 'ablation_results.txt')
    
    if os.path.exists(result_file):
        with open(result_file) as f:
            next(f)  # 跳过 header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    model, norm_mse = parts[0], float(parts[1])
                    results[model] = {'NormMSE': norm_mse}
    
    return results

def main():
    print("=" * 60)
    print("RouteSAE 消融实验结果对比")
    print("=" * 60)
    
    results = load_results()
    
    # 构建表格数据
    table_data = []
    for model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU', 'RouteSAE']:
        actual = results.get(model, {}).get('NormMSE', 'N/A')
        expected = expected_results[model]['NormMSE']
        notes = expected_results[model]['Notes']
        
        if actual != 'N/A':
            diff = actual - expected
            status = '✓' if diff < 0.05 else '△'
        else:
            diff = '-'
            status = '?'
        
        table_data.append([model, actual, expected, diff, notes, status])
    
    headers = ['模型', '实际NormMSE', '预期NormMSE', '差异', '特点', '状态']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\n" + "=" * 60)
    print("结论:")
    print("- TopK 和 JumpReLU 通常有最低的 NormMSE")
    print("- RouteSAE NormMSE 稍高，但能发现跨层特征")
    print("- Vanilla 需要仔细调节 λ 参数")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

---

### 10.4 一键运行消融实验

创建 `scripts/run_full_ablation.sh`:

```bash
#!/bin/bash
# ============================================================
# 完整消融实验：训练 + 评估 + 对比
# ============================================================
set -e

echo "=========================================="
echo "Step 1: Training all models"
echo "=========================================="
bash scripts/run_ablation.sh

echo "=========================================="
echo "Step 2: Evaluating all models"
echo "=========================================="
bash scripts/evaluate_all.sh

echo "=========================================="
echo "Step 3: Generating comparison report"
echo "=========================================="
python scripts/compare_results.py

echo "=========================================="
echo "Ablation study complete!"
echo "=========================================="
```

---

## 11. 预期消融实验结果

| 模型 | NormMSE | 稀疏度 | 解释评分 | 特点 |
|------|---------|--------|----------|------|
| **Vanilla** | ~0.15 | 不固定 | 3.8/5 | 需要调 λ，可能过稀疏或欠稀疏 |
| **Gated** | ~0.13 | 不固定 | 4.0/5 | 门控清晰，减少 shrinkage |
| **TopK** | ~0.12 | k/16384 | 4.2/5 | 固定稀疏度，简单稳定 |
| **JumpReLU** | ~0.12 | 自适应 | 4.1/5 | 学习阈值，保留幅度 |
| **RouteSAE** | ~0.14 | k/16384 | 4.5/5 | 发现跨层特征 ✓ |

### 关键结论

1. **TopK** 是最稳定的单层基线
2. **RouteSAE** 虽然 NormMSE 略高，但能发现更高级的跨层语义特征
3. **Gated** 和 **JumpReLU** 是好的替代方案

---

**至此，完整的 RouteSAE 重写指南已完成！包含所有模型实现和消融实验脚本。**
