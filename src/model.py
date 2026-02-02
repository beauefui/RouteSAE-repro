import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopK(nn.Module):
    """
    TopK 稀疏自编码器
    
    原理:
        1. 编码: x -> 高维稀疏表示 (只保留 top-k 个激活)
        2. 解码: 稀疏表示 -> 重构 x
    
    为什么用 TopK 而不是 L1 正则化?
        - 精确控制稀疏度 (恰好 k 个非零特征)
        - 无需调节 L1 系数超参数
        - 计算成本可预测

    数据流:
    x (输入)
     ↓
    x - pre_bias          ← 去中心化：减去学习到的"输入均值"
     ↓
    encoder(·)            ← 线性变换到高维空间 (无内置 bias)
     ↓
    · + latent_bias       ← 调整激活阈值，影响哪些特征更容易被 TopK 选中
     ↓
    TopK(·)               ← 稀疏化：只保留 k 个最大值，其余置零
     ↓
    decoder(·)            ← 线性变换回原始维度 (无内置 bias)
     ↓
    · + pre_bias          ← 恢复中心：加回之前减去的偏移
     ↓
    x_hat (重构输出)

    Args:
        hidden_size: 输入维度 (LLM hidden size，例如 2048)
        latent_size: 稀疏表示维度 (通常是 hidden_size 的 4-8 倍，例如 16384)
        k: 保留的激活数量 (例如 64，即只有 64 个特征非零)
    """
    
    def __init__(self, hidden_size: int, latent_size: int, k: int) -> None:
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot be larger than latent_size ({latent_size})")
            
        super().__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        
        # pre_bias: 输入偏置，用于去中心化
        # 编码时减去，解码时加回 (对称设计)
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # latent_bias: 潜在空间偏置，加在 TopK 之前
        # 正的 bias → 该特征更容易进入 top-k
        # 负的 bias → 该特征更难被激活
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        
        # 编码器和解码器 (无内置 bias，用 pre_bias/latent_bias 代替)
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """初始化权重: 解码器 = 编码器的转置"""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """计算预激活值 (TopK 之前)"""
        centered_x = x - self.pre_bias          # 去中心化
        encoded = self.encoder(centered_x)       # 线性变换
        return encoded + self.latent_bias        # 加上 latent_bias
    
    def get_latents(self, pre_acts: torch.Tensor, infer_k: Optional[int] = None, theta: Optional[float] = None) -> torch.Tensor:
        """应用 TopK 稀疏化"""
        if infer_k is not None and theta is not None:
            raise ValueError('不能同时指定 infer_k 和 theta')
        
        if theta is not None:
            # 阈值模式: 保留 > theta 的值
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        else:
            # TopK 模式
            k = infer_k if infer_k is not None else self.k
            if k > pre_acts.size(-1):
                warnings.warn(f"k ({k}) 大于 latent_size，将使用全部 latents")
                k = pre_acts.size(-1)
            
            topk_values, topk_indices = torch.topk(pre_acts, k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        
        return latents

    def encode(self, x: torch.Tensor, infer_k: Optional[int] = None, theta: Optional[float] = None) -> torch.Tensor:
        """编码: 输入 -> 稀疏表示"""
        pre_acts = self.pre_acts(x)
        return self.get_latents(pre_acts, infer_k=infer_k, theta=theta)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """解码: 稀疏表示 -> 重构"""
        return self.decoder(latents) + self.pre_bias  # 加回 pre_bias
    
    def forward(self, x: torch.Tensor, infer_k: Optional[int] = None, theta: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: x -> (latents, x_hat)"""
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class RouteSAE(nn.Module):
    """
    带路由器的多层 SAE
    
    核心思想:
        传统 SAE: 每层单独训练一个 SAE
        RouteSAE: 共享一个 SAE + 路由器，动态为每个 token 选择最佳层
    
    架构图:
    多层激活 x: [batch, seq_len, n_layers, hidden_size]
         ↓
    ┌─────────────────────────────────────────┐
    │  1. Router (路由器)                      │
    │     输入: 聚合后的激活 (sum/mean)         │
    │     输出: 每层的权重 [batch, seq, n_layers] │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │  2. Routing (路由)                       │
    │     hard: 选择权重最大的 1 层            │
    │     soft: 加权融合所有层                 │
    │     输出: routed_x [batch, seq, hidden]  │
    └─────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────┐
    │  3. SAE (TopK 稀疏自编码器)              │
    │     对 routed_x 进行编码/解码            │
    │     输出: latents, x_hat                │
    └─────────────────────────────────────────┘
    
    优势:
        - 减少训练成本 (只需训练一个 SAE)
        - 发现跨层特征
        - 知道每个特征来自哪一层
    
    Args:
        hidden_size: LLM hidden size
        n_layers: LLM 总层数
        latent_size: SAE 稀疏表示维度
        k: TopK 激活数量
    """
    
    def __init__(self, hidden_size: int, n_layers: int, latent_size: int, k: int):
        if hidden_size <= 0 or latent_size <= 0 or n_layers <= 0 or k <= 0:
            raise ValueError("All dimensions must be positive")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot exceed latent_size ({latent_size})")
        if n_layers < 4:
            raise ValueError(f"n_layers ({n_layers}) should be at least 4 for meaningful routing")
            
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
        
        # 共享 SAE: 所有层共用一个 TopK SAE
        self.sae = TopK(hidden_size, latent_size, k)
    
    def get_router_weights(self, x: torch.Tensor, aggre: str) -> torch.Tensor:

        # 聚合多层激活作为路由器输入
        if aggre == 'sum':
            router_input = x.sum(dim=2)    # [batch, seq, hidden]
        elif aggre == 'mean':
            router_input = x.mean(dim=2)   # [batch, seq, hidden]
        else:
            raise ValueError(
                f'Unsupported aggregation method: {aggre}. '
                f'Expected one of ["sum", "mean"].'
            )
        
        # 计算路由权重并用 softmax 归一化
        logits = self.router(router_input)         # [batch, seq, n_layers]
        weights = F.softmax(logits, dim=-1)        # 归一化为概率
        
        return weights
    
    def route(self, x: torch.Tensor, router_weights: torch.Tensor, routing: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行路由: 根据权重选择/融合层激活 get sae inputs
        """
        if routing == 'hard':
            # ===== 硬路由: Select single best layer per token =====
            # 找到每个 token 权重最大的层
            max_weights, target_layers = router_weights.max(dim=-1)  # [batch, seq]
            
            # One-hot 编码: 记录选了哪层
            layer_weights = F.one_hot(target_layers, num_classes=self.n_routed_layers).float()
            
            # 从 x 中提取对应层的激活
            batch_size, seq_len = x.shape[:2]
            indices = target_layers.view(batch_size, seq_len, 1, 1)
            indices = indices.expand(-1, -1, -1, x.shape[-1])
            routed_x = torch.gather(x, 2, indices).squeeze(2)  # [batch, seq, hidden]
            
            # 乘以权重 (置信度)
            routed_x = routed_x * max_weights.unsqueeze(-1)
            
        elif routing == 'soft':
            # ===== 软路由: Weighted combination of all layers =====
            # 扩展权重维度以便广播
            weights_expanded = router_weights.unsqueeze(-1)  # [batch, seq, n_layers, 1]
            
            # 加权求和所有层
            routed_x = (x * weights_expanded).sum(dim=2)  # [batch, seq, hidden]
            
            # 层权重就是路由权重本身
            layer_weights = router_weights
            
        else:
            raise ValueError(
                f'Unsupported routing method: {routing}. '
                f'Expected one of ["hard", "soft"].'
            )
        
        return layer_weights, routed_x
    
    def forward(self, 
        x: torch.Tensor, 
        aggre: str = 'sum',
        routing: str = 'hard',
        infer_k: Optional[int] = None,
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 注意: 输入 x 已经在 get_outputs 中被切片为中间层 [batch, seq, n_routed_layers, hidden]
        # 不需要再次切片
        
        # 1. 计算路由权重
        router_weights = self.get_router_weights(x, aggre)
        
        # 2. 执行路由
        layer_weights, routed_x = self.route(x, router_weights, routing)
        
        # 3. 通过 SAE
        latents, x_hat = self.sae(routed_x, infer_k=infer_k, theta=theta)
        
        return layer_weights, routed_x, latents, x_hat, router_weights


def get_model(model_type: str, hidden_size: int, latent_size: int, k: int, n_layers: int = 16) -> nn.Module:

    if model_type == 'TopK':
        return TopK(hidden_size, latent_size, k)
    elif model_type == 'RouteSAE':
        return RouteSAE(hidden_size, n_layers, latent_size, k)
    else:
        raise ValueError(f"Unknown model type: {model_type}, please use 'TopK' or 'RouteSAE'")