"""
RouteSAE 工具函数模块
"""

import os
import json
import logging
import random
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# 随机种子与配置
# ============================================

def set_seed(seed: int) -> None:
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RouteSAE-repro')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='TopK', choices=['TopK', 'RouteSAE', 'MLSAE', 'Random', 'Vanilla', 'Gated', 'JumpReLU'])
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--latent_size', type=int, default=16384)
    parser.add_argument('--k', type=int, default=64)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--n_layers', type=int, default=16)
    parser.add_argument('--l1_coeff', type=float, default=5e-4, help='L1 regularization coefficient for Vanilla SAE')
    
    # Gated/JumpReLU args
    parser.add_argument('--threshold', type=float, default=1e-3, help='Initial threshold for JumpReLU')
    parser.add_argument('--bandwidth', type=float, default=1e-3, help='Bandwidth for JumpReLU gradients')
    
    # RouteSAE 参数
    parser.add_argument('--aggre', type=str, default='sum', choices=['sum', 'mean'])
    parser.add_argument('--routing', type=str, default='hard', choices=['hard', 'soft'])
    
    # 训练参数
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max_samples', type=int, default=None)
    
    # Wandb
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='RouteSAE-repro')
    
    # 评估参数
    parser.add_argument('--SAE_path', type=str, default=None)
    parser.add_argument('--metric', type=str, default='NormMSE')
    parser.add_argument('--infer_k', type=int, default=None)
    parser.add_argument('--theta', type=float, default=None)
    
    # API 参数 (interpreter)
    parser.add_argument('--api_base', type=str, default=None)
    parser.add_argument('--api_version', type=str, default='2024-02-15-preview')
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--engine', type=str, default='gpt-4o-mini')
    
    # 其他
    parser.add_argument('--output_dir', type=str, default='/mnt/nfs/zijie/routesae_repro_outputs')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


# ============================================
# 模型加载
# ============================================

def get_language_model(model_path: str, device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """加载 LLM 和 tokenizer"""
    logger.info(f"Loading language model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        output_hidden_states=True
    ).to(device)
    
    logger.info(f"Model loaded on {device}")
    return tokenizer, language_model


def get_outputs(
    cfg: argparse.Namespace, 
    batch: Tuple[torch.Tensor, torch.Tensor], 
    language_model: nn.Module, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
    """从 batch 提取 hidden states"""
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 根据模型类型选择 hidden states
    if cfg.model in ['TopK', 'Vanilla', 'Gated', 'JumpReLU']:
        hidden_states = outputs.hidden_states[cfg.layer]
    elif cfg.model in ['RouteSAE', 'MLSAE']:
        start_layer = cfg.n_layers // 4
        end_layer = cfg.n_layers * 3 // 4 + 1
        hidden_states = torch.stack(
            outputs.hidden_states[start_layer:end_layer], dim=0
        ).permute(1, 2, 0, 3)  # [batch, seq, n_layers, hidden]
    else:
        hidden_states = outputs.hidden_states[cfg.layer]
    
    return input_ids, attention_mask, outputs, hidden_states


# ============================================
# 预处理与损失函数
# ============================================

def pre_process(hidden_states: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """归一化 hidden states (零均值单位方差)"""
    mean = hidden_states.mean(dim=-1, keepdim=True)
    std = hidden_states.std(dim=-1, keepdim=True)
    x = (hidden_states - mean) / (std + eps)
    return x, mean, std


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """归一化 MSE 损失 (除以输入方差)"""
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()


def L1_loss(latents: torch.Tensor) -> torch.Tensor:
    """L1 稀疏正则化损失"""
    return latents.abs().mean()


# ============================================
# 模型操作
# ============================================

@torch.no_grad()
def unit_norm_decoder(model: nn.Module) -> None:
    """归一化解码器权重到单位范数"""
    from .model import TopK, RouteSAE, Vanilla, Gated, JumpReLU
    
    if isinstance(model, (TopK, Vanilla, Gated, JumpReLU)):
        model.decoder.weight.data /= model.decoder.weight.data.norm(dim=0, keepdim=True)
    elif isinstance(model, RouteSAE):
        model.sae.decoder.weight.data /= model.sae.decoder.weight.data.norm(dim=0, keepdim=True)


# ============================================
# Wandb 日志
# ============================================

def wandb_init(project: str, config: Dict[str, Any], name: str) -> None:
    """初始化 wandb"""
    wandb.init(project=project, config=config, name=name)
    logger.info(f"Initialized WandB project: {project}, run: {name}")


def log_layers(layer_weights: np.ndarray) -> None:
    """记录层权重分布到 wandb"""
    data = [[i, value] for i, value in enumerate(layer_weights[:], start=1)]
    table = wandb.Table(data=data, columns=['Layer', 'Weight'])
    wandb.log({
        'Layer Weights': wandb.plot.bar(
            table=table, label='Layer', value='Weight', title='Layer Weights'
        )
    })


# ============================================
# JSON 工具
# ============================================

def save_json(data: Dict[str, Any], path: str) -> None:
    """保存数据到 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f'Saved data to {path}')


# ============================================
# SAE 干预钩子
# ============================================

def hook_SAE(
    cfg: argparse.Namespace,
    model: nn.Module,
    hooked_module: nn.Module,
    set_high: Optional[List[Tuple[int, float, int]]] = None, 
    set_low: Optional[List[Tuple[int, float, int]]] = None,
    is_zero: bool = False
) -> List[RemovableHandle]:
    """
    注册 SAE 干预钩子
    
    Args:
        set_high: [(latent_idx, val, mode)] mode=0: add, mode=1: multiply
        set_low: [(latent_idx, val, mode)] mode=0: subtract, mode=1: divide
        is_zero: 是否将激活置零
    """
    def hook(module: nn.Module, _, outputs):
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = [outputs]
        
        original_dtype = unpack_outputs[0].dtype  # 保存原始数据类型
        
        if is_zero:
            unpack_outputs[0] = torch.zeros_like(unpack_outputs[0])
        else:
            x, mu, std = pre_process(unpack_outputs[0])
            x = x.float()  # 转为 float32 进行 SAE 计算
            latents = model.encode(x, cfg.infer_k, cfg.theta)

            if set_high:
                for (latent_idx, val, mode) in set_high:
                    if mode == 0:
                        latents[..., latent_idx] += val
                    elif mode == 1:
                        latents[..., latent_idx] *= val

            if set_low:
                for (latent_idx, val, mode) in set_low:
                    if mode == 0:
                        latents[..., latent_idx] -= val
                    elif mode == 1 and val != 0:
                        latents[..., latent_idx] /= val

            x_hat = model.decode(latents)
            reconstructed = x_hat * std + mu
            unpack_outputs[0] = reconstructed.to(original_dtype)  # 还原数据类型

        return tuple(unpack_outputs) if isinstance(outputs, tuple) else unpack_outputs[0]

    return [hooked_module.register_forward_hook(hook)]


class RouteHook:
    """RouteSAE 层级干预钩子"""
    
    def __init__(
        self,
        cfg: argparse.Namespace,
        layer_idx: int,
        model: nn.Module,
        batch_layer_weights: torch.Tensor,
        set_high: Optional[List[Tuple[int, float, int]]] = None,
        set_low: Optional[List[Tuple[int, float, int]]] = None,
        is_zero: bool = False  
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.model = model
        self.batch_layer_weights = batch_layer_weights
        self.set_high = set_high or []
        self.set_low = set_low or []
        self.is_zero = is_zero  

    def __call__(self, module: nn.Module, inputs: tuple, outputs: Union[torch.Tensor, tuple]):
        layer_mask = self.batch_layer_weights[
            :, :, self.layer_idx - self.model.start_layer + 1
        ].bool()

        if not layer_mask.any():
            return outputs

        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs
        
        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        original_dtype = output_tensor.dtype # 保存原始数据类型

        if self.is_zero:
            replace_mask = layer_mask.unsqueeze(-1).expand_as(output_tensor)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = 0
        else:
            x, mu, std = pre_process(output_tensor)
            x = x.float() # 转为 float32 进行 SAE 计算
            latents = self.model.sae.encode(x, self.cfg.infer_k, self.cfg.theta)

            for (idx, val, mode) in self.set_high:
                if mode == 0:
                    latents[..., idx] += val
                elif mode == 1:
                    latents[..., idx] *= val

            for (idx, val, mode) in self.set_low:
                if mode == 0:
                    latents[..., idx] -= val
                elif mode == 1 and val != 0:
                    latents[..., idx] /= val

            x_hat = self.model.sae.decode(latents)
            reconstruct = x_hat * std + mu
            reconstruct = reconstruct.to(original_dtype) # 还原数据类型

            replace_mask = layer_mask.unsqueeze(-1).expand_as(reconstruct)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = reconstruct[replace_mask]

        return tuple(outputs) if isinstance(outputs, list) else output_tensor


def hook_RouteSAE(
    cfg: argparse.Namespace,
    model: nn.Module,
    language_model: nn.Module,
    batch_layer_weights: torch.Tensor,
    set_high: Optional[List[Tuple[int, float, int]]] = None,
    set_low: Optional[List[Tuple[int, float, int]]] = None,
    is_zero: bool = False 
) -> List[RemovableHandle]:
    """注册 RouteSAE 多层干预钩子"""
    handles = []
    num_layers = batch_layer_weights.size(-1)

    for layer_idx in range(model.start_layer - 1, num_layers + model.start_layer - 1):
        if batch_layer_weights[:, :, layer_idx - model.start_layer + 1].any():
            layer_name = f'model.layers.{layer_idx}'
            try:
                module = language_model.get_submodule(layer_name)
            except AttributeError:
                raise ValueError(f'Submodule {layer_name} not found')

            hook_fn = RouteHook(
                cfg=cfg,
                layer_idx=layer_idx,
                model=model, 
                batch_layer_weights=batch_layer_weights,
                set_high=set_high,
                set_low=set_low,
                is_zero=is_zero
            )
            handles.append(module.register_forward_hook(hook_fn))
    return handles


# ============================================
# 学习率调度器
# ============================================

class LinearWarmupLR(LambdaLR):
    """线性 warmup + 常数 + 线性 decay 学习率调度器"""
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        num_warmup_steps: int, 
        num_training_steps: int, 
        max_lr: float
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        if step < self.num_warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step < self.num_training_steps - self.num_training_steps // 5:
            # Constant LR
            return 1.0
        else:
            # Linear decay in last 20%
            decay_steps = self.num_training_steps // 5
            steps_into_decay = step - (self.num_training_steps - self.num_training_steps // 5)
            return max(0.0, 1.0 - float(steps_into_decay) / float(decay_steps))