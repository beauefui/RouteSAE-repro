"""
RouteSAE 评估模块
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from .data import create_dataloader
from .model import TopK, RouteSAE
from .utils import (
    get_language_model,
    get_outputs,
    pre_process,
    Normalized_MSE_loss,
    hook_SAE,
    hook_RouteSAE,
    log_layers,
    wandb_init
)

logger = logging.getLogger(__name__)


class Evaluator:
    """SAE 评估器"""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 加载语言模型和数据
        logger.info(f"Initializing Evaluator for {cfg.model}")
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(
            cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length,
            max_samples=cfg.max_samples
        )
        
        sae_name = os.path.splitext(os.path.basename(cfg.SAE_path))[0]
        data_name = os.path.basename(cfg.data_path)
        self.title = f'{sae_name}_{data_name}_{cfg.metric}'
        
        self.config_dict = {
            'batch_size': cfg.batch_size,
            'infer_k': cfg.infer_k,
            'theta': cfg.theta
        }
        
        # 初始化模型
        if cfg.model == 'TopK':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.hooked_module = self.language_model.get_submodule(f'model.layers.{cfg.layer-1}')
        elif cfg.model == 'RouteSAE':
            self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
            self.layer_weights = np.zeros(cfg.n_layers // 2 + 1, dtype=float)
        else:
            raise ValueError(f'Invalid model: {cfg.model}. Expected one of [TopK, RouteSAE]')
        
        # 加载预训练权重
        logger.info(f"Loading SAE weights from {cfg.SAE_path}")
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.num_batches = 0
        self.total_loss = 0.0

    def KLDiv(self, logits_original: torch.Tensor, logits_reconstruct: torch.Tensor) -> torch.Tensor:
        """计算原始和重构 logits 之间的 KL 散度"""
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_reconstruct = F.log_softmax(logits_reconstruct, dim=-1)
        return F.kl_div(log_probs_reconstruct, probs_original, reduction='batchmean')
    
    @torch.no_grad()
    def run(self) -> float:
        """执行评估循环"""
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)

        logger.info(f"Starting evaluation with metric: {self.cfg.metric}")
        
        for batch_idx, batch in enumerate(self.dataloader):
            input_ids, attention_mask, outputs, hidden_states = get_outputs(
                self.cfg, batch, self.language_model, self.device
            )
            x, _, _ = pre_process(hidden_states)
            x = x.float()  # 确保 float32，因为 LLM 可能输出 bfloat16

            # Forward pass with SAE
            if self.cfg.model == 'RouteSAE':
                batch_layer_weights, x, latents, x_hat, _ = self.model(
                    x, self.cfg.aggre, self.cfg.routing, self.cfg.infer_k, self.cfg.theta
                )
                self.layer_weights += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()
            else:  # TopK
                latents, x_hat = self.model(x, self.cfg.infer_k, self.cfg.theta)
                batch_layer_weights = None
            
            # Compute metrics
            if self.cfg.metric == 'NormMSE': 
                loss = Normalized_MSE_loss(x, x_hat)
            elif self.cfg.metric == 'KLDiv':
                logits_original = outputs.logits
                
                # Hook SAE intervention
                if self.cfg.model == 'RouteSAE':
                    if self.cfg.routing == 'soft':
                        raise ValueError('RouteSAE with soft routing not supported for KLDiv')
                    handles = hook_RouteSAE(
                        self.cfg, self.model, self.language_model, batch_layer_weights
                    )
                else:  # TopK
                    handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                
                logits_reconstruct = self.language_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                
                for handle in handles:
                    handle.remove()
                
                loss = self.KLDiv(logits_original, logits_reconstruct)
                del input_ids, attention_mask
                torch.cuda.empty_cache()
            else:
                raise ValueError(f'Invalid metric: {self.cfg.metric}. Expected [NormMSE, KLDiv]')

            self.num_batches += 1
            self.total_loss += loss.item()

            if self.cfg.use_wandb:
                wandb.log({'Batch_loss': loss.item()})
            else:
                if batch_idx % 10 == 0:
                    logger.info(f'Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_loss = self.total_loss / self.num_batches
        
        if self.cfg.use_wandb:
            wandb.log({'Avg_loss': avg_loss})
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            wandb.finish()
        
        logger.info(f"Evaluation complete. Average loss: {avg_loss:.4f}")
        return avg_loss