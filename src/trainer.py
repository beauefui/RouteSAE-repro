"""
RouteSAE 训练模块
"""

import argparse
import logging
import os

import numpy as np
import torch
import wandb
from torch.optim import Adam

from .data import create_dataloader
from .model import TopK, RouteSAE
from .utils import (
    get_language_model,
    get_outputs,
    pre_process,
    Normalized_MSE_loss,
    unit_norm_decoder,
    log_layers,
    wandb_init,
    LinearWarmupLR
)

logger = logging.getLogger(__name__)


class Trainer:
    """SAE 训练器"""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        logger.info(f"Loading LLM: {cfg.model_path}")
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(
            cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length,
            max_samples=cfg.max_samples
        )

        # 构建实验名称
        data_name = cfg.data_path.split("/")[-1]
        model_name = cfg.model_path.split("/")[-1]
        self.title = f'{model_name}_{data_name}_{cfg.latent_size}'

        self.config_dict = {
            'batch_size': cfg.batch_size,
            'num_epochs': cfg.num_epochs,
            'lr': cfg.lr,
            'steps': cfg.steps
        }       

        # 初始化模型
        if cfg.model == 'TopK':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.title = f'L{cfg.layer}_K{cfg.k}_{self.title}'
        elif cfg.model == 'MLSAE':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.title = f'ML_K{cfg.k}_{self.title}'
        elif cfg.model == 'RouteSAE':
            self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
            self.title = f'{cfg.aggre}_{cfg.routing}_K{cfg.k}_{self.title}'
            self.layer_weights = np.zeros(cfg.n_layers // 2 + 1, dtype=float)
        else:
            raise ValueError(f'Invalid model: {cfg.model}. Expected one of [TopK, RouteSAE, MLSAE]')

        self.model.to(self.device)
        self.model.train()
        
        # 优化器和调度器
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(
            self.optimizer, num_warmup_steps, num_training_steps, cfg.lr
        )
        
        logger.info(f"Model initialized: {self.title}")
    
    def run(self) -> float:
        """执行训练循环"""
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        
        logger.info("Starting training")
        curr_loss = 0.0
        unit_norm_decoder(self.model)
        
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                _, _, _, hidden_states = get_outputs(
                    self.cfg, batch, self.language_model, self.device
                )
                
                # Handling MLSAE (multi-layer training)
                if self.cfg.model == 'MLSAE':
                    # hidden_states: [batch, seq, n_layers, hidden]
                    # We iterate over the n_layers dimension
                    num_layers = hidden_states.shape[2]
                    for i in range(num_layers):
                        layer_hidden = hidden_states[:, :, i, :]
                        x, _, _ = pre_process(layer_hidden)
                        x = x.float()
                        
                        _, x_hat = self.model(x)
                        loss = Normalized_MSE_loss(x, x_hat)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        curr_loss = loss.item()
                    
                    # Scheduler step after all layers in batch (or per layer? Check logic. Reference code did step per batch loop)
                    # Reference code: self.scheduler.step() is outside inner layer loop
                    self.scheduler.step()

                else:
                    x, _, _ = pre_process(hidden_states)
                    x = x.float()  # 确保 float32，因为 LLM 可能输出 bfloat16
    
                    # Forward pass
                    if self.cfg.model == 'RouteSAE':
                        batch_layer_weights, x, _, x_hat, _ = self.model(
                            x, self.cfg.aggre, self.cfg.routing
                        )
                        self.layer_weights += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()
                    else:  # TopK
                        latents, x_hat = self.model(x)
                    
                    # Compute loss
                    loss = Normalized_MSE_loss(x, x_hat)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    curr_loss = loss.item()

                if batch_idx % self.cfg.steps == 0:
                    unit_norm_decoder(self.model)
                
                # Logging
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                else:
                    if batch_idx % self.cfg.steps == 0:
                        logger.info(f'Epoch {epoch+1}/{self.cfg.num_epochs}, Batch {batch_idx+1}, Loss: {curr_loss:.4f}')
        
        # Finalize
        if self.cfg.use_wandb:
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            wandb.finish()

        unit_norm_decoder(self.model)
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        # Modified model naming as requested: SAEType_Dataset.pt
        data_name = self.cfg.data_path.split("/")[-1]
        save_name = f"{self.cfg.model}_{data_name}.pt"
        save_path = os.path.join(self.cfg.output_dir, save_name)
        
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'Training complete. Model saved to {save_path}')
        
        return curr_loss