"""
特征提取模块 (Apply)
"""

import os
import heapq
import logging
import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm

from .utils import get_language_model, get_outputs, pre_process, save_json
from .data import create_dataloader
from .model import get_model, TopK, RouteSAE

logger = logging.getLogger(__name__)

class Applier:
    """SAE 特征提取与应用 pipeline"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # 1. 加载 SAE 模型结构
        # 注意: 这里我们需要先实例化空模型，然后加载权重
        logger.info(f"Initializing model structure: {cfg.model}")
        try:
            self.model = get_model(
                cfg.model, 
                cfg.hidden_size, 
                cfg.latent_size, 
                cfg.k, 
                cfg.n_layers
            )
        except TypeError:
            # 兼容可能的参数差异
            if cfg.model == 'TopK':
                self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            elif cfg.model == 'RouteSAE':
                self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
            else:
                raise ValueError(f"Unknown model: {cfg.model}")

        # 2. 加载预训练权重
        if not cfg.SAE_path or not os.path.exists(cfg.SAE_path):
            raise FileNotFoundError(f"SAE model not found at {cfg.SAE_path}")
            
        logger.info(f"Loading SAE weights from {cfg.SAE_path}")
        state_dict = torch.load(cfg.SAE_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # 3. 准备 Tokenizer (LLM 会在 get_context 中按需加载以节省显存，或者在这里加载)
        # 考虑到显存，我们尽量复用. 为了简单，这里暂不加载 LLM，使用时加载
        self.tokenizer = None
        self.language_model = None

    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 10.0, 
        max_length: int = 64, 
        max_per_token: int = 2, 
        lines: int = 4,
        output_path: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        提取每个潜在特征的 Top-K 激活上下文
        
        Args:
            threshold: 激活阈值
            max_length: 上下文最大长度
            max_per_token: 每个 token 类型保留的最大上下文数 (避免重复)
            lines: 每个特征至少需要的上下文数量 (少于此数的特征将被丢弃)
            output_path: 输出 JSON 路径
        
        Returns:
            (total_latents, output_path)
        """
        if output_path is None:
            # 默认输出路径
            model_name = os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]
            output_path = os.path.join(
                self.cfg.output_dir, 
                'contexts', 
                f'{model_name}_contexts.json'
            )

        logger.info(f"Extracting feature contexts (threshold={threshold})")
        
        # 加载 LLM 和数据
        if self.language_model is None:
            self.tokenizer, self.language_model = get_language_model(self.cfg.model_path, self.device)
            
        dataloader = create_dataloader(
            self.cfg.data_path, 
            self.tokenizer, 
            self.cfg.batch_size, 
            self.cfg.max_length,
            max_samples=self.cfg.max_samples # 如果配置了 max_samples
        )
        
        # 结果存储: latent_idx -> token_class -> heap(activation, context)
        latent_context_map = defaultdict(lambda: defaultdict(list))
        
        sentence_enders = {'.', '!', '?', '<|end_of_text|>', '"', '\n'}
        half_length = max_length // 2

        # 辅助函数: 寻找句子边界
        def find_sentence_bounds(seq_pos: int, tokens: List[str]) -> Tuple[int, int]:
            start_pos = seq_pos
            # 向前找
            while start_pos > 0 and tokens[start_pos - 1] not in sentence_enders:
                start_pos -= 1
            # 向后找
            end_pos = seq_pos
            while end_pos < len(tokens) - 1 and tokens[end_pos] not in sentence_enders:
                end_pos += 1
            if end_pos < len(tokens):
                end_pos += 1
            return start_pos, end_pos

        # 辅助函数: 处理并存储上下文
        def process_and_store_context(
            latent_dim: int, seq_pos: int, activation_value: float, tokens: List[str]
        ) -> None:
            start_pos, end_pos = find_sentence_bounds(seq_pos, tokens)
            sentence_tokens = tokens[start_pos:end_pos]
            sentence_length = len(sentence_tokens)
            
            # 如果句子太长，截取中心部分
            if sentence_length > max_length:
                activated_token_idx = seq_pos - start_pos
                left_context_start = max(0, activated_token_idx - half_length)
                right_context_end = min(sentence_length, activated_token_idx + half_length + 1)
                context_tokens = sentence_tokens[left_context_start:right_context_end]
                activated_token_idx -= left_context_start
            else:
                context_tokens = sentence_tokens
                activated_token_idx = seq_pos - start_pos
            
            if not (0 <= activated_token_idx < len(context_tokens)):
                return

            # 标记激活 Token
            context_tokens = context_tokens.copy()
            raw_token = context_tokens[activated_token_idx]
            
            # 清理 special tokens
            if raw_token in ['<|end_of_text|>', '']:
                return

            context_tokens[activated_token_idx] = f'<ACTIVATED>{raw_token}</ACTIVATED>'
            
            # 重建字符串
            context_text = self.tokenizer.convert_tokens_to_string(context_tokens).strip()
            
            # 获取 token 类型 (用于多样性过滤)
            token_class = raw_token.strip().lower()
            
            # 存入堆 (最小堆，保留最大的 activation)
            # 这里的逻辑是保留 Top-K 个，所以如果堆满了，通过 heappushpop 维持
            # 但我们需要多样性，所以是对每个 token_class 维护一个堆
            heap = latent_context_map[latent_dim][token_class]
            heapq.heappush(heap, (activation_value, context_text))
            
            # 如果超过该 token 类型的最大限制，移除最小激活的那个
            if len(heap) > max_per_token:
                heapq.heappop(heap)

        # 遍历数据
        logger.info(f"Processing batches...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            # 获取 LLM 隐藏层状态
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            
            # 预处理 (归一化)
            x, _, _ = pre_process(hidden_states)
            
            # 确保 x 的 dtype 与 model 一致 (解决 BFloat16 vs Float32 问题)
            model_dtype = next(self.model.parameters()).dtype
            x = x.to(dtype=model_dtype)
            
            # 获取 SAE 激活 (latents)
            if self.cfg.model == 'RouteSAE':
                # RouteSAE 返回 (layer_weights, routed_x, latents, x_hat, router_weights)
                # 我们只需要 latents
                _, _, latents, _, _ = self.model(
                    x, 
                    aggre=self.cfg.aggre, 
                    routing=self.cfg.routing, 
                    infer_k=self.cfg.infer_k, 
                    theta=self.cfg.theta
                )
            else:
                # TopK 返回 (latents, x_hat)
                latents, _ = self.model(x, infer_k=self.cfg.infer_k, theta=self.cfg.theta)
                
            # latents shape: [batch, seq_len, latent_size]
            batch_size = latents.shape[0]
            
            # 找到大于阈值的位置
            # 为了效率，可以只处理非零元素 (已经是 TopK 了，大部分是 0)
            # 这里先用 threshold 过滤
            positions = (latents > threshold)
            
            # 遍历 Batch
            for i in range(batch_size):
                # 获取当前样本的 tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                
                # 获取该样本所有激活位置 (seq_pos, latent_dim)
                # 使用 nonzero 可能会很大，但 TopK 保证了每一行只有 k 个非零
                # 加上 threshold 过滤后更少
                activated_indices = torch.nonzero(positions[i], as_tuple=False)
                
                for idx in activated_indices:
                    seq_pos, latent_dim = idx[0].item(), idx[1].item()
                    activation_value = latents[i, seq_pos, latent_dim].item()
                    
                    process_and_store_context(latent_dim, seq_pos, activation_value, tokens)

        # 整理和过滤结果
        logger.info("Filtering and sorting contexts...")
        filtered_latent_context = {}
        
        for latent_dim, token_dict in latent_context_map.items():
            # 统计该特征的总上下文数
            total_contexts = sum(len(contexts) for contexts in token_dict.values())
            
            # 如果上下文太少，忽略该特征 (可能是噪声)
            if total_contexts < lines:
                continue
                
            sorted_token_dict = {}
            for t_class, heap in token_dict.items():
                # 堆转列表并按激活值降序排序
                contexts_list = sorted(list(heap), key=lambda x: x[0], reverse=True)
                sorted_token_dict[t_class] = [
                    {'activation': act, 'context': ctx} for act, ctx in contexts_list
                ]
            
            # 按 token 字母顺序排序 token entries? 或者保持字典序
            filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

        total_latents = len(filtered_latent_context)
        # 按 latent_dim 排序
        sorted_latent_context = dict(sorted(filtered_latent_context.items(), key=lambda item: int(item[0])))

        output_data = {
            'model': self.cfg.model,
            'source_model': self.cfg.model_path,
            'sae_path': self.cfg.SAE_path,
            'total_latents': total_latents,
            'threshold': threshold,
            'max_length': max_length,
            'max_per_token': max_per_token,
            'lines': lines,
            'latent_context_map': sorted_latent_context,
        }
        
        save_json(output_data, output_path)
        logger.info(f"Extracted {total_latents} features with sufficient contexts")
        return total_latents, output_path
