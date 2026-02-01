"""
数据加载模块: OpenWebText 数据集
"""

import os
import json
import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OpenWebTextDataset(Dataset):
    def __init__(
        self, 
        folder_path: str, 
        tokenizer: AutoTokenizer,
        max_length: int,
        max_samples: Optional[int] = None,
        keyword: str = 'text'
    ):
        """
        Args:
            folder_path: 包含 .jsonl 文件的目录
            tokenizer: HuggingFace tokenizer
            max_length: 最大 token 长度
            max_samples: 最大样本数 (None = 全部)
            keyword: JSONL 中文本字段的 key
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword = keyword
        
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 查找 JSONL 文件
        file_list = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        if not file_list:
            raise ValueError(f"No .jsonl files found in {folder_path}")
        
        # 加载并分块 (预先 tokenize)
        logger.info(f"Loading dataset from {folder_path} ({len(file_list)} files)")
        self.data: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        for filename in file_list:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Loading {filename}"):
                    try:
                        record = json.loads(line.strip())
                        text = record.get(self.keyword, '')
                        if not text:
                            continue
                        
                        # 按 word 分块
                        words = text.split()
                        for i in range(0, len(words), self.max_length):
                            chunk = ' '.join(words[i:i + self.max_length])
                            
                            inputs = self.tokenizer(
                                chunk,
                                return_tensors='pt',
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True
                            )
                            input_ids = inputs['input_ids'].squeeze(0)
                            attention_mask = inputs['attention_mask'].squeeze(0)
                            self.data.append((input_ids, attention_mask))
                            
                            if max_samples and len(self.data) >= max_samples:
                                break
                                
                    except json.JSONDecodeError:
                        logger.warning(f'Error decoding JSON in file: {filepath}')
                        continue
                    
                    if max_samples and len(self.data) >= max_samples:
                        break
                        
            if max_samples and len(self.data) >= max_samples:
                break
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]


def create_dataloader(
    folder_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    max_samples: Optional[int] = None,
    keyword: str = 'text'
) -> DataLoader:
    dataset = OpenWebTextDataset(
        folder_path, 
        tokenizer, 
        max_length,
        max_samples,
        keyword
    )
    
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return input_ids, attention_mask
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )