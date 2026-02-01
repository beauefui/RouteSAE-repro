# RouteSAE-repro 项目完整文档

本文档详细解析项目中的每个 Python 文件及其功能，帮助你理解整个代码流程。

---

## 目录

1. [项目结构](#项目结构)
2. [数据流总览](#数据流总览)
3. [model.py 详解](#modelpy-详解)
4. [data.py 详解](#datapy-详解)
5. [utils.py 详解](#utilspy-详解)
6. [trainer.py 详解](#trainerpy-详解)
7. [evaluator.py 详解](#evaluatorpy-详解)
8. [interpreter.py 详解](#interpreterpy-详解)
9. [训练流程](#训练流程)
10. [评估流程](#评估流程)

---

## 项目结构

```
src/
├── __init__.py       # 包初始化
├── model.py          # SAE 模型定义 (278 行)
├── data.py           # 数据加载 (129 行)
├── utils.py          # 工具函数 (~350 行)
├── trainer.py        # 训练器 (130 行)
├── evaluator.py      # 评估器 (145 行)
└── interpreter.py    # GPT-4o 特征解释 (225 行)
```

---

## 数据流总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                           训练流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 文本数据 (.jsonl)                                               │
│       ↓                                                             │
│  2. Tokenize (data.py)                                              │
│       ↓                                                             │
│  3. LLM 前向传播，提取 hidden_states (utils.py: get_outputs)         │
│       ↓                                                             │
│  4. 归一化预处理 (utils.py: pre_process)                            │
│       ↓                                                             │
│  5. SAE 编码 → TopK 稀疏化 → 解码 (model.py)                        │
│       ↓                                                             │
│  6. 计算 Normalized MSE Loss (utils.py)                             │
│       ↓                                                             │
│  7. 反向传播更新 SAE 权重                                            │
│       ↓                                                             │
│  8. 周期性归一化解码器 (utils.py: unit_norm_decoder)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## model.py 详解

定义两种 SAE 模型：TopK 和 RouteSAE。

### TopK 类

**作用：** 标准的 TopK 稀疏自编码器

#### 核心参数

```python
class TopK(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int, k: int):
        # pre_bias: 输入偏置，编码时减去，解码时加回
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))  # [2048]
        
        # latent_bias: 潜在空间偏置，在 TopK 之前加上
        # 正的 bias → 特征更容易被选中
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))  # [16384]
        
        # 编码器/解码器 (无内置 bias)
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
```

#### pre_acts() - 预激活计算

```python
def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
    centered_x = x - self.pre_bias      # 去中心化
    encoded = self.encoder(centered_x)   # 线性变换
    return encoded + self.latent_bias    # 加上 latent_bias
```

**数学公式：** `pre_acts = W_enc @ (x - b_pre) + b_latent`

#### get_latents() - TopK 稀疏化

```python
def get_latents(self, pre_acts, infer_k=None, theta=None):
    if theta is not None:
        # 阈值模式：保留 > theta 的值
        latents = torch.where(pre_acts > theta, pre_acts, 0)
    else:
        # TopK 模式：只保留最大的 k 个
        k = infer_k or self.k
        topk_values, topk_indices = torch.topk(pre_acts, k, dim=-1)
        latents = torch.zeros_like(pre_acts)
        latents.scatter_(-1, topk_indices, topk_values)
    return latents
```

**关键点：**
- 训练时固定 k=64，保证稀疏度一致
- 推理时可以用 `infer_k` 或 `theta` 调整

#### decode() - 解码

```python
def decode(self, latents: torch.Tensor) -> torch.Tensor:
    return self.decoder(latents) + self.pre_bias  # 加回 pre_bias
```

**数学公式：** `x_hat = W_dec @ latents + b_pre`

#### forward() - 完整前向传播

```python
def forward(self, x, infer_k=None, theta=None):
    latents = self.encode(x, infer_k, theta)
    x_hat = self.decode(latents)
    return latents, x_hat
```

---

### RouteSAE 类

**作用：** 多层融合 + 路由机制的 SAE

#### 核心概念

传统方法：每层训练一个独立的 SAE
RouteSAE：共享一个 SAE + 路由器动态选择最佳层

#### 核心参数

```python
class RouteSAE(nn.Module):
    def __init__(self, hidden_size, n_layers, latent_size, k):
        # 只关注中间层 (n/4 到 3n/4)
        self.start_layer = n_layers // 4       # 16层: 第4层
        self.end_layer = n_layers * 3 // 4 + 1 # 16层: 第13层
        
        # 路由器：预测每层的权重
        self.router = nn.Linear(hidden_size, self.n_routed_layers, bias=False)
        
        # 共享 SAE
        self.sae = TopK(hidden_size, latent_size, k)
```

#### get_router_weights() - 计算路由权重

```python
def get_router_weights(self, x, aggre):
    if aggre == 'sum':
        router_input = x.sum(dim=2)   # 聚合多层
    elif aggre == 'mean':
        router_input = x.mean(dim=2)
    
    logits = self.router(router_input)     # [batch, seq, n_layers]
    weights = F.softmax(logits, dim=-1)    # 归一化为概率
    return weights
```

#### route() - 执行路由

```python
def route(self, x, router_weights, routing):
    if routing == 'hard':
        # 硬路由：选择权重最大的那一层
        max_weights, target_layers = router_weights.max(dim=-1)
        layer_weights = F.one_hot(target_layers, num_classes=self.n_routed_layers)
        routed_x = torch.gather(x, 2, indices).squeeze(2)
        routed_x = routed_x * max_weights.unsqueeze(-1)  # 乘以置信度
        
    elif routing == 'soft':
        # 软路由：加权融合所有层
        weights_expanded = router_weights.unsqueeze(-1)
        routed_x = (x * weights_expanded).sum(dim=2)
        layer_weights = router_weights
    
    return layer_weights, routed_x
```

**硬路由 vs 软路由：**
| 类型 | 计算量 | 效果 |
|------|--------|------|
| hard | 低 | 每个 token 只用一层 |
| soft | 高 | 加权融合所有层 |

---

## data.py 详解

负责加载和预处理 OpenWebText 数据集。

### OpenWebTextDataset 类

```python
class OpenWebTextDataset(Dataset):
    def __init__(self, folder_path, tokenizer, max_length, max_samples=None, keyword='text'):
        self.data = []  # 存储 (input_ids, attention_mask) 元组
        
        for filename in file_list:
            with open(filepath, 'r') as f:
                for line in f:
                    record = json.loads(line)
                    text = record.get(self.keyword, '')
                    
                    # 按 word 分块
                    words = text.split()
                    for i in range(0, len(words), max_length):
                        chunk = ' '.join(words[i:i + max_length])
                        
                        # Tokenize
                        inputs = tokenizer(
                            chunk,
                            return_tensors='pt',
                            max_length=max_length,
                            padding='max_length',
                            truncation=True
                        )
                        self.data.append((inputs['input_ids'], inputs['attention_mask']))
```

**分块策略：**
- 按 word 数量分块 (max_length 个词)
- 每个 chunk 独立 tokenize
- 这样保证每个样本大小一致

### create_dataloader()

```python
def create_dataloader(folder_path, tokenizer, batch_size, max_length, ...):
    dataset = OpenWebTextDataset(...)
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return input_ids, attention_mask
    
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
```

---

## utils.py 详解

包含所有工具函数，是项目的核心。

### 1. set_seed() - 随机种子

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

**作用：** 保证实验可复现

### 2. parse_args() - 命令行参数解析

```python
def parse_args():
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument('--model', choices=['TopK', 'RouteSAE'])
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--latent_size', type=int, default=16384)
    parser.add_argument('--k', type=int, default=64)
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    ...
```

### 3. get_language_model() - 加载 LLM

```python
def get_language_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 关键：output_hidden_states=True
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        output_hidden_states=True  # 返回每层的隐藏状态
    ).to(device)
    
    return tokenizer, language_model
```

### 4. get_outputs() - 提取 hidden states

```python
def get_outputs(cfg, batch, language_model, device):
    input_ids, attention_mask = batch
    
    with torch.no_grad():
        outputs = language_model(input_ids, attention_mask)
    
    # TopK: 取单层
    if cfg.model == 'TopK':
        hidden_states = outputs.hidden_states[cfg.layer]  # [batch, seq, hidden]
    
    # RouteSAE: 取多层
    elif cfg.model == 'RouteSAE':
        start = cfg.n_layers // 4
        end = cfg.n_layers * 3 // 4 + 1
        hidden_states = torch.stack(outputs.hidden_states[start:end])
        hidden_states = hidden_states.permute(1, 2, 0, 3)  # [batch, seq, layers, hidden]
    
    return input_ids, attention_mask, outputs, hidden_states
```

**outputs.hidden_states 结构：**
- 元组，长度 = n_layers + 1
- `[0]` = embedding 输出
- `[1]` ~ `[n_layers]` = 各层输出

### 5. pre_process() - 归一化

```python
def pre_process(hidden_states, eps=1e-6):
    mean = hidden_states.mean(dim=-1, keepdim=True)
    std = hidden_states.std(dim=-1, keepdim=True)
    x = (hidden_states - mean) / (std + eps)
    return x, mean, std
```

**作用：**
- 零均值单位方差归一化
- 返回 mean/std 供后续反归一化

### 6. Normalized_MSE_loss() - 损失函数

```python
def Normalized_MSE_loss(x, x_hat):
    mse = ((x_hat - x) ** 2).mean(dim=-1)
    variance = (x ** 2).mean(dim=-1)
    return (mse / variance).mean()
```

**公式：** `L = E[||x_hat - x||² / ||x||²]`

**优点：** 对不同 scale 的输入公平

### 7. unit_norm_decoder() - 解码器归一化

```python
@torch.no_grad()
def unit_norm_decoder(model):
    if isinstance(model, TopK):
        W = model.decoder.weight.data
        model.decoder.weight.data = W / W.norm(dim=0, keepdim=True)
    elif isinstance(model, RouteSAE):
        W = model.sae.decoder.weight.data
        model.sae.decoder.weight.data = W / W.norm(dim=0, keepdim=True)
```

**作用：** 防止特征抑制 (feature suppression)

**原理：** 如果某列范数过大，对应特征会主导重构，其他特征被抑制

### 8. LinearWarmupLR - 学习率调度器

```python
class LinearWarmupLR(LambdaLR):
    def lr_lambda(self, step):
        if step < self.num_warmup_steps:
            # Warmup: 线性增加
            return step / self.num_warmup_steps
        elif step < self.num_training_steps * 0.8:
            # 中间: 恒定
            return 1.0
        else:
            # 最后 20%: 线性衰减
            decay_progress = (step - 0.8*total) / (0.2*total)
            return 1.0 - decay_progress
```

```
LR
 │    ______________________
 │   /                      \
 │  /                        \
 │ /                          \
 └─────────────────────────────→ Step
   ↑     ↑                    ↑
   0   5%                   80%  100%
      warmup    constant    decay
```

### 9. hook_SAE() - 干预钩子

```python
def hook_SAE(cfg, model, hooked_module, set_high=None, set_low=None, is_zero=False):
    def hook(module, _, outputs):
        if is_zero:
            outputs[0] = torch.zeros_like(outputs[0])
        else:
            x, mu, std = pre_process(outputs[0])
            latents = model.encode(x, cfg.infer_k, cfg.theta)
            
            # 干预特定特征
            for (idx, val, mode) in (set_high or []):
                if mode == 0: latents[..., idx] += val
                if mode == 1: latents[..., idx] *= val
            
            x_hat = model.decode(latents)
            outputs[0] = x_hat * std + mu  # 反归一化
        
        return outputs
    
    return [hooked_module.register_forward_hook(hook)]
```

**使用场景：**
1. **KLDiv 评估：** 用 SAE 重构替换 LLM 层输出
2. **特征干预：** 放大/缩小特定特征

---

## trainer.py 详解

### Trainer.__init__()

```python
def __init__(self, cfg):
    # 1. 加载 LLM
    self.tokenizer, self.language_model = get_language_model(cfg.model_path, device)
    
    # 2. 创建数据加载器
    self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, ...)
    
    # 3. 初始化 SAE 模型
    if cfg.model == 'TopK':
        self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
    elif cfg.model == 'RouteSAE':
        self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
    
    # 4. 优化器 + 调度器
    self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
    self.scheduler = LinearWarmupLR(self.optimizer, warmup_steps, total_steps, cfg.lr)
```

### Trainer.run()

```python
def run(self):
    unit_norm_decoder(self.model)  # 初始归一化
    
    for epoch in range(self.cfg.num_epochs):
        for batch_idx, batch in enumerate(self.dataloader):
            
            # 1. 获取 hidden states
            _, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            
            # 2. 预处理
            x, _, _ = pre_process(hidden_states)
            
            # 3. SAE 前向
            if self.cfg.model == 'RouteSAE':
                layer_weights, x, _, x_hat, _ = self.model(x, cfg.aggre, cfg.routing)
            else:
                latents, x_hat = self.model(x)
            
            # 4. 计算损失
            loss = Normalized_MSE_loss(x, x_hat)
            
            # 5. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # 6. 周期性归一化解码器
            if batch_idx % self.cfg.steps == 0:
                unit_norm_decoder(self.model)
    
    # 保存模型
    torch.save(self.model.state_dict(), save_path)
```

---

## evaluator.py 详解

### Evaluator.__init__()

与 Trainer 类似，但：
1. 加载预训练的 SAE 权重
2. 设置 `model.eval()` 模式

```python
def __init__(self, cfg):
    # ... 加载 LLM 和数据 ...
    
    # 加载预训练权重
    self.model.load_state_dict(torch.load(cfg.SAE_path))
    self.model.eval()
    
    # TopK 需要 hook 的模块
    if cfg.model == 'TopK':
        self.hooked_module = language_model.get_submodule(f'model.layers.{cfg.layer-1}')
```

### Evaluator.run()

```python
@torch.no_grad()
def run(self):
    for batch in self.dataloader:
        input_ids, attention_mask, outputs, hidden_states = get_outputs(...)
        x, _, _ = pre_process(hidden_states)
        
        # SAE 前向
        latents, x_hat = self.model(x, cfg.infer_k, cfg.theta)
        
        # 计算指标
        if cfg.metric == 'NormMSE':
            loss = Normalized_MSE_loss(x, x_hat)
        
        elif cfg.metric == 'KLDiv':
            logits_original = outputs.logits
            
            # 注册 hook，替换层输出
            handles = hook_SAE(cfg, self.model, self.hooked_module)
            
            # 重新前向传播
            logits_new = self.language_model(input_ids, attention_mask).logits
            
            # 移除 hook
            for h in handles: h.remove()
            
            # 计算 KL 散度
            loss = self.KLDiv(logits_original, logits_new)
    
    return avg_loss
```

---

## interpreter.py 详解

使用 GPT-4o 对 SAE 特征进行语义解释。

### 核心流程

1. 加载特征激活数据
2. 随机采样 N 个特征
3. 对每个特征，收集高激活的 token 和上下文
4. 构造 prompt，调用 GPT-4o
5. 解析响应，分类为 low-level / high-level / undiscernible

### construct_prompt()

```python
def construct_prompt(self, tokens_info):
    prompt = '''
    We are analyzing the activation levels of features...
    
    Features are categorized as:
    A. Low-level features (word-level polysemy disambiguation)
    B. High-level features (long-range pattern formation)
    C. Undiscernible features (noise)
    
    Token: {token} | Activation: {value} | Context: {context}
    ...
    
    Provide your response in format:
    Feature category: [Low-level/High-level/Undiscernible]
    Score: [5/4/3/2/1]
    Explanation: [Your brief explanation]
    '''
    return prompt
```

---

## 训练流程

```
┌─────────────────────────────────────────────────────┐
│ 1. python train.py --model TopK --layer 12 ...      │
├─────────────────────────────────────────────────────┤
│                                                     │
│   parse_args()                                      │
│       ↓                                             │
│   Trainer.__init__()                                │
│       ├─ get_language_model()  → 加载 Llama         │
│       ├─ create_dataloader()   → 加载 OpenWebText   │
│       └─ TopK()                → 初始化 SAE         │
│       ↓                                             │
│   Trainer.run()                                     │
│       ├─ for epoch:                                 │
│       │   ├─ for batch:                             │
│       │   │   ├─ get_outputs() → hidden_states      │
│       │   │   ├─ pre_process() → 归一化             │
│       │   │   ├─ model()       → latents, x_hat     │
│       │   │   ├─ Normalized_MSE_loss()              │
│       │   │   ├─ backward()                         │
│       │   │   └─ unit_norm_decoder()                │
│       └─ torch.save()                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 评估流程

```
┌─────────────────────────────────────────────────────┐
│ 1. python evaluate.py --metric KLDiv ...            │
├─────────────────────────────────────────────────────┤
│                                                     │
│   Evaluator.__init__()                              │
│       ├─ 加载 LLM                                   │
│       ├─ 加载 SAE 权重                              │
│       └─ model.eval()                               │
│       ↓                                             │
│   Evaluator.run()                                   │
│       ├─ for batch:                                 │
│       │   ├─ get_outputs() → hidden_states, logits  │
│       │   ├─ pre_process()                          │
│       │   ├─ model()  → x_hat                       │
│       │   │                                         │
│       │   ├─ [NormMSE]: Normalized_MSE_loss()       │
│       │   │                                         │
│       │   └─ [KLDiv]:                               │
│       │       ├─ hook_SAE()  → 注册钩子             │
│       │       ├─ LLM 重新前向 → 新 logits           │
│       │       ├─ 移除钩子                           │
│       │       └─ KLDiv(old_logits, new_logits)      │
│       │                                             │
│       └─ return avg_loss                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 关键概念速查

| 概念 | 解释 | 位置 |
|------|------|------|
| TopK 稀疏化 | 只保留激活值最大的 k 个 | model.py |
| pre_bias | 对称偏置：编码减，解码加 | model.py |
| latent_bias | 控制特征激活难度 | model.py |
| Router | 预测每层重要性权重 | model.py (RouteSAE) |
| pre_process | 零均值单位方差归一化 | utils.py |
| unit_norm_decoder | 防止特征抑制 | utils.py |
| hook_SAE | 干预 LLM 层输出 | utils.py |
| LinearWarmupLR | warmup + 常数 + decay | utils.py |

---

## 入口脚本模板

### train.py

```python
from src.utils import parse_args, set_seed
from src.trainer import Trainer

if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    final_loss = trainer.run()
    print(f"Training complete. Final loss: {final_loss:.4f}")
```

### evaluate.py

```python
from src.utils import parse_args, set_seed
from src.evaluator import Evaluator

if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    evaluator = Evaluator(cfg)
    avg_loss = evaluator.run()
    print(f"Evaluation complete. Avg loss: {avg_loss:.4f}")
```

---

## 运行示例

```bash
# 激活环境
conda activate routesaerepro

# 训练 TopK SAE (第12层)
python train.py \
    --model TopK \
    --model_path /path/to/Llama-3.2-1B-Instruct \
    --data_path /path/to/openwebtext \
    --layer 12 \
    --k 64 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --batch_size 32 \
    --num_epochs 1 \
    --lr 5e-4 \
    --output_dir ./outputs

# 评估
python evaluate.py \
    --model TopK \
    --SAE_path ./outputs/L12_K64_xxx.pt \
    --metric NormMSE \
    --layer 12
```
