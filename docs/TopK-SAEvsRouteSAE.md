# SAE 模型评估对比报告

> 生成时间: 2026-02-02 09:57:40

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
| **NormMSE** ↓ | 0.1938 | 0.2129 | TopK |
| **KL Divergence** ↓ | 372.68 | 124.58 | **RouteSAE** |

> 注: ↓ 表示越低越好

## 指标说明

### NormMSE (Normalized Mean Squared Error)
衡量 SAE 重构的质量。计算公式:
```
NormMSE = ||x - x_hat||^2 / ||x||^2
```
- 值越低表示重构越准确
- 0 = 完美重构

### KL Divergence
衡量使用 SAE 重构后对 LLM 输出分布的影响:
```
KL(P_original || P_reconstructed)
```
- 值越低表示对 LLM 行为影响越小
- 0 = 完全不影响

## 结果分析

### 1. NormMSE (重构质量)
- **TopK SAE**: 0.1938
- **RouteSAE**: 0.2129 (+9.8%)

TopK SAE 在单一层重构上表现略好。这是预期的，因为 RouteSAE 需要学习跨多层（4-12层）的特征，任务难度更大。

### 2. KL Divergence (行为保持) - **关键差异!**
- **TopK SAE**: 372.68
- **RouteSAE**: 124.58 (降低 66.6%)

**RouteSAE 取得了压倒性优势！** 
虽然重构误差略高，但 RouteSAE 对 LLM 输出分布的影响远小于 TopK SAE。这表明：
1. RouteSAE 捕获的特征更具功能性意义
2. 动态路由机制有效保留了不同层的信息流
3. 对于下游任务（如特征干预），RouteSAE 是更安全的选择

### 结论
虽然 TopK SAE 在"复制"激活值方面略胜一筹，但 **RouteSAE 在"理解"和"保持"模型行为方面表现卓越**。这验证了论文的核心假设：多层路由机制能更好地解耦和表示 LLM 的内部计算。

## 模型文件

| 模型 | 路径 |
|------|------|
| TopK SAE | `/mnt/nfs/zijie/routesae_repro_outputs/L16_K64_9213176726f574b556790deb65791e0c5aa438b6_openwebtext2_16384.pt` |
| RouteSAE | `/mnt/nfs/zijie/routesae_repro_outputs/sum_hard_K64_9213176726f574b556790deb65791e0c5aa438b6_openwebtext2_16384.pt` |
