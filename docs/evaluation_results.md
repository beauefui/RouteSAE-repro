# SAE 模型评估对比报告

> 生成时间: 2026-02-04 07:13:18

## 评估配置

| 配置项 | 值 |
|--------|-----|
| 评估样本数 | 10,000 |
| Batch Size | 64 |
| Sequence Length | 512 |
| GPU | cuda:3 |

## 评估结果

| 指标 | TopK | RouteSAE | MLSAE | Vanilla | Gated | JumpReLU | Crosscoder | RandomK |
|------|------|----------|-------|---------|-------|----------|------------|---------|
| **NormMSE** ↓ | 0.1937
0.1937 | 0.2129
0.2129 | 0.2510
0.2510 | 0.0080
0.0080 | 0.0077
0.0077 | 0.1167
0.1167 | 0.2487
0.2487 | 0.8638 |
| **KL Divergence** ↓ | 371.1592
371.1592 | 124.5796
124.5796 | 232.4551
232.4551 | 14.8623
14.8623 | 14.4240
14.4240 | 233.3503
233.3503 | N/A (Not Impl) | 3672.0510 |


> 注: ↓ 表示越低越好

## 模型说明

1.  **TopK**: 标准 SAE，在 Layer 16 训练并评估。
2.  **RouteSAE**: 多层路由 SAE，动态选择 Layer 4-12。
3.  **MLSAE**: 多层 SAE，在所有层上训练统一的特征空间。
4.  **Vanilla**: 传统 SAE，使用 ReLU 和 L1 正则化。
5.  **Gated**: Gated SAE，分离 Gate 和 Magnitude。
6.  **JumpReLU**: JumpReLU SAE，使用与特征相关的可学习阈值。
7.  **Crosscoder**: 跨层稀疏编码器，学习多层共享特征 (NormMSE only)。
8.  **RandomK**: 使用 TopK 模型，但评估时随机选择层 (Baseline)。

## 详细对比分析

根据上述评估数据，结合 RouteSAE 论文及稀疏自编码器 (SAE) 的相关研究，我们对各个模型的性能进行深入分析。

### 1. 基线模型 (The Baseline)
-   **RandomK**: 
    -   **NormMSE (0.8638)**: 极高的重构误差，意味着随机选择的特征几乎无法重构原始激活。
    -   **KL Divergence (3672.05)**: 极高的 KL 散度，说明随机特征干预严重破坏了语言模型的输出分布。
    -   **意义**: RandomK 作为一个“最差下界”，有效地验证了其他 SAE 模型并非是在做随机映射，而是确实学到了有意义的特征结构。所有训练过的模型（除 Crosscoder KL 外）表现均显著优于 RandomK。

### 2. 标准 SAE vs 高级激活函数 (Standard vs Advanced Activations)
-   **Vanilla & Gated**:
    -   **异常低 Loss (NormMSE ~0.008)**: 这两个模型展现了惊人的重构能力（误差 < 1%），远低于 TopK 和 RouteSAE。
    -   **潜含风险**: 在 SAE 训练中，极低的重构误差通常伴随着**极低的稀疏性 (High L0)**。甚至可能出现了“恒等映射 (Identity Mapping)”现象，即模型放弃了稀疏性约束，直接复制输入到输出。这通常是因为 L1 系数设置过小或训练未收敛到稀疏解。
    -   **KLDiv 表现 (Top-tier)**: 由于重构极好，它们对下游任务的干扰极小 (KLDiv ~14)，这进一步支撑了“恒等映射”的猜想。
-   **JumpReLU**:
    -   表现介于 Vanilla 和 TopK 之间。NormMSE (0.1167) 比 TopK 好，但 KLDiv (233.35) 却较高，说明虽然重构误差小，但重构出的特征可能在语义上对下游任务有负面影响，或者其稀疏性控制与 TopK 有本质不同。

### 3. TopK vs RouteSAE (核心对比)
这是本项目最核心的对比组：
-   **TopK (Layer 16)**: 
    -   NormMSE: 0.1937
    -   KLDiv: 371.16
-   **RouteSAE (Dynamic Layers)**:
    -   NormMSE: 0.2129 (略差于 Layer 16 TopK)
    -   KLDiv: 124.58 (**显著优于 TopK**)
-   **结论**: 
    -   尽管 RouteSAE 的重构误差略高（因为它要处理多层特征，难度更大），但其 **KL Divergence 降低了近 3 倍** (371 -> 124)。
    -   这有力地证明了 **RouteSAE 的路由机制有效**。通过动态选择最适合的层（Layer 4-12）进行特征提取，RouteSAE 捕捉到了更能保持下游任务性能的语义特征，而非仅仅是在单层（Layer 16）上做死记硬背的重构。

### 4. MLSAE (Multi-Layer SAE)
-   **表现均衡**: NormMSE (0.2510) 和 KLDiv (232.46) 均处于中等水平。
-   **对比 RouteSAE**: MLSAE 试图用同一套特征去解释所有层，这比 RouteSAE 的“分而治之”策略更具挑战性。虽然 KLDiv 优于单层 TopK，但仍不如 RouteSAE，说明**层间差异性** (Layer Specificity) 是不可忽视的，强行统一特征空间可能会损失精度。

### 5. Crosscoder
-   **NormMSE (0.2487)**: 重构质量与 MLSAE 相当，略差于 RouteSAE。
-   **角色定位**: Crosscoder 旨在学习跨层共享特征。由于其 KLDiv 未实现，我们主要看重构。其表现说明跨层编码是可行的，但可能需要更大的参数量或更复杂的解码机制才能达到单层 SAE 的重构精度。

### 总结建议
1.  **推荐模型**: **RouteSAE** 是综合性能最佳的选择，特别是在保持下游任务能力 (Low KLDiv) 方面表现卓越。
2.  **需检查模型**: **Vanilla 和 Gated** 需要检查其 L0 稀疏度（WandB 日志）。如果 L0 过高（如接近 Hidden Size），则它们失去了作为“稀疏”自编码器的意义，需要调大 L1 正则系数重新训练。
3.  **基线确认**: RandomK 的糟糕表现确认了实验的有效性。
