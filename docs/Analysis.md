# RouteSAE 原始代码（old/）能否复现实验效果：面向导师问题的分析

本文档**只以 old/ 原始项目代码为准**回答导师关注的问题：原始代码的逻辑、训练/测试/干预流程、实验覆盖范围，以及它在多大程度上能复现论文结果。你的 `src/` 复现仅作为差异参考，不作为主依据。

---

## 1. RouteSAE 方法本身的代码逻辑是什么？如何实现？如何训练与测试？做了哪些测试与干预？

### 1.1 核心逻辑（old/src/model.py + old/src/utils.py）

**核心思想**：把多层残差流激活交给一个“轻量路由器”做**层选择/加权**，再用**共享 TopK SAE**做稀疏分解与重构。

- **输入**：LLM 多层残差流激活（中间层范围）
- **路由器**：对多层激活做 `sum/mean` 聚合 → 线性层输出每层权重 → softmax
- **路由策略**：
  - **hard routing**：每个 token 选权重最大的层
  - **soft routing**：所有层加权求和
- **共享 SAE**：路由后的激活进入共享 TopK SAE（TopK 编码 + 解码重构）
- **训练目标**：重构误差（Normalized MSE）驱动路由器与 SAE 联合学习

### 1.2 训练流程（old/src/train.py + old/src/utils.py::Trainer）

- **模型**：Llama-3.2-1B-Instruct
- **输入层范围**：中间层（1/4 到 3/4 深度）
- **预处理**：零均值 / 单位方差归一化
- **前向**：
  - 路由器 → routed_x
  - TopK SAE → latents + x_hat
- **损失**：
  - RouteSAE/TopK：Normalized MSE
  - Vanilla/Gated：MSE + L1
  - JumpReLU：MSE + L0 近似
- **稳定性技巧**：每隔固定步数对 decoder 列向量做 unit-norm

### 1.3 测试流程（old/src/evaluate.py + old/src/utils.py::Evaluater）

- **NormMSE**：重构质量
- **KLDiv**：替换中间层激活为 SAE 重构值，比较输出分布差异
- **DeltaCE / Recovered**：旧代码支持（README 和 utils 中有实现入口）
- **Random baseline**：随机路由 / 均匀权重对照

### 1.4 实际覆盖的测试与干预

**测试指标覆盖**：
- NormMSE
- KLDiv
- DeltaCE
- Recovered
- interpretable features 数量（阈值筛选）
- GPT-4o interpretation score（低/高层特征评分）
- routing 权重分布统计

**干预试验覆盖**：
- **Feature steering / clamping**：hook 对 latent 做加/乘/减/除，再比较生成输出
- **Zero ablation**：hook 直接置零作为干预对照

> 结论：**old 项目在功能层面覆盖了论文完整实验链条**（训练、评估、解释、干预）。

---

## 2. old 项目用哪些数据集？能否完全对齐论文实验？论文提到但代码没实现的有哪些？

### 2.1 old 项目使用的数据与模型

- **模型**：Llama-3.2-1B-Instruct（与论文一致）
- **训练/评估数据**：OpenWebText2（论文一致）
- **干预/对话测试**：HH-RLHF（用于多轮对话 clamping）

### 2.2 是否能完全对齐论文的实验数据规模与结果？

**代码能力层面：可以。**
old 代码的结构与指标设计与论文一致，支持：
- 100M tokens 训练、10M tokens 评估（论文设置）
- context length 512
- 论文全部评估指标与干预流程

**实际结果是否能完全对齐：取决于你是否真的用论文规模与相同超参跑过。**
如果仅在小规模数据或少量样本上跑，结果无法保证对齐论文。

### 2.3 论文提到但 old 项目未实现的内容

**从 old 代码角度看，论文主要指标都已有对应实现**，但存在以下现实差异点：

- **Crosscoder 的 KLDiv**：实现难度高，论文中也弱化处理；old 未实现
- **Random baseline 的定义差异**：论文是“均匀路由”，代码中可能是“随机选层”或“随机路由”版本
- **规模与可重复性**：论文大规模训练/评估，old 项目能跑但对计算资源要求极高

> 结论：**old 项目在逻辑上对齐论文，但是否能“完全复现论文结果”取决于实际数据规模、训练时长、资源与超参复现程度。**

---

## 3. 通过 RouteSAE，我们能学到提出新 SAE 方法后应该做哪些实验，怎么做？

这是导师关心的“方法论文评估范式”。RouteSAE 给出的范式可总结为：

### 3.1 必须完成的核心实验

1. **重构质量**：NormMSE
2. **功能保持**：KLDiv / DeltaCE / Recovered
3. **稀疏度曲线**：K/λ/阈值 vs 指标曲线
4. **解释性**：
   - interpretable features 数量（阈值过滤 + 多上下文）
   - GPT-4o interpretation score（100 feature 抽样）
5. **对比基线**：TopK / Vanilla / Gated / JumpReLU / Crosscoder / Random

### 3.2 干预实验（证明因果有效性）

6. **Feature steering / clamping**：
   - 通过 hook 修改特定 latent
   - 比较输出变化
   - 量化干预效果

### 3.3 如何实施（以 old 项目为准）

- **训练**：Trainer 直接替换 SAE 结构即可
- **评估**：Evaluater 覆盖 NormMSE + KLDiv + DeltaCE + Recovered
- **解释**：Application + Interpreter（阈值筛选 + GPT-4o）
- **干预**：clamp pipeline 对生成输出做对比

---

## 总结（面向导师的答案）

1. **old 项目代码覆盖了论文所需的核心逻辑、训练/测试/解释/干预全链条**，RouteSAE 的核心机制实现完整。
2. **与论文完全对齐的关键在于数据规模与超参是否复现**：old 项目有能力，但若没按 100M/10M token 跑，结果无法保证一致。
3. **RouteSAE 提供了一个标准 SAE 论文评估范式**：重构质量 + 功能保持 + 稀疏度曲线 + 解释性 + 干预实验 + 基线对比。

