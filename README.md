# RouteSAE-repro

> **个人学习项目** - 复现 RouteSAE 论文以深入理解稀疏自编码器的工作原理

本项目是对 [RouteSAE](https://github.com/OpenSAE/RouteSAE) 的学习性复现，仅用于个人学习和研究目的。

---

## 项目说明

本项目复现了 **Route Sparse Autoencoder (RouteSAE)** 的核心实现，包括：

- **TopK SAE**: 标准的 TopK 稀疏自编码器
- **RouteSAE**: 带路由机制的多层稀疏自编码器

主要用于学习以下内容：
1. 稀疏自编码器在 LLM 可解释性中的应用
2. TopK 稀疏化机制的实现
3. 多层路由机制的设计思路
4. SAE 训练和评估流程

---

## 项目结构

```
RouteSAE-repro/
├── src/
│   ├── model.py      # SAE 模型 (TopK, RouteSAE)
│   ├── data.py       # 数据加载
│   ├── utils.py      # 工具函数
│   ├── trainer.py    # 训练器
│   ├── evaluator.py  # 评估器
│   └── interpreter.py # GPT-4o 特征解释
├── docs/
│   └── code_walkthrough.md  # 代码详解
├── scripts/          # 运行脚本
└── old/              # 原始参考代码
```

---

## 学习资源

- [代码详解文档](docs/code_walkthrough.md) - 每个模块的详细解析

---

## 致谢

本项目基于以下工作进行学习复现：

**原始论文与代码：**
- 论文：*Route Sparse Autoencoder to Interpret Large Language Models*
- 官方仓库：[OpenSAE/RouteSAE](https://github.com/OpenSAE/RouteSAE)

---

## 引用

如果您的工作参考了 RouteSAE，请引用原始论文：

```bibtex
@inproceedings{routesae2025,
    title={Route Sparse Autoencoder to Interpret Large Language Models},
    author={Wei Shi and Sihang Li and Tao Liang and Mingyang Wan and Guojun Ma and Xiang Wang and Xiangnan He},
    booktitle={Proceedings of EMNLP},
    year={2025}
}
```

---

## 声明

- 本项目仅用于**个人学习目的**
- 所有核心思想和方法归属于原作者
- 如有任何版权问题，请联系删除

---

## License

本项目遵循原始 RouteSAE 项目的开源协议。
