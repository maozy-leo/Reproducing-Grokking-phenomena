# Reproducing Grokking Phenomena

本项目在模加法任务上复现并研究神经网络的 **Grokking（顿悟）现象**：模型先快速记忆训练集，经过更长时间训练后，测试集准确率才会突然提升。实验进一步比较了训练数据比例、模型架构、优化器、Transformer 组件和损失地形对泛化行为的影响。

## 实验内容

- **Subtask 1 — 基础复现与数据效率**：使用两层 Transformer 学习 $a+b \pmod{97}$，并扫描训练数据比例 $\alpha$。
- **Subtask 2 — 模型架构比较**：比较 MLP、Gated MLP、SIREN、CNN、RNN、LSTM 和 ResNet MLP 等架构。
- **Subtask 3 — 优化器比较**：研究 Adam、AdamW、SGD、RMSprop、Full-batch GD、权重噪声及不同权重衰减设置。
- **Subtask 4 — Transformer 组件消融**：考察操作数个数 $K$、位置编码和因果掩码的影响。
- **Subtask 5 — 损失地形可视化**：通过 PCA 将参数轨迹投影至二维平面，对比 Transformer 与简单 MLP 的优化过程。

## 代表性结果

| 基础 Grokking 曲线 | 优化器实验汇总 |
|---|---|
| ![Grokking curve](code%26results/subtask1/grokking_curve_alpha0.5.png) | ![Optimizer comparison](code%26results/subtask3/grokking_results_summary.png) |

| 多操作数与组件消融 | 损失地形 |
|---|---|
| ![Ablation results](code%26results/subtask4/combined_accuracy_curves.png) | ![Transformer loss landscape](code%26results/subtask5/transformer.png) |

## 项目结构

```text
.
├── README.md
├── doc/
│   └── report.pdf                 # 项目报告
└── code&results/
    ├── subtask1/                  # 基础复现及数据比例实验
    ├── subtask2/                  # 模型架构比较
    ├── subtask3/                  # 优化器实验
    ├── subtask4/                  # 多操作数及组件消融
    └── subtask5/                  # 损失地形可视化
```

各子目录同时包含实验脚本、训练日志（CSV）和生成的图表。

## 环境配置

建议使用 Python 3.10 或更高版本，并根据硬件环境安装合适版本的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib tqdm seaborn scikit-learn
```

大规模网格实验耗时较长，建议使用支持 CUDA 的 GPU；绘图脚本可在 CPU 环境运行。

## 运行实验

从仓库根目录执行：

```bash
# Subtask 1：基础 Grokking 实验
python 'code&results/subtask1/subtask1.py'

# Subtask 2：模型架构比较
python 'code&results/subtask2/subtask2.py'

# Subtask 3：批量优化器实验及绘图
cd 'code&results/subtask3'
python run_batch.py
python plot_best_accuracy.py
python plot_steps_grokking.py

# Subtask 4：组件消融实验
cd '../subtask4'
python run.py
python plot.py

# Subtask 5：损失地形可视化
cd '../subtask5'
python subtask5.py
```

部分训练脚本的默认步数和实验网格较大，首次运行时可先在对应脚本中减小训练步数或配置数量，以验证环境是否正常。

## 报告

完整的实验设计、结果和分析见 [`doc/report.pdf`](doc/report.pdf)。

## 声明
本项目为北京大学2025年秋季学期研本课程`机器学习数学导引`的课程大作业，作者为数学科学学院2024级本科生毛志远、蔡琳珊、陈佳鸿。
项目分工具体如下：
- 组长：毛志远，负责代码实现。
- 组员：蔡琳珊、陈佳鸿，负责报告撰写。

本项目采用 MIT LICENSE 开源。
