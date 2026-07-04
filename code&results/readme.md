本项目旨在系统性研究神经网络中的 **Grokking 现象**。实验通过模加法任务，探讨了数据效率、模型架构、优化算法以及 Transformer 组件（如位置编码 PE 和因果掩码 Mask）对模型泛化性能的影响。

## 1. 环境要求

本项目基于 Python 和 PyTorch 开发。以下为复现实验所需的最低版本要求：

- **Python**: 3.12+

- **PyTorch**: 

  ```
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  ```

- **关键依赖库**:

  ```
  pip install torch numpy pandas matplotlib tqdm seaborn sklearn
  ```

- **硬件建议**: 建议使用具有 8GB 以上显存的 GPU 以支持 Subtask 3/4 的并行任务调度。

------

## 2. 项目结构与文件说明

```
提交版/
├── subtask1/                       # 任务 1: Grokking 现象复现
│   ├── grokking_curve_alpha0.5.png # 实验结果图表
│   ├── grokking_detailed_log_a0.5.csv # 详细训练日志
│   ├── grokking_grid_plot.png      # 数据效率网格搜索结果
│   ├── grokking_results_strict.csv # 训练步数统计
│   └── subtask1.py                 # 主程序：包含 Transformer 实现与 Grid Search
├── subtask2/                       # 任务 2: 不同模型架构对比 (CNN, RNN, MLP, ResNet等)
│   ├── [Model]_log.csv             # 各个架构（如 LSTM, SIREN）的训练日志
│   ├── combined_results.png        # 所有架构的准确率对比汇总图
│   └── subtask2.py                 # 多模型架构对比脚本
├── subtask3/                       # 任务 3: 优化器性能研究
│   ├── logs_optimized/             # 存放不同优化器（AdamW, SGD等）的日志文件夹
│   ├── optimizers.py               # 核心逻辑：包含自定义优化器与 NoisyLinear
│   ├── run_batch.py                # 批量自动化运行脚本（多进程并行）
│   ├── plot_best_accuracy.py       # 绘制最高准确率对比图
│   ├── plot_steps_grokking.py      # 绘制达到 Grokking 所需步数统计图
│   └── grokking_results_summary.png# 优化器对比结果总结图
├── subtask4/                       # 任务 4: 复杂模运算与组件消融
│   ├── logs_K/                     # 存放不同 K 值、PE、Mask 配置的日志
│   ├── combined_accuracy_curves.png# 合并后的多操作数准确率曲线
│   ├── subtask4.py                 # 任务 4 核心逻辑：支持 K 个操作数及组件开关
│   ├── run.py                      # 自动化调度脚本：运行消融实验网格
│   └── plot.py                     # 绘图工具：汇总生成最终对比曲线
└── subtask5/                       # 任务 5: Loss Landscape 可视化
    ├── transformer.png             # Transformer 的 Loss 地形与优化轨迹图
    ├── simpleMLP.png               # MLP 的 Loss 地形与优化轨迹图
    └── subtask5.py                 # 主程序：轨迹收集、PCA 降维与地形绘制
```

### Subtask 1: Grokking现象复现

- **`subtask1.py`**: 核心脚本。实现了一个 2 层 Transformer 模型，用于复现 $a + b \pmod{97}$ 任务中的顿悟现象，并包含对数据占比 $\alpha$ 的网格搜索逻辑。

### Subtask 2: 架构消融实验

- **`subtask2.py`**: 架构对比脚本。实现了包括 Simple MLP, Deep MLP, Gated MLP, SIREN, 1D-CNN, RNN, LSTM, ResNet-MLP 在内的多种架构，旨在对比不同归纳偏置在代数任务上的表现。

### Subtask 3: 优化器实验

- **`optimizers.py`**: 核心逻辑文件。包含自定义的 `NoisyLinear` 和 `CustomAdamW`（支持权重衰减至初始值），定义了 12 种优化器实验配置。
- **`run_batch.py`**: 自动化批处理脚本。利用多进程并行运行不同 $\alpha$ 和 Seed 下的优化器对比实验。
- **`plot_best_accuracy.py` / `plot_steps_grokking.py`**: 绘图脚本。绘制各个优化器对比实验的结果。

### Subtask 4: 复杂模运算与组件消融

- **`subtask4.py`**: 扩展任务脚本。支持 $K$ 个操作数 ($K \in \{2, 3, 4, 5\}$) 的模加法，并提供位置编码 (PE) 和因果掩码 (Mask) 的开关。
- **`run.py`**: 调度器。用于并行执行 K, PE, Mask 三个维度的全网格消融实验。
- **`plot.py`**: 绘图脚本。汇总 `logs_K` 目录下的所有数据并生成多子图对比曲线。

### Subtask 5: Loss Landscape 可视化

- **`subtask5.py`**: 损失地形可视化脚本。分别训练 `GrokkingTransformer` 和 `SimpleMLP`，并记录训练过程中的参数轨迹。使用 **PCA (主成分分析)** 将高维参数轨迹投影到 2D 平面，并绘图。

------

## 3. 运行指南

### 复现 Subtask 1 & 2

```
# 运行基础 Grokking 实验
python subtask1/subtask1.py

# 运行多架构对比实验
python subtask2/subtask2.py
```

### 复现 Subtask 3 (优化器对比)

```
cd subtask3
# 1. 批量运行所有优化器配置（默认开启 8 进程）
python run_batch.py

# 2. 生成结果分析图表
python plot_best_accuracy.py
python plot_steps_grokking.py
```

### 复现 Subtask 4 (K 个操作数消融)

```
cd subtask4
# 自动运行 K=[2,3,4,5] 的并行实验
python run.py
```

### 复现 Subtask 5 (Loss Landscape 可视化)

```
cd subtask5
# 运行可视化实验
python subtask5.py
```



------

