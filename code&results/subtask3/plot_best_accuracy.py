import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#  配置 
EXPERIMENT_NAMES = [
    'AdamW (WD=1.0) [Baseline]',
    'Adam (No WD)',
    'AdamW (WD to Init)',
    'AdamW (High WD=3.0)',
    'AdamW (Low LR=1e-4)',
    'AdamW (Std Betas 0.999)',
    'Adam (LR=5e-3)',
    'SGD (Heavy-Ball, mom=0.9)',
    'RMSprop',
    'SGD (Nesterov, mom=0.99)',
    'Full-Batch GD',
    'AdamW (Weight Noise=0.01)'
]

LOG_DIR = './logs_optimized'

def clean_exp_name(name):
    """
    复现 optimizers.py 中的文件夹命名逻辑
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")

def load_data():
    all_records = []
    
    print("正在读取日志文件...")
    
    for exp_name in EXPERIMENT_NAMES:
        folder_name = clean_exp_name(exp_name)
        folder_path = os.path.join(LOG_DIR, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"[警告] 未找到实验文件夹: {folder_path}")
            continue
            
        # 获取该文件夹下所有csv文件
        safe_folder_path = glob.escape(folder_path) 
        csv_files = glob.glob(os.path.join(safe_folder_path, "*.csv"))
        
        for fpath in csv_files:
            try:
                # 解析文件名: alpha_{alpha}_seed_{seed}.csv
                filename = os.path.basename(fpath)
                name_parts = filename.replace('.csv', '').split('_')
                
                # 根据文件名格式提取 alpha 和 seed
                # 格式: ['alpha', '0.15', 'seed', '0']
                if len(name_parts) >= 4:
                    alpha = float(name_parts[1])
                    seed = int(name_parts[3])
                    
                    # 读取CSV
                    df = pd.read_csv(fpath)
                    
                    if not df.empty and 'val_acc' in df.columns:
                        # 获取该次实验达到的最佳验证集准确率
                        # 源代码逻辑：一旦 val_acc > 99.5 会提前停止，所以取 max 即可代表该次运行的结果
                        best_acc = df['val_acc'].max()
                        
                        all_records.append({
                            'Experiment': exp_name,
                            'Alpha': alpha,
                            'Seed': seed,
                            'BestAcc': best_acc
                        })
            except Exception as e:
                print(f"[错误] 解析文件失败 {fpath}: {e}")
                
    return pd.DataFrame(all_records)

def plot_experiments(df):
    if df.empty:
        print("没有数据可绘图。")
        return

    #  数据聚合 
    # 按 Experiment 和 Alpha 分组，计算 BestAcc 的中位数
    df_agg = df.groupby(['Experiment', 'Alpha'])['BestAcc'].median().reset_index()
    
    #  绘图设置 
    # 设置风格
    sns.set_theme(style="whitegrid")
    
    # 创建 3行 x 4列 的子图布局
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    
    # 颜色和标记
    line_color = '#1f77b4'  # 标准蓝色
    marker_style = 'o'
    
    print("正在绘图...")

    for idx, exp_name in enumerate(EXPERIMENT_NAMES):
        ax = axes[idx]
        
        # 筛选当前实验的数据
        subset = df_agg[df_agg['Experiment'] == exp_name].sort_values('Alpha')
        
        if not subset.empty:
            ax.plot(subset['Alpha'], subset['BestAcc'], 
                    marker=marker_style, linestyle='-', linewidth=2, color=line_color, markersize=6)
        
        # 子图修饰
        ax.set_title(exp_name, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylim(-5, 105)  # 准确率 0-100
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 添加一条 99.5% 的参考线（Grokking 阈值）
        ax.axhline(y=99.0, color='r', linestyle=':', alpha=0.5, linewidth=1)
        
        # 坐标轴标签
        if idx >= (n_rows - 1) * n_cols:  # 最后一行
            ax.set_xlabel('Alpha (Data Fraction)', fontsize=10)
        if idx % n_cols == 0:  # 第一列
            ax.set_ylabel('Median Best Val Acc (%)', fontsize=10)

    # 隐藏多余的子图（如果有）
    for i in range(len(EXPERIMENT_NAMES), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Grokking Experiment Results: Median Best Accuracy vs Alpha", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间
    
    output_filename = 'grokking_results_summary.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成！图片已保存为: {output_filename}")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    plot_experiments(df)