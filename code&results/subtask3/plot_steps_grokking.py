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
MAX_STEPS_LIMIT = 50000

def clean_exp_name(name):
    """复现 optimizers.py 中的文件夹命名逻辑"""
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")

def load_step_data():
    all_records = []
    print("正在读取日志文件最大步数...")
    
    for exp_name in EXPERIMENT_NAMES:
        folder_name = clean_exp_name(exp_name)
        folder_path = os.path.join(LOG_DIR, folder_name)
        
        if not os.path.exists(folder_path):
            continue
            
        safe_folder_path = glob.escape(folder_path) 
        csv_files = glob.glob(os.path.join(safe_folder_path, "*.csv"))
        
        for fpath in csv_files:
            try:
                # 解析文件名: alpha_{alpha}_seed_{seed}.csv
                filename = os.path.basename(fpath)
                name_parts = filename.replace('.csv', '').split('_')
                
                if len(name_parts) >= 4:
                    alpha = float(name_parts[1])
                    seed = int(name_parts[3])
                    
                    # 仅读取最后一行或 'step' 列来获取最大值，提高读取效率
                    df = pd.read_csv(fpath)
                    
                    if not df.empty and 'step' in df.columns:
                        max_step_in_run = df['step'].max()
                        
                        if max_step_in_run < MAX_STEPS_LIMIT:
                            steps_needed = max_step_in_run
                        else:
                            steps_needed = MAX_STEPS_LIMIT
                        
                        all_records.append({
                            'Experiment': exp_name,
                            'Alpha': alpha,
                            'Seed': seed,
                            'Steps': steps_needed
                        })
            except Exception as e:
                print(f"读取错误 {fpath}: {e}")
                
    return pd.DataFrame(all_records)

def plot_grokking_steps(df):
    if df.empty:
        print("没有数据可绘图。")
        return

    #  数据聚合 
    # 计算中位数
    df_agg = df.groupby(['Experiment', 'Alpha'])['Steps'].median().reset_index()
    
    #  绘图设置 
    sns.set_theme(style="whitegrid")
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15), sharex=True, sharey=True)
    axes = axes.flatten()
    
    print("正在绘图...")

    for idx, exp_name in enumerate(EXPERIMENT_NAMES):
        ax = axes[idx]
        subset = df_agg[df_agg['Experiment'] == exp_name].sort_values('Alpha')
        
        if not subset.empty:
            # 1. 绘制连线 (连接所有中位数点)
            ax.plot(subset['Alpha'], subset['Steps'], 
                    color='gray', linestyle='-', alpha=0.5, linewidth=1.5, zorder=1)
            
            # 2. 分组：Grok (Steps < 50000) vs Not Grok (Steps == 50000)
            grok_mask = subset['Steps'] < MAX_STEPS_LIMIT
            
            grok_data = subset[grok_mask]
            fail_data = subset[~grok_mask]
            
            # 3. 绘制成功 Grok 的点 (蓝色实心圆)
            if not grok_data.empty:
                ax.scatter(grok_data['Alpha'], grok_data['Steps'], 
                           c='#1f77b4', marker='o', s=40, label='Grokked (<50k steps)', zorder=2)

            # 4. 绘制未 Grok 的点 (红色叉号)
            if not fail_data.empty:
                ax.scatter(fail_data['Alpha'], fail_data['Steps'], 
                           c='#d62728', marker='X', s=60, label='Not Grokked (50k steps)', zorder=2)

        # 子图修饰
        ax.set_title(exp_name, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylim(-1000, MAX_STEPS_LIMIT + 3000) # 给顶部留点空间画红叉
        
        # 优化Y轴刻度显示
        yticks = [0, 10000, 20000, 30000, 40000, 50000]
        yticklabels = ['0', '10k', '20k', '30k', '40k', 'FAIL']
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        
        # 仅在第一个图显示图例
        if idx == 0:
            ax.legend(loc='lower left', framealpha=0.9, fontsize=9)

        # 坐标轴标签
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Alpha (Data Fraction)', fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel('Median Steps', fontsize=10)

    # 隐藏多余子图
    for i in range(len(EXPERIMENT_NAMES), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Grokking Efficiency: Median Steps to Reach Solution vs Alpha\n(Based on log termination step: <50k = Grokked, 50k = Not Grokked)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = 'grokking_steps_v2.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"绘图完成！图片已保存为: {output_filename}")
    plt.show()

if __name__ == "__main__":
    df = load_step_data()
    plot_grokking_steps(df)