import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import math

#  配置 
LOG_DIR = './logs_optimized_single_experiment'

# 实验名称列表
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

def get_folder_name(exp_name):
    """
    根据实验名称生成文件夹名称
    """
    return exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")

def plot_all_experiments():
    n_experiments = len(EXPERIMENT_NAMES)
    
    ncols = 4
    nrows = math.ceil(n_experiments / ncols)
    
    # 调整画布大小，宽度增加以适应4列
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), constrained_layout=True)
    axes = axes.flatten()
    
    print(f"Searching for logs in: {os.path.abspath(LOG_DIR)}")

    for i, exp_name in enumerate(EXPERIMENT_NAMES):
        ax = axes[i]
        folder_name = get_folder_name(exp_name)
        folder_path = os.path.join(LOG_DIR, folder_name)
        
        safe_folder_path = glob.escape(folder_path)
        csv_pattern = os.path.join(safe_folder_path, "*.csv")
        
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            # 调试信息：打印出原本试图寻找的路径，方便排查
            print(f"[Warning] No data found for: {exp_name}")
            print(f"   -> Looking in: {csv_pattern}")
            
            ax.text(0.5, 0.5, 'No Data Found', horizontalalignment='center', verticalalignment='center', color='gray')
            ax.set_title(exp_name, fontsize=10)
            ax.set_axis_off()
            continue
            
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if 'step' in df.columns and 'train_acc' in df.columns and 'val_acc' in df.columns:
                    ax.plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.9, linewidth=1.5)
                    ax.plot(df['step'], df['val_acc'], label='Test Acc', alpha=0.9, linewidth=1.5, linestyle='--')
                else:
                    print(f"[Error] Columns missing in {csv_file}")
            except Exception as e:
                print(f"[Error] Failed to read {csv_file}: {e}")

        # 设置子图样式
        ax.set_title(exp_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xscale('log')
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 图例设置
        ax.legend(loc='lower right', fontsize='small')

    # 隐藏多余的子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    save_path = 'experiment_results_summary_3x4.png'
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(LOG_DIR):
        print(f"Error: Log directory '{LOG_DIR}' not found. Please run 'run_single.py' first.")
    else:
        plot_all_experiments()