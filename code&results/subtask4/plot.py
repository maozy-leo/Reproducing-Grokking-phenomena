import os
import glob
import math
import re
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_results(log_dir="logs_K", output_file="combined_accuracy_curves.png"):
    """
    读取 logs_K 下的所有 CSV 文件，绘制 Train/Test Accuracy 曲线并汇总到一张大图中。
    """
    # 检查目录是否存在
    if not os.path.exists(log_dir):
        print(f"Error: Directory '{log_dir}' not found.")
        return

    # 寻找所有 CSV 文件 (格式: log_K*_PE*_Mask*.csv)
    csv_pattern = os.path.join(log_dir, "log_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {log_dir}.")
        return

    # 辅助函数：从文件名解析参数 (K, PE, Mask)
    def get_params(fname):
        base = os.path.basename(fname)
        # 正则匹配文件名中的参数
        match = re.search(r'K(\d+)_PE(\d+)_Mask(\d+)', base)
        if match:
            # 返回 tuple 以便排序: (K, PE, Mask)
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return (0, 0, 0)

    # 对文件进行排序，保证图表顺序逻辑清晰 (先按K排，再按PE，最后按Mask)
    csv_files.sort(key=get_params)

    # 计算网格大小
    n_plots = len(csv_files)
    cols = 4  # 每行显示4个图
    rows = math.ceil(n_plots / cols)
    
    print(f"Found {n_plots} experiments. Generating {rows}x{cols} grid plot...")

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    
    # 处理 axes 扁平化，方便遍历
    axes_flat = axes.flatten() if n_plots > 1 else [axes]

    for i, file_path in enumerate(csv_files):
        ax = axes_flat[i]
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 绘制曲线
            ax.plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.9, linewidth=1.5)
            ax.plot(df['step'], df['test_acc'], label='Test Acc', alpha=0.9, linewidth=1.5, linestyle='--')
            
            # 设置标题和标签
            k, pe, mask = get_params(file_path)
            pe_str = "Yes" if pe else "No"
            mask_str = "Yes" if mask else "No"
            
            title = f"K={k} | PE={pe_str} | Mask={mask_str}"
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Accuracy (%)')
            
            ax.set_xscale('log') 
            ax.set_ylim(-5, 105)
            
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc='lower right', fontsize=9)
            
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            ax.text(0.5, 0.5, "Data Load Error", ha='center', va='center')

    # 隐藏多余的空子图
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    # 保存图片
    plt.savefig(output_file, dpi=150)
    print(f"Combined plot saved to: {output_file}")

if __name__ == "__main__":
    plot_all_results()