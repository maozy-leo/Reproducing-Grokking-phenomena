import subprocess
import time
from tqdm import tqdm

#  配置并发数 
MAX_WORKERS = 8  

alphas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
seeds = [0, 1, 2, 4, 5]

experiment_names = [
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

def main():
    # 1. 生成所有任务列表
    commands = []
    for exp_name in experiment_names:
        for alpha in alphas:
            for seed in seeds:
                # 构造命令行参数
                cmd = [
                    "python", "optimizers.py",
                    "--exp_name", exp_name,
                    "--alpha", str(alpha),
                    "--seed", str(seed)
                ]
                commands.append(cmd)

    print(f"Total experiments to run: {len(commands)}")
    print(f"Max concurrent workers: {MAX_WORKERS}")

    # 2. 进程管理池
    processes = []
    
    # 使用 tqdm 显示进度条
    with tqdm(total=len(commands)) as pbar:
        while len(commands) > 0 or len(processes) > 0:
            # A. 如果还有名额且还有任务，就启动新进程
            while len(processes) < MAX_WORKERS and len(commands) > 0:
                cmd = commands.pop(0)
                p = subprocess.Popen(cmd)
                processes.append(p)
            
            # B. 检查哪些进程运行结束了
            still_running = []
            for p in processes:
                if p.poll() is None:
                    # 还在运行
                    still_running.append(p)
                else:
                    # 已经结束
                    pbar.update(1)
                    # 检查是否有错误
                    if p.returncode != 0:
                        print(f"\nTask failed with code {p.returncode}")
                        # 打印错误信息 (stderr)
                        print(p.stderr.read().decode())
            
            processes = still_running
            
            # C. 避免死循环占用 CPU
            time.sleep(0.1)

    print("\nAll experiments finished!")

if __name__ == "__main__":
    main()