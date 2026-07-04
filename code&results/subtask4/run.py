import subprocess
import itertools
import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_WORKERS = 8
PYTHON_EXEC = sys.executable
OUTPUT_DIR = "logs_K"

def run_worker(config):
    k, pe, mask = config
    cmd = [
        PYTHON_EXEC, "subtask4.py",
        "--run_single",
        "--k", str(k),
        "--pe", str(int(pe)),
        "--mask", str(int(mask)),
        "--seed", "999"
    ]
    
    print(f"[Scheduler] Submitting task: K={k}, PE={pe}, Mask={mask}")
    
    try:
        subprocess.run(cmd, check=True)
        return f"Success: K={k} PE={pe} Mask={mask}"
    except subprocess.CalledProcessError as e:
        return f"Failed: K={k} PE={pe} Mask={mask}, Error: {e}"

def main():
    # 0. 预先创建输出目录，防止多进程竞争
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 定义实验网格
    K_values = [2, 3, 4, 5]
    pe_settings = [True, False]
    mask_settings = [True, False]
    
    # 生成所有组合列表 [(2, True, True), (2, True, False), ...]
    configurations = list(itertools.product(K_values, pe_settings, mask_settings))
    
    print(f"Total experiments to run: {len(configurations)}")
    print(f"Max parallel workers: {MAX_WORKERS}")
    
    start_time = time.time()
    
    # 2. 使用进程池并行执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {executor.submit(run_worker, conf): conf for conf in configurations}
        
        for future in as_completed(futures):
            result = future.result()
            print(f"[Scheduler] {result}")

    total_time = time.time() - start_time
    print(f"\nAll experiments finished in {total_time:.2f} seconds.")

    # 3. 最后合并图片
    print("Combining plots...")
    subprocess.run([PYTHON_EXEC, "subtask4.py", "--combine_only"])
    print(f"Results are stored in the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()