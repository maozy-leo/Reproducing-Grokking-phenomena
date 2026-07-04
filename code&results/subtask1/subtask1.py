import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import pandas as pd
import math

#  全局配置 
CONFIG = {
    'vocab_size': 100,      # 0-96 (数字), 97 (+), 98 (=), 99 (PAD/EOS)
    'd_model': 128,
    'n_layers': 2,
    'n_heads': 4,
    'max_len': 10,
    'batch_size': 512,
    'max_steps': 100000,
    'target_acc': 99.0,
    'check_interval': 50,  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'results_file': 'grokking_results_strict.csv',
    'detailed_log_file': 'grokking_detailed_log_a0.5.csv',
    'weight_decay': 1.0,
    'lr': 1e-3,
    'betas': (0.9, 0.98),
    'dropout': 0.0
}

# PART 1: 数据生成

def create_dataset_in_memory(alpha, seed):
    """
    生成模加法数据集 a + b = c (mod 97)
    """
    MODULUS = 97
    
    # 生成全量样本
    all_equations = []
    nums = list(range(MODULUS))
    for a in nums:
        for b in nums:
            c = (a + b) % MODULUS
            all_equations.append([a, b, c])
    
    total_samples = len(all_equations)
    
    # 打乱与切分
    rng = np.random.RandomState(seed)
    shuffled_data = all_equations.copy()
    rng.shuffle(shuffled_data)

    train_size = int(round(total_samples * alpha))
    train_data = np.array(shuffled_data[:train_size])
    test_data = np.array(shuffled_data[train_size:])

    # 转换为 Tensor
    def prepare_tensors(data):
        if len(data) == 0: return None, None
        a, b, c = data[:, 0], data[:, 1], data[:, 2]
        
        op = np.full_like(a, 97) # '+'
        eq = np.full_like(a, 98) # '='
        
        # Input: a, +, b, =
        x = np.stack([a, op, b, eq], axis=1)
        y = c 
        return torch.LongTensor(x), torch.LongTensor(y)

    train_X, train_y = prepare_tensors(train_data)
    test_X, test_y = prepare_tensors(test_data)
    
    return train_X, train_y, test_X, test_y

# PART 2: 模型架构

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)

class GrokkingTransformerStrict(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.pos_encoder = SinusoidalPositionalEncoding(CONFIG['d_model'], CONFIG['max_len'])
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['d_model'],
            nhead=CONFIG['n_heads'],
            dim_feedforward=CONFIG['d_model'] * 4,
            dropout=CONFIG['dropout'],
            activation='relu',
            batch_first=True,
            norm_first=False 
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=CONFIG['n_layers'])
        self.fc = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'], bias=False)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        out = self.transformer_decoder(emb, mask=mask)
        last_token_out = out[:, -1, :]
        logits = self.fc(last_token_out)
        return logits

# PART 3:实验逻辑

def run_experiment(alpha, seed):
    # 只返回步数，不记录详细日志
    train_X, train_y, test_X, test_y = create_dataset_in_memory(alpha, seed)
    train_X, train_y = train_X.to(CONFIG['device']), train_y.to(CONFIG['device'])
    test_X, test_y = test_X.to(CONFIG['device']), test_y.to(CONFIG['device'])

    model = GrokkingTransformerStrict().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], betas=CONFIG['betas'])
    criterion = nn.CrossEntropyLoss()

    for step in range(CONFIG['max_steps']):
        idx = torch.randperm(train_X.size(0), device=CONFIG['device'])[:CONFIG['batch_size']]
        x_batch, y_batch = train_X[idx], train_y[idx]
        
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        if (step + 1) % CONFIG['check_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(test_X)
                val_preds = val_logits.argmax(1)
                val_acc = (val_preds == test_y).float().mean().item() * 100
            model.train()
            
            if val_acc >= CONFIG['target_acc']:
                return step + 1, True
            if torch.isnan(loss):
                return CONFIG['max_steps'], False

    return CONFIG['max_steps'], False

def run_detailed_experiment(alpha, seed, output_csv):
    print(f"Running Detailed Logging Experiment: Alpha={alpha}, Seed={seed}")
    
    # 1. 准备数据
    train_X, train_y, test_X, test_y = create_dataset_in_memory(alpha, seed)
    train_X, train_y = train_X.to(CONFIG['device']), train_y.to(CONFIG['device'])
    test_X, test_y = test_X.to(CONFIG['device']), test_y.to(CONFIG['device'])

    # 2. 初始化 CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_acc', 'test_acc', 'loss'])

    # 3. 模型设置
    model = GrokkingTransformerStrict().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], betas=CONFIG['betas'])
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(CONFIG['max_steps']), desc="Detailed Run")
    
    for step in pbar:
        # 训练步
        idx = torch.randperm(train_X.size(0), device=CONFIG['device'])[:CONFIG['batch_size']]
        x_batch, y_batch = train_X[idx], train_y[idx]
        
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        # 记录日志
        if (step + 1) % CONFIG['check_interval'] == 0:
            model.eval()
            with torch.no_grad():
                # 计算测试集准确率
                test_logits = model(test_X)
                test_preds = test_logits.argmax(1)
                test_acc = (test_preds == test_y).float().mean().item() * 100
                
                # 计算训练集准确率
                train_logits = model(train_X)
                train_preds = train_logits.argmax(1)
                train_acc = (train_preds == train_y).float().mean().item() * 100
            
            model.train()
            
            # 写入 CSV
            with open(output_csv, 'a', newline='') as f:
                csv.writer(f).writerow([step+1, train_acc, test_acc, loss.item()])
            
            pbar.set_postfix({'Train': f"{train_acc:.1f}%", 'Test': f"{test_acc:.1f}%"})


    print(f"Detailed logs saved to {output_csv}")

# PART 4: 绘图与主程序

def plot_learning_curves(log_csv, output_img='grokking_curve.png'):
    try:
        data = pd.read_csv(log_csv)
    except ImportError:
        steps, train_accs, test_accs = [], [], []
        with open(log_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row['step']))
                train_accs.append(float(row['train_acc']))
                test_accs.append(float(row['test_acc']))
        data = {'step': steps, 'train_acc': train_accs, 'test_acc': test_accs}
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['step'], data['train_acc'], label='Train Accuracy', alpha=0.9, linewidth=1.5)
    plt.plot(data['step'], data['test_acc'], label='Test Accuracy',alpha=0.9 ,linewidth=1.5, linestyle='--')
    
    plt.xlabel('Optimization Steps')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Grokking Learning Curve (Alpha=0.5)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.savefig(output_img)
    print(f"Learning curve plot saved to {output_img}")
    plt.close()

def plot_grid_results(results, output_img='grokking_grid_plot.png'):
    print("\nGenerating grid search plot...")
    plot_alphas = []
    plot_medians = []
    sorted_alphas = sorted(results.keys())
    
    for alpha in sorted_alphas:
        steps_list = results[alpha]
        if steps_list:
            median = np.median(steps_list)
            plot_alphas.append(alpha)
            plot_medians.append(median)

    plt.figure(figsize=(10, 6))
    for alpha in sorted_alphas:
        steps = results[alpha]
        plt.scatter([alpha] * len(steps), steps, color='blue', alpha=0.3, s=20)

    plt.plot(plot_alphas, plot_medians, 'r-', linewidth=2, label='Median Steps')
    plt.yscale('log')
    plt.xlabel('Data Fraction (Alpha)')
    plt.ylabel('Optimization Steps to >99% Acc')
    plt.title('Grokking Data Efficiency')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend()
    plt.savefig(output_img)
    print(f"Grid plot saved to {output_img}")
    plt.close()

if __name__ == "__main__":
    print(f"Running on: {CONFIG['device']}")
    
    #  TASK 1: 记录 alpha=0.5, seed=0 的详细曲线 
    CONFIG['max_steps'] = 10000
    CONFIG['check_interval'] = 5
    print("\n=== TASK 1: Detailed Single Run (Alpha=0.5) ===")
    run_detailed_experiment(alpha=0.5, seed=0, output_csv=CONFIG['detailed_log_file'])
    plot_learning_curves(CONFIG['detailed_log_file'], output_img='grokking_curve_alpha0.5.png')

    #  TASK 2: Grid Search 
    CONFIG['max_steps'] = 100000
    CONFIG['check_interval'] = 50
    print("\n=== TASK 2: Full Grid Search (Data Efficiency) ===")
    alphas = np.arange(0.15, 0.86, 0.05)
    seeds = range(10)

    results = {float(f"{alpha:.2f}"): [] for alpha in alphas}
    
    # 写入 Grid Search 结果头
    with open(CONFIG['results_file'], 'w', newline='') as f:
        csv.writer(f).writerow(['alpha', 'seed', 'steps', 'converged'])

    total_runs = len(alphas) * len(seeds)
    
    # 使用进度条
    with tqdm(total=total_runs, desc="Grid Search") as pbar:
        for alpha in alphas:
            alpha_key = float(f"{alpha:.2f}")
            for seed in seeds:
                steps, converged = run_experiment(alpha_key, seed)
                
                with open(CONFIG['results_file'], 'a', newline='') as f:
                    csv.writer(f).writerow([alpha_key, seed, steps, converged])
                
                if converged:
                    results[alpha_key].append(steps)
                else:
                    results[alpha_key].append(CONFIG['max_steps'])
                
                pbar.update(1)

    plot_grid_results(results)