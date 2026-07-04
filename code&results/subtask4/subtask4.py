import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from tqdm import tqdm
import csv
import pandas as pd
import math
import itertools
import argparse
import sys

CONFIG = {
    'modulus': 31,
    'd_model': 128,
    'n_layers': 2,
    'n_heads': 4,
    'batch_size': 512,
    'max_steps': 100000,
    'check_interval': 50,  
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'weight_decay': 1.0,
    'lr': 1e-3,
    'betas': (0.9, 0.98),
    'dropout': 0.0,
    'max_data_samples': 1000000
}

OUTPUT_DIR = "logs_K"


# PART 1: 泛化的数据生成

def create_dataset_with_k(k, alpha, seed):
    p = CONFIG['modulus']
    op_plus = p
    op_eq = p + 1
    total_space = p ** k
    rng = np.random.RandomState(seed)
    
    if total_space <= CONFIG['max_data_samples']:
        inputs = list(itertools.product(range(p), repeat=k))
        inputs = np.array(inputs)
    else:
        inputs = rng.randint(0, p, size=(CONFIG['max_data_samples'], k))

    targets = np.sum(inputs, axis=1) % p
    total_samples = len(inputs)
    
    X_seq = []
    for i in range(total_samples):
        seq = []
        nums = inputs[i]
        for idx, n in enumerate(nums):
            seq.append(n)
            if idx < k - 1:
                seq.append(op_plus)
            else:
                seq.append(op_eq)
        X_seq.append(seq)
    
    X_seq = np.array(X_seq)
    y_seq = targets
    
    indices = np.arange(total_samples)
    rng.shuffle(indices)
    train_size = int(round(total_samples * alpha))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_X = torch.LongTensor(X_seq[train_idx])
    train_y = torch.LongTensor(y_seq[train_idx])
    test_X = torch.LongTensor(X_seq[test_idx])
    test_y = torch.LongTensor(y_seq[test_idx])
    
    return train_X, train_y, test_X, test_y


# PART 2: 模型架构

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
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

class GrokkingTransformerFlexible(nn.Module):
    def __init__(self, vocab_size, max_len, use_pe=True):
        super().__init__()
        self.use_pe = use_pe
        self.embedding = nn.Embedding(vocab_size, CONFIG['d_model'])
        if self.use_pe:
            self.pos_encoder = SinusoidalPositionalEncoding(CONFIG['d_model'], max_len)
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
        self.fc = nn.Linear(CONFIG['d_model'], vocab_size, bias=False)

    def forward(self, x, use_mask=True):
        seq_len = x.size(1)
        emb = self.embedding(x)
        if self.use_pe:
            emb = self.pos_encoder(emb)
        if use_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        else:
            mask = torch.zeros(seq_len, seq_len).to(x.device)
        out = self.transformer_decoder(emb, mask=mask)
        last_token_out = out[:, -1, :]
        logits = self.fc(last_token_out)
        return logits


# PART 3: 实验与绘图逻辑

def plot_result(history, output_file, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['train_acc'], label='Train Acc', color='blue', alpha=0.6)
    plt.plot(history['step'], history['test_acc'], label='Test Acc', color='red', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Grokking Exp: {title}')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_file)
    plt.close()

def combine_all_plots(output_filename="combined_results.png"):
    print(f"\n--- Combining all plots in {OUTPUT_DIR} ---")
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        print(f"Directory {OUTPUT_DIR} does not exist.")
        return

    search_pattern = os.path.join(OUTPUT_DIR, "plot_*.png")
    img_files = sorted(glob.glob(search_pattern))
    
    if not img_files:
        print(f"No plot files found in {OUTPUT_DIR} to combine.")
        return

    num_plots = len(img_files)
    cols = 4
    rows = math.ceil(num_plots / cols)
    
    print(f"Found {num_plots} plots. Creating grid {rows}x{cols}...")
    fig_width = 5 * cols
    fig_height = 4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    
    # 处理 axes 数组维度的不同情况
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes.flatten()

    for i, img_path in enumerate(img_files):
        ax = axes_flat[i]
        try:
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(os.path.basename(img_path), fontsize=8)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    for j in range(num_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    
    # 将最终合并的图片也保存到子文件夹中
    final_output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(final_output_path, dpi=100)
    plt.close()
    print(f"Combined image saved to: {os.path.abspath(final_output_path)}")

def run_experiment_k(k, use_pe, use_mask, seed=42):
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    exp_name = f"K{k}_PE{int(use_pe)}_Mask{int(use_mask)}"
    
    # 修改文件路径到子文件夹
    csv_file = os.path.join(OUTPUT_DIR, f"log_{exp_name}.csv")
    img_file = os.path.join(OUTPUT_DIR, f"plot_{exp_name}.png")
    
    print(f"Start Exp: {exp_name} on {CONFIG['device']}")
    
    train_X, train_y, test_X, test_y = create_dataset_with_k(k, 0.5, seed)
    train_X = train_X.to(CONFIG['device'])
    train_y = train_y.to(CONFIG['device'])
    test_X = test_X.to(CONFIG['device'])
    test_y = test_y.to(CONFIG['device'])
    
    vocab_size = CONFIG['modulus'] + 5 
    max_len_seq = k * 2 + 5 

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_acc', 'test_acc', 'loss'])

    model = GrokkingTransformerFlexible(vocab_size, max_len_seq, use_pe=use_pe).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], betas=CONFIG['betas'])
    criterion = nn.CrossEntropyLoss()

    history = {'step': [], 'train_acc': [], 'test_acc': [], 'loss': []}

    pbar = tqdm(range(CONFIG['max_steps']), desc=exp_name, leave=False, position=0)
    
    for step in pbar:
        idx = torch.randperm(train_X.size(0), device=CONFIG['device'])[:CONFIG['batch_size']]
        x_batch, y_batch = train_X[idx], train_y[idx]
        
        optimizer.zero_grad()
        logits = model(x_batch, use_mask=use_mask)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        if (step + 1) % CONFIG['check_interval'] == 0:
            model.eval()
            with torch.no_grad():
                EVAL_SIZE = 4096
                
                # Test Acc
                if len(test_X) > EVAL_SIZE:
                    perm_test = torch.randperm(test_X.size(0), device=CONFIG['device'])
                    idx_test = perm_test[:EVAL_SIZE]
                    test_sample_x, test_sample_y = test_X[idx_test], test_y[idx_test]
                    test_acc = (model(test_sample_x, use_mask=use_mask).argmax(1) == test_sample_y).float().mean().item() * 100
                else:
                    test_acc = (model(test_X, use_mask=use_mask).argmax(1) == test_y).float().mean().item() * 100
                
                # Train Acc
                if len(train_X) > EVAL_SIZE:
                    perm_train = torch.randperm(train_X.size(0), device=CONFIG['device'])
                    idx_train = perm_train[:EVAL_SIZE]
                    train_sample_x, train_sample_y = train_X[idx_train], train_y[idx_train]
                    train_acc = (model(train_sample_x, use_mask=use_mask).argmax(1) == train_sample_y).float().mean().item() * 100
                else:
                    train_acc = (model(train_X, use_mask=use_mask).argmax(1) == train_y).float().mean().item() * 100
            
            model.train()
            
            history['step'].append(step+1)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(loss.item())
            
            with open(csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([step+1, train_acc, test_acc, loss.item()])
            
            pbar.set_postfix({'Tr': f"{train_acc:.1f}", 'Te': f"{test_acc:.1f}"})

    plot_result(history, img_file, title=f"K={k}, PE={use_pe}, Mask={use_mask}")
    print(f"Finished: {exp_name}")


# PART 4: 命令行入口


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Single Grokking Experiment")
    
    # 互斥组：要么运行单个实验，要么合并图片
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('--run_single', action='store_true', help="Run a single experiment configuration")
    group.add_argument('--combine_only', action='store_true', help="Only combine existing plots")

    # 实验参数
    parser.add_argument('--k', type=int, default=2, help="Number of operands")
    parser.add_argument('--pe', type=int, default=1, choices=[0, 1], help="Use Positional Encoding (1=True, 0=False)")
    parser.add_argument('--mask', type=int, default=1, choices=[0, 1], help="Use Causal Mask (1=True, 0=False)")
    parser.add_argument('--seed', type=int, default=999, help="Random seed")

    args = parser.parse_args()

    if args.combine_only:
        combine_all_plots()
    elif args.run_single:
        run_experiment_k(
            k=args.k,
            use_pe=bool(args.pe),
            use_mask=bool(args.mask),
            seed=args.seed
        )