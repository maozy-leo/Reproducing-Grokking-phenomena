import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import math
import glob
import pandas as pd

#  0. 环境与配置 
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

OUTPUT_DIR = 'subtask2'
ensure_dir(OUTPUT_DIR)

GLOBAL_CONFIG = {
    'alpha': 0.50,
    'seed': 0,
    'batch_size': 512,
    'embedding_dim': 128,
    'lr': 1e-3,
    'weight_decay': 1.0,
    'steps': 20000,
    'log_interval': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'p': 97
}

print(f"运行设备: {GLOBAL_CONFIG['device']}")

#  1. 数据生成 
def generate_mod_addition_data(p, alpha, seed):
    all_equations = []
    for a in range(p):
        for b in range(p):
            all_equations.append([a, b, (a + b) % p])
    
    data_np = np.array(all_equations, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(data_np)
    
    train_size = int(round(len(data_np) * alpha))
    train_data, test_data = data_np[:train_size], data_np[train_size:]
    
    return (torch.tensor(train_data[:, :2], dtype=torch.long), 
            torch.tensor(train_data[:, 2], dtype=torch.long),
            torch.tensor(test_data[:, :2], dtype=torch.long), 
            torch.tensor(test_data[:, 2], dtype=torch.long))

#  2. 模型定义 

# GLU 相关
class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc_in(x) * torch.sigmoid(self.fc_gate(x))

class GluMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.glu1 = GLU(embedding_dim * 2, hidden_dim)
        self.glu2 = GLU(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        return self.fc_out(self.glu2(self.glu1(x)))

# SIREN 相关
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        with torch.no_grad():
            limit = np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-limit, limit)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SirenMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = SineLayer(embedding_dim * 2, hidden_dim)
        self.layer2 = SineLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        return self.fc(self.layer2(self.layer1(x)))

# 其他基础模型
class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
    def forward(self, x):
        return self.net(self.embedding(x).view(x.size(0), -1))

class Simple1DCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, kernel_size=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x)).view(x.size(0), -1)
        return self.fc(x)

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        _, (h_n, _) = self.lstm(self.embedding(x))
        return self.fc(h_n[-1])

class ResNetMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_blocks=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) for _ in range(num_blocks)])
        self.head = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = torch.relu(self.proj(self.embedding(x).view(x.size(0), -1)))
        for block in self.blocks:
            x = torch.relu(x + block(x))
        return self.head(x)

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        _, h_n = self.rnn(self.embedding(x))
        return self.fc(h_n[-1])

class AdvancedRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.RNN(hidden_dim, hidden_dim, batch_first=True, bidirectional=True) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)])
        self.projs = nn.ModuleList([nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    def forward(self, x):
        x = self.input_proj(self.embedding(x))
        out = x
        for rnn, norm, proj in zip(self.layers, self.norms, self.projs):
            res = x
            out, _ = rnn(x)
            x = torch.relu(proj(norm(out)) + res)
        return self.fc(torch.mean(out, dim=1))

class DeepMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout)
        ) for _ in range(num_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.input_proj(self.embedding(x).view(x.size(0), -1))
        for block in self.blocks:
            x = x + block(x)
        return self.head(self.final_ln(x))

#  3. 训练引擎 
def train_engine(model, model_name, config):
    X_train, y_train, X_test, y_test = generate_mod_addition_data(config['p'], config['alpha'], config['seed'])
    X_train, y_train, X_test, y_test = [t.to(config['device']) for t in [X_train, y_train, X_test, y_test]]
    
    model = model.to(config['device'])
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    csv_path = os.path.join(OUTPUT_DIR, f"{model_name.replace(' ', '_')}_log.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_acc', 'test_acc', 'loss'])
        
        pbar = tqdm(range(config['steps']), desc=f"Training {model_name}")
        for step in pbar:
            model.train()
            idx = torch.randperm(X_train.size(0))[:config['batch_size']]
            optimizer.zero_grad()
            loss = criterion(model(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            
            if (step + 1) % config['log_interval'] == 0:
                model.eval()
                with torch.no_grad():
                    tr_acc = (model(X_train).argmax(1) == y_train).float().mean().item() * 100
                    te_acc = (model(X_test).argmax(1) == y_test).float().mean().item() * 100
                    writer.writerow([step + 1, tr_acc, te_acc, loss.item()])
                    pbar.set_postfix({'Tr': f"{tr_acc:.1f}%", 'Te': f"{te_acc:.1f}%"})

#  4. 汇总绘图 
def plot_combined_results():
    csv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, '*_log.csv')))
    if not csv_files: return

    num_files = len(csv_files)
    cols = 3
    rows = math.ceil(num_files / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if num_files > 1 else [axes]
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        name = os.path.basename(csv_file).replace('_log.csv', '').replace('_', ' ')
        axes[i].plot(df['step'], df['train_acc'], label='Train Acc', alpha=0.8)
        axes[i].plot(df['step'], df['test_acc'], label='Test Acc', linestyle='--', alpha=0.8)
        axes[i].set_title(name, fontweight='bold')
        axes[i].set_xscale('log')
        axes[i].set_ylim(-5, 105)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        axes[i].legend(loc='lower right', fontsize='small')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_results.png"), dpi=150)
    print(f"\n汇总图表已保存至 {OUTPUT_DIR}/combined_results.png")

#  5. 主程序 
def main():
    experiments = [
        (GluMLP, "Gated MLP", {'hidden_dim': 128}),
        (SirenMLP, "SIREN", {'hidden_dim': 128, 'steps': 50000}),
        (SimpleMLP, "Simple MLP", {'hidden_dim': 128}),
        (Simple1DCNN, "1D-CNN", {'hidden_dim': 128}),
        (SimpleLSTM, "LSTM", {'hidden_dim': 128}),
        (ResNetMLP, "ResNet MLP", {'hidden_dim': 256}),
        (SimpleRNN, "Vanilla RNN", {'hidden_dim': 128}),
        (AdvancedRNN, "Advanced RNN", {'hidden_dim': 128, 'steps': 5000}),
        (DeepMLP, "Deep MLP", {'hidden_dim': 128, 'steps': 5000})
    ]
    
    for ModelClass, name, updates in experiments:
        config = GLOBAL_CONFIG.copy()
        config.update(updates)
        
        if ModelClass in [Simple1DCNN]:
            model = ModelClass(config['p'], config['embedding_dim'], config['hidden_dim'])
        elif ModelClass in [ResNetMLP]:
            model = ModelClass(config['p'], config['embedding_dim'], config['hidden_dim'])
        elif ModelClass in [AdvancedRNN, DeepMLP]:
            model = ModelClass(config['p'], config['embedding_dim'], config['hidden_dim'])
        else:
            model = ModelClass(config['p'], config['embedding_dim'], config['hidden_dim'])
            
        train_engine(model, name, config)
    
    plot_combined_results()

if __name__ == "__main__":
    main()