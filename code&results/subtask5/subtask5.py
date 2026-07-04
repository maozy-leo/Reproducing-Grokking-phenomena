import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
import math
import os

# 0. 全局配置
CONFIG = {
    'vocab_size': 100,
    'd_model': 128,
    'hidden_dim': 128,
    'p': 97,
    'alpha': 0.5,
    'seed': 42,
    'batch_size': 512,
    'lr': 1e-3,
    'weight_decay': 1.0,
    'max_steps': 50000,
    'snapshot_interval': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"Running on {CONFIG['device']}")

def get_flat_params(model):
    """获取模型参数展平向量"""
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_flat_params(model, flat_params):
    """将向量恢复为模型参数"""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = flat_params[offset:offset + numel].view(p.size()).to(p.device)
        offset += numel

# 1. 数据生成
def create_dataset(alpha, seed, p=97):
    """生成训练集和测试集 (用于训练)"""
    all_equations = []
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            all_equations.append([a, b, c])
    
    data = np.array(all_equations)
    rng = np.random.RandomState(seed)
    rng.shuffle(data)
    
    train_size = int(len(data) * alpha)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    def prepare_xy(d):
        if len(d) == 0: return None, None
        x = torch.tensor(d[:, :2], dtype=torch.long)
        y = torch.tensor(d[:, 2], dtype=torch.long)
        return x, y

    train_X, train_y = prepare_xy(train_data)
    test_X, test_y = prepare_xy(test_data)
    return train_X, train_y, test_X, test_y

def get_full_dataset(p=97):
    """生成全量数据集 (用于 Loss Landscape 绘制)"""
    all_equations = []
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            all_equations.append([a, b, c])
    data = np.array(all_equations)
    
    x = torch.tensor(data[:, :2], dtype=torch.long)
    y = torch.tensor(data[:, 2], dtype=torch.long)
    return x, y

# 2. 模型定义

# --- Transformer ---
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

class GrokkingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG['vocab_size'], CONFIG['d_model'])
        self.pos_encoder = SinusoidalPositionalEncoding(CONFIG['d_model'], max_len=10)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG['d_model'], nhead=4, dim_feedforward=CONFIG['d_model'] * 4,
            dropout=0.0, activation='relu', batch_first=True, norm_first=False 
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.fc = nn.Linear(CONFIG['d_model'], CONFIG['vocab_size'], bias=False)

    def forward(self, x):
        # 适配输入: x [batch, 2] -> [a, +, b, =]
        batch_size = x.size(0)
        device = x.device
        a, b = x[:, 0], x[:, 1]
        op = torch.full((batch_size,), 97, device=device) # '+'
        eq = torch.full((batch_size,), 98, device=device) # '='
        
        inp = torch.stack([a, op, b, eq], dim=1)
        emb = self.pos_encoder(self.embedding(inp))
        mask = torch.triu(torch.ones(4, 4) * float('-inf'), diagonal=1).to(device)
        out = self.transformer_decoder(emb, mask=mask)
        return self.fc(out[:, -1, :])

# --- Simple MLP ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONFIG['vocab_size'], CONFIG['hidden_dim'])
        self.net = nn.Sequential(
            nn.Linear(CONFIG['hidden_dim'] * 2, CONFIG['hidden_dim']), 
            nn.ReLU(),
            nn.Linear(CONFIG['hidden_dim'], CONFIG['vocab_size'])
        )
    def forward(self, x):
        emb = self.embedding(x).view(x.size(0), -1)
        return self.net(emb)

# 3. 训练与轨迹收集
def train_and_collect_path(model_class, name):
    print(f"\nTraining {name} with alpha={CONFIG['alpha']}...")
    # 使用部分数据进行训练
    train_X, train_y, test_X, test_y = create_dataset(CONFIG['alpha'], CONFIG['seed'])
    train_X, train_y = train_X.to(CONFIG['device']), train_y.to(CONFIG['device'])
    test_X, test_y = test_X.to(CONFIG['device']), test_y.to(CONFIG['device'])
    
    model = model_class().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    weight_history = [get_flat_params(model).cpu().numpy()]
    step_history = [0]
    
    found_train_99 = None
    found_test_99 = None

    for step in range(CONFIG['max_steps']):
        model.train()
        idx = torch.randperm(train_X.size(0))[:CONFIG['batch_size']]
        optimizer.zero_grad()
        loss = criterion(model(train_X[idx]), train_y[idx])
        loss.backward()
        optimizer.step()
        
        if (step + 1) % CONFIG['snapshot_interval'] == 0 or step == 0:
            weight_history.append(get_flat_params(model).cpu().numpy())
            step_history.append(step + 1)
            
            model.eval()
            with torch.no_grad():
                tr_acc = (model(train_X).argmax(1) == train_y).float().mean().item()
                te_acc = (model(test_X).argmax(1) == test_y).float().mean().item()
            
            if tr_acc >= 0.99 and found_train_99 is None: found_train_99 = step + 1
            if te_acc >= 0.99 and found_test_99 is None: found_test_99 = step + 1

            if (step+1) % 2000 == 0:
                print(f"Step {step+1}: TrAcc {tr_acc:.2%}, TeAcc {te_acc:.2%}")
                

    return model, np.array(weight_history), step_history, found_train_99, found_test_99

# 4. PCA 可视化
def plot_loss_landscape_pca(model, weight_path, step_history, train_99_step, test_99_step, X_full, y_full, filename):
    print(f"Generating Loss Landscape for {filename} using FULL DATASET...")
    
    # 1. PCA 投影
    theta_final = weight_path[-1]
    M = weight_path - theta_final
    pca = PCA(n_components=2)
    coords = pca.fit_transform(M)
    pc1, pc2 = pca.components_[0], pca.components_[1]

    variance_ratios = pca.explained_variance_ratio_
    var_pc1 = variance_ratios[0] * 100
    var_pc2 = variance_ratios[1] * 100
    total_var = var_pc1 + var_pc2
    variance_str = f"(Explained Variance: PC1={var_pc1:.1f}%, PC2={var_pc2:.1f}%, Total={total_var:.1f}%)"
    print(f"PCA Explained Variance: PC1={var_pc1:.2f}%, PC2={var_pc2:.2f}%")
    
    # 2. 网格定义
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin_x, margin_y = (x_max - x_min) * 0.2, (y_max - y_min) * 0.2
    
    res = 30 # 分辨率
    xg = np.linspace(x_min - margin_x, x_max + margin_x, res)
    yg = np.linspace(y_min - margin_y, y_max + margin_y, res)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = np.zeros_like(XX)
    
    # 3. 计算 Grid Loss (Full Dataset)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    theta_final_t = torch.tensor(theta_final, device=CONFIG['device'])
    pc1_t = torch.tensor(pc1, device=CONFIG['device'])
    pc2_t = torch.tensor(pc2, device=CONFIG['device'])
    
    X_full = X_full.to(CONFIG['device'])
    y_full = y_full.to(CONFIG['device'])
    
    print("Computing grid losses (this may take a moment)...")
    with torch.no_grad():
        for i in range(res):
            for j in range(res):
                delta = XX[i, j] * pc1_t + YY[i, j] * pc2_t
                set_flat_params(model, theta_final_t + delta)
                
                # 使用全量数据计算 Loss
                logits = model(X_full)
                loss = criterion(logits, y_full).item()
                ZZ[i, j] = loss
                
    # 恢复模型权重
    set_flat_params(model, theta_final_t)
    
    # 4. 绘图
    plt.figure(figsize=(12, 10))
    # 使用 Log Loss 使得等高线更清晰
    cp = plt.contourf(XX, YY, np.log(ZZ + 1e-5), levels=30, cmap='viridis', alpha=0.9)
    cbar = plt.colorbar(cp)
    cbar.set_label('Log(Cross Entropy Loss) on Full Dataset')
    
    # 绘制轨迹
    plt.plot(coords[:, 0], coords[:, 1], c='white', alpha=0.5, linewidth=2, label='Optimization Path')
    plt.scatter(coords[0, 0], coords[0, 1], c='white', marker='x', s=100, label='Initial')
    plt.scatter(coords[-1, 0], coords[-1, 1], c='red', marker='*', s=200, label='Final')
    
    # 标注 99% 点
    def find_coord(step):
        idx = (np.abs(np.array(step_history) - step)).argmin()
        return coords[idx], step_history[idx]

    if train_99_step:
        c, s = find_coord(train_99_step)
        plt.scatter(c[0], c[1], c='orange', s=150, edgecolors='k', zorder=5, label=f'Train 99% (Step {s})')
    
    if test_99_step:
        c, s = find_coord(test_99_step)
        plt.scatter(c[0], c[1], c='cyan', marker='^', s=150, edgecolors='k', zorder=5, label=f'Test 99% (Step {s})')

    title_text = (
        f'Loss Landscape (PCA) - {filename.replace(".png", "")}\n'
        f'{variance_str}'
    )
    plt.title(title_text)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

# 主程序
if __name__ == "__main__":
    X_full, y_full = get_full_dataset(CONFIG['p'])
    
    # 1. Transformer
    model, weights, steps, tr99, te99 = train_and_collect_path(GrokkingTransformer, "Transformer")
    plot_loss_landscape_pca(model, weights, steps, tr99, te99, X_full, y_full, 'transformer.png')
    
    # 2. SimpleMLP
    model, weights, steps, tr99, te99 = train_and_collect_path(SimpleMLP, "SimpleMLP")
    plot_loss_landscape_pca(model, weights, steps, tr99, te99, X_full, y_full, 'simpleMLP.png')
    
    print("All tasks finished.")