import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import math
import csv
import os
import argparse
import sys

#  配置区域 
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'modulus': 97,
    'model_params': {
        'd_model': 128, 
        'n_layers': 2, 
        'n_heads': 4,
        'vocab_size': 100, 
        'max_len': 6,
        'max_steps': 50000,
        'batch_size': 512,
        'log_interval': 50
    },
    'log_dir': './logs_optimized'
}

#  模型定义 
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, weight_noise=0.0):
        super().__init__(in_features, out_features, bias)
        self.weight_noise = weight_noise
    def forward(self, input):
        if self.weight_noise > 0 and self.training:
            with torch.no_grad():
                weight_noise_tensor = torch.randn_like(self.weight) * self.weight_noise
                bias_noise_tensor = torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + weight_noise_tensor
            bias = self.bias + bias_noise_tensor if self.bias is not None else self.bias
        else:
            weight = self.weight; bias = self.bias
        return F.linear(input, weight, bias)

class NoisyLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, weight_noise=0.0):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.weight_noise = weight_noise
    def forward(self, input):
        if self.weight_noise > 0 and self.training:
            with torch.no_grad():
                weight_noise_tensor = torch.randn_like(self.weight) * self.weight_noise
                bias_noise_tensor = torch.randn_like(self.bias) * self.weight_noise if self.bias is not None else None
            weight = self.weight + weight_noise_tensor
            bias = self.bias + bias_noise_tensor if self.bias is not None else self.bias
        else:
            weight = self.weight; bias = self.bias
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

class NoisyEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, weight_noise=0.0):
        super().__init__(num_embeddings, embedding_dim)
        self.weight_noise = weight_noise
    def forward(self, input):
        if self.weight_noise > 0 and self.training:
            with torch.no_grad():
                noise = torch.randn_like(self.weight) * self.weight_noise
            weight = self.weight + noise
        else:
            weight = self.weight
        return F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

class NoisyMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, weight_noise=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = NoisyLinear(d_model, d_model, bias=False, weight_noise=weight_noise)
        self.w_k = NoisyLinear(d_model, d_model, bias=False, weight_noise=weight_noise)
        self.w_v = NoisyLinear(d_model, d_model, bias=False, weight_noise=weight_noise)
        self.fc = NoisyLinear(d_model, d_model, bias=False, weight_noise=weight_noise)
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        q = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.fc(out)

class NoisyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.0, weight_noise=0.0):
        super().__init__()
        self.attn = NoisyMultiHeadAttention(d_model, n_heads, weight_noise=weight_noise)
        self.norm1 = NoisyLayerNorm(d_model, weight_noise=weight_noise)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            NoisyLinear(d_model, dim_feedforward, bias=False, weight_noise=weight_noise),
            nn.ReLU(),
            NoisyLinear(dim_feedforward, d_model, bias=False, weight_noise=weight_noise)
        )
        self.norm2 = NoisyLayerNorm(d_model, weight_noise=weight_noise)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout2(mlp_out))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class GrokkingTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        weight_noise = config.get('weight_noise', 0.0)
        dropout = config.get('dropout', 0.0)
        self.embedding = NoisyEmbedding(self.vocab_size, self.d_model, weight_noise=weight_noise)
        self.pos_encoder = PositionalEncoding(self.d_model, config['max_len'])
        self.layers = nn.ModuleList([
            NoisyTransformerBlock(self.d_model, config['n_heads'], self.d_model * 4, dropout, weight_noise) 
            for _ in range(config['n_layers'])
        ])
        self.fc = NoisyLinear(self.d_model, self.vocab_size, bias=False, weight_noise=weight_noise)
    def forward(self, x):
        sz = x.size(1)
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(x.device)
        emb = self.pos_encoder(self.embedding(x))
        out = emb
        for layer in self.layers: out = layer(out, mask=mask)
        return self.fc(out[:, -1, :])

class CustomAdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, weight_decay_form="to_zero"):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, weight_decay_form=weight_decay_form)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p); state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['weight_decay_form'] == 'to_init': state['init'] = p.detach().clone()
                if group['weight_decay'] > 0:
                    if group['weight_decay_form'] == 'to_zero': p.mul_(1 - group['lr'] * group['weight_decay'])
                    elif group['weight_decay_form'] == 'to_init': p.add_((state['init'] - p) * (group['lr'] * group['weight_decay']))
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                step_size = group['lr'] / (1 - beta1 ** state['step'])
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

def get_data(alpha, seed, device):
    MODULUS = CONFIG['modulus']
    all_eqs = []
    for a in range(MODULUS):
        for b in range(MODULUS):
            all_eqs.append([a, 97, b, 98, (a + b) % MODULUS]) 
    
    rng = np.random.RandomState(seed)
    rng.shuffle(all_eqs)
    
    data_tensor = torch.LongTensor(all_eqs).to(device)
    n_train = int(len(all_eqs) * alpha)
    return data_tensor[:n_train], data_tensor[n_train:]

#  定义所有实验配置 
def get_experiment_config(exp_name):
    base_config = CONFIG['model_params']
    experiments = {
        'AdamW (WD=1.0) [Baseline]': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1.0 },
        'Adam (No WD)': { **base_config, 'optimizer': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0 },
        'AdamW (WD to Init)': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1.0, 'wd_form': 'to_init' },
        'AdamW (High WD=3.0)': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 3.0 },
        'AdamW (Low LR=1e-4)': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 1.0 },
        'AdamW (Std Betas 0.999)': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1.0, 'betas': (0.9, 0.999) },
        'Adam (LR=5e-3)': { **base_config, 'optimizer': 'Adam', 'lr': 5e-3, 'weight_decay': 0.0 },
        'SGD (Heavy-Ball, mom=0.9)': { **base_config, 'optimizer': 'SGD', 'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9, 'nesterov': False },
        'RMSprop': { **base_config, 'optimizer': 'RMSprop', 'lr': 1e-3, 'weight_decay': 1e-4 },
        'SGD (Nesterov, mom=0.99)': { **base_config, 'optimizer': 'SGD', 'lr': 0.05, 'weight_decay': 1e-4, 'momentum': 0.99, 'nesterov': True },
        'Full-Batch GD': { **base_config, 'optimizer': 'SGD', 'lr': 0.05, 'weight_decay': 0, 'momentum': 0, 'nesterov': False, 'batch_size': -1 },
        'AdamW (Weight Noise=0.01)': { **base_config, 'optimizer': 'AdamW', 'lr': 1e-3, 'weight_decay': 1.0, 'weight_noise': 0.01 }
    }
    return experiments.get(exp_name)

def run_experiment(exp_name, alpha, seed):
    config = get_experiment_config(exp_name)
    if config is None:
        raise ValueError(f"Experiment {exp_name} not found!")

    device = CONFIG['device']
    log_dir = CONFIG['log_dir']
    
    # 路径处理
    save_folder = os.path.join(log_dir, exp_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", ""))
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, f"alpha_{alpha}_seed_{seed}.csv")
    
    if os.path.exists(csv_path):
        print(f"Skipping existing: {csv_path}")
        return

    # 设置种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 数据
    train_data, val_data = get_data(alpha, seed, device)
    train_X, train_y = train_data[:, :-1], train_data[:, -1]
    val_X, val_y = val_data[:, :-1], val_data[:, -1]
    
    # 模型
    model = GrokkingTransformer(config).to(device)
    
    # 编译优化
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass # Fallback

    # 优化器选择
    opt_name = config['optimizer']
    lr = config['lr']
    wd = config['weight_decay']
    if opt_name == 'AdamW':
        optimizer = CustomAdamW(model.parameters(), lr=lr, weight_decay=wd, 
                                weight_decay_form=config.get('wd_form', 'to_zero'), 
                                betas=config.get('betas', (0.9, 0.98)))
    elif opt_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=config.get('betas', (0.9, 0.999)))
    elif opt_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.get('momentum', 0.0), 
                              weight_decay=wd, nesterov=config.get('nesterov', False))
    elif opt_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=config.get('momentum', 0.0))

    criterion = nn.CrossEntropyLoss()
    max_steps = config.get('max_steps', 50000)
    batch_size = config.get('batch_size', 512)
    log_interval = config.get('log_interval', 50) 

    if batch_size == -1: batch_size = len(train_X)
    n_train = len(train_X)

    print(f"Running: {exp_name} | A={alpha} | S={seed} | PID={os.getpid()}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
        
        best_val_acc = 0.0
        model.train()
        
        for step in range(max_steps + 1):
            if batch_size == n_train:
                idx = torch.arange(n_train, device=device)
            else:
                idx = torch.randint(0, n_train, (batch_size,), device=device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_X[idx])
            loss = criterion(logits, train_y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 限制梯度范数为 1.0
            optimizer.step()
            
            if step % log_interval == 0:
                model.eval()
                with torch.no_grad():
                    train_acc = (logits.argmax(dim=-1) == train_y[idx]).float().mean().item() * 100
                    train_loss = loss.item()
                    
                    val_logits = model(val_X)
                    val_loss = criterion(val_logits, val_y).item()
                    val_acc = (val_logits.argmax(dim=-1) == val_y).float().mean().item() * 100
                    
                    writer.writerow([step, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}"])
                    f.flush()
                    
                    if val_acc > best_val_acc: best_val_acc = val_acc
                    if best_val_acc > 99.5 and val_acc > 99.0: break
                model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    torch.set_float32_matmul_precision('high')
    run_experiment(args.exp_name, args.alpha, args.seed)