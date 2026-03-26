"""
Step 3: Deterministic Residual Learning - Robust Multi-Step Training
==================================================================
改进点：
1. Trajectory-Aware Dataset: 确保 Rollout 窗口不跨越不同轨迹的边界。通过位置跳变自动识别轨迹切换。
2. Mixed Loss: 结合 One-step Residual Loss (作为锚点防止变坏) 和 Multi-step Rollout Loss (作为长时优化目标)。
3. Input Augmentation Fix: 只对初始输入 x0 加噪以增强鲁棒性，绝对不污染未来的真值标签 (x1...xH)。
4. Curriculum Learning: beta 随训练轮次逐步增加，前期先练好单步偏差，后期再攻克多步累积。
"""

import os
import json
import copy
import h5py
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Dataset (轨迹感知 + 预归一化优化)
# ============================================================

class TrajectoryDataset(Dataset):
    """
    按轨迹加载数据，并在初始化时完成所有数据的预归一化，显著提升训练速度。
    """
    def __init__(self, mat_file, horizon=5, stats=None):
        self.mat_file = mat_file
        self.horizon = horizon
        self.physical_params = None

        # 尝试从数据集所在目录提取物理参数
        try:
            import scipy.io as sio
            data_dir = os.path.dirname(mat_file)
            import glob
            # 搜索数据集目录下任意包含 traj_case_*.mat 的子文件夹
            case_files = glob.glob(os.path.join(data_dir, 'case_*', 'traj_case_*.mat'))
            if case_files:
                sample_file = case_files[0]
                sample_m = sio.loadmat(sample_file, squeeze_me=True)
                if 'traj' in sample_m and 'params' in sample_m['traj'].dtype.names:
                    p_struct = sample_m['traj']['params'].item()
                    self.physical_params = {n: p_struct[n] for n in p_struct.dtype.names}
        except Exception as e:
            print(f"--> Warning: Could not extract physical params from dataset: {e}")

        with h5py.File(mat_file, 'r') as f:
            root = f['split_ds'] if 'split_ds' in f else f['dataset']
            X_all = np.array(root['X']).T.astype(np.float32)
            U_all = np.array(root['U']).T.astype(np.float32)
            R_all = np.array(root['R_label']).T.astype(np.float32)
            
            # 通过位置跳变识别轨迹边界
            pos = X_all[:, :30]
            dist_sq = np.sum((pos[1:] - pos[:-1])**2, axis=1)
            boundaries = np.where(dist_sq > 25.0)[0] + 1
            
            # 计算或接收统计量
            if stats is None:
                self.stats = {
                    'X_mean': np.mean(X_all, axis=0), 'X_std': np.std(X_all, axis=0) + 1e-6,
                    'U_mean': np.mean(U_all, axis=0), 'U_std': np.std(U_all, axis=0) + 1e-6,
                    'R_mean': np.mean(R_all, axis=0), 'R_std': np.std(R_all, axis=0) + 1e-6,
                }
            else:
                self.stats = stats
            
            # 关键：预归一化全部数据 (CPU 运行一次即可)
            self.X_norm = (X_all - self.stats['X_mean']) / self.stats['X_std']
            self.U_norm = (U_all - self.stats['U_mean']) / self.stats['U_std']
            self.R_norm = (R_all - self.stats['R_mean']) / self.stats['R_std']

            self.trajs_X = np.split(self.X_norm, boundaries)
            self.trajs_U = np.split(self.U_norm, boundaries)
            self.trajs_R = np.split(self.R_norm, boundaries)
            
        self.valid_indices = []
        for t_idx, t_data in enumerate(self.trajs_X):
            T = t_data.shape[0]
            if T > self.horizon:
                for s in range(T - self.horizon):
                    self.valid_indices.append((t_idx, s))

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t_idx, s_idx = self.valid_indices[idx]
        # 直接切片已归一化的数据，无需重复计算
        x_seq = self.trajs_X[t_idx][s_idx : s_idx + self.horizon + 1]
        u_seq = self.trajs_U[t_idx][s_idx : s_idx + self.horizon]
        r_seq = self.trajs_R[t_idx][s_idx : s_idx + self.horizon]

        return (
            torch.from_numpy(x_seq), 
            torch.from_numpy(u_seq), 
            torch.from_numpy(r_seq)
        )

# ============================================================
# Model
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=90, output_dim=60, hidden_dim=256, depth=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, u):
        h = self.input_proj(torch.cat([x, u], dim=-1))
        return self.head(self.blocks(h))

# ============================================================
# Ops (核心提速点：减少冗余张量创建和多步优化)
# ============================================================

def denormalize(val, m_t, s_t):
    """直接使用已在 device 上的 tensor，消除函数闭包/临时张量创建开销"""
    return val * s_t + m_t

def normalize(val, m_t, s_t):
    return (val - m_t) / s_t

def nominal_dynamics_torch(x, u, dt=0.3):
    p, v, a = x[:, :30], x[:, 30:], u
    new_p = p + v * dt + 0.5 * a * (dt**2)
    new_v = v + a * dt
    return torch.cat([new_p, new_v], dim=-1)

def compute_mixed_loss(model, x_seq, u_seq, r_lbl_seq, stats_t, horizon, alpha=1.0, beta=0.5, x0_override=None):
    """
    stats_t: 必须是已经 .to(device) 的张量字典。
    x0_override: 提供用于加噪训练的初始状态，避免 clone 整个 x_seq。
    """
    x_curr_norm = x0_override if x0_override is not None else x_seq[:, 0, :]
    
    # One-step Residual Anchor (锚点)
    r_pred_1 = model(x_curr_norm, u_seq[:, 0, :])
    anchor_loss = torch.mean((r_pred_1 - r_lbl_seq[:, 0, :])**2)
    
    # 获取统计量
    xm, xs = stats_t['X_mean'], stats_t['X_std']
    rm, rs = stats_t['R_mean'], stats_t['R_std']
    
    # Multi-step Rollout
    rollout_loss = 0
    for t in range(horizon):
        u_t_norm = u_seq[:, t, :]
        r_p_norm = model(x_curr_norm, u_t_norm)
        
        # 物理演化 (通过已在设备上的 stats_t 进行反归一化)
        x_c = denormalize(x_curr_norm, xm, xs)
        u_c = denormalize(u_t_norm, stats_t['U_mean'], stats_t['U_std'])
        r_p = denormalize(r_p_norm, rm, rs)
        
        x_next_pred = nominal_dynamics_torch(x_c, u_c) + r_p
        
        # 对比物理空间真值 (真值在 Dataset 预归一化时已就绪)
        x_true_next = denormalize(x_seq[:, t+1, :], xm, xs)
        step_err = torch.mean((x_next_pred - x_true_next)**2)
        rollout_loss += step_err * (1.0 + t/horizon)
        
        # 重新归一化进入下一步循环
        x_curr_norm = normalize(x_next_pred, xm, xs)

    return alpha * anchor_loss + beta * (rollout_loss / horizon)

# ============================================================
# Train
# ============================================================

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(TRAIN_ROOT, '..', '..'))
    COLLECT_ROOT = os.path.join(PROJECT_ROOT, 'traj', 'step3_collect')
    
    import glob
    manifests = glob.glob(os.path.join(COLLECT_ROOT, "**", "manifest.mat"), recursive=True)
    if not manifests:
        print("Error: No data found."); return
    # 按修改时间排序找到最新的数据集目录
    DATA_DIR = os.path.dirname(max(manifests, key=os.path.getmtime))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RUN_OUT_DIR = os.path.join(TRAIN_ROOT, 'out')
    BACKUP_DIR = os.path.join(RUN_OUT_DIR, timestamp)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HORIZON, BATCH_SIZE, EPOCHS, LR = 5, args.batch_size, args.epochs, args.lr # 使用解析后的参数

    # 加载数据集
    train_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_train.mat'), horizon=HORIZON)
    val_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_val.mat'), horizon=HORIZON, stats=train_ds.stats)
    
    # 启用多线程读取和锁页内存
    num_workers = 4 if os.name != 'nt' else 0 # Windows 下 num_workers > 0 容易报错，小心使用
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    # 准备统计量 Tensor (核心改进：只创建一次，并传给 loss 函数)
    stats_t = {k: torch.tensor(v, device=DEVICE) for k, v in train_ds.stats.items()}

    model = ResidualMLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda')) # 开启混合精度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    print(f"--> Starting Optimized Robust Training on {DEVICE}")
    best_val = float('inf')
    PATIENCE_LIMIT = 25
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_l = 0
        beta_curr = min(0.1 + epoch/60, 0.5)

        for x_seq, u_seq, r_lbl in train_loader:
            x_seq, u_seq, r_lbl = x_seq.to(DEVICE), u_seq.to(DEVICE), r_lbl.to(DEVICE)
            
            # 使用高效加噪：只对 x0 掩码加噪，不 clone 整个序列
            noise = torch.randn_like(x_seq[:, 0, :]) * 0.005
            x0_aug = x_seq[:, 0, :] + noise

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                loss = compute_mixed_loss(model, x_seq, u_seq, r_lbl, stats_t, HORIZON, alpha=1.0, beta=beta_curr, x0_override=x0_aug)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for x_seq, u_seq, r_lbl in val_loader:
                x_seq, u_seq, r_lbl = x_seq.to(DEVICE), u_seq.to(DEVICE), r_lbl.to(DEVICE)
                v_loss = compute_mixed_loss(model, x_seq, u_seq, r_lbl, stats_t, HORIZON, alpha=1.0, beta=0.5)
                val_l += v_loss.item()
        
        avg_train = train_l / len(train_loader); avg_val = val_l / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val; patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            for save_dir in [RUN_OUT_DIR, BACKUP_DIR]:
                torch.save(best_state_dict, os.path.join(save_dir, 'residual_model.pt'))
                with open(os.path.join(save_dir, 'scaling_stats.json'), 'w') as f:
                    json.dump({k: v.tolist() for k, v in train_ds.stats.items()}, f)
                
                # 导出物理参数以供验证 (使用 NumpyEncoder 处理 ndarray)
                if hasattr(train_ds, 'physical_params') and train_ds.physical_params:
                    import numpy as np
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            if isinstance(obj, np.generic): return obj.item()
                            return super().default(obj)
                    with open(os.path.join(save_dir, 'physical_params.json'), 'w') as f:
                        json.dump(train_ds.physical_params, f, indent=4, cls=NumpyEncoder)
        else:
            patience_counter += 1

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | B: {beta_curr:.2f} | Best: {best_val:.6f}")

        if patience_counter >= PATIENCE_LIMIT:
            print(f"--> Early Stopping at epoch {epoch+1}. Best Val: {best_val:.6f}")
            break

if __name__ == "__main__":
    train()
