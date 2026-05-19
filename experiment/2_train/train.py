"""
Step 3: Deterministic Residual Learning - Robust Multi-Step Training
==================================================================
改进点：
1. Trajectory-Aware Dataset: 确保 Rollout 窗口不跨越不同轨迹的边界。通过位置跳变自动识别轨迹切换。
2. Mixed Loss: 结合 One-step Residual Loss (作为锚点防止变坏) 和 Multi-step Rollout Loss (作为长时优化目标)。
3. Input Augmentation Fix: 只对初始输入 x0 加噪以增强鲁棒性，绝对不污染未来的真值标签 (x1...xH)。
4. Curriculum Learning: beta 随训练轮次逐步增加，前期先练好单步偏差，后期再攻克多步累积。
5. Multi-Agent Decoupling (NEW): 将 15 辆车的全耦合特征拆解为单车 4 维状态输入输出，彻底解决多步滚动发散与过拟合。
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
# Global Configuration
# ============================================================
GLOBAL_HORIZON = 10

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

# ============================================================
# Dataset (轨迹感知 + 预归一化优化)
# ============================================================

class TrajectoryDataset(Dataset):
    """
    按轨迹加载数据，并在初始化时完成所有数据的预归一化，显著提升训练速度。
    """
    def __init__(self, mat_file, horizon=GLOBAL_HORIZON, stats=None):
        self.mat_file = mat_file
        self.horizon = horizon
        self.physical_params = None

        # 尝试从数据集所在目录提取物理参数
        try:
            data_dir = os.path.dirname(mat_file)
            json_path = os.path.join(data_dir, 'physical_params.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.physical_params = json.load(f)
            else:
                import scipy.io as sio
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
            X_true_all = None
            if 'X_true' in root:
                X_true_all = np.array(root['X_true']).T.astype(np.float32)
            
            # 使用 case_id 识别轨迹边界 (更健壮)
            if 'case_id' in root:
                case_id = np.array(root['case_id']).flatten()
                boundaries = np.where(case_id[1:] != case_id[:-1])[0] + 1
            else:
                # 备选方案：通过位置跳变识别轨迹边界
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
                if X_true_all is not None:
                    self.stats['X_true_mean'] = np.mean(X_true_all, axis=0)
                    self.stats['X_true_std'] = np.std(X_true_all, axis=0) + 1e-6
            else:
                self.stats = stats
            
            # 关键：预归一化全部数据 (CPU 运行一次即可)
            self.X_norm = (X_all - self.stats['X_mean']) / self.stats['X_std']
            self.U_norm = (U_all - self.stats['U_mean']) / self.stats['U_std']
            self.R_norm = (R_all - self.stats['R_mean']) / self.stats['R_std']
            if X_true_all is not None:
                x_true_mean = self.stats.get('X_true_mean', self.stats['X_mean'])
                x_true_std = self.stats.get('X_true_std', self.stats['X_std'])
                self.X_true_norm = (X_true_all - x_true_mean) / x_true_std
            else:
                self.X_true_norm = self.X_norm

            self.trajs_X = np.split(self.X_norm, boundaries)
            self.trajs_X_true = np.split(self.X_true_norm, boundaries)
            self.trajs_U = np.split(self.U_norm, boundaries)
            self.trajs_R = np.split(self.R_norm, boundaries)
            
        self.valid_indices = []
        for t_idx, t_data in enumerate(self.trajs_X):
            T = t_data.shape[0]
            if T > self.horizon:
                for s in range(T - self.horizon):
                    self.valid_indices.append((t_idx, s))
        print(f"--> Loaded {len(self.valid_indices)} valid training samples from {mat_file}")

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t_idx, s_idx = self.valid_indices[idx]
        # 直接切片已归一化的数据，无需重复计算
        x_seq = self.trajs_X[t_idx][s_idx : s_idx + self.horizon + 1]
        x_true_seq = self.trajs_X_true[t_idx][s_idx : s_idx + self.horizon + 1]
        u_seq = self.trajs_U[t_idx][s_idx : s_idx + self.horizon]
        r_seq = self.trajs_R[t_idx][s_idx : s_idx + self.horizon]

        return (
            torch.from_numpy(x_seq), 
            torch.from_numpy(x_true_seq),
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
    def __init__(self, input_dim=4, output_dim=4, hidden_dim=128, depth=3):
        """默认改为单车 4 维输入 [vx, vy, ax, ay] 和 4 维输出 [dx, dy, dvx, dvy]"""
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, u):
        h = self.input_proj(torch.cat([x, u], dim=-1))
        return self.head(self.blocks(h))

# ============================================================
# Ops (物理动力学演化基础算子)
# ============================================================

def denormalize(val, m_t, s_t):
    return val * s_t + m_t

def normalize(val, m_t, s_t):
    return (val - m_t) / s_t

def nominal_dynamics_torch(x, u, dt, vmax, pFactor, predator, n, alpha_v, alpha_x):
    p, v, a = x[:, :2*n], x[:, 2*n:], u
    new_v = alpha_v * (v + a * dt)
    
    # 物理限速 (vmax 截断)
    vx = new_v[:, :n]
    vy = new_v[:, n:]
    v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
    
    # 计算每个 agent 的最大速度限制
    vmax_tensor = torch.full_like(v_mag, vmax)
    if predator > 0:
        vmax_tensor[:, -1] = vmax * pFactor
        
    scale = torch.clamp(vmax_tensor / v_mag, max=1.0)
    new_v_clipped = torch.cat([vx * scale, vy * scale], dim=-1)
    
    new_p = p + alpha_x * (new_v_clipped * dt) # Semi-implicit Euler with alpha mismatch
    return torch.cat([new_p, new_v_clipped], dim=-1)

# ============================================================
# Core Tensor Reshaping (乐高拆装关键函数)
# ============================================================

def forward_single_agent(model, x_nn_in, u_t_norm, n):
    """
    将大系统的状态切片与控制量重新塑形为独立的单车样本，实现高度泛化训练。
    输入：
        x_nn_in: [B, 2n]  (包含当前归一化后的 vx_1..vx_n, vy_1..vy_n)
        u_t_norm: [B, 2n] (包含当前归一化后的 ax_1..ax_n, ay_1..ay_n)
    输出：
        r_p_norm: [B, 4n] (还原拼回大系统排布的归一化残差)
    """
    B = x_nn_in.shape[0]
    
    # 1. 提取单车速度 [vx, vy] -> 构造 [B * n, 2] 的一维紧凑特征
    vx = x_nn_in[:, :n]   # [B, n]
    vy = x_nn_in[:, n:]   # [B, n]
    v_flat = torch.stack([vx, vy], dim=-1).reshape(-1, 2) # [B * n, 2]
    
    # 2. 提取单车动作 [ax, ay] -> 构造 [B * n, 2] 的一维紧凑特征
    ax = u_t_norm[:, :n]  # [B, n]
    ay = u_t_norm[:, n:]  # [B, n]
    u_flat = torch.stack([ax, ay], dim=-1).reshape(-1, 2) # [B * n, 2]
    
    # 3. 极简 4->4 网络前向传播预测
    r_flat = model(v_flat, u_flat) # 输出形状: [B * n, 4]
    
    # 4. 拼回大张量排布 [dx_all, dy_all, dvx_all, dvy_all]
    r_res = r_flat.reshape(B, n, 4)
    dx  = r_res[:, :, 0] # [B, n]
    dy  = r_res[:, :, 1] # [B, n]
    dvx = r_res[:, :, 2] # [B, n]
    dvy = r_res[:, :, 3] # [B, n]
    
    return torch.cat([dx, dy, dvx, dvy], dim=-1) # [B, 4n]

# ============================================================
# Loss Computation (混合损失闭环)
# ============================================================

def compute_mixed_loss(model, x_seq, x_true_seq, u_seq, r_lbl_seq, stats_t, horizon, alpha, beta, x0_override, vmax, pFactor, predator, dt, n, alpha_v, alpha_x):
    x_curr_norm = x0_override if x0_override is not None else x_seq[:, 0, :]
    
    # One-step Residual Anchor (单步锚点损失，引入解耦变换)
    x_nn_in = x_curr_norm[:, 2*n:]
    r_pred_1 = forward_single_agent(model, x_nn_in, u_seq[:, 0, :], n)
    anchor_loss = torch.mean((r_pred_1 - r_lbl_seq[:, 0, :])**2)
    
    # 获取统计量
    xm, xs = stats_t['X_mean'], stats_t['X_std']
    x_true_m = stats_t.get('X_true_mean', xm)
    x_true_s = stats_t.get('X_true_std', xs)
    rm, rs = stats_t['R_mean'], stats_t['R_std']
    
    # Multi-step Rollout (多步自回归滚动)
    rollout_loss = 0
    for t in range(horizon):
        u_t_norm = u_seq[:, t, :]
        x_nn_in = x_curr_norm[:, 2*n:]
        
        # 通过解耦函数获取还原后的系统级归一化预测偏差项
        r_p_norm = forward_single_agent(model, x_nn_in, u_t_norm, n)
        
        # 物理空间无噪推进
        x_c = denormalize(x_curr_norm, xm, xs)
        u_c = denormalize(u_t_norm, stats_t['U_mean'], stats_t['U_std'])
        r_p = denormalize(r_p_norm, rm, rs)
        
        x_next_pred = nominal_dynamics_torch(
            x_c, u_c, dt=dt, vmax=vmax, pFactor=pFactor, predator=predator, n=n, alpha_v=alpha_v, alpha_x=alpha_x
        ) + r_p
        
        # FIX 2-A (2026-05-19): Mirror the second vmax clamp applied in dynamics_learned.m (L87-101).
        # Without this, training rollout and MATLAB inference have different output spaces:
        #   - MATLAB: clamps velocity after (x_nom + residual)
        #   - Python: previously did NOT clamp after (nominal + r_p)
        # This asymmetry biased training in high-speed regimes where residual dvx > 0 near saturation.
        vx_p = x_next_pred[:, 2*n : 3*n]
        vy_p = x_next_pred[:, 3*n : 4*n]
        v_mag_p = torch.sqrt(vx_p**2 + vy_p**2 + 1e-8)
        vmax_p = torch.full_like(v_mag_p, vmax)
        if predator > 0:
            vmax_p[:, -1] = vmax * pFactor
        scale_p = torch.clamp(vmax_p / v_mag_p, max=1.0)
        x_next_pred = torch.cat([
            x_next_pred[:, :2*n],
            vx_p * scale_p,
            vy_p * scale_p
        ], dim=-1)

        # 全归一化对齐监督，杜绝随机噪声污染
        x_next_pred_norm = normalize(x_next_pred, x_true_m, x_true_s)
        step_err = torch.mean((x_next_pred_norm - x_true_seq[:, t+1, :])**2)
        rollout_loss += step_err * (1.0 + t/horizon)
        
        # 重新归一化进入下一步自回归演化
        x_curr_norm = normalize(x_next_pred, xm, xs)

    return alpha * anchor_loss + beta * (rollout_loss / horizon)

# ============================================================
# Train Execution Main Loop
# ============================================================

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
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
    DATA_DIR = os.path.dirname(max(manifests, key=os.path.getmtime))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RUN_OUT_DIR = os.path.join(TRAIN_ROOT, 'out')
    BACKUP_DIR = os.path.join(RUN_OUT_DIR, timestamp)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HORIZON, BATCH_SIZE, EPOCHS, LR = GLOBAL_HORIZON, args.batch_size, args.epochs, args.lr

    cfg = load_config()
    # 加载物理配置与元数据
    train_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_train.mat'), horizon=HORIZON)
    val_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_val.mat'), horizon=HORIZON, stats=train_ds.stats)
    
    num_workers = 4 if os.name != 'nt' else 0 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    print(f"--> Starting Optimized Robust Training on {DEVICE}")
    vmax_val, pFactor_val, predator_val, dt_val, n_val, alpha_v_val, alpha_x_val = 2.0, 1.40, 0, 0.3, 15, 1.0, 1.0
    if hasattr(train_ds, 'physical_params') and train_ds.physical_params:
        vmax_val = float(train_ds.physical_params.get('vmax', 2.0))
        pFactor_val = float(train_ds.physical_params.get('pFactor', 1.40))
        predator_val = int(train_ds.physical_params.get('predator', 0))
        dt_val = float(train_ds.physical_params.get('dt', 0.3))
        n_val = int(train_ds.physical_params.get('n', 15))
        alpha_v_val = float(train_ds.physical_params.get('alpha_v', 1.0))
        alpha_x_val = float(train_ds.physical_params.get('alpha_x', 1.0))

    stats_t = {k: torch.tensor(v, device=DEVICE) for k, v in train_ds.stats.items()}

    # 实例化单车超轻量化网络模型 (输入维度 4, 输出维度 4)
    model = ResidualMLP(
        input_dim=cfg['model_input_dim'],
        output_dim=cfg['model_output_dim'],
        hidden_dim=cfg['hidden_dim'],
        depth=cfg['depth']
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    best_val_loss = float('inf')
    PATIENCE_LIMIT = 25
    patience_counter = 0

    def beta_for_epoch(epoch_idx):
        if epoch_idx <= 20: return 0.0
        if epoch_idx <= 70: return (epoch_idx - 20) / 50.0
        return 1.0

    for epoch in range(EPOCHS):
        model.train()
        train_l = 0
        beta_curr = beta_for_epoch(epoch)

        for x_seq, x_true_seq, u_seq, r_lbl in train_loader:
            x_seq = x_seq.to(DEVICE)
            x_true_seq = x_true_seq.to(DEVICE)
            u_seq = u_seq.to(DEVICE)
            r_lbl = r_lbl.to(DEVICE)
            
            noise = torch.randn_like(x_seq[:, 0, :]) * 0.005
            x0_aug = x_seq[:, 0, :] + noise

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
                loss = compute_mixed_loss(
                    model, x_seq, x_true_seq, u_seq, r_lbl, stats_t, HORIZON, 
                    alpha=1.0, beta=beta_curr, x0_override=x0_aug, 
                    vmax=vmax_val, pFactor=pFactor_val, predator=predator_val, dt=dt_val, n=n_val,
                    alpha_v=alpha_v_val, alpha_x=alpha_x_val
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for x_seq, x_true_seq, u_seq, r_lbl in val_loader:
                x_seq = x_seq.to(DEVICE)
                x_true_seq = x_true_seq.to(DEVICE)
                u_seq = u_seq.to(DEVICE)
                r_lbl = r_lbl.to(DEVICE)
                v_loss = compute_mixed_loss(
                    model, x_seq, x_true_seq, u_seq, r_lbl, stats_t, HORIZON, 
                    alpha=1.0, beta=beta_curr, x0_override=None,
                    vmax=vmax_val, pFactor=pFactor_val, predator=predator_val, dt=dt_val, n=n_val,
                    alpha_v=alpha_v_val, alpha_x=alpha_x_val
                )
                val_l += v_loss.item()
        
        avg_train = train_l / len(train_loader); avg_val = val_l / len(val_loader)
        scheduler.step(avg_val)
        
        is_curriculum_stable = (beta_curr >= 1.0)

        if (not is_curriculum_stable) or (avg_val < best_val_loss):
            best_val_loss = avg_val
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            
            for save_dir in [RUN_OUT_DIR, BACKUP_DIR]:
                torch.save(best_state_dict, os.path.join(save_dir, 'residual_model.pt'))
                with open(os.path.join(save_dir, 'scaling_stats.json'), 'w') as f:
                    json.dump({k: v.tolist() for k, v in train_ds.stats.items()}, f)
                
                metrics = {"residual_variance": float(best_val_loss)}
                with open(os.path.join(save_dir, 'validation_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=4)

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

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | B: {beta_curr:.2f} | Best: {best_val_loss:.6f} | Patience: {patience_counter}")

        if patience_counter >= PATIENCE_LIMIT:
            print(f"--> Early Stopping at epoch {epoch+1}. Best Val: {best_val_loss:.6f}")
            break

if __name__ == "__main__":
    train()