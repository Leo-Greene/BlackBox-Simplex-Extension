import torch
import torch.nn as nn
import os
import glob
import shutil
import json

# 复用升级后的模型定义 (与 train3_robust.py 一致)
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
        h = self.blocks(h)
        return self.head(h)

def export_onnx():
    # 路径配置
    TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(TRAIN_DIR, 'out')
    
    # 1. 寻找最新文件夹 (20* 开头)
    subdirs = [os.path.join(OUT_DIR, d) for d in os.listdir(OUT_DIR) 
               if os.path.isdir(os.path.join(OUT_DIR, d)) and d.startswith('20')]
    
    if not subdirs:
        print(f"Error: No result folders starting with '20' found in {OUT_DIR}")
        return

    latest_dir = max(subdirs, key=os.path.getmtime)
    print(f"--> Found latest model folder: {latest_dir}")
    
    model_path = os.path.join(latest_dir, 'residual_model.pt')
    if not os.path.exists(model_path):
        # 尝试寻找上一级目录（有些逻辑可能直接保存在 out 下）
        model_path = os.path.join(OUT_DIR, 'residual_model.pt')
        if not os.path.exists(model_path):
            print(f"Error: Could not find residual_model.pt in {latest_dir} or {OUT_DIR}")
            return

    onnx_filename = 'residual_model.onnx'
    onnx_path_local = os.path.join(latest_dir, onnx_filename)
    onnx_path_root = os.path.join(OUT_DIR, onnx_filename)

    # 从物理参数文件中读取 n 以确定维度
    physical_params_path = os.path.join(latest_dir, 'physical_params.json')
    n = 15 # 默认后备值
    if os.path.exists(physical_params_path):
        with open(physical_params_path, 'r') as f:
            pp = json.load(f)
            n = int(pp.get('n', 15))

    # 2. 配置并加载模型
    input_dim = 6 * n  # 4*n (X) + 2*n (U)
    output_dim = 4 * n 
    model = ResidualMLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=256, depth=3)
    
    # 加载权重
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 3. 准备虚拟输入以便导出
    dummy_x = torch.randn(1, 4 * n)
    dummy_u = torch.randn(1, 2 * n)

    # 4. 导出 ONNX
    print(f"Exporting to ONNX...")
    torch.onnx.export(
        model, 
        (dummy_x, dummy_u), 
        onnx_path_local,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_state', 'input_action'],
        output_names=['residual']
    )

    # 5. 复制到根目录
    shutil.copy2(onnx_path_local, onnx_path_root)

    # 6. 打印绝对路径
    print(f"\nSUCCESS: ONNX exported.")
    print(f"Local Path: {os.path.abspath(onnx_path_local)}")
    print(f"Root Copy:  {os.path.abspath(onnx_path_root)}")

if __name__ == "__main__":
    export_onnx()
