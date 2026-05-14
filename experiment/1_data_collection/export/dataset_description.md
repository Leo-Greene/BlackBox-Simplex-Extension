# 数据集变量描述 (Dataset Variable Descriptions)

该文档描述了 `export_step3_dataset.m` 导出的数据集（如 `dataset_all.mat`, `dataset_train.mat` 等）在 Python/HDF5 环境下读取后的变量形状与维度意义。

## 1. 核心数据变量

假设系统中智能体数量为 $n$（例如 15 个），总样本数为 $M$（例如 300 个）。在 Python 中使用 `h5py` 读取并转置后，变量形状如下：

| 变量名 (Variable) | 形状 (Shape) | 维度构成 (Dimension Composition) | 物理意义 (Physical Meaning) |
| :--- | :--- | :--- | :--- |
| **`X`** | `[M, 4*n]` | $[x_{1..n}, y_{1..n}, vx_{1..n}, vy_{1..n}]$ | 当前时刻 $k$ 的联合状态向量。包含所有智能体的位置和速度。 |
| **`U`** | `[M, 2*n]` | $[ax_{1..n}, ay_{1..n}]$ | 当前时刻 $k$ 下发的控制指令（加速度组合）。 |
| **`R_label`** | `[M, 4*n]` | $[\Delta x_{1..n}, \Delta y_{1..n}, \Delta vx_{1..n}, \Delta vy_{1..n}]$ | **残差标签**：实际下一状态与预测名义状态之差 ($x_{true} - x_{nom}$)。 |
| **`X_next_true`** | `[M, 4*n]` | 同 `X` | 实际观测到的下一时刻 ($k+1$) 的真实状态。 |
| **`X_next_nominal`** | `[M, 4*n]` | 同 `X` | 名义模型（纯物理模型 `dynamics.m`）预测出的下一时刻状态。 |

## 2. 详细维度排布 (Feature Layout)

*   **状态向量 (`X`, `R_label`, `X_next_...`)**:
    *   索引 `0` 到 `n-1`: 所有智能体的 $x$ 坐标。
    *   索引 `n` 到 `2n-1`: 所有智能体的 $y$ 坐标。
    *   索引 `2n` 到 `3n-1`: 所有智能体的 $x$ 方向速度 $v_x$。
    *   索引 `3n` 到 `4n-1`: 所有智能体的 $y$ 方向速度 $v_y$。
*   **动作向量 (`U`)**:
    *   索引 `0` 到 `n-1`: 所有智能体的 $x$ 方向采样加速度 $a_x$。
    *   索引 `n` 到 `2n-1`: 所有智能体的 $y$ 方向采样加速度 $a_y$。

## 3. 辅助标识变量

| 变量名 | 形状 | 物理意义 |
| :--- | :--- | :--- |
| **`case_id`** | `[M, 1]` | 样本所属的仿真 Case 编号。 |
| **`step_id`** | `[M, 1]` | 样本在该 Case 轨迹中的时间步序号。 |
| **`tag`** | `[M, 1]` | 仿真场景标签（`default`, `init_perturb`, `geometry_perturb`, `boundary_focus`）。 |
| **`split`** | `[M, 1]` | 数据集划分标签（`train`, `val`, `test`）。 |

## 4. 数据集完整形状与边界处理分析 (Crucial Logic)

### 4.1 核心维度与代码证明
数据集（如 `dataset_val.mat`）的完整形状为 `(Total_Samples, Feature_Dim)`，以 $N=15$ 个 Agent 为例：
- **行 (Rows)**：代表时间采样点（跨 Case 拼接）。
- **列 (Columns)**：代表全系统状态快照（$15 \times 4 = 60$ 维）。

**证明 A：特征拼接方式 (MATLAB)**
在 `experiment/data_collection/export/export_step3_dataset.m` [L100-L105] 中：
```matlab
for k = 1:T
    % 15个Agent的量被水平拼成一行
    X(k, :) = [pos_k(1, :), pos_k(2, :), vel_k(1, :), vel_k(2, :)];
end
```
循环变量 `k` 控制行（时间步），而方括号 `[...]` 的特征拼接决定了每一行就是一个时刻的全系统“快照”。

**证明 B：Case 之间的物理断裂 (Python)**
在 `experiment/train/train.py` [L56-L59] 中：
```python
# 通过检测相邻两行（样本点）坐标的巨大跳变，识别 Case 的边界点
pos = X_all[:, :30] 
dist_sq = np.sum((pos[1:] - pos[:-1])**2, axis=1)
boundaries = np.where(dist_sq > 25.0)[0] + 1
```
这证明了 Case 1 的末尾与 Case 2 的开头是在垂直方向（行）相连的。

### 4.2 训练与验证中的处理逻辑
由于数据集是“长条拼接”状的，为了保证多步动力学预测（Rollout）的物理真实性，代码实施了以下过滤：

1.  **训练 (Python)**：利用识别出的 `boundaries` 调用 `np.split` 将矩阵切回 7 个独立的轨迹，确保 Rollout 窗口不会跨越 Case。
2.  **验证 (MATLAB)**：在 `verify_deployment.m` 中，预测前会核实起始点 `i` 到目标点 `i+Horizon` 之间是否存在断裂索引（Boundary），若存在则跳过。这保证了评估结果 100% 来源于模型性能，排除了 0% 的数据拼接伪误差。

## 5. 注意事项 (Python 读取)

由于 MATLAB `-v7.3` 格式使用 HDF5 存储，在 Python 中使用 `h5py` 读取时：
1.  **维度反转**：原始读取出的形状为 `(特征数, 样本数)`，需要使用 `.T` 或 `np.transpose()` 转换为 `(样本数, 特征数)`。
2.  **字符串处理**：`tag` 和 `split` 字段读取后通常是 `numpy.uint8` 的数组（ASCII 码），需要显式转换为字符串。
