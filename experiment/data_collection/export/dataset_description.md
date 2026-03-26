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

## 4. 注意事项 (Python 读取)

由于 MATLAB `-v7.3` 格式使用 HDF5 存储，在 Python 中使用 `h5py` 读取时：
1.  **维度反转**：原始读取出的形状为 `(特征数, 样本数)`，需要使用 `.T` 或 `np.transpose()` 转换为 `(样本数, 特征数)`。
2.  **字符串处理**：`tag` 和 `split` 字段读取后通常是 `numpy.uint8` 的数组（ASCII 码），需要显式转换为字符串。
