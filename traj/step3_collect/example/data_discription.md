- 采集环境1：plant加上了残差和误差
    - 控制器
        - 策略：二次规划
        - model：错误的，未考虑残差的预测模型
    - plant
        - model：带残差和误差
    - DM
        - model：错误的，未考虑残差的预测模型

- 数据质量：只有1条碰撞用例，说明误差太小
    - 误差设置（轻微误差）：% Unmodeled params in true_dynamics
            params.acc_scale = 0.95;
            params.acc_bias = 0;
            params.damping = 0.03;
            params.nonlinear_drift = false;
            params.noise_std = 0;