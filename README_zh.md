# 热电薄膜结构优化 (TE Film Optimization)

[English](README.md) | **中文**

本仓库专注于 3D 热电（TE）薄膜结构的计算设计与优化。核心目标是通过空间几何设计，利用两种具有不同热导率的复合材料，将底部的面外温度梯度转化为顶表面上面内温差的最大化。

## 🌟 主要特性
- **3D 稳态热传导**: 采用自定义开发的 3D 有限差分法（FDM）求解器，基于 `scipy.sparse` 高效计算大型体素网格，摆脱了对笨重商业软件的依赖。
- **顶面电极建模**: 摒弃了理想化的单点测量，采用了物理上更符合实际情况的 2D 面积平均温度作为局部电极的测量标准。
- **结构化几何库**: 支持低维度的几何族，包括直楔形 (wedge)、阶梯形 (step)、双层桥接 (double-layer)、曲线楔形 (curved wedge) 以及拱形 (arc)。这使得数据库能够由具有明确物理意义和可解释性的结构主导，而不仅仅是随机平滑噪声。
- **自动化工作流管道**: 支持自动化几何生成、3D 仿真、指标后处理以及统一的数据存储（HDF5 保存物理场数据 + CSV 保存元数据）。
- **物理边界条件**: 严格定义的物理边界条件（底部：固定高温热端；顶部和四周侧边：向环境空气的自然对流散热），确保了在千变万化的拓扑结构之间能够进行完全公平的性能比较。

## 📂 项目结构
```text
TE_film_optimization/
├── src/
│   ├── geometry/        # 3D 结构参数化与体素网格生成
│   ├── simulation/      # FDM 有限差分求解器引擎
│   ├── postprocess/     # 性能指标提取与电极搜寻
│   ├── optimization/    # 贝叶斯优化/主动学习闭环（计划中）
│   ├── data_io/         # HDF5 和 CSV 数据接口
│   └── main.py          # 自动化流水线执行脚本
├── data/
│   └── simulations/     # 本地数据存储库（包含 metadata.csv 与 HDF5 物理场文件）
├── results/             # 生成的图表和分析结果
├── plan.md              # 长期项目规划和里程碑
└── database_temperature_difference_protocol.md # 仿真设置和物理验证的核心协议
```

## 🔄 项目全链路流程图：从物理仿真到逆向设计

本项目的终极目标是**逆向设计 (Inverse Design)**：在给定薄膜厚度和环境温度等已知约束的前提下，理论上能够实现最大面内温差的最优 3D 形状是什么？我们通过极其严谨的三阶段流水线来实现这一目标：

```mermaid
flowchart TD
    %% 样式定义
    classDef phase fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000,font-weight:bold;
    classDef data fill:#e1f5fe,stroke:#1565c0,color:#000;
    classDef model fill:#e8f5e9,stroke:#2e7d32,color:#000;
    classDef action fill:#fff3e0,stroke:#f57c00,color:#000;
    classDef highlight fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000,font-weight:bold;

    subgraph Phase1 [第一阶段：海量高保真数据获取 (Data Generation)]
        direction TB
        A1[随机采样物理环境\n厚度, 热导率, 热/冷端温度]:::action --> B1
        A2[参数化几何生成\n曲线楔形, 阶梯, 双层桥接等]:::action --> B1
        B1{3D FDM 有限差分求解器\n(计算成本: 极高)}:::model --> C1
        C1[(50,000 个高保真数据\nmetadata.csv + 3D .h5)]:::data
    end

    subgraph Phase2 [第二阶段：正向代理模型训练 (Forward Training)]
        direction TB
        C1 --> D1(数据集切分 80/10/10)
        D1 --> E1[3D CNN 分支\n提取拓扑特征]:::model
        D1 --> E2[MLP 分支\n提取物理特征]:::model
        E1 --> F1((特征融合层 Fusion))
        E2 --> F1
        F1 --> G1[预测面内温差 ΔT]:::data
        G1 -->|计算 MSE Loss 反向传播| E1
        G1 -->|保存最优权重| H1[毫秒级 AI 裁判模型\nThermoNet]:::highlight
    end

    subgraph Phase3 [第三阶段：终极目标 - 逆向设计 (Inverse Design)]
        direction TB
        I1[/用户输入已知条件:\n锁定厚度 h & 环境温度/]:::data --> J1
        J1[AI 优化算法引擎\n遗传算法 / 梯度上升]:::action -->|1. 不断生成候选 3D 形状| K1
        H1 -->|部署| K1{AI 裁判打分\n(耗时: 1毫秒/次)}:::highlight
        K1 -->|2. 返回预测的温差 ΔT| J1
        J1 -->|3. 优胜劣汰, 迭代进化千百代| L1[/输出: 理论极致温差\n及其对应的最优 3D 几何形状/]:::data
    end

    %% 跨阶段连接
    Phase1 ===> Phase2
    Phase2 ===> Phase3
    
    %% 验证回环
    L1 -.->|最终高保真物理复核验证| B1
```

### 流程阶段解析
1. **第一阶段 (数据获取)**：我们用物理规则约束了几何生成器，搭配定制的 3D FDM 求解器，挂机运算积累下含有 50,000 个样本的标准数据库。这个阶段建立的是极其可靠且纯净的物理基石。
2. **第二阶段 (正向代理模型训练)**：使用双模态神经网络 `ThermoNet`。3D CNN 分支负责看懂复杂的立体骨架结构，MLP 分支负责听懂环境温度和导热条件。训练收敛后，我们将获得一个“克隆版的高速求解器”，它的预测速度比传统 FEM 仿真快约 10,000 倍。
3. **第三阶段 (逆向设计)**：这是我们的终极目标。我们将第二阶段训练好的极速代理模型作为适应度评估函数（Fitness Function）。由此，诸如遗传算法等 AI 优化引擎就能在短短几秒钟内“凭空捏造”并验证几百万种复杂的拓扑变体，像达尔文进化论一样精准筛选出能够压榨出最高面内温差的终极 3D 结构！

## 🛠 安装指南

环境要求：**Python 3.8+**。

克隆本仓库:
```bash
git clone https://github.com/zshao56/TE_film_optimization.git
cd TE_film_optimization
```

安装依赖:
```bash
pip install -r requirements.txt
```

## 🚀 快速开始

在小批量结构化几何示例上运行自动仿真流水线：
```bash
python src/main.py
```

执行完毕后：
- 核心目标指标 (`delta_T_parallel`) 和其他运行细节将被追加写入 `data/simulations/metadata.csv` 中。
- 完整的 3D 温度场和热导率场数据将被保存在 `data/simulations/fields/<sim_id>.h5` 中。

如果要进行大规模的数据集生成（优先使用结构化几何族，并辅以少量随机拓扑探索）：
```bash
python src/generate_database.py --samples 50000 --cores 8 --mode mixed --structured-ratio 0.8 --seed 42
```

你可以使用 `--mode structured` 彻底排除随机平滑拓扑，或者使用 `--mode random` 回退到纯随机生成模式。

训练代理模型：
```bash
python src/optimization/train.py --batch-size 32 --epochs 50 --seed 42
```

如果第一轮模型在高 `delta_T_parallel` 区域明显低估，可启动第二轮训练配置：
```bash
python src/optimization/train.py --batch-size 32 --epochs 80 --seed 42 --normalize-target --top-quantile 0.9 --top-weight 3.0
```

训练完成后，先在独立测试集上评估代理模型，而不是直接继续加 epoch：
```bash
python src/optimization/evaluate.py --split test --seed 42
```

评估结果会输出到 `results/evaluation/`，包括整体 MAE/RMSE/R²、按结构类型拆分的误差表、预测值与 FDM 真值散点图、高 `delta_T_parallel` 区域的单独误差指标，以及 top 区域排序命中率。只有当测试集，尤其是高温差区域，误差足够小且散点图接近对角线时，才进入代理模型辅助逆向设计。

如果要让训练机自动扫描多组欠预测惩罚参数、自动评估并生成排行榜：
```bash
python src/optimization/run_experiments.py --penalties 0.05 0.1 0.15 0.2 0.25 --batch-size 128
```

如果要让本地 advisor 根据每轮评估结果自动决定下一轮参数：
```bash
python src/optimization/run_experiments.py --adaptive --max-adaptive-runs 4 --batch-size 128
```

如果已经手动跑完一轮训练，并且当前 checkpoint 位于 `results/models/best_thermonet.pth`，可先导入这轮结果到自动排行榜：
```bash
python src/optimization/run_experiments.py --import-current-run thermonet_v6_underpredict_0p1_bs128 --no-sweep
```

自动实验输出位于 `results/experiments/`，其中 `leaderboard.csv` 会按综合评分排序，`advisor_decisions.json` 会记录每一步的决策理由；每个 run 都会保留独立的 checkpoint、评估指标、预测 CSV 和图像，避免后续训练覆盖 `best_thermonet.pth`。

进入第一版代理模型辅助逆向设计时，先用当前最佳 surrogate 批量筛候选，再用真实 FDM 复算验证：
```bash
python src/optimization/inverse_design.py screen --model-path results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth --num-candidates 100000 --top-k 500 --batch-size 256 --mode mixed --structured-ratio 0.9 --seed 20260511
```

如果目标是固定工况和固定厚度下寻找最佳材料/几何组合，则显式固定热边界条件和厚度，但不要固定 `k_low` / `k_high`：
```bash
python src/optimization/inverse_design.py screen --model-path results/experiments/thermonet_auto_adaptive_under_0p2_bs128/best_thermonet.pth --num-candidates 100000 --top-k 500 --batch-size 256 --mode mixed --structured-ratio 0.9 --seed 20260511 --fixed-h 0.001 --fixed-T-hot 350.0 --fixed-T-air 298.15 --fixed-h-c 10.0 --fixed-h-c-side 10.0
```

筛选结果会保存在 `results/inverse_design/screen_<timestamp>/`。然后选择该目录，复算前 50 个候选：
```bash
python src/optimization/inverse_design.py verify --screen-dir results/inverse_design/screen_<timestamp> --verify-count 50
```

`verify` 会自动跳过已经写入 `verified_candidates.csv` 的候选，所以可以用下面的命令把同一个 CSV 继续扩展到 surrogate top 200，不会重复计算前 50 个：
```bash
python src/optimization/inverse_design.py verify --screen-dir results/inverse_design/screen_<timestamp> --verify-count 200
```

如果要把真实 FDM 排名前 10 的结构图输出到 `verified_candidates.csv` 所在文件夹：
```bash
python src/optimization/inverse_design.py plot-top --screen-dir results/inverse_design/screen_<timestamp> --top-n 10
```

`screen` 阶段只做神经网络预测；`verify` 阶段才会运行 FDM，并把真实仿真结果写入数据库和 `verified_candidates.csv`。最终排序应看 `verified_candidates.csv` 里的真实 `fdm_delta_T`，不要看 surrogate rank。

## 📐 网格无关性与网格选择 (Grid Independence)

针对海量数据库生成（例如 50,000 个样本），选择极具性价比的网格分辨率至关重要。我们在对网格精度高度敏感的 `curved_wedge`（曲线楔形）结构上进行了**网格无关性测试**，以评估物理精度与计算时间成本：

| 网格 (nx, ny, nz) | 自由度 (总节点数) | 面内最大温差 $\Delta T_{\parallel}$ (K) | 单次求解耗时 (秒) |
| :--- | :--- | :--- | :--- |
| (40, 40, 15) | 24,000 | 3.3118 | ~3.2 |
| **(50, 50, 20)** | **50,000** | **3.5888** | **~9.3** |
| (60, 60, 30) | 108,000 | 3.5294 | ~63.2 |

**结论：** 最终我们将 `nx=50, ny=50, nz=20` 选定为系统默认网格配置。它能够极好地消除曲线界面的马赛克截断误差（捕捉到更准确的物理温度梯度），同时将有限差分矩阵的规模控制在合理区间内，使得生成 50,000 个样本的数据集只需一台 8 核电脑运行约 15-18 小时即可完成。如果进一步加密（如 60x60x30），计算时间将呈指数级爆炸，且对物理精度的提升微乎其微。

## 📜 物理规则与验证
所有的三维仿真都必须严格遵守 `database_temperature_difference_protocol.md` 中定义的规则。这确保了在物理实验条件上的一致性，使得为下游机器学习任务提供的 $\Delta T_{\parallel}$ 目标具有绝对的公平性和可比性，严防优化算法通过“钻模型空子”或生成无法加工的局部伪影来刷高数据。
