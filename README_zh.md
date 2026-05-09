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