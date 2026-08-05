# Linux CPU 数据采集指南（v3 标准数据库）

本指南面向**另一台 Linux 机器**，用 CPU 按 `DATA_GENERATION_PARAMETERS.md` 标准批量生成训练数据库（v3）。

## 0. 分支与变更摘要

```text
分支：data/v3_standard_regeneration
```

相对 v2 的主要变更（全部对齐 `DATA_GENERATION_PARAMETERS.md`）：

| 项目 | v2 | v3（本标准） |
|---|---|---|
| 厚度 | 连续 [0.4, 4] mm | **离散四档均匀 {0.5, 1, 1.5, 2} mm** |
| 材料热导率 | 随机范围 | **固定 k_low=0.089, k_high=4.492 W/m·K** |
| 面内尺寸 | 固定 1cm×1cm | **Lx∈{1,2,3,4}cm，Ly∈{1/4,1/2,1}×Lx，均匀组合** |
| 网格 | 固定 50×50×20 | **按 50 grids/cm 密度缩放（nx/ny 随尺寸）**，nz=20 |
| 曲率 | 100% 平面 | **80% 平面 + 20% 圆弧曲面（metadata 参数化）** |
| 目标指标 | 仅 delta_T_parallel | **新增 delta_T_parallel_per_area = ΔT/(Lx×Ly)，采集时直接写入 metadata；训练/逆设计以它为主目标（v3 配置 `training.target_col`）** |
| 核数 | 配置固定 24 | **自动检测 CPU 核数**（可 `--cores` 覆盖） |
| 其他 | mixed 0.85、expanded、seed 20260513、160000 | 不变 |

配置：`configs/v3_standard.json`

> 注意：曲率目前是 **metadata 参数化**（`arc_angle`/`bend_radius`/`projected_length` 等记录在 metadata 中），体素网格本身保持平面展开——与现有 solver 一致，真实网格弯曲不在本分支范围。

## 1. 环境准备（一次性）

要求 Python 3.10+（macOS 上验证过 3.13）。

```bash
# 1) 克隆分支
git clone -b data/v3_standard_regeneration https://github.com/zshao56/TE_film_optimization.git
cd TE_film_optimization

# 2) 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3) 安装 CPU 版依赖（数据生成不需要 torch，别装完整 requirements.txt）
pip install --upgrade pip
pip install -r requirements-cpu.txt
```

数据生成只依赖 numpy / scipy / pandas / h5py / tqdm。

## 2. 先小批量冒烟测试

`--samples` 现在可以覆盖配置文件里的数量（本分支新增），先用小批量确认环境正常：

```bash
python src/generate_database.py --config configs/v3_standard.json --samples 20 --cores 4
```

预期输出结尾：`Successfully added to database: 20` / `Failed: 0`。
检查 `data/simulations/metadata.csv` 已有 20 行，`data/simulations/fields/` 有 20 个 `.h5`。

## 3. 正式全量生成

### 3.1 清理旧数据（本分支配置已开启，自动执行）

`configs/v3_standard.json` 中 `clear_existing: true`、`allow_delete_data: true`，
运行时会**先删除 `data/simulations/metadata.csv` 和 `fields/*.h5`** 再从头生成。
（Linux 机器是全新 clone，本来就没有数据，不会误删。）

### 3.2 启动全量生成

直接运行（自动使用全部 CPU 核）：

```bash
python -u src/generate_database.py --config configs/v3_standard.json 2>&1 | tee logs/01_generate_v3_standard.log
```

指定核数（例如 64 核机器）：

```bash
python -u src/generate_database.py --config configs/v3_standard.json --cores 64 2>&1 | tee logs/01_generate_v3_standard.log
```

建议放后台（tmux 或 nohup）：

```bash
tmux new -s datagen
python -u src/generate_database.py --config configs/v3_standard.json --cores 64 2>&1 | tee logs/01_generate_v3_standard.log
# 断开：Ctrl-b d   重新进入：tmux attach -t datagen
```

### 3.3 断点续跑

进程中断后**直接重跑同一条命令**即可——脚本按 `metadata.csv` 中
`qc_pass=True` 的行数自动续跑，只补剩余样本，且每个样本的随机种子与首次运行一致
（`seed=20260513` 派生，样本 i 的结果可复现，不重复不遗漏）。

## 4. 单样本耗时预估（供规划）

本机（Apple Silicon，单核）实测 FDM 求解耗时：

| 尺寸 | 网格 | 单元数 | 单样本耗时 |
|---|---|---|---|
| 1cm × 0.25cm | 50×13×20 | 13k | ~1–2 s |
| 1cm × 1cm | 50×50×20 | 50k | ~3–4 s |
| 2cm × 2cm | 100×100×20 | 200k | ~12–15 s |
| 4cm × 2cm | 200×100×20 | 400k | ~25 s |
| 4cm × 4cm | 200×200×20 | 800k | ~49 s |

12 种尺寸组合均匀分布，按此估算平均约 **15–20 s/样本**（Linux 新 CPU 可能更快）：

- 160000 样本 ≈ 单核 700–900 小时
- 24 核 ≈ 30–40 小时；64 核 ≈ 12–15 小时

## 5. 生成完成后的过滤（训练用干净数据）

```bash
python src/optimization/run_configured_pipeline.py \
  --config configs/v3_standard.json \
  --stages metadata_filter
```

或直接：

```bash
python scripts/filter_physical_metadata.py \
  --metadata data/simulations/metadata.csv \
  --output data/simulations/metadata_clean.csv \
  --bad-output results/metadata/v3_standard/bad_physical_rows.csv \
  --report results/metadata/v3_standard/physical_sanity_by_scenario.csv \
  --tolerance 0.001 --max-relative-residual 1e-8 --require-qc-pass
```

训练时使用 **`data/simulations/metadata_clean.csv`**（而非未过滤的 metadata.csv）。

## 6. 数据回传（回 Mac/GPU 机器）

只回传 `data/` 和 `results/metadata/`（数据体积大，注意磁盘空间）：

```bash
# Linux 机器上打包
tar -czf v3_database.tar.gz data/simulations results/metadata/v3_standard
# 或在 Mac 上拉取（rsync 断点续传）
rsync -avzP --partial user@linux_host:/path/TE_film_optimization/data/simulations/ ./data/simulations/
```

回传后建议跑一次 `scripts/check_v2_dataset.py`（或对应 v3 检查）确认低/高温差区间都有覆盖。

## 7. 后续训练

数据回传后，在 GPU 机器上把 `configs/v3_standard.json` 的 `run` 块改为：

```json
"run": {
  "data_generation": false,
  "metadata_filter": false,
  "training": true,
  "evaluation": true,
  "real_world_benchmark": false
}
```

然后：

```bash
python -u src/optimization/run_configured_pipeline.py --config configs/v3_standard.json --stages training evaluation
```

v3 配置的 `training.target_col` 已是 `delta_T_parallel_per_area`（单位面积温差，K/m²），
训练与逆设计排名都使用该目标；`metadata.csv` 中同时保留原始 `delta_T_parallel`。
注意：loss 相关超参（`top_quantile`、`low_delta_cutoff` 等）是针对旧目标尺度调的，
换目标后可能需要重新调优（可用 `--target-col delta_T_parallel` 切回旧目标做对照实验）。

> 训练注意：v3 样本的网格尺寸随 Lx/Ly 变化（50×50×20 ~ 200×200×20），
> `training` 阶段读取 h5 时按样本实际形状处理，`metadata.csv` 的
> `geometry_parameters` 里已记录每个样本的 nx/ny/nz。
