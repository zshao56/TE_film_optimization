# 数据采集 / 仿真数据生成参数说明

本文档集中记录本项目用于构建仿真数据库时的数据生成范围、边界条件、结构参数、测量方式和质量控制规则，方便后续复现实验、训练模型和撰写论文时引用。

> 说明：本项目里的“采集数据”主要指通过有限差分热传导仿真批量生成数据库，而不是物理相机或传感器采集。当前推荐使用的参数来源是 `configs/v3_standard.json`（分支 `data/v3_standard_regeneration`，按本文档标准实现）；本文档是对该配置和相关代码逻辑的人类可读整理。

## 1. 当前推荐配置

当前 v3 数据库配置文件（按本文档标准）：

```text
configs/v3_standard.json
```

推荐运行入口：

```bash
python src/generate_database.py --config configs/v3_standard.json
```

Linux CPU 机器的完整采集步骤见 `LINUX_CPU_DATA_GENERATION.md`。

## 2. 数据库生成规模与随机性

| 参数                  |        当前值 | 说明                                 |
| ------------------- | ---------: | ---------------------------------- |
| `samples`           |   `160000` | 目标生成样本数                            |
| `cores`             |       `24` | 并行仿真进程数                            |
| `mode`              |    `mixed` | 结构化几何 + 随机拓扑混合生成                   |
| `structured_ratio`  |     `0.85` | mixed 模式下，约 85% 为结构化几何，约 15% 为随机拓扑 |
| `seed`              | `20260513` | 根随机种子，用于可复现采样                      |
| `profile`           | `expanded` | 使用扩展采样范围，包括更宽的温度、对流和边界类型           |
| `clear_existing`    |     `true` | 流程配置中允许从头生成新版数据库                   |
| `allow_delete_data` |     `true` | 允许删除旧数据后重建，避免旧 schema 与新 schema 混合 |

## 3. 计算域尺寸与网格

| 参数   |         当前值 | 单位  | 说明                           |
| ---- | ----------: | --- | ---------------------------- |
| `Lx` | `0.01-0.04` | m   | 面内 x 方向长度，即 1 -4 cm，间隔是 1 cm |
| `Ly` |    `a * Lx` | m   | 面内 y 方向长度，a的范围是 1/4，1/2，1    |
| `nx` |        `50` | -   | x 方向网格密度是 50 grids/cm        |
| `ny` |        `50` | -   | y 方向网格密度是 50 grids/cm        |
| `nz` |        `20` | -   | 厚度方向网格数                      |

注意：配置文件中记录了 `nx/ny/nz`，但当前 `src/main.py` 中求解器实际仍固定使用 `50 × 50 × 20`。因此如果以后要改网格，除了改配置文件，还需要同步检查 `src/main.py` 中的硬编码网格设置。

## 4. 薄膜厚度范围

| 参数                |                             范围 | 单位  | 采样方式 |
| ----------------- | -----------------------------: | --- | ---- |
| `thickness_range` | `0.0005, 0.001, 0.0015, 0.002` | m   | 均匀采样 |
|                   |                                |     |      |

对应厚度为：

```text
0.5mm 1mm 1.5mm 2.0mm
```

## 5. 材料热导率参数

| 参数       | 范围 / 当前值 | 单位          | 说明   |
| -------- | -------: | ----------- | ---- |
| `k_low`  |  `0.089` | W m^-1 K^-1 | 低热导率 |
| `k_high` |  `4.492` | W m^-1 K^-1 | 高热导率 |



## 6. 环境温度与对流换热

### 6.1 空气温度

| 参数            |             范围 | 单位  | 采样方式 |
| ------------- | -------------: | --- | ---- |
| `T_air_range` | `293.15 – 353` | K   | 均匀采样 |

对应摄氏温度约为：

```text
20 – 80 °C
```

### 6.2 顶面对流换热系数 `h_c`

| 对流 regime | `h_c` 范围 | 单位 | 权重 | code |
|---|---:|---|---:|---:|
| `natural` | `2 – 15` | W m^-2 K^-1 | `0.25` | `0` |
| `weak_forced` | `15 – 50` | W m^-2 K^-1 | `0.20` | `1` |
| `forced` | `50 – 150` | W m^-2 K^-1 | `0.30` | `2` |
| `strong_forced` | `150 – 500` | W m^-2 K^-1 | `0.25` | `3` |

### 6.3 侧面对流换热系数 `h_c_side`

侧面对流由顶面对流缩放得到：

```text
side_scale ~ Uniform(0.6, 1.2)
h_c_side = clip(h_c × side_scale, 当前 regime 的 h_c 下限, 当前 regime 的 h_c 上限)
```

因此 `h_c_side` 会与 `h_c` 保持同一对流 regime 的量级。

## 7. 热边界条件

数据库使用的边界条件编号：

```text
BC-001-TOP-ELECTRODE
```

物理含义：

- 下表面 `z = 0`：固定热端温度；
- 上表面 `z = h`：与空气对流换热；
- 四个侧边界：与空气对流换热；
- 主目标是在顶面寻找两个电极窗口之间的最大平均面内温差。

### 7.1 热端边界类型

| 热边界类型              |     权重 | code | 说明                   |
| ------------------ | -----: | ---: | -------------------- |
| `uniform`          | `0.30` |  `0` | 下表面温度均匀              |
| `linear_gradient`  | `0.30` |  `1` | 下表面存在 x 或 y 方向线性温度梯度 |
| `gaussian_hotspot` | `0.40` |  `2` | 下表面存在高斯热点            |


### 7.2 热端最低温升

热端温度不是直接固定在一个绝对范围，而是先相对于环境温度采样最低温升：

```text
T_hot_min = T_air + ΔT_min
```

其中 `ΔT_min` 来自混合分布：

| 温升档位   | `ΔT_min` 范围 | 单位  |     权重 |
| ------ | ----------: | --- | -----: |
| `low`  |    `1 – 20` | K   | `0.25` |
| `mid`  |  `20 – 100` | K   | `0.45` |
| `high` | `100 – 200` | K   | `0.30` |

### 7.3 线性梯度热边界

| 参数                       |        当前值 | 单位  | 说明             |
| ------------------------ | ---------: | --- | -------------- |
| `linear_amplitude_range` | `1.1 – 35` | K   | 线性梯度幅度         |
| `linear_directions`      |   `x`, `y` | -   | 梯度方向随机选择 x 或 y |

若使用相对温升模式：

```text
T_hot_map = T_hot_min + amplitude × coord
```

其中 `coord` 为归一化后的 x 或 y 坐标。

### 7.4 高斯热点热边界

| 参数                          |            范围 | 单位    | 说明            |
| --------------------------- | ------------: | ----- | ------------- |
| `gaussian_peak_delta_range` |      `5 – 45` | K     | 热点峰值相对背景的额外温升 |
| `hotspot_x_range`           | `0.15 – 0.85` | 归一化坐标 | 热点中心 x 位置     |
| `hotspot_y_range`           | `0.15 – 0.85` | 归一化坐标 | 热点中心 y 位置     |
| `hotspot_sigma_range`       | `0.06 – 0.24` | 归一化长度 | 高斯热点宽度        |

高斯热点的温度图会被归一化到从 `T_hot_min` 到 `T_hot_min + peak_delta` 的范围。

## 8. 曲率参数

当前 v2 配置中只生成平面样本：

| 参数                        |            当前值 | 说明           |
| ------------------------- | -------------: | ------------ |
| `curvature.level_weights` | `{ "0": 0.8 }` | 100% 平面      |
| `bend_axes`               |        `["x"]` | 对当前平面样本无实际影响 |
| `curvature_type`          |         `flat` | 实际记录为平面      |
| `curvature_level`         |          `0.2` | 无弯曲          |

 使用 20% 的曲面设计。（如何设计参考之前的方法）

## 9. 几何结构采样

当前 `mode = mixed` 且 `structured_ratio = 0.85`：

```text
85%：结构化几何
15%：随机平滑拓扑
```

### 9.1 结构化几何族

结构化几何从以下 5 类中随机选择：

```text
wedge
curved_wedge
step
double_layer
arc
```

默认各几何族等概率采样。

通用方向参数：

| 参数 | 范围 / 选项 | 说明 |
|---|---|---|
| `direction` | `x`, `y`, `diagonal` | `wedge`、`curved_wedge`、`step` 可选 |
| `direction` | `x`, `y` | `double_layer`、`arc` 只使用 x 或 y |
| `reverse` | `true` / `false` | 约 50% 概率反向 |

### 9.2 `wedge`

| 参数 | 范围 | 说明 |
|---|---:|---|
| `volume_fraction_target` | `0.25 – 0.75` | 高热导材料目标体积分数 |
| `wedge_slope` | `0.35 – 1.25` | 楔形界面斜率 |

### 9.3 `curved_wedge`

| 参数 | 范围 | 说明 |
|---|---:|---|
| `base_fraction` | `0.0 – 0.3` | 起始厚度比例 |
| `max_fraction` | `0.7 – 1.0` | 最大厚度比例 |
| `exponent` | `0.5 – 4.0` | 曲线指数，控制凹凸程度 |

### 9.4 `step`

| 参数 | 范围 | 说明 |
|---|---:|---|
| `step_position` | `0.25 – 0.75` | 台阶位置 |
| `low_thickness_fraction` | `0.10 – 0.45` | 薄侧高热导材料厚度比例 |
| `high_thickness_fraction` | `0.55 – 0.95` | 厚侧高热导材料厚度比例 |

### 9.5 `double_layer`

| 参数 | 范围 | 说明 |
|---|---:|---|
| `split_fraction` | `0.35 – 0.65` | 上下层分界高度 |
| `bottom_width_fraction` | `0.40 – 0.75` | 底层横向宽度比例 |
| `top_width_fraction` | `0.40 – 0.75` | 顶层横向宽度比例 |
| `bridge_width_fraction` | `0.06 – 0.18` | 上下层连接桥宽度比例 |

### 9.6 `arc`

| 参数 | 范围 | 说明 |
|---|---:|---|
| `radius_fraction` | `0.28 – 0.42` | 圆弧半径比例 |
| `center_fraction` | 由半径约束后采样 | 圆弧中心位置，避免超出边界 |
| `base_height_fraction` | `0.08 – 0.28` | 圆弧基底高度比例 |
| `arc_height_fraction` | `0.35 – 0.70` | 圆弧高度比例 |
| `channel_half_width_fraction` | `0.04 – 0.10` | 圆弧通道半宽 |

## 10. 随机平滑拓扑采样

随机拓扑用于保留一定探索性，当前在 mixed 模式中约占 15%。但是这个随机采样的结果不能有悬空的低导热材料，因为我们采用的是 DLP 打印，结构不需要额外的支撑。

| 参数 | 范围 / 选项 | 说明 |
|---|---|---|
| `volume_fraction_target` | `0.2 – 0.8` | 高热导材料目标体积分数 |
| `random_topology_style` | `isotropic`, `pillars_z`, `lamellae_xy`, `lamellae_yz`, `lamellae_xz` | 随机结构风格 |
| `base_blur` | `1.0 – 3.0` | 基础平滑尺度 |
| `high_blur` | `8.0 – 15.0` | 强方向平滑尺度 |

不同风格对应的平滑方式：

| 风格 | `blur_sigma` |
|---|---|
| `isotropic` | `base_blur` |
| `pillars_z` | `(base_blur, base_blur, high_blur)` |
| `lamellae_xy` | `(high_blur, high_blur, base_blur × 0.5)` |
| `lamellae_yz` | `(base_blur × 0.5, high_blur, high_blur)` |
| `lamellae_xz` | `(high_blur, base_blur × 0.5, high_blur)` |

## 11. 电极测量窗口与目标值

主优化目标为顶面两个电极窗口之间的最大平均面内温差：

```text
delta_T_parallel = T_hot_electrode_avg - T_cold_electrode_avg
```

最终追求的目标是**单位面积的温差产出**（面积 = 样品总面积 Lx × Ly，单位 K/m²）：

```text
delta_T_parallel_per_area = delta_T_parallel / (Lx × Ly)
```

该指标在数据生成时直接写入 metadata（`delta_T_parallel_per_area` 列），同时保留原始 `delta_T_parallel` 供分析和验证。训练与逆设计以 `delta_T_parallel_per_area` 为主目标（见 `configs/v3_standard.json` 的 `training.target_col`）。

当前测量参数由 `src/main.py` 自动按样品尺寸设置：

| 参数                  | 对 1 cm × 1 cm 样品的值 | 单位  |
| ------------------- | -----------------: | --- |
| `measurement_wx`    |            `0.001` | m   |
| `measurement_wy`    |            `0.001` | m   |
| `electrode_min_gap` |            `0.001` | m   |

也就是当前 1 cm 样品中：

```text
电极窗口：1 mm × 1 mm
两个电极中心最小间距：1 mm
```

后处理会在顶面候选位置中搜索所有合法电极窗口对，选择平均温差最大的两个窗口。

## 12. 输出文件与 metadata 字段

每个成功样本会写入：

```text
data/simulations/metadata.csv
data/simulations/fields/<simulation_id>.h5
```

v2 metadata 中应记录的关键字段包括：

| 字段 | 含义 |
|---|---|
| `simulation_id` | 仿真样本 ID |
| `geometry_type` | 几何类型 |
| `geometry_parameters` | 几何、边界、采样参数 JSON |
| `thickness_h` | 薄膜厚度 |
| `length_Lx`, `length_Ly` | 面内尺寸 |
| `k_low`, `k_high`, `k_ratio` | 材料热导率参数 |
| `T_hot`, `T_air` | 平均热端温度和环境空气温度 |
| `h_c`, `h_c_side` | 顶面和侧面对流换热系数 |
| `database_profile` | 数据库 profile，v2 应为 `expanded` |
| `scenario_id` | 由曲率、对流 regime、热边界类型组成的场景 ID |
| `convection_regime` | 对流 regime |
| `hot_boundary_type` | 热边界类型 |
| `T_hot_min`, `T_hot_max`, `T_hot_min_delta`, `T_hot_amplitude` | 热端温度图统计量 |
| `curvature_type`, `curvature_level` | 曲率信息 |
| `measurement_wx`, `measurement_wy`, `electrode_min_gap` | 电极测量设置 |
| `delta_T_parallel` | 主目标，顶面最大平均面内温差 |
| `delta_T_parallel_per_area` | 单位面积温差目标 = `delta_T_parallel / (Lx × Ly)`，训练/逆设计主目标 |
| `qc_pass` | 后处理和物理检查是否通过 |
| `field_file` | HDF5 场文件路径 |
| `solver_relative_residual` | 求解器相对残差 |
| `solver_bounds_pass`, `surface_bounds_pass` | 温度物理边界检查 |

## 13. 质量控制与过滤

v2 配置中的 metadata 过滤参数：

| 参数                      |                                                                          当前值 | 说明                       |
| ----------------------- | ---------------------------------------------------------------------------: | ------------------------ |
| `metadata`              |                                              `data/simulations/metadata.csv` | 原始 metadata              |
| `output`                |                                        `data/simulations/metadata_clean.csv` | 过滤后的训练 metadata          |
| `bad_output`            |           `results/metadata/v2_flat_unified_thickness/bad_physical_rows.csv` | 未通过物理检查的样本               |
| `report`                | `results/metadata/v2_flat_unified_thickness/physical_sanity_by_scenario.csv` | 按场景统计的检查报告               |
| `tolerance`             |                                                                      `0.001` | 温度边界容忍度                  |
| `max_relative_residual` |                                                                       `1e-8` | 最大相对残差                   |
| `require_qc_pass`       |                                                                       `true` | 只保留 `qc_pass = true` 的样本 |

训练时应优先使用：

```text
data/simulations/metadata_clean.csv
```

而不是未过滤的：

```text
data/simulations/metadata.csv
```

