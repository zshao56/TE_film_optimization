# 仿真数据库构建与温差计算规范

## 目的

本文档定义热电薄膜结构优化项目的仿真数据库构建方法，重点说明不同结构之间如何一致、可复核地计算面内温差。该温差指标是机器学习训练、结构排序和后续实验验证的核心输出，必须避免因边界条件、采样位置、网格质量或局部尖角造成的虚假提升。

## 核心原则

1. **不同结构必须在相同物理条件下比较**
   - 相同薄膜厚度 $h$ 以及面内尺寸
   - 相同材料参数 $\kappa_{\mathrm{low}}$, $\kappa_{\mathrm{high}}$
   - 相同边界条件（固定的热端与空气自然对流冷端环境温度）
   - 相同温差后处理方法

2. **主指标必须使用区域平均温度，而不是单点温度**
   - 单点温度对网格、尖角、材料界面和数值插值非常敏感。
   - 使用顶面电极窗口的二维面积平均温度定义面内温差。

3. **仅考虑 $\Delta T_{\parallel}$ 作为优化目标**
   - 热端和冷端的温度都是作为已知变量输入。
   - 在已知的温度、尺寸等条件下，直接追求最大化面内温差。

4. **所有数据库记录必须保存足够元数据**
   - 只保存最终温差不够。
   - 必须同时保存几何参数、边界条件编号、网格信息、收敛状态、温度场和热流场路径。

5. **机器学习只能使用通过质量控制的数据**
   - 未收敛、网格质量差、边界条件不匹配、测量区域无效的样本不得进入训练集。

## 物理模型

采用 3D 稳态热传导模型。计算域包含 $x, y, z$ 三个维度（以便体现不同材料三维结构变化的更大影响）：
$$
\Omega = [0,L_x]\times[0,L_y]\times[0,h]
$$

其中 $x, y$ 是面内方向，$z$ 是厚度方向。材料分布由结构参数 $\theta$ 决定：
$$
\kappa(x,y,z;\theta) \in \{\kappa_{\mathrm{low}},\kappa_{\mathrm{high}}\}
$$

控制方程为：
$$
\nabla \cdot \left(\kappa(x,y,z;\theta)\nabla T(x,y,z)\right)=0
$$

热流密度为：
$$
\mathbf{q}(x,y,z)=-\kappa(x,y,z;\theta)\nabla T(x,y,z)
$$

## 推荐边界条件

数据库中必须给每组边界条件分配唯一 `boundary_condition_id`。

### 基准边界条件 BC-001-TOP-ELECTRODE

该边界条件假设基材的上表面和侧面边界均为暴露在自然对流下的冷端，下表面为固定的热端。温度值作为变量以适用于不同的场景。

- 下表面 $z=0$：固定热端。
  $$
  T(x,y,0)=T_{\mathrm{hot}}
  $$

- 上表面 $z=h$：空气自然对流冷端。顶面通过对流与环境冷空气换热：
  $$
  -\kappa\nabla T\cdot\mathbf{n}=h_c(T-T_{\mathrm{air}})
  $$

- 侧边边界（即 $x=0, L_x$ 所在的平面以及 $y=0, L_y$ 所在的平面）：同样暴露于空气中，考虑自然对流散热：
  $$
  -\kappa\nabla T\cdot\mathbf{n}=h_{c,side}(T-T_{\mathrm{air}})
  $$

## 面内温差的主定义

### 测量区域

主指标应对应实际电极可读出的温度。测量位于薄膜的上表面（二维面积）。

定义一个可滑动的顶层电极窗口面积：
$$
\Gamma^{elec}(x_c, y_c) =
\left[x_c-\frac{w_x}{2}, x_c+\frac{w_x}{2}\right]\times\left[y_c-\frac{w_y}{2}, y_c+\frac{w_y}{2}\right]
$$

该窗口位于 $z=h$ 表面上，不考虑厚度 $t$。合法位置满足：
$$
\frac{w_x}{2}\le x_c\le L_x-\frac{w_x}{2}, \quad \frac{w_y}{2}\le y_c\le L_y-\frac{w_y}{2}
$$

其中：
- $w_x, w_y$ 是电极接触区的二维面内尺寸。
- 两个电极窗口不能重叠，且应满足最小间距约束：
  $$
  \sqrt{(x_{c,1}-x_{c,2})^2 + (y_{c,1}-y_{c,2})^2} \ge s_{\min}
  $$

电极窗口尺寸和最小间距必须对所有结构保持一致。允许搜索电极位置，但不能改变窗口尺寸来让某个结构看起来更好。

### 区域平均温度

给定窗口中心 $(x_c, y_c)$，窗口平均温度为面积积分：
$$
\overline{T}_{elec}(x_c, y_c)=
\frac{1}{|\Gamma^{elec}|}
\int_{\Gamma^{elec}}T(x,y,h)\,d\Gamma
$$

在所有合法且不重叠的窗口对中，寻找平均温度差最大的两个窗口：
$$
((x_a^\*,y_a^\*),(x_b^\*,y_b^\*))=
\arg\max_{(x_a,y_a),(x_b,y_b)}
\left|\overline{T}_{elec}(x_a,y_a)-\overline{T}_{elec}(x_b,y_b)\right|
$$

然后按温度高低重新命名窗口中心，定义高温端和低温端：
$$
\overline{T}_{hot}=
\overline{T}_{elec}(x_{hot},y_{hot})
$$

$$
\overline{T}_{cold}=
\overline{T}_{elec}(x_{cold},y_{cold})
$$

主面内温差作为优化目标定义为：
$$
\Delta T_{\parallel}=\overline{T}_{hot}-\overline{T}_{cold}
$$

因为 $T_{\mathrm{hot}}$ 和 $T_{\mathrm{air}}$ 已经作为已知条件给出，最大化 $\Delta T_{\parallel}$ 直接反映了结构在特定条件下的转化性能。

## 为什么不用单点温差

不推荐使用单点极值：
- 顶面单点温度容易受边界条件、数值插值和局部尖角影响。
- 最大/最小单点可能落在不可制造的极小区域，无法代表真实电极读数。
- 不同结构的局部尖峰数量不同，单点最大差会放大数值伪影。
- 网格细化后单点最大值和最小值可能变化较大。

## 诊断指标

为方便专家评估，每个样本除主指标外建议保存以下诊断量。

### 热流重定向指标

为了判断结构是否真正把面外热流转化为面内热流，建议记录：
$$
R_q=
\frac{\int_{\Omega}\sqrt{q_x^2+q_y^2}\,d\Omega}
{\int_{\Omega}|q_z|\,d\Omega}
$$

其中 $q_x, q_y$ 是面内热流分量，$q_z$ 是面外热流分量。

## 后处理算法

每个仿真样本应执行同一套后处理流程。

1. 读取几何、材料、边界条件和温度场。
2. 检查仿真是否收敛。
3. 检查网格质量是否满足阈值。
4. 构造固定尺寸的可滑动顶层二维电极窗口 $\Gamma^{elec}(x_c, y_c)$。
5. 在所有合法电极位置上计算窗口面积平均温度 $\overline{T}_{elec}(x_c,y_c)$。
6. 搜索满足不重叠和最小间距约束的窗口对，得到热冷端中心。
7. 计算主优化目标：
   $$
   \Delta T_{\parallel}=\overline{T}_{hot}-\overline{T}_{cold}
   $$
8. 计算热流重定向指标等诊断量。
9. 将标量结果写入 `metadata.csv`。
10. 将场数据写入 HDF5。

## 区域平均的数值实现建议

对于 FEM 结果，稳妥的方法是对有限元表面单元做积分：
$$
\overline{T}_{\Gamma_m}=
\frac{\sum_e \int_{\Gamma_m\cap e}T_e(x,y,h)\,d\Gamma}
{\sum_e |\Gamma_m\cap e|}
$$
其中 $e$ 是表面有限元单元，$\Gamma_m$ 是某个候选电极窗口。

## 数据目录结构

推荐结构：

```text
data/
  simulations/
    metadata.csv
    boundary_conditions.csv
    materials.csv
    fields/
      sim_000001.h5
      sim_000002.h5
results/
  figures/
    sim_000001_temperature.png
    sim_000001_heat_flux.png
    sim_000001_measurement_regions.png
  optimized_structures/
src/
  geometry/
  simulation/
  postprocess/
  optimization/
```

## `metadata.csv` 字段

每个样本一行。

| 字段 | 含义 | 单位或格式 |
|---|---|---|
| `simulation_id` | 仿真唯一编号 | `sim_000001` |
| `geometry_type` | 结构类型 | `wedge`, `step`, `wave`, `freeform` |
| `geometry_parameters` | 结构参数 | JSON 字符串 |
| `thickness_h` | 薄膜厚度 | m |
| `length_Lx` | 面内长度 x | m |
| `length_Ly` | 面内长度 y | m |
| `k_low` | 低热导材料热导率 | W m^-1 K^-1 |
| `k_high` | 高热导材料热导率 | W m^-1 K^-1 |
| `boundary_condition_id` | 边界条件编号 | `BC-001-TOP-ELECTRODE` |
| `T_hot` | 下表面热端温度 | K |
| `T_air` | 上表面自然对流空气冷端温度 | K |
| `measurement_wx` | 电极窗口面内尺寸 x | m |
| `measurement_wy` | 电极窗口面内尺寸 y | m |
| `electrode_min_gap` | 两个电极窗口的最小间距 $s_{\min}$ | m |
| `x_hot_electrode` | 高温电极窗口中心 x | m |
| `y_hot_electrode` | 高温电极窗口中心 y | m |
| `x_cold_electrode` | 低温电极窗口中心 x | m |
| `y_cold_electrode` | 低温电极窗口中心 y | m |
| `T_hot_electrode_avg` | 高温电极窗口平均温度 | K |
| `T_cold_electrode_avg` | 低温电极窗口平均温度 | K |
| `delta_T_parallel` | 面内最大平均温差 (主要优化目标) | K |
| `heat_flux_redirect_ratio` | 热流重定向指标 $R_q$ | 无量纲 |
| `mesh_element_count` | 单元数量 | integer |
| `qc_pass` | 是否通过质量控制 | true/false |
| `field_file` | HDF5 场数据路径 | path |

## HDF5 场数据内容

每个仿真样本保存一个 HDF5 文件。必须能从中恢复 3D 几何、温度场和热流场信息。

## 质量控制标准

样本必须满足以下条件才可进入训练集。

1. **求解器收敛**
2. **能量守恒检查**
3. **网格收敛检查**
4. **测量区域有效**
   - 所有候选 $\Gamma^{elec}$ 必须完全位于计算域表面内。
5. **几何有效**
   - 结构连通、不自交，包含 x,y,z 3D 信息。
6. **边界条件一致**

## 推荐结论

针对不同的已知环境设定（$T_{\mathrm{hot}}$ 和 $T_{\mathrm{air}}$ 作为变量），将二维表面测量的：
$$
\Delta T_{\parallel}=\overline{T}_{hot}-\overline{T}_{cold}
$$
作为唯一的优化目标。抛弃厚度维度的测温体，采用更符合实际的二维薄膜表面读数。确保三维结构的几何差异充分反映在性能结果上。
