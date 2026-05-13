# 改进实施计划

## 阶段1：快速诊断（1-2天）

### 1.1 数据分析
- [ ] 运行 `analysis_best_structures.py` 检查几何类型分布
- [ ] 统计各ΔT区间的样本数量
  - 0-10 K: ?
  - 10-50 K: ?
  - 50-100 K: ?
  - >100 K: ?
- [ ] 检查engine场景在训练集中的覆盖度

### 1.2 模型诊断
- [ ] 绘制预测误差 vs 真实ΔT的散点图（按场景着色）
- [ ] 分析残差的系统性模式
- [ ] 检查验证集上的分场景性能

## 阶段2：针对性改进（3-5天）

### 2.1 数据增强（优先级：高）
```bash
# 生成额外的高ΔT样本
python src/data_generation/generate_targeted_samples.py \
  --target_range high \
  --min_delta_t 100 \
  --num_samples 20000 \
  --focus_scenarios engine,battery

# 生成低ΔT高精度样本
python src/data_generation/generate_targeted_samples.py \
  --target_range low \
  --max_delta_t 15 \
  --num_samples 10000 \
  --focus_scenarios skin,glass
```

### 2.2 损失函数改进（优先级：高）
- [ ] 实现分段加权损失
- [ ] 添加相对误差项：`loss = MSE + α * MAPE`
- [ ] 增加反低估惩罚：对高ΔT区域的低估加倍惩罚

### 2.3 后处理校准（优先级：中）
- [ ] 基于验证集拟合校准曲线
- [ ] 按场景类型分别校准
- [ ] 实现ensemble预测（多个checkpoint平均）

## 阶段3：重新训练（2-3天）

### 3.1 训练配置
```json
{
  "data": {
    "train_samples": 100000,  // 原有80k + 新增20k
    "augmentation": {
      "high_delta_t_oversample": 2.0,
      "low_delta_t_oversample": 1.5
    }
  },
  "loss": {
    "type": "adaptive_weighted_mse",
    "high_region_weight": 5.0,  // 从3.0提高到5.0
    "low_region_weight": 2.0,   // 新增
    "underprediction_penalty": 2.0  // 新增
  },
  "training": {
    "epochs": 100,  // 从80增加到100
    "early_stopping_patience": 15,  // 从12增加到15
    "lr_schedule": "cosine_with_warmup"
  }
}
```

### 3.2 验证策略
- [ ] 每10个epoch保存checkpoint
- [ ] 在验证集上评估多个指标：
  - 整体MAE/RMSE
  - 高ΔT区域MAE/bias
  - 低ΔT区域相对误差
  - Spearman相关系数
- [ ] 选择综合性能最佳的checkpoint（不只看validation loss）

## 阶段4：全面评估（1-2天）

### 4.1 Benchmark重跑
- [ ] 使用新模型重新运行5个场景
- [ ] 增加验证候选数：50 → 100
- [ ] 记录每个场景的top-10结构

### 4.2 对比分析
```python
# 对比v1和v2的改进
metrics_comparison = {
    "overall": ["MAE", "RMSE", "R2"],
    "high_region": ["bias", "precision", "recall"],
    "low_region": ["relative_MAE"],
    "benchmark": ["engine_bias", "skin_relative_error"]
}
```

### 4.3 物理验证
- [ ] 对可疑的最佳结构用更精细网格重新FDM验证
- [ ] 检查是否存在数值不稳定性
- [ ] 与文献中的类似结构对比

## 阶段5：文档和报告（1天）

- [ ] 更新结果摘要
- [ ] 生成对比图表
- [ ] 撰写改进说明
- [ ] 准备论文补充材料

## 预期改进目标

| 指标 | 当前 (v1) | 目标 (v2) | 改进幅度 |
|------|-----------|-----------|----------|
| 整体测试MAE | 2.87 K | < 2.5 K | -13% |
| 高区域bias | -3.74 K | > -2.0 K | +47% |
| Engine场景bias | -26.57 K | > -15 K | +44% |
| Skin相对误差 | 73.29% | < 40% | -45% |
| Top-10% recall | 92.2% | > 95% | +3% |

## 风险和备选方案

### 风险1：数据增强后过拟合
- **检测**：训练集和验证集性能差距扩大
- **应对**：增加dropout、数据增强、early stopping

### 风险2：改进后整体性能下降
- **检测**：新模型在某些指标上退步
- **应对**：使用ensemble（v1 + v2加权平均）

### 风险3：最佳结构仍然不合理
- **检测**：物理验证失败或与理论矛盾
- **应对**：
  1. 检查FDM求解器实现
  2. 与商业软件对比验证
  3. 咨询领域专家

## 资源需求

- **计算资源**：
  - 数据生成：~50 GPU小时
  - 模型训练：~100 GPU小时
  - Benchmark验证：~200 CPU小时

- **时间估计**：
  - 总计：10-15天
  - 关键路径：数据生成 → 训练 → 评估

## 成功标准

**必须满足**：
1. Engine场景bias < -15 K
2. Skin场景相对误差 < 50%
3. 整体测试R² > 0.96

**期望满足**：
1. 所有场景的最佳结构通过物理验证
2. Top-10% recall > 95%
3. 无明显的系统性偏差模式
