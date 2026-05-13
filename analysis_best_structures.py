#!/usr/bin/env python3
"""
分析最佳结构的合理性
检查是否存在数据偏差或物理不合理性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_geometry_distribution(metadata_path):
    """分析训练数据中各几何类型的分布"""
    df = pd.read_csv(metadata_path)

    # 统计各几何类型
    geo_stats = df.groupby('geometry_type').agg({
        'delta_t_parallel': ['count', 'mean', 'std', 'max'],
        'k_ratio': 'mean',
        'h_c': 'mean'
    }).round(2)

    print("=== 训练数据中各几何类型的统计 ===")
    print(geo_stats)

    # 检查random_smoothed是否有特殊优势
    random_smooth = df[df['geometry_type'] == 'random_smoothed']
    others = df[df['geometry_type'] != 'random_smoothed']

    print("\n=== Random Smoothed vs Others ===")
    print(f"Random Smoothed 平均ΔT: {random_smooth['delta_t_parallel'].mean():.2f} K")
    print(f"其他类型平均ΔT: {others['delta_t_parallel'].mean():.2f} K")
    print(f"Random Smoothed 最大ΔT: {random_smooth['delta_t_parallel'].max():.2f} K")
    print(f"其他类型最大ΔT: {others['delta_t_parallel'].max():.2f} K")

    return geo_stats

def check_physical_consistency(benchmark_results_path):
    """检查benchmark结果的物理一致性"""

    # 读取benchmark结果
    results = pd.read_csv(benchmark_results_path)

    print("\n=== 物理一致性检查 ===")

    # 检查1：ΔT是否与h_c正相关
    if 'h_c' in results.columns:
        corr = results[['h_c', 'fdm_delta_t']].corr().iloc[0, 1]
        print(f"对流系数h_c与ΔT的相关性: {corr:.3f}")
        if corr < 0.3:
            print("⚠️ 警告：相关性较弱，可能存在问题")

    # 检查2：ΔT是否与k_ratio正相关
    if 'k_ratio' in results.columns:
        corr = results[['k_ratio', 'fdm_delta_t']].corr().iloc[0, 1]
        print(f"导热比k_ratio与ΔT的相关性: {corr:.3f}")
        if corr < 0.5:
            print("⚠️ 警告：相关性较弱，可能存在问题")

    # 检查3：各场景的最佳结构是否物理合理
    for scenario in results['scenario'].unique():
        scenario_data = results[results['scenario'] == scenario]
        best_idx = scenario_data['fdm_delta_t'].idxmax()
        best = scenario_data.loc[best_idx]

        print(f"\n场景: {scenario}")
        print(f"  最佳几何: {best['geometry_type']}")
        print(f"  FDM ΔT: {best['fdm_delta_t']:.2f} K")
        print(f"  预测 ΔT: {best['predicted_delta_t']:.2f} K")
        print(f"  误差: {best['predicted_delta_t'] - best['fdm_delta_t']:.2f} K")

def suggest_validation_experiments():
    """建议额外的验证实验"""

    suggestions = """

=== 建议的验证实验 ===

1. **交叉验证最佳结构**
   - 对每个场景的top-3结构重新用更精细的FDM网格验证
   - 检查是否存在数值误差导致的排序问题

2. **物理极限检查**
   - 计算理论最大ΔT（基于傅里叶定律）
   - 验证FDM结果是否超过物理上限

3. **几何敏感性分析**
   - 对最佳结构进行小扰动
   - 检查性能是否平滑变化（排除数值奇点）

4. **对比简单基准**
   - 与解析解（如均匀薄膜）对比
   - 确认复杂几何确实优于简单结构

5. **独立验证**
   - 使用不同的求解器（如COMSOL）验证关键案例
   - 排除FDM实现的系统性错误
    """

    print(suggestions)

if __name__ == "__main__":
    # 设置路径
    base_path = Path("results/expanded_rebuild_v1_summary")

    # 分析几何分布
    metadata_path = "data/simulations/metadata_clean.csv"
    if Path(metadata_path).exists():
        analyze_geometry_distribution(metadata_path)

    # 检查物理一致性
    benchmark_path = base_path / "benchmark_summary.csv"
    if benchmark_path.exists():
        check_physical_consistency(benchmark_path)

    # 输出建议
    suggest_validation_experiments()
