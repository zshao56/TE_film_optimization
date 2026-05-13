#!/usr/bin/env python3
"""
检查metadata文件的实际结构
"""

import pandas as pd
from pathlib import Path

def inspect_metadata(metadata_path):
    """检查metadata文件的列名和基本信息"""

    if not Path(metadata_path).exists():
        print(f"❌ 文件不存在: {metadata_path}")
        return None

    df = pd.read_csv(metadata_path)

    print("=== Metadata 文件信息 ===")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n列名列表:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\n前5行数据:")
    print(df.head())

    print(f"\n数据类型:")
    print(df.dtypes)

    # 查找可能的目标列
    print(f"\n查找可能的Delta T列:")
    delta_t_candidates = [col for col in df.columns if 'delta' in col.lower() or 'dt' in col.lower()]
    if delta_t_candidates:
        print(f"  找到候选列: {delta_t_candidates}")
        for col in delta_t_candidates:
            print(f"    {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    else:
        print("  未找到明显的Delta T列")

    # 查找几何类型列
    print(f"\n查找几何类型列:")
    geo_candidates = [col for col in df.columns if 'geo' in col.lower() or 'type' in col.lower()]
    if geo_candidates:
        print(f"  找到候选列: {geo_candidates}")
        for col in geo_candidates:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                print(f"    {col}: {df[col].nunique()} 个唯一值")
                print(f"      值: {df[col].unique()[:10]}")
    else:
        print("  未找到明显的几何类型列")

    return df

if __name__ == "__main__":
    # 尝试多个可能的路径
    possible_paths = [
        "data/simulations/metadata_clean.csv",
        "data/simulations/metadata.csv",
        "data/metadata_clean.csv",
        "data/metadata.csv",
    ]

    df = None
    for path in possible_paths:
        print(f"\n尝试读取: {path}")
        df = inspect_metadata(path)
        if df is not None:
            break

    if df is None:
        print("\n❌ 未找到任何metadata文件")
        print("请手动指定文件路径:")
        print("  python inspect_metadata.py /path/to/metadata.csv")
