import sys
import os
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from geometry.structured_library import generate_curved_wedge_structure
from simulation.custom_solver import Custom3DFDMSolver
from postprocess.metrics import find_best_electrodes

def run_grid_independence_test():
    # 固定的物理参数和几何参数（测试曲线面，因为曲线最容易受到网格精度影响）
    Lx, Ly, h = 0.01, 0.01, 0.002
    k_low, k_high = 0.5, 150.0
    T_hot = 350.0
    T_air = 298.15
    h_c = 10.0
    h_c_side = 10.0

    # 我们要测试的网格列表
    resolutions = [
        (40, 40, 15),   # 当前默认（粗糙）
        (50, 50, 20),   # 轻微加密
        (60, 60, 30),   # 适度加密
    ]

    print(f"{'Mesh (nx, ny, nz)':<20} | {'DoF (Nodes)':<12} | {'Delta T (K)':<12} | {'Time (s)':<10}")
    print("-" * 62)

    for nx, ny, nz in resolutions:
        # 生成一个绝对一样的曲线几何体
        geom_params = generate_curved_wedge_structure(
            Lx, Ly, h, k_low, k_high, nx, ny, nz,
            base_fraction=0.1, max_fraction=0.9, exponent=2.5, direction="x"
        )

        start_time = time.time()

        # 运行模拟
        solver = Custom3DFDMSolver(geom_params, T_hot, T_air, h_c, h_c_side, nx=nx, ny=ny, nz=nz)
        mesh_data, field_data = solver.solve()
        solver.cleanup()

        # 计算指标
        wx, wy = 0.05 * Lx, 0.05 * Ly
        s_min = 0.05 * Lx
        X, Y = np.meshgrid(mesh_data['x'], mesh_data['y'], indexing='ij')

        best = find_best_electrodes(field_data['temperature_surface'], X, Y, Lx, Ly, wx, wy, s_min)
        
        end_time = time.time()
        
        dt = best['delta_T_parallel'] if best else 0.0
        calc_time = end_time - start_time
        nodes = nx * ny * nz

        print(f"{str((nx, ny, nz)):<20} | {nodes:<12} | {dt:<12.4f} | {calc_time:<10.2f}")

if __name__ == '__main__':
    run_grid_independence_test()
