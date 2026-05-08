import sys
import os
import numpy as np

# 1. 动态获取【项目根目录】的绝对路径 (基于当前脚本在 src/ 目录下)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 2. 将根目录加入环境变量
if project_root not in sys.path:
    sys.path.append(project_root)

# 3. 规范化定义数据读写路径
DATA_DIR = os.path.join(project_root, 'data')
RESULTS_DIR = os.path.join(project_root, 'results')

import matplotlib.pyplot as plt
from geometry.random_structure import generate_random_structure

# 尝试导入 my_toolbox，如果没有则使用基础 matplotlib
try:
    from my_toolbox.plot_utils import adobe_figure, COLORS
    use_adobe = True
except ImportError:
    use_adobe = False

def plot_styles():
    Lx, Ly, h = 0.01, 0.01, 0.002
    k_low, k_high = 0.5, 150.0
    nx, ny, nz = 30, 30, 15  # Slightly lower res for faster 3D plotting

    styles = ['isotropic', 'pillars_z', 'lamellae_xy', 'lamellae_xz']
    
    fig = plt.figure(figsize=(20, 15))
    
    for i, style in enumerate(styles):
        print(f"Generating style: {style}...")
        
        base_blur = 2.0
        high_blur = 10.0
        
        if style == 'isotropic':
            blur_sigma = base_blur
        elif style == 'pillars_z':
            blur_sigma = (base_blur, base_blur, high_blur)
        elif style == 'lamellae_xy':
            blur_sigma = (high_blur, high_blur, base_blur * 0.5)
        elif style == 'lamellae_xz':
            blur_sigma = (high_blur, base_blur * 0.5, high_blur)
            
        geom = generate_random_structure(
            Lx, Ly, h, k_low, k_high, nx, ny, nz, 
            volume_fraction_target=0.5, blur_sigma=blur_sigma
        )
        
        mask = geom['mask_3d']
        
        # Plotting voxels
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Color the voxels (True values)
        colors = np.empty(mask.shape, dtype=object)
        colors[mask] = '#FF6B6B' if not use_adobe else COLORS[0]
        
        ax.voxels(mask, facecolors=colors, edgecolor='k', linewidth=0.1, alpha=0.9)
        
        ax.set_title(f"Topology: {style}", fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Thickness)')
        ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    
    os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'figures', 'random_topologies_preview.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")

def plot_single_structure(geom, sim_id):
    """
    Utility function to plot and save a single 3D structure during massive generation.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    mask = geom['mask_3d']
    
    colors = np.empty(mask.shape, dtype=object)
    colors[mask] = '#4ECDC4' if not use_adobe else COLORS[1 % len(COLORS)]
    
    ax.voxels(mask, facecolors=colors, edgecolor='k', linewidth=0.1, alpha=0.9)
    
    ax.set_title(f"Structure: {sim_id}\nVol Frac: {geom.get('volume_fraction_target', 0):.2f}", fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Thickness)')
    ax.view_init(elev=30, azim=45)
    
    save_dir = os.path.join(RESULTS_DIR, 'figures', 'database_preview')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{sim_id}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    plot_styles()
