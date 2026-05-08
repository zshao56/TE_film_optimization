import os
import sys
import uuid
import json
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # Requires pip install tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from geometry.random_structure import generate_random_structure
from main import run_simulation_pipeline

def generate_single_sample(args):
    """
    Worker function to generate and simulate a single random structure.
    """
    Lx, Ly, nx, ny, nz = args
    
    # 1. Randomize physics and dimensions
    h = np.random.uniform(0.0005, 0.002) # Film thickness: 0.5 mm to 2.0 mm
    k_low = np.random.uniform(0.1, 1.0)
    k_high = np.random.uniform(50.0, 400.0)
    
    env_params = {
        'T_hot': np.random.uniform(320.0, 400.0),
        'T_air': np.random.uniform(290.0, 310.0),
        'h_c': np.random.uniform(5.0, 25.0),
        'h_c_side': np.random.uniform(5.0, 25.0)
    }
    
    # 2. Randomize volume fraction
    vol_frac = np.random.uniform(0.2, 0.8)
    
    # 3. Randomize topology style (Isotropic vs Anisotropic)
    style = np.random.choice(['isotropic', 'pillars_z', 'lamellae_xy', 'lamellae_yz', 'lamellae_xz'])
    base_blur = np.random.uniform(1.0, 3.0)
    high_blur = np.random.uniform(8.0, 15.0)
    
    if style == 'isotropic':
        blur_sigma = base_blur
    elif style == 'pillars_z':
        # Stretch along Z to form columns/pillars
        blur_sigma = (base_blur, base_blur, high_blur)
    elif style == 'lamellae_xy':
        # Stretch along X and Y to form stacked planes (lamellae)
        blur_sigma = (high_blur, high_blur, base_blur * 0.5)
    elif style == 'lamellae_yz':
        blur_sigma = (base_blur * 0.5, high_blur, high_blur)
    elif style == 'lamellae_xz':
        blur_sigma = (high_blur, base_blur * 0.5, high_blur)
    
    random_geom = generate_random_structure(
        Lx, Ly, h, k_low, k_high, nx, ny, nz, 
        volume_fraction_target=vol_frac, blur_sigma=blur_sigma, env_params=env_params
    )
    
    sim_id = f"sim_rand_{uuid.uuid4().hex[:8]}"
    
    try:
        run_simulation_pipeline(random_geom, sim_id)
        return True, sim_id
    except Exception as e:
        return False, str(e)

def build_massive_database(num_samples, max_workers=None):
    """
    Generate a large database using multiprocessing.
    """
    # Fixed in-plane dimensions and resolution
    Lx, Ly = 0.01, 0.01 # 1cm x 1cm
    nx, ny, nz = 40, 40, 15
    
    print(f"Starting massive database generation: {num_samples} samples.")
    if max_workers:
        print(f"Using {max_workers} CPU cores.")
    
    # Prepare arguments for each task
    tasks = [(Lx, Ly, nx, ny, nz) for _ in range(num_samples)]
    
    success_count = 0
    fail_count = 0
    
    # Use ProcessPoolExecutor to run physical simulations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single_sample, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=num_samples, desc="Simulating"):
            success, result = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\nSimulation failed: {result}")
                
    print("\n--- Generation Complete ---")
    print(f"Successfully added to database: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a massive database of 3D TE film structures.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of random structures to generate.")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use. Defaults to all available.")
    
    args = parser.parse_args()
    build_massive_database(args.samples, args.cores)
