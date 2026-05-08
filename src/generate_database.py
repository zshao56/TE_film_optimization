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
    Lx, Ly, h, k_low, k_high, nx, ny, nz = args
    
    # Randomize volume fraction (e.g., 30% to 70%) and blur level for topology diversity
    vol_frac = np.random.uniform(0.3, 0.7)
    blur = np.random.uniform(1.0, 4.0)
    
    random_geom = generate_random_structure(
        Lx, Ly, h, k_low, k_high, nx, ny, nz, 
        volume_fraction_target=vol_frac, blur_sigma=blur
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
    # Fixed physics/dimension constants for the database
    Lx, Ly, h = 0.01, 0.01, 0.002 # 1cm x 1cm x 2mm
    k_low, k_high = 0.5, 150.0
    nx, ny, nz = 40, 40, 15
    
    print(f"Starting massive database generation: {num_samples} samples.")
    if max_workers:
        print(f"Using {max_workers} CPU cores.")
    
    # Prepare arguments for each task
    tasks = [(Lx, Ly, h, k_low, k_high, nx, ny, nz) for _ in range(num_samples)]
    
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
