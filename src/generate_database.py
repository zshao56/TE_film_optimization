import os
import sys
import uuid
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # Requires pip install tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from geometry.random_structure import generate_random_structure
from geometry.structured_library import sample_structured_structure
from main import run_simulation_pipeline


def _sample_environment(rng):
    return {
        'T_hot': float(rng.uniform(308.0, 373.0)),
        'T_air': float(rng.uniform(290.0, 298.0)),
        'h_c': float(rng.uniform(5.0, 25.0)),
        'h_c_side': float(rng.uniform(5.0, 25.0))
    }


def _sample_random_smoothed_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng):
    # Random topology remains useful as a minority exploration source, but it
    # should not dominate the database because most samples are not interpretable.
    vol_frac = float(rng.uniform(0.2, 0.8))

    style = rng.choice(['isotropic', 'pillars_z', 'lamellae_xy', 'lamellae_yz', 'lamellae_xz'])
    base_blur = float(rng.uniform(1.0, 3.0))
    high_blur = float(rng.uniform(8.0, 15.0))

    if style == 'isotropic':
        blur_sigma = base_blur
    elif style == 'pillars_z':
        blur_sigma = (base_blur, base_blur, high_blur)
    elif style == 'lamellae_xy':
        blur_sigma = (high_blur, high_blur, base_blur * 0.5)
    elif style == 'lamellae_yz':
        blur_sigma = (base_blur * 0.5, high_blur, high_blur)
    elif style == 'lamellae_xz':
        blur_sigma = (high_blur, base_blur * 0.5, high_blur)
    else:
        raise ValueError(f"Unsupported random topology style: {style}")

    geom = generate_random_structure(
        Lx, Ly, h, k_low, k_high, nx, ny, nz,
        volume_fraction_target=vol_frac, blur_sigma=blur_sigma, env_params=env_params, rng=rng
    )
    geom['random_topology_style'] = str(style)
    return geom


def _sample_geometry(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng, mode, structured_ratio):
    if mode == 'random':
        source = 'random'
    elif mode == 'structured':
        source = 'structured'
    elif mode == 'mixed':
        source = 'structured' if rng.random() < structured_ratio else 'random'
    else:
        raise ValueError(f"Unsupported generation mode: {mode}")

    if source == 'structured':
        return sample_structured_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params=env_params, rng=rng)

    return _sample_random_smoothed_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng)


def generate_single_sample(args):
    """
    Worker function to generate and simulate a single structure.
    """
    index, Lx, Ly, nx, ny, nz, mode, structured_ratio, seed = args
    rng = np.random.default_rng(seed)
    
    # 1. Randomize physics and dimensions
    h = float(rng.uniform(0.0005, 0.002)) # Film thickness: 0.5 mm to 2.0 mm
    k_low = float(rng.uniform(0.08, 0.5))
    k_high = float(rng.uniform(1.0, 5.0))
    
    env_params = _sample_environment(rng)
    geom = _sample_geometry(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng, mode, structured_ratio)
    geom['sample_seed'] = int(seed)

    sim_id = f"sim_{geom['geometry_type']}_{uuid.uuid4().hex[:8]}"
    
    # Visual check: Save a 3D plot every 10 samples
    if index % 10 == 0:
        from visualize_random_structures import plot_single_structure
        plot_single_structure(geom, sim_id)
    
    try:
        run_simulation_pipeline(geom, sim_id)
        return True, sim_id
    except Exception as e:
        return False, str(e)

def build_massive_database(num_samples, max_workers=None, mode='mixed', structured_ratio=0.8, seed=None):
    """
    Generate a large database using multiprocessing, with auto-resume capability.
    """
    if not 0.0 <= structured_ratio <= 1.0:
        raise ValueError("structured_ratio must be between 0 and 1.")

    # Fixed in-plane dimensions and resolution
    Lx, Ly = 0.01, 0.01 # 1cm x 1cm
    nx, ny, nz = 50, 50, 20
    
    # 1. Check existing database for resume capability
    metadata_path = os.path.join(current_dir, '..', 'data', 'simulations', 'metadata.csv')
    existing_count = 0
    if os.path.exists(metadata_path):
        try:
            df = pd.read_csv(metadata_path)
            # Count only successful runs
            existing_count = len(df[df['qc_pass'] == True])
        except Exception as e:
            print(f"Warning: Could not read {metadata_path}. Starting from scratch. Error: {e}")
            existing_count = 0

    remaining_samples = num_samples - existing_count

    if remaining_samples <= 0:
        print(f"Database already contains {existing_count} successful samples. Target of {num_samples} reached. Exiting.")
        return

    print(f"Target database size: {num_samples} samples.")
    if existing_count > 0:
        print(f"Found {existing_count} existing samples. Resuming and generating {remaining_samples} more...")
    else:
        print(f"Starting fresh database generation: {remaining_samples} samples.")
        
    print(f"Generation mode: {mode}. Structured ratio for mixed mode: {structured_ratio:.2f}.")
    if max_workers:
        print(f"Using {max_workers} CPU cores.")
    
    root_sequence = np.random.SeedSequence(seed)
    # Spawn total needed (existing + remaining) to maintain reproducibility offset
    child_seeds = root_sequence.spawn(num_samples)

    # Prepare arguments for each task, offset by existing_count to avoid repeating seeds
    tasks = [
        (existing_count + i, Lx, Ly, nx, ny, nz, mode, structured_ratio, int(child_seeds[existing_count + i].generate_state(1)[0]))
        for i in range(remaining_samples)
    ]
    
    success_count = 0
    fail_count = 0
    
    # Use ProcessPoolExecutor to run physical simulations in parallel
    # On Windows, ProcessPoolExecutor must be inside the if __name__ == '__main__' guard,
    # which it effectively is because build_massive_database is called from there.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_single_sample, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=remaining_samples, desc="Simulating"):
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
    parser.add_argument("--samples", type=int, default=1000, help="Number of structures to generate.")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use. Defaults to all available.")
    parser.add_argument(
        "--mode",
        choices=["structured", "mixed", "random"],
        default="mixed",
        help="Structure source. 'mixed' uses mostly structured families plus some random-smoothed exploration."
    )
    parser.add_argument(
        "--structured-ratio",
        type=float,
        default=0.8,
        help="Fraction of structured samples in mixed mode."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional root random seed for reproducible sampling.")
    
    args = parser.parse_args()
    build_massive_database(args.samples, args.cores, args.mode, args.structured_ratio, args.seed)
