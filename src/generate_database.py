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


CONVECTION_REGIMES = {
    'natural': (2.0, 15.0, 0),
    'weak_forced': (15.0, 50.0, 1),
    'forced': (50.0, 150.0, 2),
    'strong_forced': (150.0, 500.0, 3),
}

HOT_BOUNDARY_TYPE_CODES = {
    'uniform': 0,
    'linear_gradient': 1,
    'gaussian_hotspot': 2,
}


def _sample_convection(rng, profile):
    if profile != 'expanded':
        h_c = float(rng.uniform(5.0, 25.0))
        h_c_side = float(rng.uniform(5.0, 25.0))
        return {
            'convection_regime': 'legacy_natural',
            'convection_regime_code': 0,
            'h_c': h_c,
            'h_c_side': h_c_side,
        }

    regimes = np.array(['natural', 'weak_forced', 'forced', 'strong_forced'])
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    regime = str(rng.choice(regimes, p=weights))
    low, high, code = CONVECTION_REGIMES[regime]
    h_c = float(rng.uniform(low, high))
    side_scale = float(rng.uniform(0.6, 1.2))
    h_c_side = float(np.clip(h_c * side_scale, low, high))
    return {
        'convection_regime': regime,
        'convection_regime_code': code,
        'h_c': h_c,
        'h_c_side': h_c_side,
    }


def _sample_hot_boundary(rng, profile, nx, ny):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    if profile != 'expanded':
        T_hot = float(rng.uniform(308.0, 373.0))
        hot_map = np.full((nx, ny), T_hot, dtype=float)
        return {
            'T_hot': T_hot,
            'T_hot_map': hot_map,
            'hot_boundary_type': 'uniform',
            'hot_boundary_type_code': HOT_BOUNDARY_TYPE_CODES['uniform'],
            'T_hot_min': T_hot,
            'T_hot_max': T_hot,
            'T_hot_amplitude': 0.0,
            'gradient_direction_code': 0,
            'hotspot_x': 0.0,
            'hotspot_y': 0.0,
            'hotspot_sigma': 0.0,
        }

    boundary_type = str(rng.choice(
        ['uniform', 'linear_gradient', 'gaussian_hotspot'],
        p=[0.40, 0.30, 0.30],
    ))

    if boundary_type == 'uniform':
        T_hot = float(rng.uniform(308.0, 373.0))
        hot_map = np.full((nx, ny), T_hot, dtype=float)
        gradient_direction_code = 0
        hotspot_x = hotspot_y = hotspot_sigma = 0.0
    elif boundary_type == 'linear_gradient':
        mean_temp = float(rng.uniform(318.0, 363.0))
        amplitude = float(rng.uniform(5.0, 30.0))
        direction = str(rng.choice(['x', 'y']))
        coord = X if direction == 'x' else Y
        hot_map = mean_temp + amplitude * (coord - 0.5)
        hot_map = np.clip(hot_map, 308.0, 373.0)
        gradient_direction_code = 0 if direction == 'x' else 1
        hotspot_x = hotspot_y = hotspot_sigma = 0.0
    elif boundary_type == 'gaussian_hotspot':
        base_temp = float(rng.uniform(303.0, 335.0))
        peak_temp = float(rng.uniform(max(base_temp + 10.0, 330.0), 373.0))
        hotspot_x = float(rng.uniform(0.15, 0.85))
        hotspot_y = float(rng.uniform(0.15, 0.85))
        hotspot_sigma = float(rng.uniform(0.06, 0.22))
        r2 = (X - hotspot_x) ** 2 + (Y - hotspot_y) ** 2
        hot_map = base_temp + (peak_temp - base_temp) * np.exp(-0.5 * r2 / (hotspot_sigma ** 2))
        hot_map = np.clip(hot_map, 303.0, 373.0)
        gradient_direction_code = 0
    else:
        raise ValueError(f"Unsupported hot boundary type: {boundary_type}")

    return {
        'T_hot': float(np.mean(hot_map)),
        'T_hot_map': hot_map.astype(float),
        'hot_boundary_type': boundary_type,
        'hot_boundary_type_code': HOT_BOUNDARY_TYPE_CODES[boundary_type],
        'T_hot_min': float(np.min(hot_map)),
        'T_hot_max': float(np.max(hot_map)),
        'T_hot_amplitude': float(np.max(hot_map) - np.min(hot_map)),
        'gradient_direction_code': gradient_direction_code,
        'hotspot_x': hotspot_x,
        'hotspot_y': hotspot_y,
        'hotspot_sigma': hotspot_sigma,
    }


def _sample_environment(rng, profile='legacy', nx=50, ny=50):
    env = {
        'T_air': float(rng.uniform(293.0, 303.0)),
    }
    env.update(_sample_convection(rng, profile))
    env.update(_sample_hot_boundary(rng, profile, nx, ny))
    return env


def _sample_thickness(rng, profile):
    if profile != 'expanded':
        return float(rng.uniform(0.0005, 0.002))

    band = str(rng.choice(['thin', 'medium', 'thick'], p=[0.20, 0.60, 0.20]))
    if band == 'thin':
        return float(rng.uniform(0.0001, 0.0005))
    if band == 'medium':
        return float(rng.uniform(0.0005, 0.002))
    return float(rng.uniform(0.002, 0.005))


def _sample_materials(rng, profile):
    if profile != 'expanded':
        k_low = float(rng.uniform(0.08, 0.5))
        k_high = float(rng.uniform(1.0, 5.0))
        return k_low, k_high

    k_low = float(rng.uniform(0.05, 0.8))
    ratio = float(np.exp(rng.uniform(np.log(3.0), np.log(80.0))))
    k_high = float(np.clip(k_low * ratio, 0.8, 10.0))
    if k_high <= k_low:
        k_high = float(k_low + 0.75)
    return k_low, k_high


def _sample_curvature(rng, Lx, Ly, profile):
    if profile != 'expanded':
        return {
            'curvature_type': 'flat',
            'curvature_level': 0.0,
            'arc_angle': 0.0,
            'bend_axis': 'none',
            'bend_axis_code': 0,
            'bend_radius': 0.0,
            'arc_length': Lx,
            'projected_length': Lx,
            'projected_Lx': Lx,
            'projected_Ly': Ly,
        }

    level_choice = rng.choice(['0', '025', '05', '075', '1', 'random'], p=[0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    if level_choice == 'random':
        curvature_level = float(rng.uniform(0.0, 1.0))
    else:
        curvature_level = {'0': 0.0, '025': 0.25, '05': 0.5, '075': 0.75, '1': 1.0}[str(level_choice)]

    if curvature_level <= 1e-12:
        return {
            'curvature_type': 'flat',
            'curvature_level': 0.0,
            'arc_angle': 0.0,
            'bend_axis': 'none',
            'bend_axis_code': 0,
            'bend_radius': 0.0,
            'arc_length': Lx,
            'projected_length': Lx,
            'projected_Lx': Lx,
            'projected_Ly': Ly,
        }

    bend_axis = str(rng.choice(['x', 'y']))
    bend_axis_code = 0 if bend_axis == 'x' else 1
    arc_angle = float(curvature_level * np.pi)
    arc_length = float(Lx if bend_axis == 'x' else Ly)
    radius = float(arc_length / arc_angle)
    projected_length = float(2.0 * radius * np.sin(arc_angle / 2.0))
    projected_Lx = projected_length if bend_axis == 'x' else Lx
    projected_Ly = projected_length if bend_axis == 'y' else Ly
    return {
        'curvature_type': 'cylindrical_arc',
        'curvature_level': curvature_level,
        'arc_angle': arc_angle,
        'bend_axis': bend_axis,
        'bend_axis_code': bend_axis_code,
        'bend_radius': radius,
        'arc_length': arc_length,
        'projected_length': projected_length,
        'projected_Lx': projected_Lx,
        'projected_Ly': projected_Ly,
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
    index, Lx, Ly, nx, ny, nz, mode, structured_ratio, profile, seed = args
    rng = np.random.default_rng(seed)
    
    # 1. Randomize physics and dimensions
    h = _sample_thickness(rng, profile)
    k_low, k_high = _sample_materials(rng, profile)
    
    env_params = _sample_environment(rng, profile=profile, nx=nx, ny=ny)
    geom = _sample_geometry(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng, mode, structured_ratio)
    geom.update(_sample_curvature(rng, Lx, Ly, profile))
    geom['database_profile'] = profile
    geom['k_ratio'] = float(k_high / (k_low + 1e-15))
    geom['sample_seed'] = int(seed)
    geom['scenario_id'] = (
        f"{geom.get('curvature_type', 'flat')}_"
        f"{geom.get('convection_regime', 'legacy')}_"
        f"{geom.get('hot_boundary_type', 'uniform')}"
    )

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

def build_massive_database(num_samples, max_workers=None, mode='mixed', structured_ratio=0.8, seed=None, profile='legacy'):
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
        
    print(f"Database profile: {profile}")
    print(f"Generation mode: {mode}. Structured ratio for mixed mode: {structured_ratio:.2f}.")
    if max_workers:
        print(f"Using {max_workers} CPU cores.")
    
    root_sequence = np.random.SeedSequence(seed)
    # Spawn total needed (existing + remaining) to maintain reproducibility offset
    child_seeds = root_sequence.spawn(num_samples)

    # Prepare arguments for each task, offset by existing_count to avoid repeating seeds
    tasks = [
        (existing_count + i, Lx, Ly, nx, ny, nz, mode, structured_ratio, profile, int(child_seeds[existing_count + i].generate_state(1)[0]))
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
    parser.add_argument(
        "--profile",
        choices=["legacy", "expanded"],
        default="legacy",
        help="Database parameter profile. 'expanded' adds wider convection regimes, non-uniform hot boundaries, and curvature metadata."
    )
    
    args = parser.parse_args()
    build_massive_database(args.samples, args.cores, args.mode, args.structured_ratio, args.seed, args.profile)
