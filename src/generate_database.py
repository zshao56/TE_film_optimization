import os
import sys
import uuid
import argparse
import json
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


def _load_json_config(path):
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _config_section(config, key):
    if not config:
        return {}
    return config.get(key, config)


def _nested_get(config, keys, default):
    current = config or {}
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _sample_uniform_range(rng, bounds):
    low, high = bounds
    return float(rng.uniform(float(low), float(high)))


def _weighted_choice(rng, weights):
    names = list(weights.keys())
    probs = np.array([float(weights[name]) for name in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def _sample_convection(rng, profile, sampling_config=None):
    if profile != 'expanded':
        h_range = _nested_get(sampling_config, ['legacy', 'convection', 'h_c_range'], [5.0, 25.0])
        h_side_range = _nested_get(sampling_config, ['legacy', 'convection', 'h_c_side_range'], h_range)
        h_c = _sample_uniform_range(rng, h_range)
        h_c_side = _sample_uniform_range(rng, h_side_range)
        return {
            'convection_regime': 'legacy_natural',
            'convection_regime_code': 0,
            'h_c': h_c,
            'h_c_side': h_c_side,
        }

    convection_config = (sampling_config or {}).get('convection', {})
    default_regimes = {
        regime: {
            'range': [low, high],
            'code': code,
            'weight': {'natural': 0.35, 'weak_forced': 0.25, 'forced': 0.25, 'strong_forced': 0.15}[regime],
        }
        for regime, (low, high, code) in CONVECTION_REGIMES.items()
    }
    regimes_config = convection_config.get('regimes', default_regimes)
    regime = _weighted_choice(rng, {name: cfg.get('weight', 1.0) for name, cfg in regimes_config.items()})
    low, high = regimes_config[regime].get('range', CONVECTION_REGIMES[regime][:2])
    code = int(regimes_config[regime].get('code', CONVECTION_REGIMES[regime][2]))
    h_c = float(rng.uniform(low, high))
    side_scale_range = convection_config.get('side_scale_range', [0.6, 1.2])
    side_scale = _sample_uniform_range(rng, side_scale_range)
    h_c_side = float(np.clip(h_c * side_scale, low, high))
    return {
        'convection_regime': regime,
        'convection_regime_code': code,
        'h_c': h_c,
        'h_c_side': h_c_side,
    }


def _sample_hot_boundary(rng, profile, nx, ny, sampling_config=None, T_air=None):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    if profile != 'expanded':
        temp_range = _nested_get(sampling_config, ['legacy', 'hot_boundary', 'uniform_temp_range'], [308.0, 373.0])
        T_hot = _sample_uniform_range(rng, temp_range)
        hot_map = np.full((nx, ny), T_hot, dtype=float)
        return {
            'T_hot': T_hot,
            'T_hot_map': hot_map,
            'hot_boundary_type': 'uniform',
            'hot_boundary_type_code': HOT_BOUNDARY_TYPE_CODES['uniform'],
            'T_hot_min': T_hot,
            'T_hot_max': T_hot,
            'T_hot_min_delta': None if T_air is None else float(T_hot - T_air),
            'T_hot_amplitude': 0.0,
            'gradient_direction_code': 0,
            'hotspot_x': 0.0,
            'hotspot_y': 0.0,
            'hotspot_sigma': 0.0,
        }

    hot_config = (sampling_config or {}).get('hot_boundary', {})
    type_weights = hot_config.get(
        'type_weights',
        {'uniform': 0.40, 'linear_gradient': 0.30, 'gaussian_hotspot': 0.30},
    )
    boundary_type = _weighted_choice(rng, type_weights)
    use_relative_hot_min = T_air is not None and 'hot_min_delta_T_range' in hot_config
    if use_relative_hot_min:
        hot_min_delta = _sample_uniform_range(rng, hot_config.get('hot_min_delta_T_range', [5.0, 200.0]))
        hot_min = float(T_air + hot_min_delta)
    else:
        hot_min_delta = None
        hot_min = None

    if boundary_type == 'uniform':
        T_hot = hot_min if use_relative_hot_min else _sample_uniform_range(rng, hot_config.get('uniform_temp_range', [308.0, 373.0]))
        hot_map = np.full((nx, ny), T_hot, dtype=float)
        gradient_direction_code = 0
        hotspot_x = hotspot_y = hotspot_sigma = 0.0
    elif boundary_type == 'linear_gradient':
        amplitude = _sample_uniform_range(rng, hot_config.get('linear_amplitude_range', [5.0, 30.0]))
        directions = hot_config.get('linear_directions', ['x', 'y'])
        direction = str(rng.choice(directions))
        coord = X if direction == 'x' else Y
        if use_relative_hot_min:
            hot_map = hot_min + amplitude * coord
        else:
            mean_temp = _sample_uniform_range(rng, hot_config.get('linear_mean_range', [318.0, 363.0]))
            hot_map = mean_temp + amplitude * (coord - 0.5)
            hot_clip = hot_config.get('linear_clip_range', [308.0, 373.0])
            hot_map = np.clip(hot_map, hot_clip[0], hot_clip[1])
        gradient_direction_code = 0 if direction == 'x' else 1
        hotspot_x = hotspot_y = hotspot_sigma = 0.0
    elif boundary_type == 'gaussian_hotspot':
        hotspot_x = _sample_uniform_range(rng, hot_config.get('hotspot_x_range', [0.15, 0.85]))
        hotspot_y = _sample_uniform_range(rng, hot_config.get('hotspot_y_range', [0.15, 0.85]))
        hotspot_sigma = _sample_uniform_range(rng, hot_config.get('hotspot_sigma_range', [0.06, 0.22]))
        r2 = (X - hotspot_x) ** 2 + (Y - hotspot_y) ** 2
        bump = np.exp(-0.5 * r2 / (hotspot_sigma ** 2))
        if use_relative_hot_min:
            bump_span = float(np.max(bump) - np.min(bump))
            bump = np.zeros_like(bump) if bump_span <= 1e-15 else (bump - np.min(bump)) / bump_span
            peak_delta = _sample_uniform_range(rng, hot_config.get('gaussian_peak_delta_range', [10.0, 40.0]))
            hot_map = hot_min + peak_delta * bump
        else:
            base_temp = _sample_uniform_range(rng, hot_config.get('gaussian_base_range', [303.0, 335.0]))
            peak_min = max(base_temp + hot_config.get('gaussian_min_peak_delta', 10.0), hot_config.get('gaussian_peak_floor', 330.0))
            peak_max = hot_config.get('gaussian_peak_max', 373.0)
            peak_temp = float(rng.uniform(peak_min, peak_max))
            hot_map = base_temp + (peak_temp - base_temp) * bump
            hot_clip = hot_config.get('gaussian_clip_range', [303.0, 373.0])
            hot_map = np.clip(hot_map, hot_clip[0], hot_clip[1])
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
        'T_hot_min_delta': float(np.min(hot_map) - T_air) if T_air is not None else hot_min_delta,
        'T_hot_amplitude': float(np.max(hot_map) - np.min(hot_map)),
        'gradient_direction_code': gradient_direction_code,
        'hotspot_x': hotspot_x,
        'hotspot_y': hotspot_y,
        'hotspot_sigma': hotspot_sigma,
    }


def _sample_environment(rng, profile='legacy', nx=50, ny=50, sampling_config=None):
    env = {
        'T_air': _sample_uniform_range(rng, _nested_get(sampling_config, ['environment', 'T_air_range'], [293.0, 303.0])),
    }
    env.update(_sample_convection(rng, profile, sampling_config))
    env.update(_sample_hot_boundary(rng, profile, nx, ny, sampling_config, T_air=env['T_air']))
    return env


def _sample_thickness(rng, profile, sampling_config=None):
    if profile != 'expanded':
        return _sample_uniform_range(rng, _nested_get(sampling_config, ['legacy', 'thickness_range'], [0.0005, 0.002]))

    bands = (sampling_config or {}).get(
        'thickness_bands',
        {
            'thin': {'range': [0.0001, 0.0005], 'weight': 0.20},
            'medium': {'range': [0.0005, 0.002], 'weight': 0.60},
            'thick': {'range': [0.002, 0.005], 'weight': 0.20},
        },
    )
    band = _weighted_choice(rng, {name: cfg.get('weight', 1.0) for name, cfg in bands.items()})
    return _sample_uniform_range(rng, bands[band]['range'])


def _sample_materials(rng, profile, sampling_config=None):
    if profile != 'expanded':
        k_low = _sample_uniform_range(rng, _nested_get(sampling_config, ['legacy', 'materials', 'k_low_range'], [0.08, 0.5]))
        k_high = _sample_uniform_range(rng, _nested_get(sampling_config, ['legacy', 'materials', 'k_high_range'], [1.0, 5.0]))
        return k_low, k_high

    material_config = (sampling_config or {}).get('materials', {})
    k_low = _sample_uniform_range(rng, material_config.get('k_low_range', [0.05, 0.8]))
    ratio_low, ratio_high = material_config.get('k_ratio_log_range', [3.0, 80.0])
    ratio = float(np.exp(rng.uniform(np.log(ratio_low), np.log(ratio_high))))
    k_high_min, k_high_max = material_config.get('k_high_clip', [0.8, 10.0])
    k_high = float(np.clip(k_low * ratio, k_high_min, k_high_max))
    if k_high <= k_low:
        k_high = float(k_low + material_config.get('min_k_gap', 0.75))
    return k_low, k_high


def _sample_curvature(rng, Lx, Ly, profile, sampling_config=None):
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

    curvature_config = (sampling_config or {}).get('curvature', {})
    level_weights = curvature_config.get(
        'level_weights',
        {'0': 0.25, '025': 0.15, '05': 0.20, '075': 0.15, '1': 0.15, 'random': 0.10},
    )
    level_choice = _weighted_choice(rng, level_weights)
    if level_choice == 'random':
        curvature_level = _sample_uniform_range(rng, curvature_config.get('random_level_range', [0.0, 1.0]))
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

    bend_axes = curvature_config.get('bend_axes', ['x', 'y'])
    bend_axis = str(rng.choice(bend_axes))
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
    index, Lx, Ly, nx, ny, nz, mode, structured_ratio, profile, seed, sampling_config = args
    rng = np.random.default_rng(seed)
    
    # 1. Randomize physics and dimensions
    h = _sample_thickness(rng, profile, sampling_config)
    k_low, k_high = _sample_materials(rng, profile, sampling_config)
    
    env_params = _sample_environment(rng, profile=profile, nx=nx, ny=ny, sampling_config=sampling_config)
    geom = _sample_geometry(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params, rng, mode, structured_ratio)
    geom.update(_sample_curvature(rng, Lx, Ly, profile, sampling_config))
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

def build_massive_database(num_samples, max_workers=None, mode='mixed', structured_ratio=0.8, seed=None, profile='legacy', sampling_config=None, grid_config=None):
    """
    Generate a large database using multiprocessing, with auto-resume capability.
    """
    if not 0.0 <= structured_ratio <= 1.0:
        raise ValueError("structured_ratio must be between 0 and 1.")

    grid_config = grid_config or {}
    Lx = float(grid_config.get('Lx', 0.01))
    Ly = float(grid_config.get('Ly', 0.01))
    nx = int(grid_config.get('nx', 50))
    ny = int(grid_config.get('ny', 50))
    nz = int(grid_config.get('nz', 20))
    
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
        (
            existing_count + i,
            Lx,
            Ly,
            nx,
            ny,
            nz,
            mode,
            structured_ratio,
            profile,
            int(child_seeds[existing_count + i].generate_state(1)[0]),
            sampling_config or {},
        )
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
    parser.add_argument("--config", type=str, default=None, help="Optional JSON pipeline/config file. Uses the data_generation section when present.")
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
    config = _load_json_config(args.config)
    data_config = _config_section(config, 'data_generation')
    sampling_config = data_config.get('sampling', {})
    grid_config = data_config.get('grid', {})
    samples = int(data_config.get('samples', args.samples))
    cores = data_config.get('cores', args.cores)
    cores = None if cores is None else int(cores)
    mode = data_config.get('mode', args.mode)
    structured_ratio = float(data_config.get('structured_ratio', args.structured_ratio))
    seed = data_config.get('seed', args.seed)
    seed = None if seed is None else int(seed)
    profile = data_config.get('profile', args.profile)
    build_massive_database(samples, cores, mode, structured_ratio, seed, profile, sampling_config, grid_config)
