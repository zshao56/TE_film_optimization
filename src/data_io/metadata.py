import os
import sys
import pandas as pd
import h5py
import csv
import time
import random
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

DATA_DIR = os.path.join(project_root, 'data')
METADATA_FILE = os.path.join(DATA_DIR, 'simulations', 'metadata.csv')
FIELDS_DIR = os.path.join(DATA_DIR, 'simulations', 'fields')

REQUIRED_COLUMNS = [
    'simulation_id', 'geometry_type', 'geometry_parameters', 'thickness_h', 
    'length_Lx', 'length_Ly', 'k_low', 'k_high', 'boundary_condition_id', 
    'T_hot', 'T_air', 'measurement_wx', 'measurement_wy', 'electrode_min_gap',
    'x_hot_electrode', 'y_hot_electrode', 'x_cold_electrode', 'y_cold_electrode',
    'T_hot_electrode_avg', 'T_cold_electrode_avg', 'delta_T_parallel',
    'heat_flux_redirect_ratio', 'mesh_element_count', 'qc_pass', 'field_file',
    'database_profile', 'scenario_id', 'k_ratio',
    'convection_regime', 'convection_regime_code', 'h_c', 'h_c_side',
    'hot_boundary_type', 'hot_boundary_type_code', 'T_hot_min', 'T_hot_max',
    'T_hot_amplitude', 'gradient_direction_code', 'hotspot_x', 'hotspot_y',
    'hotspot_sigma',
    'curvature_type', 'curvature_level', 'arc_angle', 'bend_axis',
    'bend_axis_code', 'bend_radius', 'arc_length', 'projected_length',
    'projected_Lx', 'projected_Ly'
]

def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(FIELDS_DIR, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results', 'optimized_structures'), exist_ok=True)

def append_metadata(record: dict):
    """
    Append a simulation record to metadata.csv safely across multiple processes.
    """
    ensure_dirs()
    
    # Fill missing optional fields with None to align with REQUIRED_COLUMNS
    for col in REQUIRED_COLUMNS:
        if col not in record:
            record[col] = None
            
    # Remove extra keys that shouldn't be in the CSV
    clean_record = {k: v for k, v in record.items() if k in REQUIRED_COLUMNS}
    
    file_exists = os.path.exists(METADATA_FILE)
    
    # Retry mechanism for Windows file locking issues during concurrent process writes
    max_retries = 20
    for i in range(max_retries):
        try:
            with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
                # Write header if file is totally new or empty
                if not file_exists or os.path.getsize(METADATA_FILE) == 0:
                    writer.writeheader()
                    file_exists = True # Prevent writing header multiple times
                
                writer.writerow(clean_record)
            # print(f"Appended record {record.get('simulation_id')} to {METADATA_FILE}")
            break
        except Exception as e:
            if i == max_retries - 1:
                print(f"CRITICAL: Failed to write metadata after {max_retries} retries: {e}")
                raise
            # Backoff randomly to avoid collision sync
            time.sleep(random.uniform(0.01, 0.1))

def save_h5_fields(simulation_id: str, data: dict):
    """
    Save 3D field data and mesh into an HDF5 file.
    `data` dict should contain arrays or nested dicts for:
    mesh, fields, postprocess
    """
    ensure_dirs()
    file_path = os.path.join(FIELDS_DIR, f"{simulation_id}.h5")
    
    with h5py.File(file_path, 'w') as f:
        for group_name, group_data in data.items():
            grp = f.create_group(group_name)
            for key, val in group_data.items():
                if val is not None:
                    grp.create_dataset(key, data=val)
                    
    return file_path
