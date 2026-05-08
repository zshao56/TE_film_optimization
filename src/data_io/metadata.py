import os
import sys
import pandas as pd
import h5py
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
    'heat_flux_redirect_ratio', 'mesh_element_count', 'qc_pass', 'field_file'
]

def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(FIELDS_DIR, exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'results', 'optimized_structures'), exist_ok=True)

def append_metadata(record: dict):
    """
    Append a simulation record to metadata.csv.
    """
    ensure_dirs()
    
    # Fill missing optional fields with None to align with REQUIRED_COLUMNS
    for col in REQUIRED_COLUMNS:
        if col not in record:
            record[col] = None
            
    df_new = pd.DataFrame([record])
    
    if os.path.exists(METADATA_FILE):
        df_existing = pd.read_csv(METADATA_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    # Ensure column order
    df_combined = df_combined[[col for col in REQUIRED_COLUMNS if col in df_combined.columns]]
    df_combined.to_csv(METADATA_FILE, index=False)
    print(f"Appended record {record.get('simulation_id')} to {METADATA_FILE}")

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
