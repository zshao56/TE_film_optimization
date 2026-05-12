import os
import sys
import uuid
import json
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from geometry.structured_library import sample_structured_structure
from simulation.custom_solver import Custom3DFDMSolver
from postprocess.metrics import find_best_electrodes
from data_io.metadata import append_metadata, save_h5_fields

def run_simulation_pipeline(geom_params, sim_id):
    Lx, Ly, h = geom_params['Lx'], geom_params['Ly'], geom_params['h']
    
    # Boundary Conditions setup (extract from params if available, else default)
    T_hot = geom_params.get('T_hot', 350.0)  # K
    T_air = geom_params.get('T_air', 298.15) # K
    h_c = geom_params.get('h_c', 10.0)       # Convection coeff top
    h_c_side = geom_params.get('h_c_side', 10.0)# Convection coeff side
    
    # Solver resolution
    nx, ny, nz = 50, 50, 20
    
    print(f"Running simulation {sim_id} for {geom_params['geometry_type']}...")
    
    # Solve 3D Heat Equation
    solver = Custom3DFDMSolver(geom_params, T_hot, T_air, h_c, h_c_side, nx=nx, ny=ny, nz=nz)
    try:
        mesh_data, field_data = solver.solve()
    finally:
        solver.cleanup()
    
    # Postprocess (2D Top Surface Area Measurement)
    wx, wy = 0.05 * Lx, 0.05 * Ly
    s_min = 0.05 * Lx
    
    x_coords = mesh_data['x']
    y_coords = mesh_data['y']
    # Create 2D meshgrid for the surface
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    best_electrodes = find_best_electrodes(
        field_data['temperature_surface'], X, Y, Lx, Ly, wx, wy, s_min
    )
    
    qc_pass = best_electrodes is not None
    
    # Save HDF5
    h5_data = {
        'mesh': mesh_data,
        'fields': field_data,
        'postprocess': best_electrodes if best_electrodes else {}
    }
    h5_path = save_h5_fields(sim_id, h5_data)
    
    # Prepare parameters for JSON serialization (remove large arrays and unwrap numpy scalars).
    json_params = {}
    for key, value in geom_params.items():
        if key in {'mask_3d', 'T_hot_map'}:
            continue
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            json_params[key] = value.item()
        else:
            json_params[key] = value
    
    # Save Metadata
    metadata_record = {
        'simulation_id': sim_id,
        'geometry_type': geom_params['geometry_type'],
        'geometry_parameters': json.dumps(json_params),
        'thickness_h': h,
        'length_Lx': Lx,
        'length_Ly': Ly,
        'k_low': geom_params.get('k_low', geom_params.get('k_val')),
        'k_high': geom_params.get('k_high', geom_params.get('k_val')),
        'k_ratio': geom_params.get('k_ratio'),
        'boundary_condition_id': 'BC-001-TOP-ELECTRODE',
        'T_hot': T_hot,
        'T_air': T_air,
        'h_c': h_c,
        'h_c_side': h_c_side,
        'database_profile': geom_params.get('database_profile', 'legacy'),
        'scenario_id': geom_params.get('scenario_id'),
        'convection_regime': geom_params.get('convection_regime'),
        'convection_regime_code': geom_params.get('convection_regime_code'),
        'hot_boundary_type': geom_params.get('hot_boundary_type'),
        'hot_boundary_type_code': geom_params.get('hot_boundary_type_code'),
        'T_hot_min': geom_params.get('T_hot_min'),
        'T_hot_max': geom_params.get('T_hot_max'),
        'T_hot_amplitude': geom_params.get('T_hot_amplitude'),
        'gradient_direction_code': geom_params.get('gradient_direction_code'),
        'hotspot_x': geom_params.get('hotspot_x'),
        'hotspot_y': geom_params.get('hotspot_y'),
        'hotspot_sigma': geom_params.get('hotspot_sigma'),
        'curvature_type': geom_params.get('curvature_type'),
        'curvature_level': geom_params.get('curvature_level'),
        'arc_angle': geom_params.get('arc_angle'),
        'bend_axis': geom_params.get('bend_axis'),
        'bend_axis_code': geom_params.get('bend_axis_code'),
        'bend_radius': geom_params.get('bend_radius'),
        'arc_length': geom_params.get('arc_length'),
        'projected_length': geom_params.get('projected_length'),
        'projected_Lx': geom_params.get('projected_Lx'),
        'projected_Ly': geom_params.get('projected_Ly'),
        'measurement_wx': wx,
        'measurement_wy': wy,
        'electrode_min_gap': s_min,
        'qc_pass': qc_pass,
        'field_file': h5_path,
        'solver_method_code': field_data.get('solver_method_code'),
        'solver_info': field_data.get('solver_info'),
        'solver_relative_residual': field_data.get('solver_relative_residual'),
        'solver_temperature_min': field_data.get('solver_temperature_min'),
        'solver_temperature_max': field_data.get('solver_temperature_max'),
        'solver_lower_bound': field_data.get('solver_lower_bound'),
        'solver_upper_bound': field_data.get('solver_upper_bound'),
        'solver_bounds_pass': field_data.get('solver_bounds_pass'),
        'surface_bounds_pass': field_data.get('surface_bounds_pass'),
        'surface_min_bound_violation': field_data.get('surface_min_bound_violation'),
        'surface_max_bound_violation': field_data.get('surface_max_bound_violation')
    }
    
    if best_electrodes:
        metadata_record.update({
            'x_hot_electrode': best_electrodes['x_hot'],
            'y_hot_electrode': best_electrodes['y_hot'],
            'x_cold_electrode': best_electrodes['x_cold'],
            'y_cold_electrode': best_electrodes['y_cold'],
            'T_hot_electrode_avg': best_electrodes['T_hot_avg'],
            'T_cold_electrode_avg': best_electrodes['T_cold_avg'],
            'delta_T_parallel': best_electrodes['delta_T_parallel']
        })
        
    append_metadata(metadata_record)
    print(f"Finished {sim_id}. Delta T_parallel: {metadata_record.get('delta_T_parallel')}\n")
    return metadata_record

if __name__ == "__main__":
    Lx, Ly, h = 0.01, 0.01, 0.002 # 1cm x 1cm x 2mm
    k_low, k_high = 0.5, 150.0
    nx, ny, nz = 50, 50, 20
    
    print("Generating a batch of 5 structured structures for the database...")
    rng = np.random.default_rng(20260508)
    for i in range(5):
        geom = sample_structured_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, rng=rng)
        
        run_simulation_pipeline(geom, f"sim_{geom['geometry_type']}_{uuid.uuid4().hex[:8]}")
