import numpy as np

class Dummy3DSolver:
    """
    A placeholder for the actual 3D FEM/FVM solver (e.g., COMSOL API, FEniCS).
    """
    def __init__(self, geometry_params, T_hot, T_air, h_c, h_c_side):
        self.geom = geometry_params
        self.T_hot = T_hot
        self.T_air = T_air
        self.h_c = h_c
        self.h_c_side = h_c_side
        
    def solve(self):
        """
        Mock solver that returns dummy 3D fields.
        """
        Lx = self.geom['Lx']
        Ly = self.geom['Ly']
        h = self.geom['h']
        
        # Mock mesh
        nx, ny, nz = 50, 50, 10
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, h, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Mock temperature field: linear gradient from T_hot at bottom to T_air at top
        T_field = self.T_hot - (self.T_hot - self.T_air) * (Z / h)
        
        # Mock top surface temperature for electrode measurement
        T_surface = T_field[:, :, -1]
        
        # Add some dummy variation if it's a wedge, so Delta T is not exactly 0
        if self.geom['geometry_type'] == '3d_wedge':
            T_surface += np.sin(X/Lx * np.pi) * 5.0
            
        mesh_data = {
            'x': x,
            'y': y,
            'z': z
        }
        
        field_data = {
            'temperature': T_field,
            'temperature_surface': T_surface
        }
        
        return mesh_data, field_data
