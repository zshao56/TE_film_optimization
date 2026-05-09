import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator
import scipy.sparse.linalg as spla
import time

class Custom3DFDMSolver:
    """
    A custom 3D Finite Difference Method (FDM) solver for steady-state heat conduction.
    Uses numpy and scipy.sparse for efficient vectorized assembly and solving.
    """
    def __init__(self, geometry_params, T_hot, T_air, h_c, h_c_side, nx=40, ny=40, nz=15):
        self.geom = geometry_params
        self.T_hot = T_hot
        self.T_air = T_air
        self.h_c = h_c
        self.h_c_side = h_c_side
        
        # Grid resolution
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def _build_kappa_field(self):
        """
        Creates a 3D array of thermal conductivity based on the geometry parameters.
        """
        kappa = np.ones((self.nx, self.ny, self.nz)) * self.geom.get('k_low', 1.0)
        
        g_type = self.geom.get('geometry_type', 'uniform')
        
        if g_type == 'uniform':
            kappa *= self.geom.get('k_val', 1.0) / self.geom.get('k_low', 1.0)
            
        elif g_type == '3d_wedge':
            # Simple wedge example: High k in a diagonal half
            x = np.linspace(0, 1, self.nx)
            y = np.linspace(0, 1, self.ny)
            z = np.linspace(0, 1, self.nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            # condition for wedge
            mask = (X + Y) > 1.0
            kappa[mask] = self.geom.get('k_high', 10.0)
            
        elif 'mask_3d' in self.geom:
            # Use the explicit boolean mask passed from the geometry generator
            mask = self.geom['mask_3d']
            # Ensure the mask matches the solver resolution
            if mask.shape == (self.nx, self.ny, self.nz):
                kappa[mask] = self.geom.get('k_high', 10.0)
            else:
                raise ValueError("The shape of 'mask_3d' does not match the solver grid resolution.")
            
        return kappa

    def solve(self):
        Lx = self.geom['Lx']
        Ly = self.geom['Ly']
        h = self.geom['h']
        
        nx, ny, nz = self.nx, self.ny, self.nz
        dx = Lx / nx
        dy = Ly / ny
        dz = h / nz
        
        kappa = self._build_kappa_field()
        N = nx * ny * nz
        
        print("Constructing 3D FDM matrix...")
        t0 = time.time()
        
        row = []
        col = []
        data = []
        
        def idx(i, j, k):
            return i * ny * nz + j * nz + k
            
        def k_int(k1, k2):
            # Harmonic mean
            return 2.0 * k1 * k2 / (k1 + k2 + 1e-15)

        def get_U_eff(h_c_val, kappa_val, delta):
            return (h_c_val * kappa_val) / (kappa_val + h_c_val * delta / 2.0 + 1e-15)

        I, J, K = np.mgrid[0:nx, 0:ny, 0:nz]
        m = idx(I, J, K)
        
        A_center = np.zeros_like(kappa)
        b_matrix = np.zeros_like(kappa)
        
        # --- Internal X direction ---
        valid_x = I < nx - 1
        iv, jv, kv = I[valid_x], J[valid_x], K[valid_x]
        m1 = idx(iv, jv, kv)
        m2 = idx(iv+1, jv, kv)
        k_x = k_int(kappa[iv, jv, kv], kappa[iv+1, jv, kv])
        coeff_x = k_x / (dx**2)
        
        row.extend(m1.ravel()); col.extend(m2.ravel()); data.extend(-coeff_x.ravel())
        row.extend(m2.ravel()); col.extend(m1.ravel()); data.extend(-coeff_x.ravel())
        
        A_center[iv, jv, kv] += coeff_x
        A_center[iv+1, jv, kv] += coeff_x
        
        # --- Internal Y direction ---
        valid_y = J < ny - 1
        iv, jv, kv = I[valid_y], J[valid_y], K[valid_y]
        m1 = idx(iv, jv, kv)
        m2 = idx(iv, jv+1, kv)
        k_y = k_int(kappa[iv, jv, kv], kappa[iv, jv+1, kv])
        coeff_y = k_y / (dy**2)
        
        row.extend(m1.ravel()); col.extend(m2.ravel()); data.extend(-coeff_y.ravel())
        row.extend(m2.ravel()); col.extend(m1.ravel()); data.extend(-coeff_y.ravel())
        
        A_center[iv, jv, kv] += coeff_y
        A_center[iv, jv+1, kv] += coeff_y
        
        # --- Internal Z direction ---
        valid_z = K < nz - 1
        iv, jv, kv = I[valid_z], J[valid_z], K[valid_z]
        m1 = idx(iv, jv, kv)
        m2 = idx(iv, jv, kv+1)
        k_z = k_int(kappa[iv, jv, kv], kappa[iv, jv, kv+1])
        coeff_z = k_z / (dz**2)
        
        row.extend(m1.ravel()); col.extend(m2.ravel()); data.extend(-coeff_z.ravel())
        row.extend(m2.ravel()); col.extend(m1.ravel()); data.extend(-coeff_z.ravel())
        
        A_center[iv, jv, kv] += coeff_z
        A_center[iv, jv, kv+1] += coeff_z
        
        # --- Boundary Conditions ---
        
        # Bottom (z=0): Dirichlet T = T_hot
        k_bottom = kappa[:, :, 0]
        coeff_b = k_bottom / (dz**2 / 2.0)
        A_center[:, :, 0] += coeff_b
        b_matrix[:, :, 0] += coeff_b * self.T_hot
        
        # Top (z=H): Robin (Convection)
        k_top = kappa[:, :, -1]
        U_eff_top = get_U_eff(self.h_c, k_top, dz)
        coeff_t = U_eff_top / dz
        A_center[:, :, -1] += coeff_t
        b_matrix[:, :, -1] += coeff_t * self.T_air
        
        # Sides (x=0)
        k_x0 = kappa[0, :, :]
        U_eff_x0 = get_U_eff(self.h_c_side, k_x0, dx)
        coeff_x0 = U_eff_x0 / dx
        A_center[0, :, :] += coeff_x0
        b_matrix[0, :, :] += coeff_x0 * self.T_air

        # Sides (x=L)
        k_xL = kappa[-1, :, :]
        U_eff_xL = get_U_eff(self.h_c_side, k_xL, dx)
        coeff_xL = U_eff_xL / dx
        A_center[-1, :, :] += coeff_xL
        b_matrix[-1, :, :] += coeff_xL * self.T_air
        
        # Sides (y=0)
        k_y0 = kappa[:, 0, :]
        U_eff_y0 = get_U_eff(self.h_c_side, k_y0, dy)
        coeff_y0 = U_eff_y0 / dy
        A_center[:, 0, :] += coeff_y0
        b_matrix[:, 0, :] += coeff_y0 * self.T_air
        
        # Sides (y=L)
        k_yL = kappa[:, -1, :]
        U_eff_yL = get_U_eff(self.h_c_side, k_yL, dy)
        coeff_yL = U_eff_yL / dy
        A_center[:, -1, :] += coeff_yL
        b_matrix[:, -1, :] += coeff_yL * self.T_air
        
        # Add diagonal to matrix
        row.extend(m.ravel())
        col.extend(m.ravel())
        data.extend(A_center.ravel())
        
        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        b = b_matrix.ravel()
        
        # Performance logging for debug
        # print(f"Matrix built in {time.time() - t0:.2f} s. Solving system (N={N})...")
        t0 = time.time()
        
        # PERFORMANCE OPTIMIZATION: 
        # `spsolve` (direct solver) is too slow for 3D matrices > 40,000 nodes, especially on Windows where memory/cache behaves differently.
        # We switch to an iterative solver `bicgstab` with Incomplete LU (ILU) preconditioning.
        try:
            # 1. Create ILU preconditioner
            ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=10)
            M_x = lambda x: ilu.solve(x)
            M = spla.LinearOperator((N, N), M_x)
            
            # 2. Iterative solve
            T_vec, info = spla.bicgstab(A, b, M=M, tol=1e-5, maxiter=3000)
            
            if info > 0:
                print(f"Warning: bicgstab did not converge within maxiter ({info}). Falling back to direct solver.")
                T_vec = spsolve(A, b)
            elif info < 0:
                print(f"Warning: bicgstab illegal input/breakdown ({info}). Falling back to direct solver.")
                T_vec = spsolve(A, b)
                
        except RuntimeError:
            # If ILU fails (e.g. exactly singular matrix layout), fallback to direct solver
            print("Warning: ILU preconditioner failed. Falling back to direct solver.")
            T_vec = spsolve(A, b)
            
        # print(f"System solved in {time.time() - t0:.2f} s.")
        
        T_field = T_vec.reshape((nx, ny, nz))
        
        # Compute exact top surface temperature for metric calculation
        T_surface = T_field[:, :, -1] - U_eff_top * (T_field[:, :, -1] - self.T_air) * (dz / 2.0) / (k_top + 1e-15)
        
        x_coords = np.linspace(dx/2, Lx - dx/2, nx)
        y_coords = np.linspace(dy/2, Ly - dy/2, ny)
        z_coords = np.linspace(dz/2, h - dz/2, nz)
        
        mesh_data = {
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        }
        
        field_data = {
            'temperature': T_field,
            'temperature_surface': T_surface,
            'kappa': kappa
        }
        
        return mesh_data, field_data
        
    def cleanup(self):
        pass
