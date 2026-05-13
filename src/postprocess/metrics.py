import numpy as np

def compute_area_average_temperature(T_surface, x_coords, y_coords, xc, yc, wx, wy):
    """
    Compute the area average temperature within a 2D window on the top surface.
    T_surface: 2D array or structured data of temperature at z=h
    x_coords, y_coords: corresponding coordinate arrays
    xc, yc: window center
    wx, wy: window dimensions
    """
    # Create mask for the window
    mask_x = (x_coords >= xc - wx/2) & (x_coords <= xc + wx/2)
    mask_y = (y_coords >= yc - wy/2) & (y_coords <= yc + wy/2)
    mask = mask_x & mask_y
    
    if not np.any(mask):
        return np.nan
        
    # Simple average for structured grids. 
    # For FEM, this should be an element surface integral.
    return np.mean(T_surface[mask])

def find_best_electrodes(T_surface, x_coords, y_coords, Lx, Ly, wx, wy, s_min):
    """
    Search for the best hot and cold electrode positions on the top surface
    to maximize Delta T_parallel.

    Vectorized: precompute 400 candidate temperatures, then use numpy
    broadcasting for pairwise distance / temperature-difference comparison.
    """
    x_centers = np.linspace(wx / 2, Lx - wx / 2, 20)
    y_centers = np.linspace(wy / 2, Ly - wy / 2, 20)

    # Step 1: precompute temperatures for all 20x20 = 400 candidates
    nc = len(x_centers) * len(y_centers)
    temps = np.empty(nc)
    pos_x = np.empty(nc)
    pos_y = np.empty(nc)

    k = 0
    for xc in x_centers:
        for yc in y_centers:
            pos_x[k] = xc
            pos_y[k] = yc
            temps[k] = compute_area_average_temperature(
                T_surface, x_coords, y_coords, xc, yc, wx, wy
            )
            k += 1

    valid = np.isfinite(temps)
    if valid.sum() < 2:
        return None
    temps, pos_x, pos_y = temps[valid], pos_x[valid], pos_y[valid]

    # Step 2: vectorised pairwise distance and |ΔT|
    ddx = pos_x[:, None] - pos_x[None, :]
    ddy = pos_y[:, None] - pos_y[None, :]
    dist = np.sqrt(ddx ** 2 + ddy ** 2)
    abs_diff = np.abs(temps[:, None] - temps[None, :])
    abs_diff[dist < s_min] = -1.0

    best_flat = int(np.argmax(abs_diff))
    i, j = np.unravel_index(best_flat, abs_diff.shape)
    if abs_diff[i, j] <= 0:
        return None
    if temps[i] < temps[j]:
        i, j = j, i

    return {
        'x_hot': float(pos_x[i]), 'y_hot': float(pos_y[i]),
        'T_hot_avg': float(temps[i]),
        'x_cold': float(pos_x[j]), 'y_cold': float(pos_y[j]),
        'T_cold_avg': float(temps[j]),
        'delta_T_parallel': float(temps[i] - temps[j]),
    }
