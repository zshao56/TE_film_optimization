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
    Search for the best hot and cold electrode positions on the top surface to maximize Delta T_parallel.
    """
    # Define search grid (e.g., discretize the valid area)
    x_centers = np.linspace(wx/2, Lx - wx/2, 20)
    y_centers = np.linspace(wy/2, Ly - wy/2, 20)
    
    best_diff = -1
    best_hot = None
    best_cold = None
    
    candidates = []
    
    for xc1 in x_centers:
        for yc1 in y_centers:
            T1 = compute_area_average_temperature(T_surface, x_coords, y_coords, xc1, yc1, wx, wy)
            if np.isnan(T1): continue
            
            for xc2 in x_centers:
                for yc2 in y_centers:
                    # Check minimum distance constraint
                    dist = np.sqrt((xc1 - xc2)**2 + (yc1 - yc2)**2)
                    if dist < s_min:
                        continue
                        
                    T2 = compute_area_average_temperature(T_surface, x_coords, y_coords, xc2, yc2, wx, wy)
                    if np.isnan(T2): continue
                    
                    diff = abs(T1 - T2)
                    if diff > best_diff:
                        best_diff = diff
                        if T1 >= T2:
                            best_hot = (xc1, yc1, T1)
                            best_cold = (xc2, yc2, T2)
                        else:
                            best_hot = (xc2, yc2, T2)
                            best_cold = (xc1, yc1, T1)
                            
    if best_hot and best_cold:
        return {
            'x_hot': best_hot[0], 'y_hot': best_hot[1], 'T_hot_avg': best_hot[2],
            'x_cold': best_cold[0], 'y_cold': best_cold[1], 'T_cold_avg': best_cold[2],
            'delta_T_parallel': best_hot[2] - best_cold[2]
        }
    return None
