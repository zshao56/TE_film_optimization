import numpy as np
from scipy.ndimage import gaussian_filter

def generate_random_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, volume_fraction_target=0.5, blur_sigma=2.0, env_params=None, rng=None):
    """
    Generate a 3D random structure by applying a Gaussian blur to uniform noise,
    then thresholding it to reach a target volume fraction of the high-k material.
    
    blur_sigma can be a scalar (isotropic) or a tuple of 3 floats (anisotropic) 
    to generate directional features like pillars or lamellae.
    """
    rng = np.random.default_rng() if rng is None else rng

    # 1. Generate uniform random noise
    noise = rng.random((nx, ny, nz))
    
    # 2. Smooth the noise
    smoothed_noise = gaussian_filter(noise, sigma=blur_sigma)
    
    # 3. Find the threshold to match the desired volume fraction
    threshold = np.percentile(smoothed_noise, (1.0 - volume_fraction_target) * 100)
    
    # 4. Create boolean mask
    mask = smoothed_noise >= threshold
    
    params = {
        'geometry_type': 'random_smoothed',
        'Lx': Lx,
        'Ly': Ly,
        'h': h,
        'k_low': k_low,
        'k_high': k_high,
        'volume_fraction_target': volume_fraction_target,
        'blur_sigma': blur_sigma if isinstance(blur_sigma, float) else list(blur_sigma),
        'mask_3d': mask 
    }
    
    if env_params:
        params.update(env_params)
        
    return params
