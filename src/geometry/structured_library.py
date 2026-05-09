import numpy as np


def _as_float(value):
    return float(np.asarray(value).item())


def _as_bool(value):
    return bool(np.asarray(value).item())


def _normalized_grid(nx, ny, nz):
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    z = (np.arange(nz) + 0.5) / nz
    return np.meshgrid(x, y, z, indexing="ij")


def _direction_coordinate(X, Y, direction, reverse=False):
    if direction == "x":
        coord = X
    elif direction == "y":
        coord = Y
    elif direction == "diagonal":
        coord = 0.5 * (X + Y)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return 1.0 - coord if reverse else coord


def _base_params(geometry_type, Lx, Ly, h, k_low, k_high):
    return {
        "geometry_type": geometry_type,
        "Lx": _as_float(Lx),
        "Ly": _as_float(Ly),
        "h": _as_float(h),
        "k_low": _as_float(k_low),
        "k_high": _as_float(k_high),
    }


def _finalize(mask, params, env_params=None):
    mask = np.asarray(mask, dtype=bool)
    params["mask_3d"] = mask
    params["volume_fraction_actual"] = _as_float(mask.mean())
    if env_params:
        params.update(env_params)
    return params


def generate_wedge_structure(
    Lx,
    Ly,
    h,
    k_low,
    k_high,
    nx,
    ny,
    nz,
    volume_fraction_target=0.5,
    wedge_slope=0.8,
    direction="x",
    reverse=False,
    env_params=None,
):
    """
    High-k material fills a bottom-connected wedge with thickness varying in-plane.
    This is a low-dimensional, manufacturable alternative to free random blobs.
    """
    X, Y, Z = _normalized_grid(nx, ny, nz)
    coord = _direction_coordinate(X, Y, direction, reverse=reverse)

    interface = volume_fraction_target + wedge_slope * (coord - 0.5)
    interface = np.clip(interface, 0.0, 1.0)
    mask = Z <= interface

    params = _base_params("wedge", Lx, Ly, h, k_low, k_high)
    params.update(
        {
            "volume_fraction_target": _as_float(volume_fraction_target),
            "wedge_slope": _as_float(wedge_slope),
            "direction": direction,
            "reverse": _as_bool(reverse),
        }
    )
    return _finalize(mask, params, env_params)


def generate_curved_wedge_structure(
    Lx,
    Ly,
    h,
    k_low,
    k_high,
    nx,
    ny,
    nz,
    base_fraction=0.1,
    max_fraction=0.9,
    exponent=2.0,
    direction="x",
    reverse=False,
    env_params=None,
):
    """
    High-k material fills a bottom-connected region defined by a polynomial curve.
    exponent = 1.0 -> straight line (linear wedge)
    exponent > 1.0 -> concave curve (dips down, thinner in the middle)
    exponent < 1.0 -> convex curve (bulges up, thicker in the middle)
    """
    X, Y, Z = _normalized_grid(nx, ny, nz)
    coord = _direction_coordinate(X, Y, direction, reverse=reverse)

    interface = base_fraction + (max_fraction - base_fraction) * (coord ** exponent)
    interface = np.clip(interface, 0.0, 1.0)
    mask = Z <= interface

    params = _base_params("curved_wedge", Lx, Ly, h, k_low, k_high)
    params.update(
        {
            "base_fraction": _as_float(base_fraction),
            "max_fraction": _as_float(max_fraction),
            "exponent": _as_float(exponent),
            "direction": direction,
            "reverse": _as_bool(reverse),
            "volume_fraction_target": _as_float(mask.mean()),
        }
    )
    return _finalize(mask, params, env_params)


def generate_step_structure(
    Lx,
    Ly,
    h,
    k_low,
    k_high,
    nx,
    ny,
    nz,
    step_position=0.5,
    low_thickness_fraction=0.25,
    high_thickness_fraction=0.75,
    direction="x",
    reverse=False,
    env_params=None,
):
    """
    Piecewise-constant wedge: two bottom-connected high-k thickness regions.
    """
    X, Y, Z = _normalized_grid(nx, ny, nz)
    coord = _direction_coordinate(X, Y, direction, reverse=reverse)
    interface = np.where(coord < step_position, low_thickness_fraction, high_thickness_fraction)
    mask = Z <= interface

    params = _base_params("step", Lx, Ly, h, k_low, k_high)
    params.update(
        {
            "step_position": _as_float(step_position),
            "low_thickness_fraction": _as_float(low_thickness_fraction),
            "high_thickness_fraction": _as_float(high_thickness_fraction),
            "direction": direction,
            "reverse": _as_bool(reverse),
            "volume_fraction_target": _as_float(mask.mean()),
        }
    )
    return _finalize(mask, params, env_params)


def generate_double_layer_structure(
    Lx,
    Ly,
    h,
    k_low,
    k_high,
    nx,
    ny,
    nz,
    split_fraction=0.5,
    bottom_width_fraction=0.55,
    top_width_fraction=0.55,
    bridge_width_fraction=0.12,
    direction="x",
    reverse=False,
    env_params=None,
):
    """
    Two laterally offset high-k layers with a narrow vertical bridge.

    The offset makes the family more useful for in-plane temperature splitting than
    a perfectly uniform horizontal bilayer, which would be nearly symmetric.
    """
    X, Y, Z = _normalized_grid(nx, ny, nz)
    coord = _direction_coordinate(X, Y, direction, reverse=reverse)

    bottom_layer = (Z <= split_fraction) & (coord <= bottom_width_fraction)
    top_layer = (Z > split_fraction) & (coord >= 1.0 - top_width_fraction)
    bridge_center = 0.5 * (bottom_width_fraction + (1.0 - top_width_fraction))
    bridge = (
        np.abs(coord - bridge_center) <= 0.5 * bridge_width_fraction
    ) & (
        np.abs(Z - split_fraction) <= max(0.5 / nz, 0.04)
    )
    mask = bottom_layer | top_layer | bridge

    params = _base_params("double_layer", Lx, Ly, h, k_low, k_high)
    params.update(
        {
            "split_fraction": _as_float(split_fraction),
            "bottom_width_fraction": _as_float(bottom_width_fraction),
            "top_width_fraction": _as_float(top_width_fraction),
            "bridge_width_fraction": _as_float(bridge_width_fraction),
            "direction": direction,
            "reverse": _as_bool(reverse),
            "volume_fraction_target": _as_float(mask.mean()),
        }
    )
    return _finalize(mask, params, env_params)


def generate_arc_structure(
    Lx,
    Ly,
    h,
    k_low,
    k_high,
    nx,
    ny,
    nz,
    center_fraction=0.5,
    radius_fraction=0.45,
    base_height_fraction=0.2,
    arc_height_fraction=0.55,
    channel_half_width_fraction=0.07,
    direction="x",
    reverse=False,
    env_params=None,
):
    """
    High-k arcuate channel extruded through the transverse direction.
    """
    X, Y, Z = _normalized_grid(nx, ny, nz)
    coord = _direction_coordinate(X, Y, direction, reverse=reverse)

    normalized_radius = (coord - center_fraction) / max(radius_fraction, 1e-12)
    inside_span = np.abs(normalized_radius) <= 1.0
    curve = base_height_fraction + arc_height_fraction * (1.0 - normalized_radius**2)
    curve = np.clip(curve, 0.0, 1.0)

    arc_channel = inside_span & (np.abs(Z - curve) <= channel_half_width_fraction)

    left_anchor_x = center_fraction - radius_fraction
    right_anchor_x = center_fraction + radius_fraction
    left_curve_height = np.clip(base_height_fraction, 0.0, 1.0)
    right_curve_height = np.clip(base_height_fraction, 0.0, 1.0)
    anchor_width = max(channel_half_width_fraction, 1.0 / max(nx, ny))
    anchors = (
        (np.abs(coord - left_anchor_x) <= anchor_width) & (Z <= left_curve_height + channel_half_width_fraction)
    ) | (
        (np.abs(coord - right_anchor_x) <= anchor_width) & (Z <= right_curve_height + channel_half_width_fraction)
    )

    mask = arc_channel | anchors

    params = _base_params("arc", Lx, Ly, h, k_low, k_high)
    params.update(
        {
            "center_fraction": _as_float(center_fraction),
            "radius_fraction": _as_float(radius_fraction),
            "base_height_fraction": _as_float(base_height_fraction),
            "arc_height_fraction": _as_float(arc_height_fraction),
            "channel_half_width_fraction": _as_float(channel_half_width_fraction),
            "direction": direction,
            "reverse": _as_bool(reverse),
            "volume_fraction_target": _as_float(mask.mean()),
        }
    )
    return _finalize(mask, params, env_params)


STRUCTURED_FAMILIES = ("wedge", "curved_wedge", "step", "double_layer", "arc")


def sample_structured_structure(Lx, Ly, h, k_low, k_high, nx, ny, nz, env_params=None, family=None, rng=None):
    """
    Sample one structure from a low-dimensional library of physically interpretable
    geometry families.
    """
    rng = np.random.default_rng() if rng is None else rng
    if family is None:
        family = rng.choice(STRUCTURED_FAMILIES)
    family = str(family)

    direction = str(rng.choice(["x", "y", "diagonal"]))
    reverse = _as_bool(rng.integers(0, 2))

    if family == "wedge":
        return generate_wedge_structure(
            Lx,
            Ly,
            h,
            k_low,
            k_high,
            nx,
            ny,
            nz,
            volume_fraction_target=rng.uniform(0.25, 0.75),
            wedge_slope=rng.uniform(0.35, 1.25),
            direction=direction,
            reverse=reverse,
            env_params=env_params,
        )

    if family == "curved_wedge":
        return generate_curved_wedge_structure(
            Lx,
            Ly,
            h,
            k_low,
            k_high,
            nx,
            ny,
            nz,
            base_fraction=rng.uniform(0.0, 0.3),
            max_fraction=rng.uniform(0.7, 1.0),
            exponent=rng.uniform(0.5, 4.0),
            direction=direction,
            reverse=reverse,
            env_params=env_params,
        )

    if family == "step":
        low_fraction = rng.uniform(0.10, 0.45)
        high_fraction = rng.uniform(0.55, 0.95)
        return generate_step_structure(
            Lx,
            Ly,
            h,
            k_low,
            k_high,
            nx,
            ny,
            nz,
            step_position=rng.uniform(0.25, 0.75),
            low_thickness_fraction=low_fraction,
            high_thickness_fraction=high_fraction,
            direction=direction,
            reverse=reverse,
            env_params=env_params,
        )

    if family == "double_layer":
        return generate_double_layer_structure(
            Lx,
            Ly,
            h,
            k_low,
            k_high,
            nx,
            ny,
            nz,
            split_fraction=rng.uniform(0.35, 0.65),
            bottom_width_fraction=rng.uniform(0.40, 0.75),
            top_width_fraction=rng.uniform(0.40, 0.75),
            bridge_width_fraction=rng.uniform(0.06, 0.18),
            direction=str(rng.choice(["x", "y"])),
            reverse=reverse,
            env_params=env_params,
        )

    if family == "arc":
        radius_fraction = rng.uniform(0.28, 0.42)
        center_low = radius_fraction + 0.05
        center_high = 0.95 - radius_fraction
        center_fraction = 0.5 if center_low >= center_high else rng.uniform(center_low, center_high)
        return generate_arc_structure(
            Lx,
            Ly,
            h,
            k_low,
            k_high,
            nx,
            ny,
            nz,
            center_fraction=center_fraction,
            radius_fraction=radius_fraction,
            base_height_fraction=rng.uniform(0.08, 0.28),
            arc_height_fraction=rng.uniform(0.35, 0.70),
            channel_half_width_fraction=rng.uniform(0.04, 0.10),
            direction=str(rng.choice(["x", "y"])),
            reverse=reverse,
            env_params=env_params,
        )

    raise ValueError(f"Unknown structured family: {family}")
