from __future__ import annotations

import numpy as np
import scipy.optimize as opt
import pandas as pd
import trimesh
from .argo_vpp_nb_mesh import (
    immersed_mesh_for_waterline,
    safe_mesh_center,
    safe_mesh_volume,
)
from .argo_vpp_nb_models import (
    C,
    HEEL_RANGE_MAX_DEG,
    HEEL_RANGE_MIN_DEG,
    MassProperties,
    PhysicalConstants,
    rad_to_deg,
    safe_div,
)
from .argo_vpp_nb_tables import (
    TableDomainError,
    bounded_interpolate_1d,
)

__all__ = ['analysis_angle_values', 'combine_meshes', 'compute_gz_curve', 'find_waterline_for_target_volume', 'interpolate_gz', 'interpolate_gz_state', 'moment_about_x', 'rotate_mesh_about_x', 'rotate_points_about_x']

def analysis_angle_values(min_deg: float, max_deg: float, steps: int) -> np.ndarray:
    """Return an inclusive angle sweep defined by min/max and interval count."""

    if not np.isfinite(min_deg) or not np.isfinite(max_deg) or max_deg < min_deg:
        raise ValueError(f"Invalid angle range: min={min_deg}, max={max_deg}.")
    if int(steps) != steps or steps < 1:
        raise ValueError(f"Angle sweep steps must be a positive integer; got {steps!r}.")
    return np.linspace(float(min_deg), float(max_deg), int(steps) + 1)

def rotate_points_about_x(points: np.ndarray, origin: np.ndarray, phi_rad: float) -> np.ndarray:
    """Rotate points around the longitudinal x axis through origin."""

    pts = np.asarray(points, dtype=float) - origin
    cphi = np.cos(phi_rad)
    sphi = np.sin(phi_rad)
    out = pts.copy()
    out[:, 1] = cphi * pts[:, 1] - sphi * pts[:, 2]
    out[:, 2] = sphi * pts[:, 1] + cphi * pts[:, 2]
    return out + origin

def rotate_mesh_about_x(mesh: trimesh.Trimesh, origin: np.ndarray, phi_rad: float) -> trimesh.Trimesh:
    """Return a rotated copy of a mesh around the boat x axis."""

    vertices = rotate_points_about_x(mesh.vertices, origin, phi_rad)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)

def combine_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Concatenate meshes without changing their coordinates."""

    return trimesh.util.concatenate(meshes)

def find_waterline_for_target_volume(mesh: trimesh.Trimesh, target_volume: float, z_bounds: tuple[float, float], tol: float = 1e-7) -> tuple[float, trimesh.Trimesh]:
    """Find horizontal waterline that gives target immersed volume."""

    z_low, z_high = z_bounds

    def volume_at(z: float) -> float:
        immersed, _, _ = immersed_mesh_for_waterline(mesh, z, cap=True)
        return safe_mesh_volume(immersed)

    v_low = volume_at(z_low)
    v_high = volume_at(z_high)
    if target_volume < v_low - tol or target_volume > v_high + tol:
        raise ValueError(
            f"Volume cible {target_volume:.6g} m3 hors bornes [{v_low:.6g}, {v_high:.6g}] pour la recherche de flottaison."
        )

    def residual(z: float) -> float:
        return volume_at(z) - target_volume

    z_star = float(opt.brentq(residual, z_low, z_high, xtol=tol, rtol=tol, maxiter=80))
    immersed, _, _ = immersed_mesh_for_waterline(mesh, z_star, cap=True)
    return z_star, immersed

def moment_about_x(point: np.ndarray, force: np.ndarray, origin: np.ndarray) -> float:
    """Signed x-axis moment of a force applied at point around origin."""

    return float(np.cross(np.asarray(point) - np.asarray(origin), np.asarray(force))[0])

def compute_gz_curve(
    buoyancy_mesh: trimesh.Trimesh,
    masses: MassProperties,
    target_volume_m3: float,
    constants: PhysicalConstants = C,
    max_angle_deg: float = HEEL_RANGE_MAX_DEG,
    min_angle_deg: float = HEEL_RANGE_MIN_DEG,
    step_deg: float | None = 1.0,
    angle_count: int | None = None,
    waterline_tol: float = 1e-7,
) -> pd.DataFrame:
    """Compute signed GZ curve and store the waterline for every heel angle."""

    rows = []
    G = masses.center_of_gravity
    weight_total = masses.total_mass_kg * constants.G
    component_positions = masses.component_table[["x_G_m", "y_G_m", "z_G_m"]].to_numpy(dtype=float)
    component_masses = masses.component_table["mass_kg"].to_numpy(dtype=float)
    component_names = list(masses.component_table["component"])
    if angle_count is not None:
        angles = np.linspace(float(min_angle_deg), float(max_angle_deg), max(2, int(angle_count)))
    else:
        step = float(step_deg or 1.0)
        steps = max(1, int(round((float(max_angle_deg) - float(min_angle_deg)) / step)))
        angles = analysis_angle_values(min_angle_deg, max_angle_deg, steps)

    for phi_deg in angles:
        phi_rad = float(np.deg2rad(phi_deg))
        rotated = rotate_mesh_about_x(buoyancy_mesh, G, phi_rad)
        z_min = float(rotated.bounds[0, 2]) - 0.05 * max(rotated.extents[2], 1e-6)
        z_max = float(rotated.bounds[1, 2]) + 0.05 * max(rotated.extents[2], 1e-6)
        z_wl, immersed = find_waterline_for_target_volume(rotated, target_volume_m3, (z_min, z_max), tol=waterline_tol)
        B = safe_mesh_center(immersed)
        buoyancy = np.array([0.0, 0.0, constants.RHO_WATER * constants.G * target_volume_m3])
        M_buoyancy = moment_about_x(B, buoyancy, G)
        weight_moments = {}
        rotated_components = rotate_points_about_x(component_positions, G, phi_rad)
        for name, mass, pos in zip(component_names, component_masses, rotated_components):
            weight_force = np.array([0.0, 0.0, -mass * constants.G])
            weight_moments[f"M_{name}_Nm"] = moment_about_x(pos, weight_force, G)
        M_total_signed = M_buoyancy + sum(weight_moments.values())
        M_restoring = 0.0 if abs(phi_rad) < 1e-12 else -np.sign(phi_rad) * M_total_signed
        GZ_signed = safe_div(M_restoring, weight_total, default=np.nan)
        rows.append({
            "heel_deg": float(phi_deg),
            "phi_deg": float(phi_deg),
            "phi_rad": phi_rad,
            "waterline_z_m": z_wl,
            "volume_m3": safe_mesh_volume(immersed),
            "B_x_m": float(B[0]),
            "B_y_m": float(B[1]),
            "B_z_m": float(B[2]),
            "GZ_m": GZ_signed,
            "GZ_signed_m": GZ_signed,
            "M_hydrostatic_Nm": M_restoring,
            "M_hydrostatic_raw_signed_Nm": M_total_signed,
            "M_hydrostatic_restoring_Nm": M_restoring,
            "M_buoyancy_Nm": M_buoyancy,
            **weight_moments,
        })
        print(phi_deg)
    return pd.DataFrame(rows)

def interpolate_gz_state(gz_curve: pd.DataFrame, heel_deg: float) -> dict[str, float]:
    """Interpolate signed righting moment, GZ and waterline at a given heel angle."""

    phi = float(heel_deg)
    if phi < -1e-12:
        raise TableDomainError("GZ: phi negatif non supporte par la convention actuelle.")
    angle_col = "heel_deg" if "heel_deg" in gz_curve.columns else "phi_deg"
    out = {"heel_deg": phi}
    for src, dst in [
        ("GZ_signed_m", "GZ_signed_m"),
        ("M_hydrostatic_restoring_Nm", "M_hydrostatic_restoring_Nm"),
        ("M_hydrostatic_raw_signed_Nm", "M_hydrostatic_raw_signed_Nm"),
        ("waterline_z_m", "waterline_z_m"),
        ("B_x_m", "B_x_m"),
        ("B_y_m", "B_y_m"),
        ("B_z_m", "B_z_m"),
    ]:
        if src in gz_curve.columns:
            x_values = gz_curve[angle_col].to_numpy(dtype=float)
            y_values = gz_curve[src].to_numpy(dtype=float)
            if np.count_nonzero(np.isfinite(x_values) & np.isfinite(y_values)) < 2:
                out[dst] = np.nan
                continue
            out[dst] = float(bounded_interpolate_1d(x_values, y_values, phi, "GZ", angle_col))
    return out

def interpolate_gz(gz_curve: pd.DataFrame, phi_rad: float) -> tuple[float, float]:
    """Compatibility wrapper returning signed GZ and hydrostatic righting moment."""

    state = interpolate_gz_state(gz_curve, rad_to_deg(phi_rad))
    return state.get("GZ_signed_m", np.nan), state.get("M_hydrostatic_restoring_Nm", np.nan)
