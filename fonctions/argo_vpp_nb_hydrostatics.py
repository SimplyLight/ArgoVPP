from __future__ import annotations

from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
import numpy as np
import pandas as pd
import trimesh
from shapely.ops import triangulate, unary_union
from .argo_vpp_nb_mesh import (
    immersed_mesh_for_waterline,
    safe_mesh_center,
    safe_mesh_volume,
    surface_area_centroid,
    waterplane_polygon_from_loops,
)
from .argo_vpp_nb_models import (
    HullHydrostatics,
    safe_div,
)
from .argo_vpp_nb_tables import (
    fn_domain,
)

__all__ = ['DELFT_LIMITS', 'check_delft_domain', 'compute_hull_hydrostatics', 'compute_lateral_projected_area', 'section_area_at_x']

def section_area_at_x(mesh: trimesh.Trimesh, x_station: float, waterline_z: float) -> float:
    """Submerged transverse section area at x, clipping section segments at z=waterline_z."""

    clipped_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in mesh.triangles:
        x = tri[:, 0]
        if x_station < x.min() - 1e-12 or x_station > x.max() + 1e-12:
            continue
        intersections: list[np.ndarray] = []
        for i in range(3):
            p0 = tri[i]
            p1 = tri[(i + 1) % 3]
            d0 = p0[0] - x_station
            d1 = p1[0] - x_station
            if abs(d0) <= 1e-10:
                intersections.append(p0)
            if d0 * d1 < -1e-20:
                t = -d0 / (d1 - d0)
                intersections.append(p0 + t * (p1 - p0))
        unique: list[np.ndarray] = []
        for point in intersections:
            if not any(np.linalg.norm(point[[1, 2]] - other[[1, 2]]) < 1e-8 for other in unique):
                unique.append(point)
        if len(unique) < 2:
            continue
        p0 = unique[0][[1, 2]]
        p1 = unique[1][[1, 2]]
        z0 = p0[1]
        z1 = p1[1]
        below0 = z0 <= waterline_z + 1e-10
        below1 = z1 <= waterline_z + 1e-10
        if below0 and below1:
            clipped_segments.append((p0, p1))
        elif below0 != below1 and abs(z1 - z0) > 1e-14:
            t = (waterline_z - z0) / (z1 - z0)
            pint = p0 + t * (p1 - p0)
            clipped_segments.append((p0, pint) if below0 else (pint, p1))
    if not clipped_segments:
        return 0.0
    points = np.vstack([np.asarray(seg) for seg in clipped_segments])
    if len(points) < 3:
        return 0.0
    hull = MultiPoint(points).convex_hull
    return 0.0 if hull.is_empty else float(hull.area)

DELFT_LIMITS = {
    "LWL/BWL": (2.73, 5.00),
    "BWL/TC": (2.46, 19.38),
    "LWL/volume^(1/3)": (4.34, 8.50),
    "C_P": (0.52, 0.60),
    "C_M": (0.65, 0.78),
    "A_WP/volume^(2/3)": (3.78, 12.67),
    "LCB_pct_midship": (-8.2, 0.0),
    "LCF_pct_midship": (-9.5, -1.8),
}

def compute_lateral_projected_area(meshes: list[trimesh.Trimesh], waterline_z: float) -> float:
    """Projected immersed lateral area on the x-z plane without double counting overlap."""

    polys = []
    for mesh in meshes:
        immersed, _, wet_triangles = immersed_mesh_for_waterline(mesh, waterline_z, cap=False)
        triangles = wet_triangles if len(wet_triangles) else immersed.triangles
        for tri in triangles:
            xz = tri[:, [0, 2]]
            poly = Polygon(xz)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 1e-12:
                polys.append(poly)
    if not polys:
        return 0.0
    return float(unary_union(polys).area)

def check_delft_domain(hydro_values: dict[str, float], tables: dict[str, pd.DataFrame] | None = None) -> tuple[bool, list[str]]:
    """Check Delft validity domain and return visible warnings."""

    warnings_out: list[str] = []
    for name, bounds in DELFT_LIMITS.items():
        value = hydro_values.get(name, np.nan)
        lo, hi = bounds
        if not np.isfinite(value) or value < lo or value > hi:
            warnings_out.append(f"{name} = {value:.5g} hors domaine Delft [{lo}, {hi}]")
    if tables:
        for key in ["bare_hull", "keel", "heeled_residuary"]:
            lo, hi = fn_domain(tables, key)
            if not np.isfinite(lo) or not np.isfinite(hi):
                warnings_out.append(f"Domaine Fn non determine pour {key}")
    return len(warnings_out) == 0, warnings_out

def compute_hull_hydrostatics(mesh: trimesh.Trimesh, draft_m: float, tables: dict[str, pd.DataFrame], section_count: int = 180) -> tuple[HullHydrostatics, trimesh.Trimesh, list[np.ndarray], np.ndarray]:
    """Compute hydrostatic quantities from the upright hull STL and imposed draft."""

    if draft_m <= 0:
        raise ValueError("Le tirant d'eau de coque doit etre strictement positif.")
    z_min = float(mesh.bounds[0, 2])
    z_wl = z_min + float(draft_m)
    z_max = float(mesh.bounds[1, 2])
    if z_wl >= z_max:
        max_draft = z_max - z_min
        raise ValueError(f"Tirant d'eau T_C={draft_m:.4f} m trop grand pour ce STL: hauteur disponible {max_draft:.4f} m. Choisir T_C < {max_draft:.4f} m.")
    immersed, loops, wet_triangles = immersed_mesh_for_waterline(mesh, z_wl, cap=True)
    if len(wet_triangles) == 0:
        raise ValueError("Aucune surface immergee detectee au tirant d'eau impose.")
    volume = safe_mesh_volume(immersed)
    if volume <= 0:
        raise ValueError("Volume immerge nul ou negatif apres fermeture du plan d'eau.")

    S_C, _ = surface_area_centroid(wet_triangles)
    waterplane = waterplane_polygon_from_loops(loops)
    A_WP = float(waterplane.area) if not waterplane.is_empty else 0.0
    if A_WP <= 0:
        raise ValueError("Aire de flottaison nulle: verifier le STL et le tirant d'eau.")
    centroid = waterplane.centroid
    LCF = float(centroid.x)
    LCB = float(safe_mesh_center(immersed, wet_triangles)[0])
    minx, miny, _ = mesh.bounds[0]
    maxx, maxy, _ = mesh.bounds[1]
    if not waterplane.is_empty:
        wb = np.asarray(waterplane.bounds, dtype=float)
        x_min_wl, y_min_wl, x_max_wl, y_max_wl = wb
    else:
        x_min_wl, y_min_wl, x_max_wl, y_max_wl = minx, miny, maxx, maxy
    LWL = float(x_max_wl - x_min_wl)
    BWL = float(y_max_wl - y_min_wl)
    x_mid_wl = 0.5 * (x_min_wl + x_max_wl)
    LCB_pct_midship = 100.0 * safe_div(LCB - x_mid_wl, LWL, default=np.nan)
    LCF_pct_midship = 100.0 * safe_div(LCF - x_mid_wl, LWL, default=np.nan)
    x_stations = np.linspace(x_min_wl, x_max_wl, max(50, int(section_count)))
    areas = np.array([section_area_at_x(mesh, x, z_wl) for x in x_stations])
    AMS_CB = float(np.nanmax(areas)) if len(areas) else np.nan
    C_M = safe_div(AMS_CB, BWL * draft_m)
    C_P = safe_div(volume, AMS_CB * LWL)
    x_FP = float(x_max_wl)
    LCB_fpp = x_FP - LCB
    LCF_fpp = x_FP - LCF
    B_CB = BWL
    values = {
        "LWL/BWL": safe_div(LWL, BWL),
        "BWL/TC": safe_div(BWL, draft_m),
        "LWL/volume^(1/3)": safe_div(LWL, volume ** (1 / 3)),
        "C_P": C_P,
        "C_M": C_M,
        "A_WP/volume^(2/3)": safe_div(A_WP, volume ** (2 / 3)),
        "LCB_pct_midship": LCB_pct_midship,
        "LCF_pct_midship": LCF_pct_midship,
    }
    valid, delft_warnings = check_delft_domain(values, tables)
    hydro = HullHydrostatics(
        LWL=LWL,
        BWL=BWL,
        TC=float(draft_m),
        volume_C=volume,
        S_C=S_C,
        A_WP=A_WP,
        LCB=LCB,
        LCF=LCF,
        AMS_CB=AMS_CB,
        C_M=C_M,
        C_P=C_P,
        x_FP=x_FP,
        LCB_fpp=LCB_fpp,
        LCF_fpp=LCF_fpp,
        LCB_pct_midship=LCB_pct_midship,
        LCF_pct_midship=LCF_pct_midship,
        waterline_z=z_wl,
        B_CB=B_CB,
        delft_valid=valid,
        delft_warnings=delft_warnings,
    )
    return hydro, immersed, loops, wet_triangles
