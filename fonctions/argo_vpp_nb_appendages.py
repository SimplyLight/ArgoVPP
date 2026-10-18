from __future__ import annotations

from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
import numpy as np
import pandas as pd
import trimesh
import warnings
from .argo_vpp_nb_mesh import (
    safe_mesh_center,
    safe_mesh_volume,
    surface_area_centroid,
)
from .argo_vpp_nb_models import (
    FoilGeometry,
    HullHydrostatics,
    safe_div,
)
from .argo_vpp_nb_tables import (
    naca4_coordinates,
    normalize_naca_code,
)

__all__ = ['automatic_positions', 'build_foil_mesh_from_contour', 'chord_at_span', 'hull_z_at_x', 'parse_contour_text', 'planform_properties']

def parse_contour_text(text: str, label: str = "contour") -> np.ndarray:
    """Parse a closed x_local,span_local contour from a multiline text area."""

    points: list[tuple[float, float]] = []
    for line_no, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.replace(";", ",").split(",")]
        if len(parts) != 2:
            raise ValueError(f"{label}: ligne {line_no} invalide, format attendu x,y")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise ValueError(f"{label}: au moins trois points sont necessaires.")
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-10:
        pts = np.vstack([pts, pts[0]])
    poly = Polygon(pts)
    if not poly.is_valid:
        raise ValueError(f"{label}: contour non valide ou auto-intersectant.")
    if abs(poly.area) <= 1e-12:
        raise ValueError(f"{label}: aire en plan nulle.")
    return pts

def chord_at_span(contour: np.ndarray, span_value: float) -> tuple[float, float, float]:
    """Return leading edge, trailing edge and chord at a span station."""

    xs: list[float] = []
    pts = np.asarray(contour, dtype=float)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        s0, s1 = p0[1], p1[1]
        if abs(s0 - s1) < 1e-12:
            if abs(span_value - s0) < 1e-10:
                xs.extend([p0[0], p1[0]])
            continue
        if (span_value >= min(s0, s1) - 1e-10) and (span_value <= max(s0, s1) + 1e-10):
            t = (span_value - s0) / (s1 - s0)
            if -1e-10 <= t <= 1 + 1e-10:
                xs.append(float(p0[0] + t * (p1[0] - p0[0])))
    xs_unique = sorted(set(round(x, 10) for x in xs))
    if len(xs_unique) < 2:
        return np.nan, np.nan, 0.0
    le = float(min(xs_unique))
    te = float(max(xs_unique))
    return le, te, te - le

def planform_properties(contour: np.ndarray, sections: int = 100) -> tuple[float, np.ndarray, float, pd.DataFrame]:
    """Compute planform area, centroid, mean chord and chord bands."""

    poly = Polygon(contour)
    area = abs(float(poly.area))
    centroid = np.array([float(poly.centroid.x), float(poly.centroid.y)])
    span_min = float(np.min(contour[:, 1]))
    span_max = float(np.max(contour[:, 1]))
    span = span_max - span_min
    stations = np.linspace(span_min, span_max, max(50, sections))
    rows = []
    for s in stations:
        le, te, chord = chord_at_span(contour, float(s))
        rows.append({"span": float(s), "leading_edge": le, "trailing_edge": te, "chord": chord})
    bands = pd.DataFrame(rows)
    mean_chord = safe_div(area, span, default=0.0)
    return area, centroid, mean_chord, bands

def build_foil_mesh_from_contour(
    name: str,
    contour: np.ndarray,
    naca: str,
    root_position: np.ndarray,
    direction: str,
    profile_points: int = 100,
    span_sections: int = 60,
) -> FoilGeometry:
    """Generate a watertight NACA solid from a planform contour.

    Local contour coordinates are chordwise x and span. Thickness is placed along global y.
    For `direction='up'`, span is mapped to +z; for `direction='down'`, span is mapped to -z.
    """

    if direction not in {"up", "down"}:
        raise ValueError("direction doit valoir 'up' ou 'down'.")
    contour = np.asarray(contour, dtype=float)
    area, centroid2, mean_chord, chord_sections = planform_properties(contour, sections=span_sections)
    span_min = float(np.min(contour[:, 1]))
    span_max = float(np.max(contour[:, 1]))
    span = span_max - span_min
    if span <= 0 or mean_chord <= 0:
        raise ValueError(f"{name}: contour impossible, span ou corde moyenne nul.")

    profile, thickness_ratio = naca4_coordinates(naca, n=max(80, profile_points))
    stations = np.linspace(span_min, span_max, max(50, span_sections))
    vertices: list[np.ndarray] = []
    valid_sections: list[int] = []
    section_meta = []
    sign = 1.0 if direction == "up" else -1.0
    for s in stations:
        le, te, chord = chord_at_span(contour, float(s))
        if chord <= 1e-8 or not np.isfinite(chord):
            continue
        z_global = root_position[2] + sign * (s - span_min)
        xs = root_position[0] + le + profile[:, 0] * chord
        ys = root_position[1] + profile[:, 1] * chord
        zs = np.full_like(xs, z_global)
        valid_sections.append(len(vertices))
        vertices.extend(np.column_stack([xs, ys, zs]))
        section_meta.append((s, le, te, chord))
    n_profile = len(profile)
    if len(valid_sections) < 2:
        raise ValueError(f"{name}: pas assez de sections valides pour generer le solide.")

    faces: list[list[int]] = []
    n_sections = len(valid_sections)
    for j in range(n_sections - 1):
        base0 = j * n_profile
        base1 = (j + 1) * n_profile
        for i in range(n_profile - 1):
            faces.append([base0 + i, base0 + i + 1, base1 + i + 1])
            faces.append([base0 + i, base1 + i + 1, base1 + i])
    # Root and tip caps
    for base, reverse in [(0, True), ((n_sections - 1) * n_profile, False)]:
        center = len(vertices)
        vertices.append(np.mean(np.asarray(vertices[base:base + n_profile]), axis=0))
        for i in range(n_profile - 1):
            tri = [center, base + i, base + i + 1]
            faces.append(tri[::-1] if reverse else tri)

    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=int), process=True)
    if not mesh.is_watertight:
        warnings.warn(f"{name}: le solide genere n'est pas parfaitement etanche selon trimesh.")
    external_area = float(mesh.area)
    volume = safe_mesh_volume(mesh)
    _, surface_centroid = surface_area_centroid(mesh.triangles)
    aspect_ratio = safe_div(span**2, area, default=np.nan)
    wetted_area = external_area
    return FoilGeometry(
        name=name,
        contour=contour,
        naca=normalize_naca_code(naca),
        planform_area_m2=area,
        external_area_m2=external_area,
        wetted_area_m2=wetted_area,
        span_m=span,
        mean_chord_m=mean_chord,
        volume_m3=volume,
        volume_centroid=safe_mesh_center(mesh),
        surface_centroid=np.asarray(surface_centroid, dtype=float),
        aspect_ratio=aspect_ratio,
        thickness_ratio=thickness_ratio,
        root_position=np.asarray(root_position, dtype=float),
        mesh=mesh,
        chord_sections=chord_sections,
    )

def hull_z_at_x(mesh: trimesh.Trimesh, x: float, top: bool) -> float:
    """Estimate hull bottom or top z near a longitudinal station."""

    vertices = mesh.vertices
    length = max(mesh.extents[0], 1e-6)
    window = 0.04 * length
    mask = np.abs(vertices[:, 0] - x) <= window
    if not np.any(mask):
        idx = int(np.argmin(np.abs(vertices[:, 0] - x)))
        return float(vertices[idx, 2])
    z_values = vertices[mask, 2]
    return float(np.max(z_values) if top else np.min(z_values))

def automatic_positions(mesh: trimesh.Trimesh, hydro: HullHydrostatics) -> dict[str, np.ndarray]:
    """Compute automatic root positions for wing, centerboard and rudder."""

    bounds = mesh.bounds
    x_mid = hydro.LCB if np.isfinite(hydro.LCB) else 0.5 * (bounds[0, 0] + bounds[1, 0])
    x_rudder = bounds[0, 0] + 0.08 * max(bounds[1, 0] - bounds[0, 0], 1e-6)
    return {
        "wing": np.array([x_mid, 0.0, hull_z_at_x(mesh, x_mid, top=True)]),
        "centerboard": np.array([x_mid, 0.0, hull_z_at_x(mesh, x_mid, top=False)]),
        "rudder": np.array([x_rudder, 0.0, hull_z_at_x(mesh, x_rudder, top=False)]),
    }
