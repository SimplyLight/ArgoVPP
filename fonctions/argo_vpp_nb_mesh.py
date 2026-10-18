from __future__ import annotations

from typing import Any, Callable
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from pathlib import Path
from IPython.display import clear_output, display
import numpy as np
import pandas as pd
from shapely.ops import triangulate, unary_union
import trimesh
import warnings

__all__ = ['available_stl_files', 'cap_faces_from_loops', 'cleanup_largest_component', 'clip_triangle_below_z', 'immersed_mesh_for_waterline', 'infer_stl_unit_from_extents', 'interpolate_edge_to_z', 'load_hull_stl', 'merge_waterline_segments', 'point_key_2d', 'print_mesh_diagnostics', 'resolve_stl_path', 'safe_mesh_center', 'safe_mesh_volume', 'surface_area_centroid', 'triangle_area_array', 'unit_scale', 'validate_mesh', 'waterline_segments_for_triangle', 'waterplane_polygon_from_loops']

def unit_scale(unit: str) -> float:
    """Return STL unit scale to metres."""

    mapping = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    if unit not in mapping:
        raise ValueError(f"Unite STL invalide: {unit!r}")
    return mapping[unit]

def infer_stl_unit_from_extents(extents: np.ndarray) -> str:
    """Infer STL units from raw bounding-box dimensions."""

    values = np.asarray(extents, dtype=float)
    max_extent = float(np.nanmax(np.abs(values))) if values.size else 0.0
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        return "m"
    if max_extent > 100.0:
        return "mm"
    if max_extent > 20.0:
        return "cm"
    return "m"

def available_stl_files(search_dirs: tuple[str, ...] = ("Coques",)) -> list[str]:
    """Return STL files available from the usual project folders."""

    files: list[str] = []
    seen: set[str] = set()
    cwd = Path.cwd()
    for folder_name in search_dirs:
        folder = Path(folder_name)
        if not folder.exists():
            continue
        candidates = [folder] if folder.is_file() and folder.suffix.lower() == ".stl" else sorted(folder.glob("*.stl"))
        for path in candidates:
            try:
                rel = path.resolve().relative_to(cwd.resolve())
                value = str(rel).replace("\\", "/")
            except ValueError:
                value = str(path)
            key = value.lower()
            if key not in seen:
                seen.add(key)
                files.append(value)
    return files

def resolve_stl_path(path: str | Path) -> Path:
    """Resolve a hull STL path, accepting old widgets that only contain the file name."""

    candidate = Path(path)
    if candidate.exists():
        return candidate
    for folder in (Path("Coques"), Path("coque"), Path("test_coque_stl"), Path(".")):
        alternative = folder / candidate.name
        if alternative.exists():
            return alternative
    raise FileNotFoundError(f"STL introuvable: {path}. Dossiers cherches: Coques/, coque/, test_coque_stl/, ./")

def safe_mesh_volume(mesh: trimesh.Trimesh, signed: bool = False) -> float:
    """Return a mesh volume without letting trimesh mass-property warnings leak to the notebook."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            value = float(mesh.volume)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            value = 0.0
    if not np.isfinite(value):
        value = 0.0
    return value if signed else abs(value)

def safe_mesh_center(mesh: trimesh.Trimesh, fallback_triangles: np.ndarray | None = None) -> np.ndarray:
    """Return centre of volume when valid, otherwise an area/vertex centroid fallback."""

    if safe_mesh_volume(mesh) > 1e-12:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                center = np.asarray(mesh.center_mass, dtype=float)
                if center.shape == (3,) and np.all(np.isfinite(center)):
                    return center
            except (ValueError, ZeroDivisionError, FloatingPointError):
                pass
    if fallback_triangles is not None and len(fallback_triangles):
        tri = np.asarray(fallback_triangles, dtype=float)
        areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if np.sum(areas) > 0:
            return np.average(tri.mean(axis=1), axis=0, weights=areas)
    if len(mesh.vertices):
        return np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    return np.full(3, np.nan)

def cleanup_largest_component(mesh: trimesh.Trimesh, max_removed_ratio: float = 0.05) -> trimesh.Trimesh:
    """Remove tiny disconnected STL fragments while refusing large multi-body hulls."""

    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    largest = max(components, key=lambda part: len(part.faces))
    removed_faces = int(len(mesh.faces) - len(largest.faces))
    removed_ratio = removed_faces / max(len(mesh.faces), 1)
    if removed_faces <= 0:
        return mesh
    if removed_ratio > max_removed_ratio:
        raise ValueError(
            f"STL compose de {len(components)} volumes significatifs; "
            f"{removed_faces} faces seraient supprimees ({removed_ratio:.1%})."
        )
    print(
        f"Nettoyage STL: conservation du plus grand composant, "
        f"{removed_faces} faces parasites retirees sur {len(mesh.faces)}."
    )
    return trimesh.Trimesh(vertices=np.asarray(largest.vertices, dtype=float), faces=np.asarray(largest.faces, dtype=int), process=True)

def load_hull_stl(path: str | Path, unit: str = "auto", orientation: str = "x_min = poupe, x_max = proue") -> trimesh.Trimesh:
    """Load, scale and orient a hull STL in the internal boat frame."""

    path = resolve_stl_path(path)
    mesh = trimesh.load_mesh(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Le fichier ne contient pas un maillage STL unique: {path}")
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise ValueError("Le maillage STL ne contient pas de triangles valides.")

    selected_unit = infer_stl_unit_from_extents(mesh.extents) if unit == "auto" else unit
    if unit == "auto":
        print(f"Unite STL auto: {selected_unit} (dimensions brutes {np.asarray(mesh.extents, dtype=float)})")
    vertices = np.asarray(mesh.vertices, dtype=float) * unit_scale(selected_unit)
    if orientation == "x_min = proue, x_max = poupe":
        vertices[:, 0] *= -1.0
        vertices[:, 0] -= vertices[:, 0].min()
    elif orientation != "x_min = poupe, x_max = proue":
        raise ValueError(f"Orientation longitudinale invalide: {orientation}")

    oriented = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=int), process=True)
    oriented = cleanup_largest_component(oriented)
    diagnostics = validate_mesh(oriented)
    if diagnostics["fix_normals_suggested"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            oriented.fix_normals()
        print("Correction legere appliquee: orientation/coherence des normales via trimesh.fix_normals().")
    return oriented

def validate_mesh(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Return explicit STL diagnostics without silently repairing severe defects."""

    edge_inverse = np.asarray(getattr(mesh, "edges_unique_inverse", []), dtype=int)
    edge_counts = np.bincount(edge_inverse) if edge_inverse.size else np.zeros(0, dtype=int)
    boundary_edges = int(np.sum(edge_counts == 1))
    nonmanifold_edges = int(np.sum(edge_counts > 2))
    components = mesh.split(only_watertight=False)
    component_faces = [int(len(part.faces)) for part in components]
    diagnostics = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "volume_m3": safe_mesh_volume(mesh) if mesh.is_watertight else np.nan,
        "bounds_min": mesh.bounds[0].copy(),
        "bounds_max": mesh.bounds[1].copy(),
        "extents_m": mesh.extents.copy(),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "component_count": int(len(components)),
        "largest_component_faces": max(component_faces) if component_faces else 0,
        "errors": [],
        "fix_normals_suggested": False,
    }
    if diagnostics["faces"] <= 0:
        diagnostics["errors"].append("Aucune face triangulaire.")
    if np.any(mesh.area_faces <= 1e-14):
        diagnostics["errors"].append("Triangles degeneres detectes.")
    if boundary_edges:
        diagnostics["errors"].append(f"{boundary_edges} aretes ouvertes detectees.")
    if nonmanifold_edges:
        diagnostics["errors"].append(f"{nonmanifold_edges} aretes non-manifold detectees.")
    if not diagnostics["is_watertight"]:
        diagnostics["errors"].append("Maillage non etanche: le calcul hydrostatique exact peut echouer.")
    if not diagnostics["is_winding_consistent"]:
        diagnostics["errors"].append("Normales non coherentes.")
        diagnostics["fix_normals_suggested"] = True
    return diagnostics

def print_mesh_diagnostics(mesh: trimesh.Trimesh) -> pd.DataFrame:
    """Display mesh control quantities requested by the notebook brief."""

    d = validate_mesh(mesh)
    rows = {
        "faces": d["faces"],
        "vertices": d["vertices"],
        "watertight": d["is_watertight"],
        "normales coherentes": d["is_winding_consistent"],
        "volume STL m3": d["volume_m3"],
        "L x m": d["extents_m"][0],
        "B y m": d["extents_m"][1],
        "H z m": d["extents_m"][2],
        "composants": d["component_count"],
        "aretes ouvertes": d["boundary_edges"],
        "aretes non-manifold": d["nonmanifold_edges"],
        "erreurs": "; ".join(d["errors"]) if d["errors"] else "aucune erreur bloquante detectee",
    }
    df = pd.DataFrame([rows])
    display(df)
    return df

def triangle_area_array(triangles: np.ndarray) -> np.ndarray:
    """Area of each triangle in an array shaped (n, 3, 3)."""

    tri = np.asarray(triangles, dtype=float)
    if tri.size == 0:
        return np.zeros(0)
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

def surface_area_centroid(triangles: np.ndarray) -> tuple[float, np.ndarray]:
    """Area and area-weighted centroid of triangular surface."""

    tri = np.asarray(triangles, dtype=float)
    if tri.size == 0:
        return 0.0, np.full(3, np.nan)
    areas = triangle_area_array(tri)
    total = float(np.sum(areas))
    if total <= 0:
        return 0.0, np.full(3, np.nan)
    centroids = tri.mean(axis=1)
    return total, np.average(centroids, axis=0, weights=areas)

def interpolate_edge_to_z(p0: np.ndarray, p1: np.ndarray, z: float) -> np.ndarray:
    """Linear intersection of an edge with horizontal plane z."""

    dz = p1[2] - p0[2]
    if abs(dz) < 1e-14:
        return p0.copy()
    t = (z - p0[2]) / dz
    return p0 + t * (p1 - p0)

def clip_triangle_below_z(tri: np.ndarray, z: float, eps: float = 1e-10) -> list[np.ndarray]:
    """Clip one triangle by the half-space Z <= z and return triangles."""

    pts = [np.asarray(p, dtype=float) for p in tri]
    below = [p[2] <= z + eps for p in pts]
    if all(below):
        return [np.asarray(pts)]
    if not any(below):
        return []
    poly: list[np.ndarray] = []
    for i in range(3):
        current = pts[i]
        nxt = pts[(i + 1) % 3]
        cur_in = below[i]
        next_in = below[(i + 1) % 3]
        if cur_in:
            poly.append(current)
        if cur_in != next_in:
            poly.append(interpolate_edge_to_z(current, nxt, z))
    if len(poly) < 3:
        return []
    if len(poly) == 3:
        return [np.asarray(poly)]
    return [np.asarray([poly[0], poly[i], poly[i + 1]]) for i in range(1, len(poly) - 1)]

def waterline_segments_for_triangle(tri: np.ndarray, z: float, eps: float = 1e-10) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return intersection segment(s) between a triangle and the horizontal water plane."""

    tri = np.asarray(tri, dtype=float)
    d = tri[:, 2] - z
    points: list[np.ndarray] = []
    for i in range(3):
        p0, p1 = tri[i], tri[(i + 1) % 3]
        d0, d1 = d[i], d[(i + 1) % 3]
        if abs(d0) <= eps and abs(d1) <= eps:
            points.extend([p0, p1])
        elif abs(d0) <= eps:
            points.append(p0)
        elif d0 * d1 < -eps * eps:
            points.append(interpolate_edge_to_z(p0, p1, z))
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.linalg.norm(point - q) < 1e-8 for q in unique):
            unique.append(point)
    if len(unique) >= 2:
        return [(unique[0], unique[1])]
    return []

def point_key_2d(point: np.ndarray, tol: float) -> tuple[int, int]:
    """Hash a 3D waterline point by x-y coordinates."""

    return (int(round(point[0] / tol)), int(round(point[1] / tol)))

def merge_waterline_segments(segments: list[tuple[np.ndarray, np.ndarray]], tol: float = 1e-6) -> list[np.ndarray]:
    """Merge waterline segments into closed loops."""

    if not segments:
        return []
    key_to_index: dict[tuple[int, int], int] = {}
    points: list[np.ndarray] = []

    def add_point(p: np.ndarray) -> int:
        key = point_key_2d(p, tol)
        if key in key_to_index:
            return key_to_index[key]
        points.append(p)
        idx = len(points) - 1
        key_to_index[key] = idx
        return idx

    edges: list[tuple[int, int]] = []
    for a, b in segments:
        ia = add_point(a)
        ib = add_point(b)
        if ia != ib:
            edges.append((ia, ib))

    neighbors: dict[int, list[int]] = {}
    for ia, ib in edges:
        neighbors.setdefault(ia, []).append(ib)
        neighbors.setdefault(ib, []).append(ia)

    loops: list[np.ndarray] = []
    used_edges: set[tuple[int, int]] = set()
    for start, nxt in edges:
        edge_key = tuple(sorted((start, nxt)))
        if edge_key in used_edges:
            continue
        loop = [start]
        prev = start
        cur = nxt
        used_edges.add(edge_key)
        for _ in range(len(edges) + 5):
            loop.append(cur)
            if cur == start:
                break
            candidates = [k for k in neighbors.get(cur, []) if k != prev]
            if not candidates:
                break
            next_idx = candidates[0]
            for cand in candidates:
                if tuple(sorted((cur, cand))) not in used_edges:
                    next_idx = cand
                    break
            used_edges.add(tuple(sorted((cur, next_idx))))
            prev, cur = cur, next_idx
        if len(loop) >= 4 and loop[0] == loop[-1]:
            coords = np.asarray([points[i] for i in loop], dtype=float)
            if Polygon(coords[:, :2]).area > 1e-10:
                loops.append(coords)
    return loops

def cap_faces_from_loops(loops: list[np.ndarray], z: float) -> np.ndarray:
    """Triangulate horizontal waterplane loops into cap triangles."""

    cap_triangles: list[np.ndarray] = []
    for loop in loops:
        coords2 = loop[:, :2]
        poly = Polygon(coords2)
        if not poly.is_valid:
            poly = poly.buffer(0.0)
        if poly.is_empty or poly.area <= 0:
            continue
        polygons = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        for geom in polygons:
            for tri in triangulate(geom):
                if not geom.covers(tri.representative_point()):
                    continue
                xy = np.asarray(tri.exterior.coords[:3], dtype=float)
                xyz = np.column_stack([xy[:, 0], xy[:, 1], np.full(3, z)])
                cap_triangles.append(xyz[::-1])
    if not cap_triangles:
        return np.empty((0, 3, 3), dtype=float)
    return np.asarray(cap_triangles, dtype=float)

def immersed_mesh_for_waterline(mesh: trimesh.Trimesh, waterline_z: float, eps: float = 1e-10, cap: bool = True) -> tuple[trimesh.Trimesh, list[np.ndarray], np.ndarray]:
    """Clip a mesh below a horizontal waterline and optionally close it with cap faces."""

    clipped: list[np.ndarray] = []
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in mesh.triangles:
        clipped.extend(clip_triangle_below_z(tri, waterline_z, eps=eps))
        segments.extend(waterline_segments_for_triangle(tri, waterline_z, eps=eps))
    hull_triangles = np.asarray(clipped, dtype=float) if clipped else np.empty((0, 3, 3), dtype=float)
    loops = merge_waterline_segments(segments)
    cap_triangles = cap_faces_from_loops(loops, waterline_z) if cap else np.empty((0, 3, 3), dtype=float)
    all_triangles = np.vstack([hull_triangles, cap_triangles]) if len(cap_triangles) else hull_triangles
    if len(all_triangles) == 0:
        immersed = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=int), process=False)
    else:
        immersed = trimesh.Trimesh(vertices=all_triangles.reshape(-1, 3), faces=np.arange(len(all_triangles) * 3).reshape(-1, 3), process=False)
        try:
            if safe_mesh_volume(immersed) > 1e-12:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    immersed.fix_normals()
        except ValueError:
            pass
    return immersed, loops, hull_triangles

def waterplane_polygon_from_loops(loops: list[np.ndarray]) -> Polygon | MultiPolygon:
    """Build a shapely waterplane polygon from waterline loops."""

    polygons = []
    for loop in loops:
        poly = Polygon(loop[:, :2])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)
    if not polygons:
        return Polygon()
    return unary_union(polygons)
