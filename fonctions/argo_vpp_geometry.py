from __future__ import annotations
from .argo_vpp_common import np, pd, re

def parse_point(text: str) -> tuple[float, float]:
    clean = text.strip().replace(";", ",").replace("\t", ",")
    parts = [p for p in re.split(r"[,\s]+", clean) if p]
    if len(parts) != 2:
        raise ValueError(f"Invalid point: {text!r}")
    return float(parts[0]), float(parts[1])

def parse_points_text(text: str) -> list[tuple[float, float]]:
    points = []
    for raw in text.splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#"):
            points.append(parse_point(raw))
    if len(points) < 3:
        raise ValueError("Contour must contain at least three points.")
    return points

def close_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    closed = [(float(x), float(y)) for x, y in points]
    if not np.allclose(closed[0], closed[-1]):
        closed.append(closed[0])
    return closed

def polygon_area(points: list[tuple[float, float]]) -> float:
    pts = close_polygon(points)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5

def intersect_segment_with_y(
    p0: tuple[float, float], p1: tuple[float, float], y_line: float
) -> float | None:
    x0, y0 = p0
    x1, y1 = p1
    if (y0 - y_line) * (y1 - y_line) > 0:
        return None
    if y1 == y0:
        return None
    t = (y_line - y0) / (y1 - y0)
    if 0.0 <= t <= 1.0:
        return float(x0 + t * (x1 - x0))
    return None

def intersections_with_line_y(points: list[tuple[float, float]], y_line: float) -> list[float]:
    xs: list[float] = []
    for p0, p1 in zip(points[:-1], points[1:]):
        value = intersect_segment_with_y(p0, p1, y_line)
        if value is not None:
            xs.append(value)
    return xs

def compute_cmoy_from_contour(points: list[tuple[float, float]], n_bands: int = 12) -> tuple[float, pd.DataFrame]:
    pts = close_polygon(points)
    ys = [p[1] for p in pts]
    y_min, y_max = min(ys), max(ys)
    if y_max <= y_min:
        raise ValueError("Contour span is zero.")
    rows = []
    chords = []
    dy = (y_max - y_min) / int(n_bands)
    for i in range(int(n_bands)):
        y0 = y_min + i * dy
        y1 = y0 + dy
        ym = 0.5 * (y0 + y1)
        xs0 = intersections_with_line_y(pts, y0)
        xs1 = intersections_with_line_y(pts, y1)
        xsm = intersections_with_line_y(pts, ym)
        chord_root = max(xs0) - min(xs0) if len(xs0) >= 2 else 0.0
        chord_tip = max(xs1) - min(xs1) if len(xs1) >= 2 else 0.0
        chord_mid = max(xsm) - min(xsm) if len(xsm) >= 2 else 0.5 * (chord_root + chord_tip)
        chord_avg = 0.5 * (chord_root + chord_tip)
        chords.append(chord_avg)
        rows.append(
            {
                "band": i + 1,
                "y0": y0,
                "y1": y1,
                "ym": ym,
                "chord_root": chord_root,
                "chord_tip": chord_tip,
                "chord_mid": chord_mid,
                "c_moy_i": chord_avg,
                "area_i": chord_avg * dy,
            }
        )
    df = pd.DataFrame(rows)
    c_moy = float(np.sum(chords) / float(n_bands))
    return float(c_moy), df

def reynolds_number(rho: float, velocity: float, chord: float, mu: float) -> float:
    return float(rho * velocity * chord / mu)
