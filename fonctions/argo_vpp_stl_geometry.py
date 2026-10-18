from __future__ import annotations
from .argo_vpp_common import Any, Path, append_geometry_log, asdict, math, np, re, struct
from .argo_vpp_geometry import close_polygon, intersections_with_line_y, parse_points_text
from .argo_vpp_models import GeometryState, SimulationCase

def placeholder_hull_mesh() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([-0.75, -0.45, 0.45, 0.75, 0.45, -0.45])
    y = np.array([0.0, -0.12, -0.12, 0.0, 0.12, 0.12])
    z = np.array([0.06, 0.0, 0.0, 0.06, 0.0, 0.0])
    vertices = np.column_stack([x, y, z])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1], [1, 5, 4], [1, 4, 2], [2, 4, 3]])
    return vertices, faces

def parse_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    triangles: list[list[float]] = []
    if raw[:5].lower() == b"solid" and b"facet" in raw[:2048].lower():
        text = raw.decode("utf-8", errors="ignore")
        verts = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(verts) >= 3:
            vertices = np.asarray(verts, dtype=float)
            faces = np.arange(len(vertices), dtype=int).reshape((-1, 3))
            return vertices, faces
    if len(raw) < 84:
        raise ValueError("Invalid STL file.")
    n_tri = struct.unpack("<I", raw[80:84])[0]
    offset = 84
    for _ in range(n_tri):
        if offset + 50 > len(raw):
            break
        chunk = raw[offset : offset + 50]
        coords = struct.unpack("<12fH", chunk)
        triangles.extend([coords[3:6], coords[6:9], coords[9:12]])
        offset += 50
    vertices = np.asarray(triangles, dtype=float)
    faces = np.arange(len(vertices), dtype=int).reshape((-1, 3))
    return vertices, faces

def apply_stl_unit(vertices: np.ndarray, unit: str) -> np.ndarray:
    scale = {"mm": 0.001, "cm": 0.01, "m": 1.0}.get(unit, 1.0)
    return vertices.astype(float) * scale

def mesh_dimensions(vertices: np.ndarray) -> dict[str, float]:
    if vertices.size == 0:
        return {"Lx": 0.0, "Ly": 0.0, "Lz": 0.0, "x_mid": 0.0, "z_min": 0.0, "z_max": 0.0, "mins": np.zeros(3), "maxs": np.zeros(3)}
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return {
        "Lx": float(maxs[0] - mins[0]),
        "Ly": float(maxs[1] - mins[1]),
        "Lz": float(maxs[2] - mins[2]),
        "x_mid": float(0.5 * (mins[0] + maxs[0])),
        "z_min": float(mins[2]),
        "z_max": float(maxs[2]),
        "mins": mins,
        "maxs": maxs,
    }

def axis_index(axis_name: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}.get(str(axis_name).upper(), 0)

def detect_longitudinal_axis(vertices: np.ndarray, vertical_axis: str = "Z") -> str:
    if vertices.size == 0:
        return "X"
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    v_idx = axis_index(vertical_axis)
    extents = extents.astype(float)
    extents[v_idx] = -1.0
    return ["X", "Y", "Z"][int(np.argmax(extents))]

def hull_frame(case: SimulationCase) -> dict[str, np.ndarray | int | float]:
    long_idx = axis_index(case.longitudinal_axis)
    vertical_idx = axis_index(case.vertical_axis)
    if vertical_idx == long_idx:
        vertical_idx = 2 if long_idx != 2 else 1
    trans_idx = ({0, 1, 2} - {long_idx, vertical_idx}).pop()

    long_vec = np.zeros(3)
    trans_vec = np.zeros(3)
    vertical_up_vec = np.zeros(3)
    long_vec[long_idx] = 1.0
    trans_vec[trans_idx] = 1.0
    vertical_up_vec[vertical_idx] = 1.0 if case.vertical_positive_up else -1.0
    return {
        "long_idx": long_idx,
        "trans_idx": trans_idx,
        "vertical_idx": vertical_idx,
        "long_vec": long_vec,
        "trans_vec": trans_vec,
        "vertical_up_vec": vertical_up_vec,
        "vertical_down_vec": -vertical_up_vec,
    }

def local_hull_position(
    vertices: np.ndarray | None,
    case: SimulationCase,
    longitudinal_value: float | None = None,
    transverse_value: float = 0.0,
    vertical_surface: str = "bottom",
) -> np.ndarray:
    frame = hull_frame(case)
    point = np.zeros(3)
    if vertices is not None and vertices.size:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        point = 0.5 * (mins + maxs)
        long_idx = int(frame["long_idx"])
        trans_idx = int(frame["trans_idx"])
        vertical_idx = int(frame["vertical_idx"])
        if longitudinal_value is not None:
            point[long_idx] = longitudinal_value
        point[trans_idx] = 0.5 * (mins[trans_idx] + maxs[trans_idx]) + transverse_value
        local = vertices
        long_span = max(float(maxs[long_idx] - mins[long_idx]), 1e-9)
        trans_span = max(float(maxs[trans_idx] - mins[trans_idx]), 1e-9)
        long_tol = max(0.03 * long_span, 1e-4)
        trans_tol = max(0.10 * trans_span, 1e-4)
        mask = np.abs(local[:, long_idx] - point[long_idx]) <= long_tol
        mask &= np.abs(local[:, trans_idx] - point[trans_idx]) <= trans_tol
        candidates = local[mask]
        if candidates.size == 0:
            dist = np.abs(local[:, long_idx] - point[long_idx]) / long_span
            dist += np.abs(local[:, trans_idx] - point[trans_idx]) / trans_span
            candidates = local[np.argsort(dist)[: max(24, min(len(local), 200))]]
        if vertical_surface == "top":
            point[vertical_idx] = float(candidates[:, vertical_idx].max()) if case.vertical_positive_up else float(candidates[:, vertical_idx].min())
        elif vertical_surface == "bottom":
            point[vertical_idx] = float(candidates[:, vertical_idx].min()) if case.vertical_positive_up else float(candidates[:, vertical_idx].max())
    elif longitudinal_value is not None:
        point[int(frame["long_idx"])] = longitudinal_value
        point += np.asarray(frame["trans_vec"], dtype=float) * transverse_value
    return point

def longitudinal_bounds(vertices: np.ndarray, case: SimulationCase) -> tuple[float, float]:
    idx = axis_index(case.longitudinal_axis)
    if vertices.size == 0:
        return -0.5, 0.5
    return float(vertices[:, idx].min()), float(vertices[:, idx].max())

def longitudinal_from_percent(vertices: np.ndarray | None, case: SimulationCase, percent: float) -> float:
    lo, hi = longitudinal_bounds(vertices if vertices is not None else np.zeros((0, 3)), case)
    pct = float(np.clip(percent, 0.0, 100.0)) / 100.0
    if case.bow_positive:
        return lo + pct * (hi - lo)
    return hi - pct * (hi - lo)

def longitudinal_percent_from_value(vertices: np.ndarray | None, case: SimulationCase, value: float) -> float:
    lo, hi = longitudinal_bounds(vertices if vertices is not None else np.zeros((0, 3)), case)
    length = max(hi - lo, 1e-9)
    if case.bow_positive:
        return float(np.clip(100.0 * (float(value) - lo) / length, 0.0, 100.0))
    return float(np.clip(100.0 * (hi - float(value)) / length, 0.0, 100.0))

def default_keel_longitudinal(vertices: np.ndarray | None, case: SimulationCase) -> float:
    if case.keel_position_mode == "Manual" and case.keel_x != 0.0:
        return case.keel_x
    if case.keel_position_mode == "Manual":
        return longitudinal_from_percent(vertices, case, case.keel_position_percent)
    if vertices is not None and vertices.size:
        lo, hi = longitudinal_bounds(vertices, case)
        return 0.5 * (lo + hi)
    return 0.0

def default_rudder_longitudinal(vertices: np.ndarray | None, case: SimulationCase) -> float:
    if case.rudder_position_mode == "Manual" and case.rudder_x != 0.0:
        return case.rudder_x
    if case.rudder_position_mode == "Manual":
        return longitudinal_from_percent(vertices, case, case.rudder_position_percent)
    if vertices is not None and vertices.size:
        lo, hi = longitudinal_bounds(vertices, case)
        length = max(hi - lo, 1e-9)
        rear_offset = np.clip(case.rudder_stern_margin_percent, 0.0, 50.0) * 0.01 * length
        return lo + rear_offset if case.bow_positive else hi - rear_offset
    return -0.45 if case.bow_positive else 0.45

def naca_00xx_thickness(x_over_c: np.ndarray, thickness_ratio: float) -> np.ndarray:
    x = np.clip(x_over_c, 0.0, 1.0)
    t = float(thickness_ratio)
    return 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def profile_digits_from_name(profile_name: str) -> str:
    match = re.search(r"NACA\s*([0-9]{4})", profile_name.upper().replace("-", ""))
    return match.group(1) if match else "0012"

def naca_4digit_surfaces(profile_name: str, x_over_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    digits = profile_digits_from_name(profile_name)
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    x = np.clip(np.asarray(x_over_c, dtype=float), 0.0, 1.0)
    yt = naca_00xx_thickness(x, t)
    if m <= 0.0 or p <= 0.0:
        return yt, -yt
    yc = np.where(
        x < p,
        m / (p**2) * (2.0 * p * x - x**2),
        m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
    )
    dyc = np.where(
        x < p,
        2.0 * m / (p**2) * (p - x),
        2.0 * m / ((1.0 - p) ** 2) * (p - x),
    )
    theta = np.arctan(dyc)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)
    return yu, yl

def contour_chord_sections(
    contour_points: list[tuple[float, float]],
    n_span: int,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    pts = close_polygon(contour_points)
    ys = np.asarray([p[1] for p in pts], dtype=float)
    y_min = float(ys.min())
    y_max = float(ys.max())
    if y_max <= y_min:
        raise ValueError("Sail contour span is zero.")
    y_levels = np.linspace(y_min, y_max, max(2, int(n_span)))
    sections: list[tuple[float, float, float]] = []
    for y in y_levels:
        xs = sorted(intersections_with_line_y(pts, float(y)))
        if len(xs) < 2:
            continue
        x_le = float(min(xs))
        x_te = float(max(xs))
        chord = x_te - x_le
        if chord > 1e-8:
            sections.append((float(y), x_le, chord))
    if len(sections) < 2:
        raise ValueError("Sail contour does not provide enough spanwise chord sections for a 3D mesh.")
    return np.asarray([s[0] for s in sections], dtype=float), sections

def make_sail_volume_mesh(
    contour_points: list[tuple[float, float]],
    case: SimulationCase,
    root_origin: np.ndarray,
    n_chord: int = 34,
    n_span: int = 18,
) -> tuple[np.ndarray, np.ndarray]:
    frame = hull_frame(case)
    long_vec = np.asarray(frame["long_vec"], dtype=float)
    trans_vec = np.asarray(frame["trans_vec"], dtype=float)
    vertical_up_vec = np.asarray(frame["vertical_up_vec"], dtype=float)
    _, sections = contour_chord_sections(contour_points, n_span)
    y_min = min(y for y, _, _ in sections)
    xdist = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, max(3, int(n_chord)))))
    upper, lower = naca_4digit_surfaces(case.aero_profile, xdist)
    vertices: list[np.ndarray] = []
    for y, x_le, chord in sections:
        leading = root_origin + vertical_up_vec * (y - y_min) + long_vec * x_le
        for surface in (upper, lower):
            for x, t in zip(xdist, surface):
                vertices.append(leading + long_vec * (float(x) * chord) + trans_vec * (float(t) * chord))
    verts = np.asarray(vertices, dtype=float)

    nsec = len(sections)
    nch = len(xdist)

    def idx(j: int, side: int, i: int) -> int:
        return j * 2 * nch + side * nch + i

    faces: list[list[int]] = []
    for j in range(nsec - 1):
        for i in range(nch - 1):
            faces.append([idx(j, 0, i), idx(j, 0, i + 1), idx(j + 1, 0, i + 1), idx(j + 1, 0, i)])
            faces.append([idx(j, 1, i + 1), idx(j, 1, i), idx(j + 1, 1, i), idx(j + 1, 1, i + 1)])
        faces.append([idx(j, 0, 0), idx(j + 1, 0, 0), idx(j + 1, 1, 0), idx(j, 1, 0)])
        faces.append([idx(j, 0, nch - 1), idx(j, 1, nch - 1), idx(j + 1, 1, nch - 1), idx(j + 1, 0, nch - 1)])
    for j in (0, nsec - 1):
        faces.append([idx(j, 0, i) for i in range(nch)] + [idx(j, 1, i) for i in range(nch - 1, -1, -1)])
    if abs(case.rotation_deg) > 1e-12:
        verts = rotate_about_axis(verts, root_origin, vertical_up_vec, case.rotation_deg)
    if abs(case.inclination_deg) > 1e-12:
        verts = rotate_about_axis(verts, root_origin, trans_vec, case.inclination_deg)
    return verts, np.asarray(faces, dtype=object)

def rotate_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    if abs(angle) < 1e-12:
        return points
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    rel = points - origin
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = (
        rel * cos_a
        + np.cross(axis, rel) * sin_a
        + axis * np.dot(rel, axis)[:, None] * (1.0 - cos_a)
    )
    return origin + rotated

def make_naca_appendage_mesh(
    origin: np.ndarray,
    case: SimulationCase,
    root_chord: float,
    tip_chord: float,
    span: float,
    thickness_ratio: float,
    sweep: float,
    angle_deg: float = 0.0,
    pitch_deg: float = 0.0,
    n_chord: int = 28,
    n_span: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    frame = hull_frame(case)
    long_vec = np.asarray(frame["long_vec"], dtype=float)
    trans_vec = np.asarray(frame["trans_vec"], dtype=float)
    down_vec = np.asarray(frame["vertical_down_vec"], dtype=float)
    xdist = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_chord)))
    points: list[np.ndarray] = []
    for j in range(n_span):
        eta = j / max(n_span - 1, 1)
        chord = root_chord + eta * (tip_chord - root_chord)
        section_origin = origin + down_vec * (span * eta) + long_vec * (sweep * eta)
        leading = section_origin - long_vec * (0.25 * chord)
        thickness = naca_00xx_thickness(xdist, thickness_ratio) * chord
        for side in (1.0, -1.0):
            for x, th in zip(xdist, thickness):
                point = leading + long_vec * (x * chord) + trans_vec * (side * th)
                points.append(point)
    vertices = np.asarray(points, dtype=float)

    def idx(j: int, side: int, i: int) -> int:
        return j * 2 * n_chord + side * n_chord + i

    faces: list[list[int]] = []
    for j in range(n_span - 1):
        for i in range(n_chord - 1):
            faces.append([idx(j, 0, i), idx(j, 0, i + 1), idx(j + 1, 0, i + 1), idx(j + 1, 0, i)])
            faces.append([idx(j, 1, i + 1), idx(j, 1, i), idx(j + 1, 1, i), idx(j + 1, 1, i + 1)])
    for j in range(n_span - 1):
        faces.append([idx(j, 0, 0), idx(j + 1, 0, 0), idx(j + 1, 1, 0), idx(j, 1, 0)])
        faces.append([idx(j, 0, n_chord - 1), idx(j, 1, n_chord - 1), idx(j + 1, 1, n_chord - 1), idx(j + 1, 0, n_chord - 1)])
    for j in (0, n_span - 1):
        faces.append([idx(j, 0, i) for i in range(n_chord)] + [idx(j, 1, i) for i in range(n_chord - 1, -1, -1)])
    if abs(angle_deg) > 1e-12:
        vertices = rotate_about_axis(vertices, origin, down_vec, angle_deg)
    if abs(pitch_deg) > 1e-12:
        vertices = rotate_about_axis(vertices, origin, trans_vec, pitch_deg)
    return vertices, np.asarray(faces, dtype=object)

def appendage_below_hull_report(vertices: np.ndarray, root: np.ndarray, case: SimulationCase) -> dict[str, Any]:
    frame = hull_frame(case)
    vertical_up_vec = np.asarray(frame["vertical_up_vec"], dtype=float)
    local_vertical = vertices @ vertical_up_vec
    root_vertical = float(np.dot(root, vertical_up_vec))
    center_vertical = float(np.dot(vertices.mean(axis=0), vertical_up_vec))
    lowest_vertical = float(local_vertical.min())
    highest_vertical = float(local_vertical.max())
    below_fraction = float(np.mean(local_vertical <= root_vertical + 1e-7))
    valid = center_vertical < root_vertical and lowest_vertical < root_vertical and below_fraction >= 0.80
    return {
        "valid": bool(valid),
        "root_vertical": root_vertical,
        "center_vertical": center_vertical,
        "lowest_vertical": lowest_vertical,
        "highest_vertical": highest_vertical,
        "below_fraction": below_fraction,
    }

def compute_keel_autofit_case(
    case: SimulationCase,
    vertices: np.ndarray | None,
    faces: np.ndarray | None = None,
) -> tuple[SimulationCase, np.ndarray, dict[str, Any]]:
    if vertices is None or not np.asarray(vertices).size:
        raise ValueError("No hull mesh available for keel auto-fit.")
    new_case = SimulationCase(**asdict(case))
    new_case.longitudinal_axis = detect_longitudinal_axis(np.asarray(vertices), new_case.vertical_axis)
    lo, hi = longitudinal_bounds(np.asarray(vertices), new_case)
    center = 0.5 * (lo + hi)
    contact = local_hull_position(np.asarray(vertices), new_case, center, 0.0, "bottom")
    new_case.keel_position_mode = "Auto-fit"
    new_case.keel_position_percent = 50.0
    new_case.keel_x = float(center)
    new_case.keel_transverse_offset = 0.0
    new_case.keel_vertical_offset = 0.0
    new_case.keel_yaw_deg = 0.0
    new_case.keel_pitch_deg = 0.0
    root_chord = new_case.keel_root_chord if new_case.keel_root_chord > 0 else new_case.S_derive / max(new_case.hauteur_quille, 1e-9)
    tip_chord = new_case.keel_tip_chord if new_case.keel_tip_chord > 0 else 0.65 * root_chord
    keel_vertices, _ = make_naca_appendage_mesh(
        contact,
        new_case,
        root_chord=root_chord,
        tip_chord=tip_chord,
        span=new_case.hauteur_quille,
        thickness_ratio=new_case.keel_thickness_ratio,
        sweep=new_case.keel_sweep,
        angle_deg=new_case.keel_yaw_deg,
        pitch_deg=new_case.keel_pitch_deg,
    )
    report = appendage_below_hull_report(keel_vertices, contact, new_case)
    append_geometry_log(
        "[keel auto-fit solve] "
        f"frame={hull_frame(new_case)}, lo={lo:.6g}, hi={hi:.6g}, center={center:.6g}, "
        f"contact={contact.tolist()}, report={report}"
    )
    if not bool(report["valid"]):
        raise ValueError(f"Keel auto-fit validation failed: {report}")
    return new_case, contact, report

def compute_rudder_autofit_case(
    case: SimulationCase,
    vertices: np.ndarray | None,
    faces: np.ndarray | None = None,
) -> tuple[SimulationCase, np.ndarray, dict[str, Any]]:
    if vertices is None or not np.asarray(vertices).size:
        raise ValueError("No hull mesh available for rudder auto-fit.")
    new_case = SimulationCase(**asdict(case))
    new_case.longitudinal_axis = detect_longitudinal_axis(np.asarray(vertices), new_case.vertical_axis)
    rudder_long = default_rudder_longitudinal(np.asarray(vertices), new_case)
    contact = local_hull_position(np.asarray(vertices), new_case, rudder_long, 0.0, "bottom")
    new_case.rudder_position_mode = "Auto-fit"
    new_case.rudder_x = float(rudder_long)
    new_case.rudder_position_percent = longitudinal_percent_from_value(np.asarray(vertices), new_case, rudder_long)
    new_case.rudder_transverse_offset = 0.0
    new_case.rudder_vertical_offset = 0.0
    new_case.rudder_pitch_deg = 0.0
    rudder_vertices, _ = make_naca_appendage_mesh(
        contact,
        new_case,
        root_chord=new_case.rudder_root_chord,
        tip_chord=new_case.rudder_tip_chord,
        span=new_case.rudder_span,
        thickness_ratio=new_case.rudder_thickness_ratio,
        sweep=new_case.rudder_sweep,
        angle_deg=new_case.rudder_angle_deg,
        pitch_deg=new_case.rudder_pitch_deg,
    )
    report = appendage_below_hull_report(rudder_vertices, contact, new_case)
    append_geometry_log(
        "[rudder auto-fit solve] "
        f"frame={hull_frame(new_case)}, target_long={rudder_long:.6g}, contact={contact.tolist()}, report={report}"
    )
    if not bool(report["valid"]):
        raise ValueError(f"Rudder auto-fit validation failed: {report}")
    return new_case, contact, report

def faces_to_pyvista(faces: np.ndarray) -> np.ndarray:
    packed: list[int] = []
    for face in faces:
        packed.append(len(face))
        packed.extend([int(i) for i in face])
    return np.asarray(packed, dtype=np.int64)

def evaluate_geometry_state(case: SimulationCase, hull_vertices: np.ndarray | None = None) -> GeometryState:
    hull_loaded = bool(case.stl_path) and hull_vertices is not None and hull_vertices.size > 0
    hull_valid = hull_loaded
    sail_defined = bool(case.contour_text.strip())
    try:
        sail_valid = sail_defined and len(parse_points_text(case.contour_text)) >= 3
    except Exception:
        sail_valid = False
    keel_defined = bool(case.keel_defined)
    keel_valid = keel_defined and case.S_derive > 0.0 and case.hauteur_quille > 0.0
    rudder_defined = bool(case.rudder_defined)
    rudder_valid = rudder_defined
    return GeometryState(
        hull_loaded=hull_loaded,
        sail_defined=sail_defined,
        keel_defined=keel_defined,
        rudder_defined=rudder_defined,
        hull_valid=hull_valid,
        sail_valid=sail_valid,
        keel_valid=keel_valid,
        rudder_valid=rudder_valid,
        hull_visible=hull_valid and case.show_hull,
        sail_visible=sail_valid and case.show_sail,
        keel_visible=keel_valid and case.show_keel,
        rudder_visible=rudder_valid and case.show_rudder,
    )
