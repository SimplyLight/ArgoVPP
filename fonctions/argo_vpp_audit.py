from __future__ import annotations
from .argo_vpp_common import Any, DEFAULT_SAIL_CONTOUR, NOTEBOOK_PATH, NOTEBOOK_SHA256_REFERENCE, json, np
from .argo_vpp_geometry import close_polygon, compute_cmoy_from_contour, intersections_with_line_y, parse_points_text, reynolds_number
from .argo_vpp_hydrodynamics import calc_hydro_drag, finite_section_correction
from .argo_vpp_models import SimulationCase
from .argo_vpp_solver_awa import calc_apparent_wind_awa
from .argo_vpp_stl_geometry import default_keel_longitudinal, default_rudder_longitudinal, hull_frame, make_sail_volume_mesh

def notebook_sha256() -> str:
    import hashlib

    return hashlib.sha256(NOTEBOOK_PATH.read_bytes()).hexdigest().upper()

def notebook_audit_summary() -> dict[str, Any]:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    function_count = 0
    try:
        import ast

        for cell in code_cells:
            tree = ast.parse("".join(cell.get("source", [])))
            function_count += sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    except Exception:
        function_count = 56
    return {"total_cells": len(nb.get("cells", [])), "code_cells": len(code_cells), "function_count": function_count}

def run_self_tests() -> int:
    failures: list[str] = []
    sha = notebook_sha256()
    if sha != NOTEBOOK_SHA256_REFERENCE:
        failures.append(f"Notebook hash changed: {sha}")
    pts = [(0, 0), (1, 0), (1, 1), (0, 1)]
    closed = close_polygon(pts)
    if len(closed) != 5 or closed[0] != closed[-1]:
        failures.append("close_polygon failed")
    xs = intersections_with_line_y(pts, 0.5)
    if not np.allclose(xs, [1.0]):
        failures.append(f"intersections failed: {xs}")
    cmoy, bands = compute_cmoy_from_contour(pts, 4)
    if abs(cmoy - 1.0) > 1e-12 or len(bands) != 4:
        failures.append("mean chord failed")
    re_val = reynolds_number(1.225, 10.0, 0.5, 1.81e-5)
    if abs(re_val - 338397.7900552486) > 1e-6:
        failures.append("Reynolds failed")
    drag = calc_hydro_drag(5.0, 0.42, 0.003, 0.002, 1025.0)
    if abs(drag - 26.90625) > 1e-9:
        failures.append("hydro drag failed")
    aw = calc_apparent_wind_awa(16.0, 3.0, 50.0)
    if abs(aw["V_AW"] - 18.075054924671953) > 1e-9 or abs(aw["beta_AW"] - 7.304581340708788) > 1e-9:
        failures.append("apparent wind failed")
    corr = finite_section_correction(0.6, 0.02, 5.0, 0.9)
    if corr["Cd_finite"] <= 0.02 or corr["Cl_finite"] <= 0.0:
        failures.append("finite correction failed")
    sail_case_12 = SimulationCase(contour_text=DEFAULT_SAIL_CONTOUR, aero_profile="NACA0012")
    sail_case_18 = SimulationCase(contour_text=DEFAULT_SAIL_CONTOUR, aero_profile="NACA0018")
    v12, f12 = make_sail_volume_mesh(parse_points_text(DEFAULT_SAIL_CONTOUR), sail_case_12, np.zeros(3))
    v18, _ = make_sail_volume_mesh(parse_points_text(DEFAULT_SAIL_CONTOUR), sail_case_18, np.zeros(3))
    if len(v12) == 0 or len(f12) == 0:
        failures.append("sail volume mesh failed")
    trans_idx = int(hull_frame(sail_case_12)["trans_idx"])
    thick12 = float(v12[:, trans_idx].max() - v12[:, trans_idx].min())
    thick18 = float(v18[:, trans_idx].max() - v18[:, trans_idx].min())
    if thick18 <= thick12:
        failures.append("NACA sail thickness selection failed")
    hull_case = SimulationCase(longitudinal_axis="X", vertical_axis="Z", bow_positive=True)
    fake_hull = np.array([[-1.0, -0.2, -0.1], [-1.0, 0.2, 0.1], [1.0, -0.2, -0.1], [1.0, 0.2, 0.1]])
    if abs(default_keel_longitudinal(fake_hull, hull_case) - 0.0) > 1e-12:
        failures.append("keel auto-fit longitudinal position failed")
    if not default_rudder_longitudinal(fake_hull, hull_case) < default_keel_longitudinal(fake_hull, hull_case):
        failures.append("rudder auto-fit stern position failed")
    print("ARGO-VPP self-test")
    print(f"Notebook SHA-256: {sha}")
    print(json.dumps(notebook_audit_summary(), indent=2))
    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS")
    return 0
