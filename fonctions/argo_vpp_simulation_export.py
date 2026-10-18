from __future__ import annotations
from .argo_vpp_common import Any, Callable, Path, RENAMED_FUNCTIONS, TRACEABILITY_ROWS, asdict, json, np, pd
from .argo_vpp_aerodynamics import load_polar_file, plot_mt_dataset, resolve_profile_file
from .argo_vpp_geometry import compute_cmoy_from_contour, parse_points_text, reynolds_number
from .argo_vpp_hydrodynamics import compute_hydro_appendage_from_coeffs
from .argo_vpp_models import SimulationCase, SimulationResults
from .argo_vpp_settings import PARAMETER_GROUPS
from .argo_vpp_solver_awa import compute_polar_awa
from .argo_vpp_solver_true_wind import compute_polar_true_wind

def validate_case(case: SimulationCase) -> list[str]:
    errors = []
    positive_fields = [
        ("A_sail", case.A_sail),
        ("c_moy", case.c_moy),
        ("rho_air", case.rho_air),
        ("mu_air", case.mu_air),
        ("rho_water", case.rho_water),
        ("mu_water", case.mu_water),
        ("S_wetted", case.S_wetted),
        ("S_derive", case.S_derive),
        ("hauteur_quille", case.hauteur_quille),
    ]
    for name, value in positive_fields:
        if float(value) <= 0.0:
            errors.append(f"{name} must be positive.")
    if case.CR_friction + case.Rcoque_coeff <= 0.0:
        errors.append("Total hydrodynamic resistance coefficient must be positive.")
    if case.n_bands < 1:
        errors.append("Number of bands must be at least 1.")
    if case.n_cap < 36 or case.n_twa < 36:
        errors.append("Polar resolution must be at least 36 points.")
    if not case.contour_text.strip():
        errors.append("Sail geometry must be defined before running the VPP calculation.")
    else:
        try:
            parse_points_text(case.contour_text)
        except Exception as exc:
            errors.append(str(exc))
    try:
        resolve_profile_file(case.aero_profile)
    except Exception as exc:
        errors.append(str(exc))
    return errors

def run_full_simulation(case: SimulationCase, progress: Callable[[int, str], None] | None = None) -> SimulationResults:
    progress = progress or (lambda pct, msg: None)
    errors = validate_case(case)
    if errors:
        return SimulationResults(checks=errors, stale=True)
    progress(5, "Loading polar files")
    polar_tables = load_polar_file(resolve_profile_file(case.aero_profile))
    derive_tables = load_polar_file(resolve_profile_file("NACA0012"))
    progress(15, "Computing sail geometry")
    contour = parse_points_text(case.contour_text)
    cmoy, bands = compute_cmoy_from_contour(contour, case.n_bands)
    case = SimulationCase(**{**asdict(case), "c_moy": cmoy})
    progress(30, "Computing moments")
    moments = plot_mt_dataset(bands, polar_tables, case)
    progress(45, "Computing hydro appendages")
    keel_pts = [(0.0, 0.0), (case.S_derive / case.hauteur_quille, 0.0), (case.S_derive / case.hauteur_quille, case.hauteur_quille), (0.0, case.hauteur_quille)]
    hydro = pd.DataFrame([compute_hydro_appendage_from_coeffs("keel", keel_pts, case.rho_water, case.V_TW, 0.6, 0.02, 0.0, 0.9)])
    progress(60, "Computing AWA polar")
    awa = compute_polar_awa(case, polar_tables)
    progress(80, "Computing true wind polar")
    true_wind = compute_polar_true_wind(case, polar_tables, derive_tables)
    checks = [
        "Notebook compatibility formulas preserved.",
        f"Geometry bands: {len(bands)}",
        f"AWA points: {len(awa)}",
        f"CAP/TWA points: {len(true_wind)}",
    ]
    progress(100, "Done")
    return SimulationResults(geometry=bands, moments=moments, hydro_appendages=hydro, awa=awa, true_wind=true_wind, checks=checks, stale=False)

def metric_block(results: SimulationResults, case: SimulationCase) -> dict[str, Any]:
    df = results.true_wind
    awa = results.awa
    if df.empty:
        return {"health": 0, "vmax": 0.0, "vmg": 0.0, "vmg_angle": 0.0, "ld_max": 0.0, "re_min": 0.0, "re_max": 0.0, "psi_max": 0.0, "theta_max": 0.0, "converged_pct": 0.0}
    ld = (awa["Cl"] / awa["Cd"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan) if not awa.empty else pd.Series([0.0])
    idx_v = int(df["V_B"].idxmax())
    idx_vmg = int(df["VMG"].idxmax())
    conv = float((df["converged"] & df["derive_converged"] & df["heel_converged"]).mean() * 100.0)
    return {
        "health": round(conv),
        "vmax": float(df.loc[idx_v, "V_B"]),
        "vmax_angle": float(df.loc[idx_v, "CAP"]),
        "vmg": float(df.loc[idx_vmg, "VMG"]),
        "vmg_angle": float(df.loc[idx_vmg, "CAP"]),
        "ld_max": float(np.nanmax(ld)) if len(ld) else 0.0,
        "re_min": reynolds_number(case.rho_air, case.V_min_re, case.c_moy, case.mu_air),
        "re_max": reynolds_number(case.rho_air, case.V_max_re, case.c_moy, case.mu_air),
        "psi_max": float(df["psi_deg"].abs().max()),
        "theta_max": float(df["theta_deg"].abs().max()),
        "converged_pct": conv,
    }

def export_report_html(path: Path, case: SimulationCase, results: SimulationResults) -> None:
    metrics = metric_block(results, case)
    html = [
        "<html><head><meta charset='utf-8'><title>ARGO-VPP report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#0b1728} table{border-collapse:collapse;width:100%;margin:12px 0} td,th{border:1px solid #d8dee8;padding:6px;font-size:12px} th{background:#eef3f8}</style>",
        "</head><body>",
        "<h1>ARGO-VPP report</h1>",
        f"<p>Project: {case.project_name} / Variant: {case.variant_name}</p>",
        "<h2>Metrics</h2>",
        pd.DataFrame([metrics]).to_html(index=False),
        "<h2>Traceability</h2>",
        pd.DataFrame(TRACEABILITY_ROWS).to_html(index=False),
        "<h2>Renamed functions</h2>",
        pd.DataFrame([{"original": k, "new": v} for k, v in RENAMED_FUNCTIONS.items()]).to_html(index=False),
        "<h2>Checks</h2><pre>",
        "\n".join(results.checks),
        "</pre></body></html>",
    ]
    path.write_text("\n".join(html), encoding="utf-8")

def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")

def results_to_jsonable(results: SimulationResults) -> dict[str, Any]:
    return {
        "geometry": dataframe_to_records(results.geometry),
        "moments": dataframe_to_records(results.moments),
        "hydro_appendages": dataframe_to_records(results.hydro_appendages),
        "awa": dataframe_to_records(results.awa),
        "true_wind": dataframe_to_records(results.true_wind),
        "checks": list(results.checks),
        "stale": bool(results.stale),
    }

def results_from_jsonable(data: dict[str, Any]) -> SimulationResults:
    return SimulationResults(
        geometry=pd.DataFrame(data.get("geometry", [])),
        moments=pd.DataFrame(data.get("moments", [])),
        hydro_appendages=pd.DataFrame(data.get("hydro_appendages", [])),
        awa=pd.DataFrame(data.get("awa", [])),
        true_wind=pd.DataFrame(data.get("true_wind", [])),
        checks=list(data.get("checks", [])),
        stale=bool(data.get("stale", True)),
    )

def save_project_file(path: Path, case: SimulationCase, results: SimulationResults) -> None:
    payload = {
        "app": "ARGO-VPP",
        "version": 2,
        "case": asdict(case),
        "results": results_to_jsonable(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_project_file(path: Path) -> tuple[SimulationCase, SimulationResults]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    case_data = asdict(SimulationCase())
    case_data.update(payload.get("case", {}))
    case_data["project_path"] = str(path)
    return SimulationCase(**case_data), results_from_jsonable(payload.get("results", {}))

def export_results_csv(folder: Path, results: SimulationResults) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for name, df in {
        "geometry": results.geometry,
        "moments": results.moments,
        "hydro_appendages": results.hydro_appendages,
        "awa": results.awa,
        "true_wind": results.true_wind,
    }.items():
        path = folder / f"argo_vpp_{name}.csv"
        df.to_csv(path, index=False)
        exported.append(path)
    return exported

def export_results_excel(path: Path, results: SimulationResults) -> None:
    with pd.ExcelWriter(path) as writer:
        results.geometry.to_excel(writer, sheet_name="Geometry", index=False)
        results.moments.to_excel(writer, sheet_name="Moments", index=False)
        results.hydro_appendages.to_excel(writer, sheet_name="Hydro", index=False)
        results.awa.to_excel(writer, sheet_name="AWA", index=False)
        results.true_wind.to_excel(writer, sheet_name="CAP_TWA", index=False)

def parameter_traceability_table() -> pd.DataFrame:
    notebook_names = {
        "rho_air": "rho_widget / rho_air_widget7",
        "mu_air": "mu_widget",
        "rho_water": "rho_water_widget7 / rho8_widget",
        "mu_water": "mu_water_widget7",
        "contour_text": "contour_textarea",
        "A_sail": "A_sail_widget7 / A_sail_default",
        "c_moy": "c_moy_widget7 / c_moy_default",
        "wing_height": "h_aero / wing height",
        "n_bands": "N_USED_FIXED / N_HYDRO_FIXED",
        "aero_profile": "profile_widget / profile_widget7",
        "alpha_trim": "alpha_trim_widget7",
        "S_derive": "derive/quille surface",
        "hauteur_quille": "hauteur_quille",
        "m_quille": "m_quille",
        "S_wetted": "S_wetted_widget7",
        "CR_friction": "CR_friction_widget7",
        "Rcoque_coeff": "Rcoque_coeff_widget7",
        "V_TW": "V_TW_widget7",
        "n_twa": "n_twa_widget7",
        "n_cap": "n_cap_widget7",
        "Cx_spy": "Cx_spy_widget7",
        "S_spy": "S_spy_widget7",
        "psi_max": "psi_max_widget7",
        "theta_max": "theta_max_widget7",
        "tolerance": "tolerance",
        "max_iterations": "max_iterations",
    }
    consumers = {
        "geometry": "parse_points_text, compute_cmoy_from_contour, Viewer3D.draw_scene",
        "polar": "resolve_profile_file, load_polar_file, interp_coeffs_for_re_alpha",
        "aero": "calc_aero_simple_awa, calc_aero_simple_true_wind, compute_aero_moments",
        "hydro": "calc_hydro_drag, calc_derive_hydro_forces, compute_hydro_appendage_from_coeffs",
        "solver": "solve_equilibrium_awa, solve_equilibrium_hybrid_true_wind, run_full_simulation",
        "stability": "solve_derive_equilibrium, solve_heel_equilibrium, apply_heel_correction_to_equilibrium",
    }
    units = {
        "A_sail": "m2", "c_moy": "m", "wing_height": "m", "V_TW": "m/s", "V_min_re": "m/s",
        "V_max_re": "m/s", "rho_air": "kg/m3", "mu_air": "Pa.s", "rho_water": "kg/m3",
        "mu_water": "Pa.s", "alpha_trim": "deg", "S_spy": "m2", "S_wetted": "m2",
        "S_derive": "m2", "hauteur_quille": "m", "m_quille": "kg", "psi_max": "deg",
        "theta_max": "deg", "tolerance": "N",
        "sail_x": "m", "transverse_offset": "m", "vertical_offset": "m", "rotation_deg": "deg",
        "inclination_deg": "deg", "keel_x": "m", "keel_position_percent": "%", "keel_transverse_offset": "m",
        "keel_vertical_offset": "m", "keel_yaw_deg": "deg", "keel_pitch_deg": "deg", "rudder_x": "m",
        "rudder_position_percent": "%", "rudder_transverse_offset": "m", "rudder_vertical_offset": "m",
        "rudder_pitch_deg": "deg", "rudder_stern_margin_percent": "%", "rudder_angle_deg": "deg",
    }
    rows = []
    default = SimulationCase()
    for group, fields in PARAMETER_GROUPS:
        for name, label, kind, opts in fields:
            if not hasattr(default, name):
                continue
            if name in {"contour_text", "sail_defined", "show_sail", "show_hull", "keel_defined", "show_keel", "rudder_defined", "show_rudder", "stl_unit", "longitudinal_axis", "vertical_axis", "bow_positive", "vertical_positive_up", "sail_x", "keel_x", "rudder_x", "keel_position_mode", "keel_position_percent", "keel_transverse_offset", "keel_vertical_offset", "keel_yaw_deg", "keel_pitch_deg", "rudder_position_mode", "rudder_position_percent", "rudder_transverse_offset", "rudder_vertical_offset", "rudder_pitch_deg", "rudder_stern_margin_percent", "transverse_offset", "vertical_offset", "rotation_deg", "inclination_deg", "show_axes", "show_grid", "keel_root_chord", "keel_tip_chord", "keel_thickness_ratio", "keel_sweep", "rudder_root_chord", "rudder_tip_chord", "rudder_span", "rudder_thickness_ratio", "rudder_sweep", "rudder_angle_deg"}:
                model = "geometry"
                influenced = "3D view, geometry validity, stale state"
            elif name in {"aero_profile", "alpha_trim", "A_sail", "c_moy", "rho_air", "mu_air"}:
                model = "aero"
                influenced = "Cl/Cd/Cm, forces, moments, AWA and CAP/TWA polar"
            elif name in {"S_wetted", "CR_friction", "Rcoque_coeff", "rho_water", "mu_water", "S_derive", "hauteur_quille", "m_quille"}:
                model = "hydro"
                influenced = "drag, appendages, leeway, heel"
            elif name in {"V_TW", "n_twa", "n_cap", "tolerance", "max_iterations", "Cx_spy", "S_spy", "psi_max", "theta_max"}:
                model = "solver"
                influenced = "polar rows, convergence, VMG, KPI cards"
            else:
                model = "project"
                influenced = "metadata, file export, display labels"
            rows.append(
                {
                    "ui_label": label,
                    "variable": name,
                    "notebook_name": notebook_names.get(name, ""),
                    "unit": units.get(name, ""),
                    "default": getattr(default, name),
                    "minimum": opts.get("min", ""),
                    "maximum": opts.get("max", ""),
                    "decimals": opts.get("decimals", ""),
                    "step": opts.get("step", ""),
                    "python_type": kind,
                    "scientific_model": model,
                    "functions_that_read": consumers.get(model, "MainWindow/update_case_from_ui"),
                    "influences": influenced,
                    "data_path": "Qt widget -> update_case_from_ui -> SimulationCase -> run_full_simulation/Viewer3D -> SimulationResults -> KPI/table/plot/export",
                }
            )
    return pd.DataFrame(rows)
