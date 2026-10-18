from __future__ import annotations

from typing import Any, Callable
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields
import json
import numpy as np
import scipy.optimize as opt
import pandas as pd
import trimesh
import warnings
from .argo_vpp_nb_aerodynamics import (
    apparent_wind,
)
from .argo_vpp_nb_appendages import (
    build_foil_mesh_from_contour,
    chord_at_span,
)
from .argo_vpp_nb_hydrodynamics import (
    _scale_low_fn,
    bare_hull_residuary,
    centerboard_residuary,
    direct_hull_wetted_surface_delta,
    froude_number,
    orc_appendage_viscous,
    rail_under_drag,
    wetted_surface_heel_delta,
)
from .argo_vpp_nb_hydrostatics import (
    check_delft_domain,
)
from .argo_vpp_nb_models import (
    AeroState,
    BoatGeometryState,
    C,
    CLR_DRAFT_FACTOR,
    EquilibriumState,
    FoilGeometry,
    GeometryInputs,
    HEEL_RANGE_MAX_DEG,
    HEEL_RANGE_MIN_DEG,
    HEEL_RANGE_STEPS,
    HullHydrostatics,
    MassProperties,
    PhysicalConstants,
    ResistanceBreakdown,
    SolverSettings,
    TWA_RANGE_MAX_DEG,
    TWA_RANGE_MIN_DEG,
    TWA_RANGE_STEPS,
    deg_to_rad,
    rad_to_deg,
    safe_div,
)
from .argo_vpp_nb_stability import (
    analysis_angle_values,
    interpolate_gz_state,
    rotate_points_about_x,
)
from .argo_vpp_nb_tables import (
    NACA_POLAR_CACHE_DIAGNOSTICS,
    PolarDomainError,
    TableDomainError,
    bounded_table_row,
    cached_naca_polar,
    finite_wing_from_profile,
    fn_domain,
    global_common_alpha_domain,
    interpolate_naca_polar,
    load_delft_tables,
    normalize_naca_code,
    table_domain,
)

__all__ = ['AERO_MODEL_TO_SAIL_DATA', 'AdvancedARGOSettings', 'DEFAULT_MANUAL_MICROTRANSAT', 'DEFAULT_MANUAL_PYTHON_VPP_DARING', 'EnvironmentSettings', 'KeelInput', 'MainSailInput', 'ManualVPPInput', 'PEDRO_HAZEN_AWA', 'PEDRO_HAZEN_TABLES', 'PEDRO_ORC_AWA', 'PEDRO_ORC_TABLES', 'PEDRO_SPIN_AWA', 'PEDRO_SPIN_TABLES', 'RESULT_COLUMNS', 'RigInput', 'RudderInput', 'SAIL_DATA_TO_DEFAULT_AERO_MODEL', 'SailSetInput', 'YachtParticulars', '_dataclass_kwargs', '_failure_reasons', '_interp_pedro_table', '_manual_foil_from_input', '_manual_gz_curve', '_manual_sail_contour', '_manual_trapezoid_contour', 'aerodynamic_state', 'aerodynamic_state_pedro', 'alpha_candidates_for_condition', 'build_manual_preview_hull', 'build_model_from_manual_input', 'clone_manual_input', 'compute_pedro_sail_geometry', 'equilibrium_state', 'export_vpp_results_csv', 'get_pedro_sail_coefficients', 'heeled_hull_residuary_delta', 'heeled_keel_residuary_delta', 'hull_viscous_resistance', 'induced_hydrodynamic_drag', 'is_pedro_aero_model', 'manual_input_from_dict', 'manual_input_from_json', 'manual_input_to_json', 'normalize_aero_selection', 'normalize_solver_method', 'pedro_fractionality_coefficient', 'pedro_headsail_usage_factor', 'process_specifications', 'resistance_breakdown', 'resolve_manual_aero_model', 'rmv_correction', 'row_from_solution', 'scaled_residual_vector', 'section_area_at_x', 'solve_condition_with_alpha_mode', 'solve_one_condition', 'solve_vpp_grid', 'solver_heel_bounds', 'solver_speed_bounds', 'tws_to_mps', 'wind_speed_at_height']

def rmv_correction(V_B: float, heel_rad: float, state: BoatGeometryState) -> float:
    """OceanVPP dynamic righting correction.

    Units: displacement mass in kg; LWL, B_CB and AMS_CB in metres; heel angle in radians;
    speed-length ratio SLR = V_B / sqrt(LWL) with V_B in m/s. The returned value is signed
    by the formula factor and is added to the available righting moment. A negative factor is
    therefore a loss of dynamic stability, not forced positive.
    """

    if not state.settings.use_rmv:
        return 0.0
    hydro = state.hydro
    disp = state.masses.total_mass_kg
    SLR = safe_div(V_B, np.sqrt(max(hydro.LWL, 1e-9)), default=0.0)
    factor = 1.0 - 6.25 * safe_div(hydro.B_CB, np.sqrt(max(hydro.AMS_CB, 1e-12)), default=0.0) - 2.1
    return float((5.955e-5 / 3.0) * disp * hydro.LWL * factor * SLR * abs(heel_rad))

def equilibrium_state(
    V_B: float,
    heel_rad: float,
    TWS_mps: float,
    TWA_deg: float,
    state: BoatGeometryState,
    alpha_override_deg: float | None = None,
    flat_override: float | None = None,
    reef_override: float | None = None,
) -> EquilibriumState:
    """Evaluate the 2-DOF residuals using the waterline interpolated at heel."""

    gz_state = interpolate_gz_state(state.gz_curve, rad_to_deg(heel_rad))
    z_WL = float(gz_state["waterline_z_m"])
    aero = aerodynamic_state(TWS_mps, TWA_deg, V_B, heel_rad, state, alpha_override_deg=alpha_override_deg, flat_override=flat_override, reef_override=reef_override, waterline_z_m=z_WL)
    resistance = resistance_breakdown(V_B, heel_rad, aero.F_side_N, state)
    z_CLR = z_WL - CLR_DRAFT_FACTOR * float(state.T_max_m)
    z_CE = float(aero.z_CE_m)
    if is_pedro_aero_model(getattr(aero, "aero_model", "")) and getattr(state.settings, "pedro_hardcoded_hydroheel", True):
        M_aero_about_WL = aero.M_aero_about_WL_Nm
        M_hydroheel = aero.pedro_hydroheel_Nm if np.isfinite(aero.pedro_hydroheel_Nm) else 0.0
    else:
        M_aero_about_WL = aero.F_side_N * (z_CE - z_WL)
        M_hydroheel = aero.F_side_N * (z_WL - z_CLR)
    M_heel = M_aero_about_WL + M_hydroheel
    M_hydrostatic = float(gz_state["M_hydrostatic_restoring_Nm"])
    RMV = rmv_correction(V_B, heel_rad, state)
    M_right = M_hydrostatic + RMV
    invalid_parts: list[str] = []
    failure_codes: list[str] = []
    if not aero.naca_valid:
        invalid_parts.append(aero.naca_warning or "polaire NACA invalide")
        failure_codes.append("naca_invalid")
    if not resistance.delft_valid:
        invalid_parts.append(resistance.invalid_reason or "domaine Delft invalide")
        failure_codes.append("delft_component_invalid")
    if not state.hydrostatic_geometry_valid:
        invalid_parts.append("invalid_hydrostatic_geometry")
        failure_codes.append("invalid_hydrostatic_geometry")
    return EquilibriumState(
        aero=aero,
        resistance=resistance,
        M_hydroheel_Nm=M_hydroheel,
        M_heel_total_Nm=M_heel,
        M_hydrostatic_Nm=M_hydrostatic,
        RMV_Nm=RMV,
        M_right_total_Nm=M_right,
        force_residual_N=aero.F_drive_N - resistance.R_total_N,
        moment_residual_Nm=M_heel - M_right,
        Fn=froude_number(V_B, state.hydro.LWL),
        z_CE_m=z_CE,
        z_WL_m=z_WL,
        z_CLR_m=z_CLR,
        M_aero_about_WL_Nm=M_aero_about_WL,
        table_domain_valid=resistance.table_domain_valid,
        naca_valid=aero.naca_valid,
        hydrostatic_geometry_valid=bool(state.hydrostatic_geometry_valid),
        failure_reason_code=";".join(failure_codes),
        invalid_reason="; ".join(part for part in invalid_parts if part),
    )

def scaled_residual_vector(
    x: np.ndarray,
    TWS_mps: float,
    TWA_deg: float,
    state: BoatGeometryState,
    alpha_override_deg: float | None = None,
    flat_override: float | None = None,
    reef_override: float | None = None,
) -> np.ndarray:
    """Scaled residual vector [force, moment] for least_squares."""

    V_B = float(x[0])
    heel = float(x[1])
    try:
        eq = equilibrium_state(V_B, heel, TWS_mps, TWA_deg, state, alpha_override_deg=alpha_override_deg, flat_override=flat_override, reef_override=reef_override)
    except (TableDomainError, PolarDomainError):
        return np.array([1e6, 1e6], dtype=float)
    force_scale = max(0.5 * C.RHO_AIR * TWS_mps**2 * state.wing.planform_area_m2, 1.0)
    moment_scale = max(force_scale * max(state.wing.span_m, 1.0), 1.0)
    return np.array([
        eq.force_residual_N / force_scale,
        eq.moment_residual_Nm / moment_scale,
    ])

def solve_condition_with_alpha_mode(TWS_mps: float, TWA_deg: float, state: BoatGeometryState, x0: np.ndarray | None = None) -> dict[str, Any]:
    """Solve one wind condition in manual or automatic alpha mode."""

    if is_pedro_aero_model(getattr(state.settings, "aero_model", "naca_wing")):
        mi = getattr(state, "manual_input", None)
        sailset = mi.sailset if mi is not None else SailSetInput()
        if sailset.optimize_flat or sailset.optimize_flat_reef:
            flat_values = np.linspace(float(sailset.flat_min), 1.0, 8)
        else:
            flat_values = np.array([float(sailset.flat)])
        if sailset.optimize_reef or sailset.optimize_flat_reef:
            reef_values = np.linspace(float(sailset.reef_min), 1.0, 8)
        else:
            reef_values = np.array([float(sailset.reef)])
        best_valid: dict[str, Any] | None = None
        best_invalid: dict[str, Any] | None = None
        valid_count = 0
        candidate_count = int(len(flat_values) * len(reef_values))
        seed = x0
        for flat in flat_values:
            for reef in reef_values:
                solution = solve_one_condition(TWS_mps, TWA_deg, state, seed, flat_override=float(flat), reef_override=float(reef))
                solution["flat_reef_candidate_count"] = candidate_count
                if solution.get("success"):
                    valid_count += 1
                    seed = np.array([solution["boat_speed_mps"], solution["heel_rad"]])
                    if best_valid is None or solution["boat_speed_mps"] > best_valid["boat_speed_mps"]:
                        best_valid = solution
                elif best_invalid is None:
                    best_invalid = solution
        best = best_valid if best_valid is not None else best_invalid
        if best is None:
            return {"boat_speed_mps": np.nan, "heel_rad": np.nan, "success": False, "message": "solver_no_convergence", "eq": None, "failure_reason_code": "solver_no_convergence", "invalid_reason": "solver_no_convergence", "flat_reef_candidate_count": candidate_count, "flat_reef_valid_candidate_count": valid_count}
        best["flat_reef_candidate_count"] = candidate_count
        best["flat_reef_valid_candidate_count"] = valid_count
        best["alpha_candidate_count"] = 0
        best["alpha_valid_candidate_count"] = 0
        best["alpha_mode"] = "flat_reef"
        return best

    candidates, candidate_count = alpha_candidates_for_condition(state)
    if not candidates:
        return {
            "boat_speed_mps": np.nan,
            "heel_rad": np.nan,
            "success": False,
            "message": "Aucune incidence dans le domaine de la polaire NACA.",
            "eq": None,
            "invalid_reason": "Aucune incidence dans le domaine de la polaire NACA.",
            "alpha_candidate_count": candidate_count,
            "alpha_valid_candidate_count": 0,
            "alpha_mode": state.inputs.wing_alpha_mode,
        }
    best_valid: dict[str, Any] | None = None
    best_invalid: dict[str, Any] | None = None
    valid_count = 0
    seed = x0
    for alpha in candidates:
        solution = solve_one_condition(TWS_mps, TWA_deg, state, seed, alpha_override_deg=alpha)
        solution["alpha_candidate_count"] = candidate_count
        solution["alpha_mode"] = state.inputs.wing_alpha_mode
        if solution.get("success"):
            valid_count += 1
            seed = np.array([solution["boat_speed_mps"], solution["heel_rad"]])
            if best_valid is None or solution["boat_speed_mps"] > best_valid["boat_speed_mps"]:
                best_valid = solution
        else:
            if best_invalid is None:
                best_invalid = solution
    best = best_valid if best_valid is not None else best_invalid
    best["alpha_valid_candidate_count"] = valid_count
    best["alpha_candidate_count"] = candidate_count
    best["alpha_mode"] = state.inputs.wing_alpha_mode
    return best

RESULT_COLUMNS = [
    "TWS_kn", "TWS_mps", "TWA_deg", "AWA_deg", "AWS_mps", "AWS_kn",
    "boat_speed_mps", "boat_speed_kn", "heel_deg", "Fn", "alpha_wing_deg",
    "wing_orientation_deg", "alpha_mode", "alpha_candidate_count",
    "alpha_valid_candidate_count", "Cl_wing", "Cd_wing", "Cm_wing",
    "CD_profile", "CD_induced", "Re_wing", "Re_lower", "Re_upper",
    "Re_reference", "Re_domain_min", "Re_domain_max", "alpha_domain_min_deg",
    "alpha_domain_max_deg", "profile_naca", "polar_file", "naca_valid",
    "naca_warning", "F_drive_N", "F_side_aero_N", "F_side_signed_N",
    "F_side_hydro_assumed_N", "R_hull_viscous_N", "R_centerboard_viscous_N",
    "R_rudder_viscous_N", "R_hull_residuary_N", "R_centerboard_residuary_N",
    "R_heel_friction_delta_N", "R_heel_hull_residuary_delta_N",
    "R_heel_keel_residuary_delta_N", "R_induced_N", "T_eff_m", "q_water_Pa",
    "R_rail_under_N", "R_total_N", "z_CE_m", "z_WL_m", "z_CLR_m", "T_max_m",
    "M_aero_Nm", "M_aero_about_WL_Nm", "M_hydroheel_Nm", "M_heel_total_Nm",
    "M_hydrostatic_Nm", "RMV_Nm", "M_right_total_Nm", "force_residual_N",
    "moment_residual_Nm", "delft_valid", "table_domain_valid",
    "solver_success", "invalid_reason", "solver_message",
]

def solver_heel_bounds(state: BoatGeometryState) -> tuple[float, float]:
    """Apply the global heel range, limited by available GZ and hydrodynamic table data."""

    gz_hi = float(state.gz_curve["phi_deg"].max())
    _, eff_hi = table_domain(state.delft_tables, "effective_span", "heel_deg")
    hi = min(HEEL_RANGE_MAX_DEG, gz_hi, eff_hi)
    if state.settings.wetted_surface_mode == "Delft":
        _, wet_hi = table_domain(state.delft_tables, "wetted_surface_heel", "heel_deg")
        hi = min(hi, wet_hi)
    if HEEL_RANGE_MIN_DEG >= hi:
        raise ValueError(f"Heel range min {HEEL_RANGE_MIN_DEG:g} deg must be below its data-limited max {hi:g} deg.")
    return deg_to_rad(HEEL_RANGE_MIN_DEG), deg_to_rad(hi)

def alpha_candidates_for_condition(state: BoatGeometryState) -> tuple[list[float], int]:
    """Build alpha candidates using the global common alpha domain of the selected polar."""

    profile_code = normalize_naca_code(state.inputs.wing_naca)
    _, polar = cached_naca_polar(profile_code, state.inputs.polar_folder)
    alpha_common_min, alpha_common_max = global_common_alpha_domain(polar)
    mode = str(state.inputs.wing_alpha_mode).strip().lower()
    if mode == "automatique":
        step = max(float(state.inputs.alpha_step_deg), 1e-6)
        lo = max(float(state.inputs.alpha_min_deg), alpha_common_min)
        hi = min(float(state.inputs.alpha_max_deg), alpha_common_max)
        if lo > hi:
            return [], 0
        values = np.arange(lo, hi + 0.5 * step, step)
        values = values[(values >= alpha_common_min - 1e-12) & (values <= alpha_common_max + 1e-12)]
        return [float(v) for v in values], int(len(values))
    return [float(state.inputs.fixed_alpha_deg)], 1

def _failure_reasons(result: Any, eq: EquilibriumState, state: BoatGeometryState) -> tuple[bool, str, str]:
    reasons = []
    codes = []
    if not bool(getattr(result, "success", False)):
        codes.append("solver_no_convergence")
        reasons.append("solver_no_convergence")
    if eq.aero.F_drive_N < 0:
        codes.append("negative_driving_force")
        reasons.append("negative_driving_force")
    if abs(eq.force_residual_N) > state.settings.force_tol_N:
        codes.append("force_residual_too_high")
        reasons.append(f"force_residual_too_high: {eq.force_residual_N:.4g} N")
    if abs(eq.moment_residual_Nm) > state.settings.moment_tol_Nm:
        codes.append("moment_residual_too_high")
        reasons.append(f"moment_residual_too_high: {eq.moment_residual_Nm:.4g} N.m")
    if not eq.naca_valid:
        codes.append("naca_invalid")
    if not eq.table_domain_valid or not eq.resistance.delft_valid:
        codes.append("delft_component_invalid")
    if not eq.hydrostatic_geometry_valid:
        codes.append("invalid_hydrostatic_geometry")
    if eq.invalid_reason:
        reasons.append(eq.invalid_reason)
    success = not codes or codes == ["negative_driving_force"] and abs(eq.force_residual_N) <= state.settings.force_tol_N
    success = bool(getattr(result, "success", False)) and abs(eq.force_residual_N) <= state.settings.force_tol_N and abs(eq.moment_residual_Nm) <= state.settings.moment_tol_Nm and eq.naca_valid and eq.table_domain_valid and eq.resistance.delft_valid and eq.hydrostatic_geometry_valid
    return success, "; ".join(dict.fromkeys(codes)), "; ".join(dict.fromkeys(reasons))

def solve_one_condition(
    TWS_mps: float,
    TWA_deg: float,
    state: BoatGeometryState,
    x0: np.ndarray | None = None,
    alpha_override_deg: float | None = None,
    flat_override: float | None = None,
    reef_override: float | None = None,
    solver_method: str | None = None,
) -> dict[str, Any]:
    """Solve one (TWS, TWA, alpha) condition with physical speed/heel lower bounds."""

    speed_lo, speed_hi = solver_speed_bounds(state)
    heel_lo, heel_hi = solver_heel_bounds(state)
    bounds = ([speed_lo, heel_lo], [speed_hi, heel_hi])
    method = str(solver_method or getattr(state.settings, "solver_method", "least_squares_multistart")).strip().lower()
    if method not in {"least_squares_multistart", "least_squares_single"}:
        raise ValueError(f"Unsupported solver_method {method!r}; choose least_squares_multistart or least_squares_single.")
    seeds: list[np.ndarray] = []
    if x0 is not None and np.all(np.isfinite(x0)):
        seeds.append(np.clip(x0, bounds[0], bounds[1]))
    if method == "least_squares_single":
        if not seeds:
            seeds.append(np.clip(np.array([speed_lo + 0.35 * (speed_hi - speed_lo), deg_to_rad(4.0)]), bounds[0], bounds[1]))
        else:
            seeds = seeds[:1]
    else:
        for frac in [0.0, 0.15, 0.35, 0.7, 1.0]:
            speed_seed = speed_lo + frac * (speed_hi - speed_lo)
            for heel_deg in sorted(set([0.0, 0.5, 2.0, 4.0, 8.0, 20.0, rad_to_deg(heel_hi)])):
                seeds.append(np.clip(np.array([speed_seed, deg_to_rad(heel_deg)]), bounds[0], bounds[1]))

    best_valid: dict[str, Any] | None = None
    best_invalid: dict[str, Any] | None = None
    for seed in seeds:
        iteration_count = 0

        def count_solver_iteration(intermediate_result: Any) -> None:
            nonlocal iteration_count
            iteration_count += 1

        try:
            result = opt.least_squares(
                scaled_residual_vector,
                seed,
                bounds=bounds,
                args=(TWS_mps, TWA_deg, state, alpha_override_deg, flat_override, reef_override),
                max_nfev=state.settings.max_nfev,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
                callback=count_solver_iteration,
            )
            eq = equilibrium_state(float(result.x[0]), float(result.x[1]), TWS_mps, TWA_deg, state, alpha_override_deg=alpha_override_deg, flat_override=flat_override, reef_override=reef_override)
            success, code, reason = _failure_reasons(result, eq, state)
            candidate = {
                "result": result,
                "eq": eq,
                "success": success,
                "message": str(result.message),
                "cost": float(result.cost),
                "solver_iterations": iteration_count,
                "boat_speed_mps": float(result.x[0]),
                "heel_rad": float(result.x[1]),
                "failure_reason_code": "" if success else code,
                "invalid_reason": "" if success else (reason or "solver_no_convergence"),
            }
            if success:
                if best_valid is None or candidate["boat_speed_mps"] > best_valid["boat_speed_mps"]:
                    best_valid = candidate
            elif best_invalid is None or candidate["cost"] < best_invalid["cost"]:
                best_invalid = candidate
        except (TableDomainError, PolarDomainError, FileNotFoundError, ValueError, FloatingPointError, RuntimeError) as exc:
            text = str(exc)
            code = "naca_invalid" if "naca_" in text else ("delft_component_invalid" if "delft_" in text or "heel_" in text else "numerical_error")
            candidate = {"result": None, "eq": None, "success": False, "message": text, "cost": np.inf, "boat_speed_mps": np.nan, "heel_rad": np.nan, "solver_iterations": iteration_count, "failure_reason_code": code, "invalid_reason": text}
            if best_invalid is None:
                best_invalid = candidate
    best = best_valid if best_valid is not None else best_invalid
    if best is None or best["eq"] is None:
        return {"boat_speed_mps": np.nan, "heel_rad": np.nan, "success": False, "message": "solver_no_convergence", "eq": None, "failure_reason_code": "solver_no_convergence", "invalid_reason": "solver_no_convergence", "solver_method": method, "solver_iterations": np.nan}
    out = {"boat_speed_mps": float(best["boat_speed_mps"]), "heel_rad": float(best["heel_rad"]), "success": bool(best["success"]), "message": best["message"], "eq": best["eq"], "failure_reason_code": best.get("failure_reason_code", ""), "invalid_reason": best.get("invalid_reason", ""), "solver_method": method, "solver_iterations": best.get("solver_iterations", np.nan)}
    if flat_override is not None:
        out["flat"] = float(flat_override)
    if reef_override is not None:
        out["reef"] = float(reef_override)
    return out

RESULT_COLUMNS = [
    "TWS_kn", "TWS_mps", "TWA_deg", "AWA_deg", "AWS_mps", "AWS_kn",
    "boat_speed_mps", "boat_speed_kn", "heel_deg", "Fn", "alpha_wing_deg",
    "wing_orientation_deg", "alpha_mode", "alpha_candidate_count",
    "alpha_valid_candidate_count", "Cl_wing", "Cd_wing", "Cm_wing",
    "CD_profile", "CD_induced", "Re_wing", "Re_lower", "Re_upper",
    "Re_low_used", "Re_high_used", "Re_reference", "Re_domain_min", "Re_domain_max",
    "alpha_domain_min_deg", "alpha_domain_max_deg", "alpha_min_valid_deg",
    "alpha_max_valid_deg", "profile_naca", "polar_file", "naca_valid",
    "naca_warning", "naca_cache_file_reads", "naca_cache_hits",
    "F_drive_N", "F_side_aero_N", "F_side_signed_N",
    "F_side_hydro_assumed_N", "R_hull_viscous_N", "R_centerboard_viscous_N",
    "R_rudder_viscous_N", "R_hull_residuary_N", "R_centerboard_residuary_N",
    "R_heel_friction_delta_N", "R_heel_hull_residuary_delta_N",
    "R_heel_keel_residuary_delta_N", "R_induced_N", "T_eff_m", "q_water_Pa",
    "R_rail_under_N", "R_total_N", "hull_residuary_valid",
    "appendage_residuary_valid", "heel_friction_valid", "heel_hull_residuary_valid",
    "heel_appendage_residuary_valid", "effective_span_valid",
    "used_low_fn_physical_extension", "low_fn_extension_tables",
    "z_CE_m", "z_WL_m", "z_CLR_m", "T_max_m",
    "M_aero_Nm", "M_aero_about_WL_Nm", "M_hydroheel_Nm", "M_heel_total_Nm",
    "M_hydrostatic_Nm", "RMV_Nm", "M_right_total_Nm", "force_residual_N",
    "moment_residual_Nm", "solver_iterations", "delft_valid", "delft_geometry_valid",
    "table_domain_valid", "hydrostatic_geometry_valid", "hydrostatic_boolean_union_used",
    "mesh_overlap_detected", "estimated_overlap_volume_m3", "solver_success",
    "failure_reason_code", "invalid_reason", "solver_message",
]

@dataclass
class EnvironmentSettings:
    twa_min_deg: float = TWA_RANGE_MIN_DEG
    twa_max_deg: float = TWA_RANGE_MAX_DEG
    twa_steps: int = TWA_RANGE_STEPS
    tws_min: float = 8.0
    tws_max: float = 12.0
    tws_unit: str = "knots"
    Hs_m: float = 0.0
    Ts_s: float = 0.0

@dataclass
class YachtParticulars:
    name: str = "Daring"
    Lwl_m: float = 7.01
    displaced_volume_m3: float = 1.95
    Bwl_m: float = 1.70
    Tc_m: float = 0.45
    Swet_m2: float = 11.5
    Tmax_m: float = 1.35
    Amax_m2: float = 0.38
    displacement_mass_kg: float = 2000.0
    Ff_m: float = 0.75
    Fa_m: float = 0.55
    Boa_m: float = 1.98
    Loa_m: float = 9.9
    hull_roughness_ks_um: float = 150.0
    Cp: float | None = None
    Cm: float | None = None
    LCB_pct_midship: float = -4.0
    LCF_pct_midship: float | None = None

@dataclass
class MainSailInput:
    main_sail_type: str = "main"
    name: str = "MN1"
    P_m: float = 10.8
    E_m: float = 3.3
    roach: float = 0.1
    BAD_m: float = 0.8
    sail_area_m2: float | None = None
    span_m: float | None = None
    mean_chord_m: float | None = None
    aspect_ratio: float | None = None
    naca_profile: str = "NACA0012"
    alpha_min_deg: float = -5.0
    alpha_max_deg: float = 20.0
    alpha_step_deg: float = 1.0
    fixed_alpha_deg: float = 6.0
    efficiency_factor_e: float = 0.85

@dataclass
class KeelInput:
    keel_type: str = "short"
    Lkeel_m: float = 1.2
    Dkeel_m: float = 0.9
    thickness_ratio: float = 0.15
    root_chord_m: float | None = None
    tip_chord_m: float | None = None
    span_m: float | None = None
    x_position_m: float | None = None
    mass_kg: float = 0.0
    wetted_area_m2: float | None = None

@dataclass
class RudderInput:
    Cu_m: float = 0.32
    Cl_m: float = 0.18
    b_m: float = 0.75
    thickness_ratio: float = 0.12
    x_position_m: float | None = None
    mass_kg: float = 0.0
    wetted_area_m2: float | None = None

@dataclass
class RigInput:
    I_m: float = 0.0
    J_m: float = 0.0
    LPG_m: float = 0.0
    SL_m: float = 0.0
    EHM_m: float | None = None
    EMDC_m: float | None = None
    HBI_m: float = 0.0
    BAS_m: float | None = None
    fractionality: float | None = None
    backstay_config: str = "none"
    checkstays: bool = False
    low_tech_sailcloth: bool = False

@dataclass
class SailSetInput:
    sailset: str = "main_only"
    spinnaker_type: str = "symmetric"
    reference_area_mode: str = "active_area"
    flat: float = 1.0
    reef: float = 1.0
    optimize_flat_reef: bool = False
    optimize_flat: bool = False
    optimize_reef: bool = False
    flat_min: float = 0.42
    reef_min: float = 0.70

@dataclass
class AdvancedARGOSettings:
    geometry_input_mode: str = "manual"
    hydro_model: str = "oceanvpp_delft"
    mass_mode: str = "fixed_loading"
    aero_model: str = "pedro_orc_low"
    use_wind_gradient: bool = False
    wind_reference_height_m: float = 10.0
    wind_roughness_z0_m: float = 0.005
    delft_table_folder: str = "Keuning_coefficients_tables/VPP_coefficients_tables"
    polar_folder: str = "Profil NACA"
    manual_GM_m: float = 0.55
    orc_LSM1_m: float | None = None
    pedro_hardcoded_hydroheel: bool = True

@dataclass
class ManualVPPInput:
    solver: SolverSettings = field(default_factory=lambda: SolverSettings(geometry_input_mode="manual", mass_mode="fixed_loading"))
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    yacht: YachtParticulars = field(default_factory=YachtParticulars)
    main_sail: MainSailInput = field(default_factory=MainSailInput)
    rig: RigInput = field(default_factory=RigInput)
    sailset: SailSetInput = field(default_factory=SailSetInput)
    keel: KeelInput = field(default_factory=KeelInput)
    rudder: RudderInput = field(default_factory=RudderInput)
    advanced: AdvancedARGOSettings = field(default_factory=AdvancedARGOSettings)

def _dataclass_kwargs(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {key: value for key, value in dict(data).items() if key in allowed}

def manual_input_from_dict(data: dict[str, Any]) -> ManualVPPInput:
    data = dict(data)
    solver_data = data.get("solver", {})
    advanced_data = data.get("advanced", {})
    return ManualVPPInput(
        solver=SolverSettings(**_dataclass_kwargs(SolverSettings, solver_data)),
        environment=EnvironmentSettings(**_dataclass_kwargs(EnvironmentSettings, data.get("environment", {}))),
        yacht=YachtParticulars(**_dataclass_kwargs(YachtParticulars, data.get("yacht", {}))),
        main_sail=MainSailInput(**_dataclass_kwargs(MainSailInput, data.get("main_sail", {}))),
        rig=RigInput(**_dataclass_kwargs(RigInput, data.get("rig", {}))),
        sailset=SailSetInput(**_dataclass_kwargs(SailSetInput, data.get("sailset", {}))),
        keel=KeelInput(**_dataclass_kwargs(KeelInput, data.get("keel", {}))),
        rudder=RudderInput(**_dataclass_kwargs(RudderInput, data.get("rudder", {}))),
        advanced=AdvancedARGOSettings(**_dataclass_kwargs(AdvancedARGOSettings, advanced_data)),
    )

def clone_manual_input(manual_input: ManualVPPInput) -> ManualVPPInput:
    return manual_input_from_dict(asdict(manual_input))

def manual_input_to_json(manual_input: ManualVPPInput, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(asdict(manual_input), indent=2), encoding="utf-8")
    return path

def manual_input_from_json(path: str | Path) -> ManualVPPInput:
    return manual_input_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

def export_vpp_results_csv(results: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path

def tws_to_mps(value: float, unit: str) -> float:
    unit_normalized = str(unit).strip().lower()
    if unit_normalized in {"kn", "knot", "knots"}:
        return float(value) * C.KNOT_TO_MPS
    if unit_normalized in {"m/s", "mps", "ms-1"}:
        return float(value)
    raise ValueError(f"Unsupported TWS unit: {unit!r}")

def process_specifications(manual_input: ManualVPPInput | None = None) -> dict[str, Any]:
    mi = normalize_aero_selection(DEFAULT_MANUAL_PYTHON_VPP_DARING if manual_input is None else manual_input)
    yacht = mi.yacht
    sail = mi.main_sail
    aero_model = resolve_manual_aero_model(mi.solver.sail_coefficient_data_source, mi.advanced.aero_model)
    keel = mi.keel
    rudder = mi.rudder
    if is_pedro_aero_model(aero_model):
        pedro_geom = compute_pedro_sail_geometry(mi)
        sail_area = pedro_geom["A_ref"]
    else:
        pedro_geom = {}
        sail_area = sail.sail_area_m2 or 0.5 * sail.P_m * sail.E_m * (1.0 + sail.roach)
    keel_projected = 0.5 * ((keel.root_chord_m or keel.Lkeel_m) + (keel.tip_chord_m or 0.6 * (keel.root_chord_m or keel.Lkeel_m))) * (keel.span_m or keel.Dkeel_m)
    rudder_projected = 0.5 * (rudder.Cu_m + rudder.Cl_m) * rudder.b_m
    hydro_mass = C.RHO_WATER * yacht.displaced_volume_m3
    warnings_out = [part for part in str(mi.solver.setup_warning_message).split("; ") if part]
    solver_method, solver_warning = normalize_solver_method(mi.solver.solver_method)
    if solver_warning:
        warnings_out.append(solver_warning)
    source_requested = str(mi.solver.sail_coefficient_data_source or "").strip().lower()
    aero_requested = str(mi.advanced.aero_model or "").strip().lower()
    allowed_aero = {"naca_wing", "pedro_hazen", "pedro_orc_low", "pedro_orc_high", "pedro_orc_medium"}
    if aero_requested and aero_requested not in allowed_aero:
        warnings_out.append(f"Unsupported aero_model {mi.advanced.aero_model!r}; using {aero_model!r}.")
    if source_requested and source_requested not in {"naca", "hazen", "orc", "custom"}:
        warnings_out.append(f"Unsupported sail_coefficient_data_source {mi.solver.sail_coefficient_data_source!r}; using {aero_model!r}.")
    if abs(yacht.displacement_mass_kg - hydro_mass) / max(hydro_mass, 1.0) > 0.05:
        warnings_out.append(f"Mass-volume mismatch: hydrostatic mass {hydro_mass:.1f} kg vs input {yacht.displacement_mass_kg:.1f} kg.")
    if yacht.hull_roughness_ks_um > 0:
        warnings_out.append("Hull roughness input stored but no explicit roughness correction is implemented.")
    if mi.environment.Hs_m > 0 or mi.environment.Ts_s > 0:
        warnings_out.append("Wave inputs stored but added resistance in waves is not implemented in this simplified model.")
    if mi.sailset.sailset == "main_only" and mi.sailset.reference_area_mode == "pedro_nominal":
        warnings_out.append("main_only with pedro_nominal uses A_main + A_fore as reference area; for strict mainsail-only comparison use active_area.")
    if mi.advanced.hydro_model == "orc_simplified":
        warnings_out.append("ORC simplified mode only changes selected hydrodynamic formulas. It is not the full ORC 2026 VPP.")
    return {
        "geometry_input_mode": mi.advanced.geometry_input_mode,
        "hydro_model": mi.advanced.hydro_model,
        "mass_mode": mi.advanced.mass_mode,
        "solver_method": solver_method,
        "sail_coefficient_data_source": mi.solver.sail_coefficient_data_source,
        "aero_model": aero_model,
        "sailset": mi.sailset.sailset,
        "tws_min_mps": tws_to_mps(mi.environment.tws_min, mi.environment.tws_unit),
        "tws_max_mps": tws_to_mps(mi.environment.tws_max, mi.environment.tws_unit),
        "sail_area_m2": sail_area,
        "pedro_A_main_m2": pedro_geom.get("A_main", np.nan),
        "pedro_A_ref_m2": pedro_geom.get("A_ref", np.nan),
        "keel_wetted_area_m2": keel.wetted_area_m2 or 2.0 * keel_projected,
        "rudder_wetted_area_m2": rudder.wetted_area_m2 or 2.0 * rudder_projected,
        "hydrostatic_mass_kg": hydro_mass,
        "warnings": warnings_out,
    }

def _manual_trapezoid_contour(root_chord: float, tip_chord: float, span: float) -> np.ndarray:
    taper_offset = 0.5 * max(root_chord - tip_chord, 0.0)
    return np.array([
        [0.0, 0.0],
        [root_chord, 0.0],
        [taper_offset + tip_chord, span],
        [taper_offset, span],
        [0.0, 0.0],
    ], dtype=float)

def _manual_sail_contour(sail: MainSailInput) -> np.ndarray:
    area = sail.sail_area_m2 or 0.5 * sail.P_m * sail.E_m * (1.0 + sail.roach)
    span = sail.span_m or sail.P_m
    base = 2.0 * area / max(span, 1e-9)
    return np.array([[0.0, 0.0], [base, 0.0], [0.0, span], [0.0, 0.0]], dtype=float)

def build_manual_preview_hull(yacht: YachtParticulars) -> trimesh.Trimesh:
    """Build a visual-only rounded hull approximation from manual yacht particulars."""

    dimensions = {
        "Lwl_m": yacht.Lwl_m,
        "Bwl_m": yacht.Bwl_m,
        "Tc_m": yacht.Tc_m,
        "Amax_m2": yacht.Amax_m2,
        "Boa_m": yacht.Boa_m,
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in dimensions.values()):
        raise ValueError("Lwl, Bwl, Tc, Amax et Boa doivent etre finis et strictement positifs pour l'apercu 3D.")
    if any(not np.isfinite(value) or value < 0.0 for value in [yacht.Ff_m, yacht.Fa_m]):
        raise ValueError("Les franc-bords avant et arriere doivent etre finis et positifs ou nuls pour l'apercu 3D.")

    fullness = float(np.clip(yacht.Amax_m2 / (yacht.Bwl_m * yacht.Tc_m), 0.05, 0.95))
    depth_fraction = np.linspace(0.0, 1.0, 81)
    candidate_exponents = np.geomspace(0.15, 20.0, 120)
    profiles = np.power(
        np.maximum(1.0 - depth_fraction[None, :] ** candidate_exponents[:, None], 0.0),
        1.0 / candidate_exponents[:, None],
    )
    profile_areas = np.sum(
        0.5 * (profiles[:, 1:] + profiles[:, :-1]) * np.diff(depth_fraction),
        axis=1,
    )
    exponent = float(candidate_exponents[int(np.argmin(np.abs(profile_areas - fullness)))])
    stations = np.linspace(0.005, 0.995, 41)
    section_depths = np.linspace(1.0, 0.0, 17)
    deck_fractions = np.linspace(0.25, 1.0, 5)
    section_size = len(section_depths) + len(deck_fractions) + 1 + len(deck_fractions) - 1 + len(section_depths) - 2
    vertices = np.empty((len(stations) * section_size + 2, 3), dtype=float)
    sections: list[np.ndarray] = []

    for station_index, station in enumerate(stations):
        longitudinal_fullness = max(0.04, float(np.sin(np.pi * station) ** 0.55))
        waterline_half_beam = 0.5 * yacht.Bwl_m * longitudinal_fullness
        deck_half_beam = 0.5 * yacht.Boa_m * longitudinal_fullness
        freeboard = float(yacht.Fa_m + station * (yacht.Ff_m - yacht.Fa_m))
        section: list[tuple[float, float]] = [(0.0, -yacht.Tc_m)]

        for depth in section_depths[1:]:
            half_beam = waterline_half_beam * max(1.0 - depth**exponent, 0.0) ** (1.0 / exponent)
            section.append((half_beam, -float(depth) * yacht.Tc_m))
        for height_fraction in deck_fractions:
            half_beam = waterline_half_beam + (deck_half_beam - waterline_half_beam) * height_fraction
            section.append((float(half_beam), freeboard * float(height_fraction)))
        section.append((-float(deck_half_beam), freeboard))
        for height_fraction in deck_fractions[-2::-1]:
            half_beam = waterline_half_beam + (deck_half_beam - waterline_half_beam) * height_fraction
            section.append((-float(half_beam), freeboard * float(height_fraction)))
        for depth in section_depths[-2:0:-1]:
            half_beam = waterline_half_beam * max(1.0 - depth**exponent, 0.0) ** (1.0 / exponent)
            section.append((-half_beam, -float(depth) * yacht.Tc_m))

        if len(section) != section_size:
            raise RuntimeError("Le profil transversal de coque a une taille inattendue.")
        sections.append(np.asarray(section, dtype=float))
        start = station_index * section_size
        vertices[start:start + section_size, 0] = station * yacht.Lwl_m
        vertices[start:start + section_size, 1:] = sections[-1]

    start_cap_index = len(stations) * section_size
    end_cap_index = start_cap_index + 1
    vertices[start_cap_index] = vertices[:section_size].mean(axis=0)
    vertices[end_cap_index] = vertices[-section_size - 2:-2].mean(axis=0)
    faces: list[tuple[int, int, int]] = []
    for station_index in range(len(stations) - 1):
        current_start = station_index * section_size
        next_start = current_start + section_size
        for section_index in range(section_size):
            next_section_index = (section_index + 1) % section_size
            current_vertex = current_start + section_index
            next_vertex = current_start + next_section_index
            current_longitudinal = next_start + section_index
            next_longitudinal = next_start + next_section_index
            faces.extend([
                (current_vertex, next_vertex, next_longitudinal),
                (current_vertex, next_longitudinal, current_longitudinal),
            ])
    last_start = (len(stations) - 1) * section_size
    for section_index in range(section_size):
        next_section_index = (section_index + 1) % section_size
        faces.append((start_cap_index, next_section_index, section_index))
        faces.append((end_cap_index, last_start + section_index, last_start + next_section_index))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=int), process=True)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError("La coque approximative generee pour l'apercu 3D n'est pas un volume ferme valide.")
    return mesh

def _manual_gz_curve(total_mass_kg: float, volume_m3: float, waterline_z: float, gm_m: float) -> pd.DataFrame:
    angles = analysis_angle_values(HEEL_RANGE_MIN_DEG, HEEL_RANGE_MAX_DEG, HEEL_RANGE_STEPS)
    rows = []
    for phi_deg in angles:
        phi = deg_to_rad(phi_deg)
        gz = float(gm_m * np.sin(phi) * np.cos(phi))
        moment = float(total_mass_kg * C.G * gz)
        rows.append({
            "heel_deg": float(phi_deg),
            "phi_deg": float(phi_deg),
            "phi_rad": phi,
            "waterline_z_m": waterline_z,
            "volume_m3": volume_m3,
            "B_x_m": np.nan,
            "B_y_m": np.nan,
            "B_z_m": np.nan,
            "GZ_m": gz,
            "GZ_signed_m": gz,
            "M_hydrostatic_Nm": moment,
            "M_hydrostatic_raw_signed_Nm": -moment,
            "M_hydrostatic_restoring_Nm": moment,
            "M_buoyancy_Nm": np.nan,
        })
    return pd.DataFrame(rows)

def _manual_foil_from_input(
    name: str,
    contour: np.ndarray,
    naca: str,
    root_position: np.ndarray,
    direction: str,
    wetted_area_m2: float | None = None,
    thickness_ratio: float | None = None,
) -> FoilGeometry:
    foil = build_foil_mesh_from_contour(name, contour, naca, root_position, direction)
    if wetted_area_m2 is not None and wetted_area_m2 > 0:
        foil.wetted_area_m2 = float(wetted_area_m2)
        foil.external_area_m2 = float(wetted_area_m2)
    if thickness_ratio is not None and thickness_ratio > 0:
        foil.thickness_ratio = float(thickness_ratio)
    return foil

def build_model_from_manual_input(manual_input: ManualVPPInput) -> BoatGeometryState:
    mi = normalize_aero_selection(clone_manual_input(manual_input))
    yp = mi.yacht
    env = mi.environment
    sail = mi.main_sail
    keel_in = mi.keel
    rudder_in = mi.rudder
    adv = mi.advanced
    setup_warnings: list[str] = [part for part in str(mi.solver.setup_warning_message).split("; ") if part]
    mi.solver.solver_method, solver_warning = normalize_solver_method(mi.solver.solver_method)
    if solver_warning:
        setup_warnings.append(solver_warning)

    for name, value in {
        "Lwl_m": yp.Lwl_m,
        "Bwl_m": yp.Bwl_m,
        "Tc_m": yp.Tc_m,
        "Swet_m2": yp.Swet_m2,
        "Tmax_m": yp.Tmax_m,
        "Amax_m2": yp.Amax_m2,
        "displaced_volume_m3": yp.displaced_volume_m3,
    }.items():
        if value <= 0:
            raise ValueError(f"Manual input {name} must be positive.")

    hydro_mass = C.RHO_WATER * yp.displaced_volume_m3
    mass_mode = adv.mass_mode
    if mass_mode not in {"design_ballast", "fixed_loading"}:
        raise ValueError("mass_mode must be 'design_ballast' or 'fixed_loading'.")
    total_mass = hydro_mass if mass_mode == "design_ballast" else float(yp.displacement_mass_kg or hydro_mass)
    appendage_masses = [float(keel_in.mass_kg), float(rudder_in.mass_kg)]
    if not all(np.isfinite(mass) and mass >= 0.0 for mass in appendage_masses):
        raise ValueError("Manual keel and rudder masses must be finite and non-negative.")
    if sum(appendage_masses) > total_mass:
        raise ValueError("Manual keel and rudder masses cannot exceed the boat's total mass.")
    if abs(total_mass - hydro_mass) / max(hydro_mass, 1.0) > 0.05:
        setup_warnings.append(f"Mass-volume mismatch: hydrostatic mass {hydro_mass:.1f} kg vs fixed mass {total_mass:.1f} kg.")
    if yp.hull_roughness_ks_um > 0:
        setup_warnings.append("Hull roughness input stored but no explicit roughness correction is implemented.")
    if env.Hs_m > 0 or env.Ts_s > 0:
        setup_warnings.append("Wave inputs stored but added resistance in waves is not implemented in this simplified model.")
    if adv.hydro_model == "orc_simplified":
        setup_warnings.append("ORC simplified mode only changes selected hydrodynamic formulas. It is not the full ORC 2026 VPP.")
        if adv.orc_LSM1_m is None:
            setup_warnings.append("ORC simplified mode: LSM1 unavailable, using LWL as fallback.")

    volume_for_hydro = total_mass / C.RHO_WATER if mass_mode == "fixed_loading" else yp.displaced_volume_m3
    if mass_mode == "fixed_loading" and abs(volume_for_hydro - yp.displaced_volume_m3) / max(yp.displaced_volume_m3, 1e-9) > 0.01:
        setup_warnings.append("Manual fixed_loading adjusted displaced volume from fixed mass; scalar hull shape was not re-cut from STL.")

    cp = float(yp.Cp) if yp.Cp is not None else safe_div(volume_for_hydro, yp.Amax_m2 * yp.Lwl_m, default=np.nan)
    cm = float(yp.Cm) if yp.Cm is not None else safe_div(yp.Amax_m2, yp.Bwl_m * yp.Tc_m, default=np.nan)
    awp = 0.75 * yp.Lwl_m * yp.Bwl_m
    x_mid = 0.5 * yp.Lwl_m
    lcb_pct = float(yp.LCB_pct_midship)
    lcf_pct = float(yp.LCF_pct_midship) if yp.LCF_pct_midship is not None else -5.0
    lcb = x_mid + 0.01 * lcb_pct * yp.Lwl_m
    lcf = x_mid + 0.01 * lcf_pct * yp.Lwl_m
    values = {
        "LWL/BWL": safe_div(yp.Lwl_m, yp.Bwl_m),
        "BWL/TC": safe_div(yp.Bwl_m, yp.Tc_m),
        "LWL/volume^(1/3)": safe_div(yp.Lwl_m, volume_for_hydro ** (1 / 3)),
        "C_P": cp,
        "C_M": cm,
        "A_WP/volume^(2/3)": safe_div(awp, volume_for_hydro ** (2 / 3)),
        "LCB_pct_midship": lcb_pct,
        "LCF_pct_midship": lcf_pct,
    }
    tables = load_delft_tables(adv.delft_table_folder)
    delft_valid, delft_warnings = check_delft_domain(values, tables)
    delft_warnings = list(delft_warnings) + setup_warnings

    hydro = HullHydrostatics(
        LWL=float(yp.Lwl_m),
        BWL=float(yp.Bwl_m),
        TC=float(yp.Tc_m),
        volume_C=float(volume_for_hydro),
        S_C=float(yp.Swet_m2),
        A_WP=float(awp),
        LCB=float(lcb),
        LCF=float(lcf),
        AMS_CB=float(yp.Amax_m2),
        C_M=float(cm),
        C_P=float(cp),
        x_FP=float(yp.Lwl_m),
        LCB_fpp=float(yp.Lwl_m - lcb),
        LCF_fpp=float(yp.Lwl_m - lcf),
        LCB_pct_midship=float(lcb_pct),
        LCF_pct_midship=float(lcf_pct),
        waterline_z=0.0,
        B_CB=float(yp.Bwl_m),
        delft_valid=bool(delft_valid),
        delft_warnings=delft_warnings,
    )

    inputs = GeometryInputs(
        stl_path="",
        hull_draft_m=float(yp.Tc_m),
        wing_contour_text="",
        centerboard_contour_text="",
        rudder_contour_text="",
        wing_naca=sail.naca_profile,
        centerboard_naca="NACA0010",
        rudder_naca="NACA0012",
        ballast_mode="automatique" if mass_mode == "design_ballast" else "manuel",
        manual_ballast_kg=0.0,
        fixed_alpha_deg=float(sail.fixed_alpha_deg),
        wing_alpha_mode="automatique",
        alpha_min_deg=float(sail.alpha_min_deg),
        alpha_max_deg=float(sail.alpha_max_deg),
        alpha_step_deg=float(sail.alpha_step_deg),
        oswald_efficiency=float(sail.efficiency_factor_e),
        delft_table_folder=adv.delft_table_folder,
        polar_folder=adv.polar_folder,
    )
    settings = SolverSettings(**_dataclass_kwargs(SolverSettings, asdict(mi.solver)))
    settings.geometry_input_mode = "manual"
    settings.hydro_model = adv.hydro_model
    settings.mass_mode = mass_mode
    settings.aero_model = resolve_manual_aero_model(mi.solver.sail_coefficient_data_source, adv.aero_model)
    settings.use_wind_gradient = bool(adv.use_wind_gradient)
    settings.wind_reference_height_m = float(adv.wind_reference_height_m)
    settings.wind_roughness_z0_m = float(adv.wind_roughness_z0_m)
    settings.orc_LSM1_m = adv.orc_LSM1_m
    settings.pedro_hardcoded_hydroheel = bool(adv.pedro_hardcoded_hydroheel)
    settings.setup_warning_message = "; ".join(setup_warnings)

    hull = trimesh.creation.box(extents=[yp.Lwl_m, yp.Bwl_m, yp.Tc_m])
    hull.apply_translation([0.5 * yp.Lwl_m, 0.0, -0.5 * yp.Tc_m])
    immersed_hull = hull.copy()

    wing_root = np.array([0.45 * yp.Lwl_m, 0.0, float(sail.BAD_m)])
    wing = _manual_foil_from_input("aile", _manual_sail_contour(sail), sail.naca_profile, wing_root, "up")
    if sail.sail_area_m2 is not None and sail.sail_area_m2 > 0:
        wing.planform_area_m2 = float(sail.sail_area_m2)
    if sail.span_m is not None and sail.span_m > 0:
        wing.span_m = float(sail.span_m)
    if sail.mean_chord_m is not None and sail.mean_chord_m > 0:
        wing.mean_chord_m = float(sail.mean_chord_m)
    else:
        wing.mean_chord_m = safe_div(wing.planform_area_m2, max(wing.span_m, 1e-9), default=wing.mean_chord_m)
    wing.aspect_ratio = float(sail.aspect_ratio) if sail.aspect_ratio is not None else safe_div(wing.span_m**2, wing.planform_area_m2, default=wing.aspect_ratio)

    keel_root_chord = keel_in.root_chord_m or keel_in.Lkeel_m
    keel_tip_chord = keel_in.tip_chord_m or 0.6 * keel_root_chord
    keel_span = keel_in.span_m or keel_in.Dkeel_m
    keel_x = keel_in.x_position_m if keel_in.x_position_m is not None else 0.45 * yp.Loa_m
    keel_contour = _manual_trapezoid_contour(keel_root_chord, keel_tip_chord, keel_span)
    keel_root = np.array([float(keel_x), 0.0, -float(yp.Tc_m)])
    centerboard = _manual_foil_from_input("derive", keel_contour, "NACA0010", keel_root, "down", keel_in.wetted_area_m2, keel_in.thickness_ratio)

    rudder_x = rudder_in.x_position_m if rudder_in.x_position_m is not None else 0.90 * yp.Loa_m
    rudder_contour = _manual_trapezoid_contour(rudder_in.Cu_m, rudder_in.Cl_m, rudder_in.b_m)
    rudder_root = np.array([float(rudder_x), 0.0, -float(yp.Tc_m)])
    rudder = _manual_foil_from_input("safran", rudder_contour, "NACA0012", rudder_root, "down", rudder_in.wetted_area_m2, rudder_in.thickness_ratio)

    fixed_rows = []
    appendage_mass = max(0.0, keel_in.mass_kg) + max(0.0, rudder_in.mass_kg)
    hull_load_mass = max(total_mass - appendage_mass, 0.0)
    fixed_rows.append({"component": "manual hull and loading", "mass_kg": hull_load_mass, "x_G_m": 0.48 * yp.Lwl_m, "y_G_m": 0.0, "z_G_m": -0.35 * yp.Tmax_m, "area_m2": yp.Swet_m2})
    if keel_in.mass_kg > 0:
        fixed_rows.append({"component": "manual keel", "mass_kg": keel_in.mass_kg, "x_G_m": float(keel_x), "y_G_m": 0.0, "z_G_m": -yp.Tmax_m, "area_m2": centerboard.wetted_area_m2})
    if rudder_in.mass_kg > 0:
        fixed_rows.append({"component": "manual rudder", "mass_kg": rudder_in.mass_kg, "x_G_m": float(rudder_x), "y_G_m": 0.0, "z_G_m": -yp.Tc_m - 0.5 * rudder_in.b_m, "area_m2": rudder.wetted_area_m2})
    component_table = pd.DataFrame(fixed_rows)
    cg = np.average(component_table[["x_G_m", "y_G_m", "z_G_m"]].to_numpy(float), axis=0, weights=component_table["mass_kg"].to_numpy(float))
    masses = MassProperties(
        component_table=component_table,
        total_mass_kg=float(component_table["mass_kg"].sum()),
        center_of_gravity=np.asarray(cg, dtype=float),
        ballast_mass_kg=max(0.0, total_mass - hull_load_mass),
        ballast_position=np.array([float(keel_x), 0.0, -yp.Tmax_m]),
        displacement_mass_kg=hydro_mass,
        mass_error_kg=float(component_table["mass_kg"].sum() - hydro_mass),
        mass_error_rel=safe_div(float(component_table["mass_kg"].sum() - hydro_mass), hydro_mass, default=0.0),
    )

    total_volume = float(volume_for_hydro)
    gz_curve = _manual_gz_curve(masses.total_mass_kg, total_volume, hydro.waterline_z, adv.manual_GM_m)
    lateral_area = float(yp.Lwl_m * yp.Tc_m + centerboard.planform_area_m2 + rudder.planform_area_m2)
    z_min_submerged = float(-max(yp.Tmax_m, yp.Tc_m + keel_span, yp.Tc_m + rudder_in.b_m))

    state = BoatGeometryState(
        inputs=inputs,
        settings=settings,
        hull_mesh=hull,
        immersed_hull_mesh=immersed_hull,
        hydro=hydro,
        wing=wing,
        centerboard=centerboard,
        rudder=rudder,
        masses=masses,
        total_displaced_volume_m3=total_volume,
        T_max_m=float(yp.Tmax_m),
        z_min_submerged_m=z_min_submerged,
        total_draft_m=float(yp.Tmax_m),
        lateral_area_m2=lateral_area,
        gz_curve=gz_curve,
        delft_tables=tables,
        hydrostatic_boolean_union_used=False,
        mesh_overlap_detected=False,
        estimated_overlap_volume_m3=0.0,
        hydrostatic_geometry_valid=True,
        wetted_surface_cache={},
    )
    state.manual_input = mi
    return state

DEFAULT_MANUAL_PYTHON_VPP_DARING = ManualVPPInput(
    solver=SolverSettings(
        solver_method="least_squares_multistart",
        sail_coefficient_data_source="orc",
        geometry_input_mode="manual",
        hydro_model="oceanvpp_delft",
        mass_mode="fixed_loading",
        use_wind_gradient=False,
    ),
    environment=EnvironmentSettings(tws_min=8.0, tws_max=12.0, tws_unit="knots", Hs_m=0.0, Ts_s=0.0),
    yacht=YachtParticulars(),
    main_sail=MainSailInput(),
    keel=KeelInput(),
    rudder=RudderInput(),
    rig=RigInput(I_m=10.2, J_m=3.0, LPG_m=3.3, SL_m=0.0, EHM_m=11.6, EMDC_m=0.0, HBI_m=0.0),
    sailset=SailSetInput(sailset="main_only", reference_area_mode="active_area", flat=1.0, reef=1.0, optimize_flat_reef=False),
    advanced=AdvancedARGOSettings(geometry_input_mode="manual", hydro_model="oceanvpp_delft", mass_mode="fixed_loading", aero_model="pedro_orc_low", pedro_hardcoded_hydroheel=True),
)

DEFAULT_MANUAL_MICROTRANSAT = ManualVPPInput(
    solver=SolverSettings(
        solver_method="least_squares_multistart",
        sail_coefficient_data_source="naca",
        geometry_input_mode="manual",
        hydro_model="oceanvpp_delft",
        mass_mode="design_ballast",
        use_wind_gradient=True,
    ),
    environment=EnvironmentSettings(tws_min=6.0, tws_max=14.0, tws_unit="knots", Hs_m=0.0, Ts_s=0.0),
    yacht=YachtParticulars(
        name="Microtransat example",
        Lwl_m=2.40,
        displaced_volume_m3=0.095,
        Bwl_m=0.55,
        Tc_m=0.16,
        Swet_m2=1.45,
        Tmax_m=0.95,
        Amax_m2=0.075,
        displacement_mass_kg=97.0,
        Ff_m=0.22,
        Fa_m=0.18,
        Boa_m=0.65,
        Loa_m=2.70,
        hull_roughness_ks_um=80.0,
        Cp=0.53,
        Cm=0.70,
        LCB_pct_midship=-4.0,
        LCF_pct_midship=-5.0,
    ),
    main_sail=MainSailInput(
        main_sail_type="rigid_wing",
        name="ARGO wing",
        P_m=2.4,
        E_m=0.45,
        roach=0.0,
        BAD_m=0.05,
        sail_area_m2=0.85,
        span_m=2.4,
        naca_profile="NACA0012",
        alpha_min_deg=-5.0,
        alpha_max_deg=18.0,
        alpha_step_deg=1.0,
        fixed_alpha_deg=6.0,
        efficiency_factor_e=0.85,
    ),
    rig=RigInput(I_m=0.0, J_m=0.0, LPG_m=0.0, SL_m=0.0, EHM_m=2.45, EMDC_m=0.0, HBI_m=0.0),
    sailset=SailSetInput(sailset="main_only", reference_area_mode="active_area", flat=1.0, reef=1.0, optimize_flat_reef=False),
    keel=KeelInput(keel_type="daggerboard", Lkeel_m=0.18, Dkeel_m=0.75, thickness_ratio=0.12, mass_kg=0.0),
    rudder=RudderInput(Cu_m=0.10, Cl_m=0.07, b_m=0.45, thickness_ratio=0.12, mass_kg=0.0),
    advanced=AdvancedARGOSettings(geometry_input_mode="manual", hydro_model="oceanvpp_delft", mass_mode="design_ballast", aero_model="naca_wing", use_wind_gradient=True, manual_GM_m=0.42, pedro_hardcoded_hydroheel=False),
)

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

def wind_speed_at_height(V_ref_mps: float, z_CE_above_water_m: float, z_ref_m: float = 10.0, z0_m: float = 0.005) -> float:
    """Log-law true wind gradient from reference height to the sail centre of effort."""

    z = max(float(z_CE_above_water_m), 0.1)
    z_ref = max(float(z_ref_m), float(z0_m) * 1.01)
    z0 = max(float(z0_m), 1e-6)
    return float(V_ref_mps * np.log(z / z0) / np.log(z_ref / z0))

PEDRO_HAZEN_AWA = np.array([27.0, 50.0, 80.0, 100.0, 180.0])

PEDRO_HAZEN_TABLES = {
    "main": {"CD": [0.02, 0.15, 0.80, 1.00, 0.90], "CL": [1.50, 1.50, 0.95, 0.85, 0.00]},
    "jib": {"CD": [0.02, 0.25, 0.15, 0.00, 0.00], "CL": [1.50, 0.50, 0.30, 0.00, 0.00]},
    "spin": {"CD": [0.00, 0.25, 0.90, 1.20, 0.66], "CL": [0.00, 1.50, 1.00, 0.85, 0.00]},
}

PEDRO_ORC_AWA = np.array([0.0, 7.0, 9.0, 12.0, 28.0, 60.0, 90.0, 120.0, 150.0, 180.0])

PEDRO_ORC_TABLES = {
    "main": {
        "CDlow": [0.04310, 0.02586, 0.02328, 0.02328, 0.03259, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483],
        "CLlow": [0.00000, 0.86207, 1.05172, 1.16379, 1.34698, 1.35345, 1.26724, 0.93103, 0.38793, -0.11207],
        "CDhi": [0.03448, 0.01724, 0.01466, 0.01466, 0.02586, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483],
        "CLhi": [0.00000, 0.94828, 1.13793, 1.25000, 1.42681, 1.38319, 1.26724, 0.93103, 0.38793, -0.11207],
    },
    "jib": {
        "CDlow": [0.000, 0.050, 0.032, 0.031, 0.037, 0.250, 0.350, 0.730, 0.950, 0.900],
        "CLlow": [0.000, 0.000, 1.000, 1.375, 1.450, 1.450, 1.250, 0.400, 0.000, -0.100],
        "CDhi": [0.000, 0.050, 0.032, 0.031, 0.037, 0.250, 0.350, 0.730, 0.950, 0.900],
        "CLhi": [0.000, 0.000, 1.100, 1.475, 1.500, 1.450, 1.250, 0.400, 0.000, -0.100],
    },
}

PEDRO_SPIN_AWA = np.array([28.0, 41.0, 50.0, 60.0, 67.0, 75.0, 100.0, 115.0, 130.0, 150.0, 170.0, 180.0])

PEDRO_SPIN_TABLES = {
    "symmetric": {
        "CD": [0.182, 0.267, 0.337, 0.417, 0.465, 0.506, 0.588, 0.615, 0.636, 0.650, 0.650, 0.640],
        "CL": [-0.024, 0.660, 0.861, 0.992, 1.026, 1.026, 0.920, 0.805, 0.635, 0.360, 0.110, 0.000],
    },
    "asymmetric": {
        "CD": [0.154, 0.239, 0.309, 0.389, 0.436, 0.477, 0.545, 0.566, 0.475, 0.352, 0.290, 0.262],
        "CL": [0.017, 0.698, 0.899, 1.040, 1.072, 1.075, 0.985, 0.805, 0.372, 0.100, 0.020, 0.000],
    },
}

AERO_MODEL_TO_SAIL_DATA = {
    "naca_wing": "naca",
    "pedro_hazen": "hazen",
    "pedro_orc_low": "orc",
    "pedro_orc_high": "orc",
    "pedro_orc_medium": "orc",
}

SAIL_DATA_TO_DEFAULT_AERO_MODEL = {"naca": "naca_wing", "hazen": "pedro_hazen", "orc": "pedro_orc_low"}

def normalize_solver_method(method: str) -> tuple[str, str]:
    requested = str(method or "").strip().lower()
    if requested in {"iterative", "least_squares"}:
        return "least_squares_multistart", f"Legacy solver_method={requested} normalized to least_squares_multistart."
    if requested in {"least_squares_multistart", "least_squares_single"}:
        return requested, ""
    raise ValueError(f"Unsupported solver_method {requested!r}; choose least_squares_multistart or least_squares_single.")

def resolve_manual_aero_model(sail_source: str = "", aero_model: str | None = None) -> str:
    """Resolve the aerodynamic model, with Aero model as the primary field."""

    requested = str(aero_model or "").strip().lower()
    source = str(sail_source or "").strip().lower()
    if source == "custom":
        raise ValueError("Custom sail coefficient tables are not implemented yet. Choose naca, hazen or orc.")
    if requested in AERO_MODEL_TO_SAIL_DATA:
        return requested
    return SAIL_DATA_TO_DEFAULT_AERO_MODEL.get(source, "naca_wing")

def normalize_aero_selection(settings: Any) -> Any:
    """Return a coherent Aero model / Sail data selection and record corrections."""

    is_manual = isinstance(settings, ManualVPPInput)
    normalized = clone_manual_input(settings) if is_manual else (dict(settings) if isinstance(settings, dict) else settings)
    if is_manual:
        requested = str(normalized.advanced.aero_model or "").strip().lower()
        source = str(normalized.solver.sail_coefficient_data_source or "").strip().lower()
    elif isinstance(normalized, dict):
        requested = str(normalized.get("aero_model", "") or "").strip().lower()
        source = str(normalized.get("sail_coefficient_data_source", normalized.get("sail_data", "")) or "").strip().lower()
    else:
        requested = str(getattr(normalized, "aero_model", "") or "").strip().lower()
        source = str(getattr(normalized, "sail_coefficient_data_source", "") or "").strip().lower()
    if source == "custom":
        raise ValueError("Custom sail coefficient tables are not implemented yet. Choose naca, hazen or orc.")
    model = resolve_manual_aero_model(source, requested)
    coherent_source = AERO_MODEL_TO_SAIL_DATA[model]
    correction = ""
    if requested != model or source != coherent_source:
        correction = f"Aero selection normalized to aero_model={model} and sail data={coherent_source}."
    if is_manual:
        normalized.advanced.aero_model = model
        normalized.solver.aero_model = model
        normalized.solver.sail_coefficient_data_source = coherent_source
        if correction:
            existing = [part for part in str(normalized.solver.setup_warning_message).split("; ") if part]
            normalized.solver.setup_warning_message = "; ".join(dict.fromkeys(existing + [correction]))
    elif isinstance(normalized, dict):
        normalized["aero_model"] = model
        normalized["sail_coefficient_data_source"] = coherent_source
        if "sail_data" in normalized:
            normalized["sail_data"] = coherent_source
        if correction:
            normalized["warning_message"] = "; ".join(part for part in [str(normalized.get("warning_message", "")), correction] if part)
    else:
        normalized.aero_model = model
        normalized.sail_coefficient_data_source = coherent_source
        if correction and hasattr(normalized, "setup_warning_message"):
            normalized.setup_warning_message = "; ".join(part for part in [str(normalized.setup_warning_message), correction] if part)
    return normalized

def is_pedro_aero_model(model: str) -> bool:
    return str(model).startswith("pedro_")

def _interp_pedro_table(x_nodes: np.ndarray, y_nodes: list[float], awa_deg: float, label: str) -> tuple[float, str, str]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    awa = float(awa_deg)
    warning = ""
    x_eval = awa
    if awa < float(x.min()) or awa > float(x.max()):
        x_eval = float(np.clip(awa, float(x.min()), float(x.max())))
        warning = f"{label}: AWA {awa:.3g} deg outside table domain; clipped to {x_eval:.3g} deg."
    try:
        from scipy.interpolate import PchipInterpolator
        value = float(PchipInterpolator(x, y, extrapolate=False)(x_eval))
        method = "PchipInterpolator clipped" if warning else "PchipInterpolator"
    except Exception:
        value = float(np.interp(x_eval, x, y))
        method = "np.interp clipped" if warning else "np.interp"
        warning = "; ".join(part for part in [warning, "PchipInterpolator unavailable; used np.interp."] if part)
    if not np.isfinite(value):
        value = float(np.interp(x_eval, x, y))
        method = "np.interp clipped" if warning else "np.interp"
    return value, method, warning

def pedro_headsail_usage_factor(awa_deg: float) -> float:
    """Smooth activation window for the jib in the simplified Pedro sail-plan model."""

    awa = abs(float(awa_deg))
    # Avoid the ORC jib table's near-discontinuous jump at very small AWA and remove the jib deep downwind.
    factor = np.interp(awa, [0.0, 10.0, 25.0, 120.0, 145.0, 180.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    return float(np.clip(factor, 0.0, 1.0))

def pedro_fractionality_coefficient(fractionality: float | None) -> tuple[float, str]:
    if fractionality is None or not np.isfinite(float(fractionality)) or float(fractionality) <= 0.0:
        return 0.0, "fractionality missing; falling back to ORC low coefficients."
    x = min(0.3, max(0.0, 1.0 / float(fractionality) - 1.0))
    return float(np.sqrt(np.sin(np.pi / 0.6 * x))), ""

def get_pedro_sail_coefficients(source: str, awa_deg: float, sail: str, set_type: str = "low", fractionality: float | None = None) -> dict[str, Any]:
    source = str(source).strip().lower()
    sail_key = str(sail).strip().lower().replace("spinnaker", "spin")
    warnings_out: list[str] = []
    if source == "pedro_hazen":
        table = PEDRO_HAZEN_TABLES[sail_key]
        CL, method_cl, warn_cl = _interp_pedro_table(PEDRO_HAZEN_AWA, table["CL"], awa_deg, f"Hazen {sail_key} CL")
        CD, method_cd, warn_cd = _interp_pedro_table(PEDRO_HAZEN_AWA, table["CD"], awa_deg, f"Hazen {sail_key} CD")
        warnings_out.extend([w for w in [warn_cl, warn_cd] if w])
        return {"CL": CL, "CDp": CD, "coefficient_source": f"hazen_{sail_key}", "interpolation_method": method_cl if method_cl == method_cd else f"{method_cl}/{method_cd}", "warning": "; ".join(warnings_out)}
    if sail_key == "spin":
        spin_type = "asymmetric" if "asym" in set_type else "symmetric"
        table = PEDRO_SPIN_TABLES[spin_type]
        CL, method_cl, warn_cl = _interp_pedro_table(PEDRO_SPIN_AWA, table["CL"], awa_deg, f"ORC {spin_type} spin CL")
        CD, method_cd, warn_cd = _interp_pedro_table(PEDRO_SPIN_AWA, table["CD"], awa_deg, f"ORC {spin_type} spin CD")
        warnings_out.extend([w for w in [warn_cl, warn_cd] if w])
        return {"CL": CL, "CDp": CD, "coefficient_source": f"orc_{spin_type}_spin", "interpolation_method": method_cl if method_cl == method_cd else f"{method_cl}/{method_cd}", "warning": "; ".join(warnings_out)}
    table = PEDRO_ORC_TABLES[sail_key]
    mode = "low"
    if source == "pedro_orc_high":
        mode = "hi"
    elif source == "pedro_orc_medium":
        fcoef, warn = pedro_fractionality_coefficient(fractionality)
        if warn:
            warnings_out.append(warn)
            mode = "low"
        else:
            low = get_pedro_sail_coefficients("pedro_orc_low", awa_deg, sail_key, set_type=set_type, fractionality=fractionality)
            high = get_pedro_sail_coefficients("pedro_orc_high", awa_deg, sail_key, set_type=set_type, fractionality=fractionality)
            CL = low["CL"] * (1.0 - fcoef / 2.0) + high["CL"] * (fcoef / 2.0)
            CD = low["CDp"] * (1.0 - fcoef / 2.0) + high["CDp"] * (fcoef / 2.0)
            warnings_out.extend([w for w in [low.get("warning", ""), high.get("warning", "")] if w])
            return {"CL": float(CL), "CDp": float(CD), "coefficient_source": f"orc_medium_{sail_key}", "interpolation_method": f"ORC medium fcoef={fcoef:.4g}", "warning": "; ".join(dict.fromkeys(warnings_out))}
    suffix = "hi" if mode == "hi" else "low"
    CL, method_cl, warn_cl = _interp_pedro_table(PEDRO_ORC_AWA, table[f"CL{suffix}"], awa_deg, f"ORC {sail_key} CL{suffix}")
    CD, method_cd, warn_cd = _interp_pedro_table(PEDRO_ORC_AWA, table[f"CD{suffix}"], awa_deg, f"ORC {sail_key} CD{suffix}")
    warnings_out.extend([w for w in [warn_cl, warn_cd] if w])
    return {"CL": CL, "CDp": CD, "coefficient_source": f"orc_{suffix}_{sail_key}", "interpolation_method": method_cl if method_cl == method_cd else f"{method_cl}/{method_cd}", "warning": "; ".join(warnings_out)}

def compute_pedro_sail_geometry(manual_input: ManualVPPInput) -> dict[str, Any]:
    sail = manual_input.main_sail
    rig = manual_input.rig
    sailset = manual_input.sailset
    yacht = manual_input.yacht
    warnings_out: list[str] = []
    A_main = 0.5 * float(sail.P_m) * float(sail.E_m)
    I = float(rig.I_m or 0.0)
    J = float(rig.J_m or 0.0)
    LPG = float(rig.LPG_m or 0.0)
    SL = float(rig.SL_m or 0.0)
    if sailset.sailset == "main_jib" and (I <= 0 or J <= 0 or LPG <= 0):
        raise ValueError("Pedro main_jib requires positive I_m, J_m and LPG_m.")
    if sailset.sailset == "main_spinnaker" and (I <= 0 or J <= 0 or SL <= 0):
        raise ValueError("Pedro main_spinnaker requires positive I_m, J_m and SL_m.")
    A_jib = 0.5 * np.sqrt(I**2 + J**2) * LPG if I > 0 and J > 0 and LPG > 0 else 0.0
    A_fore = 0.5 * I * J if I > 0 and J > 0 else 0.0
    A_spi = 1.15 * SL * J if SL > 0 and J > 0 else 0.0
    zCoE_main = 0.39 * float(sail.P_m) + float(sail.BAD_m)
    zCoE_jib = 0.39 * I if I > 0 else 0.0
    zCoE_spi = 0.59 * I if I > 0 else 0.0
    if sailset.sailset == "main_jib":
        active_area = A_main + A_jib
        zCoE = safe_div(zCoE_main * A_main + zCoE_jib * A_jib, active_area, default=zCoE_main)
    elif sailset.sailset == "main_spinnaker":
        active_area = A_main + A_spi
        zCoE = safe_div(zCoE_main * A_main + zCoE_spi * A_spi, active_area, default=zCoE_main)
    else:
        active_area = A_main
        zCoE = zCoE_main
    A_nom = A_main + A_fore
    A_ref = A_nom if sailset.reference_area_mode == "pedro_nominal" else active_area
    if sailset.sailset == "main_only" and sailset.reference_area_mode == "pedro_nominal":
        warnings_out.append("main_only with pedro_nominal uses A_main + A_fore as reference area; for strict mainsail-only comparison use active_area.")
    if A_ref <= 0:
        raise ValueError("Pedro sail reference area must be positive.")
    EHM = float(rig.EHM_m) if rig.EHM_m is not None and float(rig.EHM_m) > 0 else float(sail.P_m + sail.BAD_m)
    EMDC = float(rig.EMDC_m) if rig.EMDC_m is not None and float(rig.EMDC_m) > 0 else 0.0
    if EMDC <= 0.0:
        warnings_out.append("EMDC missing; mast drag contribution ignored.")
    BAS = float(rig.BAS_m or 0.0)
    fractionality = rig.fractionality
    if fractionality is None and float(sail.P_m + BAS) > 0 and I > 0:
        fractionality = I / float(sail.P_m + BAS)
    return {"A_main": float(A_main), "A_jib": float(A_jib), "A_fore": float(A_fore), "A_spi": float(A_spi), "A_nom": float(A_nom), "A_ref": float(A_ref), "active_area": float(active_area), "zCoE_main": float(zCoE_main), "zCoE_jib": float(zCoE_jib), "zCoE_spi": float(zCoE_spi), "zCoE": float(zCoE), "EHM": EHM, "EMDC": EMDC, "AvgFB": 0.5 * (float(yacht.Ff_m) + float(yacht.Fa_m)), "Bmax": float(yacht.Boa_m), "HBI": float(rig.HBI_m or 0.0), "fractionality": fractionality, "warnings": warnings_out}

def aerodynamic_state_pedro(
    TWS_mps: float,
    TWA_deg: float,
    boat_speed_mps: float,
    heel_rad: float,
    state: BoatGeometryState,
    flat: float | None = None,
    reef: float | None = None,
    sailset: str | None = None,
    source: str | None = None,
    waterline_z_m: float | None = None,
) -> AeroState:
    manual_input = getattr(state, "manual_input", None)
    if manual_input is None:
        raise ValueError("Pedro aerodynamic model requires a ManualVPPInput attached to state.manual_input.")
    mi = clone_manual_input(manual_input)
    if sailset is not None:
        mi.sailset.sailset = sailset
    model = resolve_manual_aero_model(mi.solver.sail_coefficient_data_source, source or getattr(state.settings, "aero_model", mi.advanced.aero_model))
    if not is_pedro_aero_model(model):
        model = "pedro_orc_low"
    flat_value = float(mi.sailset.flat if flat is None else flat)
    reef_value = float(mi.sailset.reef if reef is None else reef)
    geom = compute_pedro_sail_geometry(mi)
    z_WL = float(state.hydro.waterline_z if waterline_z_m is None else waterline_z_m)
    z_CE_base = geom["zCoE"] * reef_value * np.cos(float(heel_rad))
    z_CE_abs = z_WL + geom["HBI"] + z_CE_base
    TWS_local = float(TWS_mps)
    if getattr(state.settings, "use_wind_gradient", False):
        TWS_local = wind_speed_at_height(TWS_mps, z_CE_abs - z_WL, getattr(state.settings, "wind_reference_height_m", 10.0), getattr(state.settings, "wind_roughness_z0_m", 0.005))
    AWS, AWA = apparent_wind(TWS_local, boat_speed_mps, TWA_deg, heel_rad)
    spin_type = mi.sailset.spinnaker_type
    set_type = spin_type if mi.sailset.sailset == "main_spinnaker" else model.split("pedro_orc_")[-1]
    main = get_pedro_sail_coefficients(model, AWA, "main", set_type=set_type, fractionality=geom["fractionality"])
    jib = {"CL": 0.0, "CDp": 0.0, "coefficient_source": "", "interpolation_method": "", "warning": ""}
    spin = {"CL": 0.0, "CDp": 0.0, "coefficient_source": "", "interpolation_method": "", "warning": ""}
    if mi.sailset.sailset == "main_jib":
        jib = get_pedro_sail_coefficients(model, AWA, "jib", set_type=set_type, fractionality=geom["fractionality"])
        jib_usage = pedro_headsail_usage_factor(AWA)
        if jib_usage < 1.0:
            jib = dict(jib)
            jib["CL"] *= jib_usage
            jib["CDp"] *= jib_usage
            jib["interpolation_method"] = "; ".join(part for part in [jib["interpolation_method"], f"jib_usage={jib_usage:.3g}"] if part)
    if mi.sailset.sailset == "main_spinnaker":
        spin = get_pedro_sail_coefficients(model, AWA, "spin", set_type=spin_type, fractionality=geom["fractionality"])
    A_ref = geom["A_ref"]
    CL_lift = (main["CL"] * geom["A_main"] + jib["CL"] * geom["A_jib"] + spin["CL"] * geom["A_spi"]) / A_ref
    CDp_weighted = (main["CDp"] * geom["A_main"] + jib["CDp"] * geom["A_jib"] + spin["CDp"] * geom["A_spi"]) / A_ref
    CL = CL_lift * flat_value * reef_value**2
    CDp = CDp_weighted * reef_value**2
    AR_height = 1.1 * (geom["EHM"] + geom["AvgFB"]) if AWA <= 45.0 else 1.1 * geom["EHM"]
    AR = max(safe_div(AR_height**2, A_ref, default=0.1), 0.1)
    CDi = CL**2 * (1.0 / (np.pi * AR) + 0.005)
    CD0 = 1.13 * safe_div(geom["Bmax"] * geom["AvgFB"] + geom["EHM"] * geom["EMDC"], A_ref, default=0.0)
    CD = CDp + CDi + CD0
    q = 0.5 * C.RHO_AIR * AWS**2
    lift = q * CL * A_ref
    drag = q * CD * A_ref
    awa_rad = deg_to_rad(AWA)
    F_drive = float(lift * np.sin(awa_rad) - drag * np.cos(awa_rad))
    F_heel = float(lift * np.cos(awa_rad) + drag * np.sin(awa_rad))
    F_side_signed = float(F_heel * np.cos(float(heel_rad)))
    F_side = abs(F_side_signed)
    z_CLR = z_WL - CLR_DRAFT_FACTOR * float(state.T_max_m)
    if getattr(state.settings, "pedro_hardcoded_hydroheel", True):
        zclr_pedro = 2.3 * 0.43
        lever = (z_CE_base + geom["HBI"] + zclr_pedro) * reef_value
        M_aero_about_WL = F_heel * lever
        pedro_hydroheel = 2.3 * 0.43 * F_heel
    else:
        M_aero_about_WL = F_side * (z_CE_abs - z_WL)
        pedro_hydroheel = np.nan
    coeff_sources = ", ".join(part for part in [main["coefficient_source"], jib["coefficient_source"], spin["coefficient_source"]] if part)
    methods = ", ".join(dict.fromkeys(part for part in [main["interpolation_method"], jib["interpolation_method"], spin["interpolation_method"]] if part))
    warnings_out = list(dict.fromkeys(list(geom["warnings"]) + [w for w in [main.get("warning", ""), jib.get("warning", ""), spin.get("warning", "")] if w]))
    return AeroState(
        AWS_mps=AWS, AWA_deg=AWA, CL=CL, CD=CD, CM=0.0, Re=np.nan,
        wing_orientation_deg=np.nan, F_drive_N=F_drive, F_side_N=F_side,
        F_side_signed_N=F_side_signed, M_aero_Nm=M_aero_about_WL,
        center_of_effort=np.array([state.wing.surface_centroid[0], 0.0, z_CE_abs]),
        profile_naca="", polar_file="", alpha_wing_deg=np.nan,
        alpha_domain_min_deg=np.nan, alpha_domain_max_deg=np.nan,
        alpha_min_valid_deg=np.nan, alpha_max_valid_deg=np.nan,
        Re_domain_min=np.nan, Re_domain_max=np.nan, Re_lower=np.nan, Re_upper=np.nan,
        Re_low_used=np.nan, Re_high_used=np.nan, Re_reference=np.nan,
        CD_profile=CDp, CD_induced=CDi, naca_valid=True,
        naca_warning="; ".join(warnings_out), naca_cache_file_reads=int(NACA_POLAR_CACHE_DIAGNOSTICS["physical_file_reads"]),
        naca_cache_hits=int(NACA_POLAR_CACHE_DIAGNOSTICS["cache_hits"]), z_CE_m=z_CE_abs,
        z_WL_m=z_WL, z_CLR_m=z_CLR, M_aero_about_WL_Nm=M_aero_about_WL,
        aero_model=model, sailset=mi.sailset.sailset, flat=flat_value, reef=reef_value,
        CL_main=main["CL"], CD_main=main["CDp"], CL_jib=jib["CL"], CD_jib=jib["CDp"],
        CL_spin=spin["CL"], CD_spin=spin["CDp"], CL_lift_weighted=CL_lift,
        CDp_weighted=CDp_weighted, CDi=CDi, CD0=CD0, CD_total=CD,
        A_ref=A_ref, A_main=geom["A_main"], A_jib=geom["A_jib"], A_fore=geom["A_fore"], A_spi=geom["A_spi"],
        zCoE_sailplan_m=z_CE_base, coefficient_source=coeff_sources,
        interpolation_method=methods, pedro_hydroheel_Nm=pedro_hydroheel,
    )

def aerodynamic_state(
    TWS_mps: float,
    TWA_deg: float,
    boat_speed_mps: float,
    heel_rad: float,
    state: BoatGeometryState,
    alpha_override_deg: float | None = None,
    flat_override: float | None = None,
    reef_override: float | None = None,
    waterline_z_m: float | None = None,
) -> AeroState:
    """Compute signed aero forces, optionally applying a vertical true-wind gradient."""

    aero_model = getattr(state.settings, "aero_model", "naca_wing")
    if is_pedro_aero_model(aero_model):
        return aerodynamic_state_pedro(
            TWS_mps,
            TWA_deg,
            boat_speed_mps,
            heel_rad,
            state,
            flat=flat_override,
            reef=reef_override,
            source=aero_model,
            waterline_z_m=waterline_z_m,
        )

    wing = state.wing
    alpha = float(state.inputs.fixed_alpha_deg if alpha_override_deg is None else alpha_override_deg)
    ce_heeled = rotate_points_about_x(np.asarray(wing.surface_centroid, dtype=float).reshape(1, 3), state.masses.center_of_gravity, heel_rad)[0]
    z_WL = float(state.hydro.waterline_z if waterline_z_m is None else waterline_z_m)
    z_CLR = z_WL - CLR_DRAFT_FACTOR * float(state.T_max_m)
    z_CE = float(ce_heeled[2])
    TWS_local = float(TWS_mps)
    if getattr(state.settings, "use_wind_gradient", False):
        TWS_local = wind_speed_at_height(
            TWS_mps,
            z_CE - z_WL,
            getattr(state.settings, "wind_reference_height_m", 10.0),
            getattr(state.settings, "wind_roughness_z0_m", 0.005),
        )
    AWS, AWA = apparent_wind(TWS_local, boat_speed_mps, TWA_deg, heel_rad)
    profile_code = normalize_naca_code(state.inputs.wing_naca)
    polar_path, polar_df = cached_naca_polar(profile_code, state.inputs.polar_folder)
    Re = safe_div(C.RHO_AIR * AWS * wing.mean_chord_m, C.MU_AIR, default=0.0)
    polar = interpolate_naca_polar(polar_df, alpha, Re, table_name=f"{profile_code} ({polar_path.name})")
    finite = finite_wing_from_profile(polar["CL"], polar["CD"], wing.aspect_ratio, state.inputs.oswald_efficiency)
    CL = finite["CL"]
    CD = finite["CD"]
    CM = float(polar["CM"])
    q = 0.5 * C.RHO_AIR * AWS**2
    lift = q * wing.planform_area_m2 * CL
    drag = q * wing.planform_area_m2 * CD
    awa_rad = deg_to_rad(AWA)
    flow = np.array([np.cos(awa_rad), np.sin(awa_rad)])
    lift_dir = np.array([np.sin(awa_rad), -np.cos(awa_rad)])
    force_xy = lift * lift_dir - drag * flow
    F_drive = float(force_xy[0])
    F_side_signed = float(force_xy[1])
    F_side = abs(F_side_signed)
    M_aero_about_WL = F_side * (z_CE - z_WL)
    return AeroState(
        AWS_mps=AWS,
        AWA_deg=AWA,
        CL=CL,
        CD=CD,
        CM=CM,
        Re=Re,
        wing_orientation_deg=AWA - alpha,
        F_drive_N=F_drive,
        F_side_N=F_side,
        F_side_signed_N=F_side_signed,
        M_aero_Nm=M_aero_about_WL,
        center_of_effort=ce_heeled,
        profile_naca=profile_code,
        polar_file=str(polar_path),
        alpha_wing_deg=alpha,
        alpha_domain_min_deg=float(polar["alpha_min"]),
        alpha_domain_max_deg=float(polar["alpha_max"]),
        alpha_min_valid_deg=float(polar["alpha_min_valid_deg"]),
        alpha_max_valid_deg=float(polar["alpha_max_valid_deg"]),
        Re_domain_min=float(polar["Re_min"]),
        Re_domain_max=float(polar["Re_max"]),
        Re_lower=float(polar["Re_lower"]),
        Re_upper=float(polar["Re_upper"]),
        Re_low_used=float(polar["Re_low_used"]),
        Re_high_used=float(polar["Re_high_used"]),
        Re_reference=float(polar["Re_reference"]) if np.isfinite(polar["Re_reference"]) else np.nan,
        CD_profile=float(finite["CD_profile"]),
        CD_induced=float(finite["CD_induced"]),
        naca_valid=True,
        naca_warning=str(polar.get("Re_warning", "")),
        naca_cache_file_reads=int(NACA_POLAR_CACHE_DIAGNOSTICS["physical_file_reads"]),
        naca_cache_hits=int(NACA_POLAR_CACHE_DIAGNOSTICS["cache_hits"]),
        z_CE_m=z_CE,
        z_WL_m=z_WL,
        z_CLR_m=z_CLR,
        M_aero_about_WL_Nm=M_aero_about_WL,
    )

def hull_viscous_resistance(V_B: float, hydro: HullHydrostatics, state: BoatGeometryState, constants: PhysicalConstants = C) -> tuple[float, list[str]]:
    """Hull viscous resistance for OceanVPP/Delft or explicitly simplified ORC mode."""

    if V_B <= 1e-9 or hydro.S_C <= 0:
        return 0.0, []
    mode = getattr(state.settings, "hydro_model", "oceanvpp_delft")
    warnings_out: list[str] = []
    if mode == "orc_simplified":
        L_ref = getattr(state.settings, "orc_LSM1_m", None)
        if L_ref is None or not np.isfinite(float(L_ref)) or float(L_ref) <= 0:
            L_ref = hydro.LWL
            warnings_out.append("ORC simplified mode: LSM1 unavailable, using LWL as fallback.")
        Re = V_B * 0.85 * float(L_ref) / constants.NU_WATER
        if Re <= 1.0:
            return 0.0, warnings_out
        Cf = 0.066 / (np.log10(Re) - 2.03) ** 2
        return float(0.5 * constants.RHO_WATER * V_B**2 * hydro.S_C * Cf * 1.05), warnings_out
    Re = V_B * 0.7 * hydro.LWL / constants.NU_WATER
    if Re <= 1.0:
        return 0.0, warnings_out
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2
    return float(0.5 * constants.RHO_WATER * V_B**2 * hydro.S_C * Cf), warnings_out

def heeled_hull_residuary_delta(V_B: float, heel_rad: float, hydro: HullHydrostatics, tables: dict[str, pd.DataFrame], constants: PhysicalConstants = C) -> tuple[float, bool, bool, str]:
    """Delft/OceanVPP heeled hull residuary correction using DSYHS LCB percent from midship.

    Table IV-4 is evaluated with LCB_pct_midship in percent of LWL from midship, e.g. -4.0.
    The correction is signed; negative deltas are retained until the final R_total validity guard.
    """

    phi = abs(float(heel_rad))
    if phi <= 1e-12:
        return 0.0, True, False, ""
    Fn = froude_number(V_B, hydro.LWL, constants)
    lo, hi = fn_domain(tables, "heeled_residuary")
    if Fn > hi + 1e-12:
        return 0.0, False, False, f"delft_fn_above_max: heeled_residuary Fn={Fn:.5g} > {hi:.5g}"
    fn_for_coeff = lo if Fn < lo - 1e-12 else Fn
    coeff = bounded_table_row(tables, "heeled_residuary", "Fn", fn_for_coeff)
    bt = safe_div(hydro.BWL, hydro.TC, default=0.0)
    lb = safe_div(hydro.LWL, hydro.BWL, default=0.0)
    lcb_table = float(hydro.LCB_pct_midship)
    rr20_scaled = (
        coeff["u0"]
        + coeff["u1"] * lb
        + coeff["u2"] * bt
        + coeff["u3"] * bt**2
        + coeff["u4_inferred"] * lcb_table
        + coeff["u5"] * lcb_table**2
    )
    # Table IV-4 publishes u0..u5 multiplied by 1000.
    rr20 = rr20_scaled / 1000.0
    value = float(rr20 * hydro.volume_C * constants.RHO_WATER * constants.G * 6.0 * phi ** 1.7)
    if Fn < lo - 1e-12:
        return _scale_low_fn(value, Fn, lo), True, True, "heeled_residuary"
    return value, True, False, ""

def heeled_keel_residuary_delta(V_B: float, heel_rad: float, hydro: HullHydrostatics, centerboard: FoilGeometry, total_draft_m: float, constants: PhysicalConstants = C) -> tuple[float, bool, bool, str]:
    """Analytic centerboard heeled residuary correction with signed deltas retained."""

    volume_k = max(centerboard.volume_m3, 0.0)
    if volume_k <= 0 or abs(heel_rad) <= 1e-12:
        return 0.0, True, False, ""
    Fn = froude_number(V_B, hydro.LWL, constants)
    ratio = safe_div(hydro.TC, total_draft_m, default=0.0)
    CH = (
        -3.5837 * ratio
        - 0.0518 * safe_div(hydro.BWL, hydro.TC, default=0.0)
        + 0.5958 * ratio * safe_div(hydro.BWL, hydro.TC, default=0.0)
        + 0.2055 * safe_div(hydro.LWL, volume_k ** (1 / 3), default=0.0)
    )
    return float(volume_k * constants.RHO_WATER * constants.G * CH * Fn**2 * abs(heel_rad)), True, False, ""

def induced_hydrodynamic_drag(V_B: float, F_side: float, heel_rad: float, hydro: HullHydrostatics, centerboard: FoilGeometry, total_draft_m: float, tables: dict[str, pd.DataFrame], constants: PhysicalConstants = C, lateral_area_m2: float | None = None) -> tuple[float, float, float, bool, str]:
    """ORC-simplified induced drag: R_i = F_H^2 / (pi q T_eff^2)."""

    F_side_abs = abs(float(F_side))
    if V_B <= 1e-8:
        if F_side_abs <= 1e-9:
            return 0.0, 0.0, 0.0, True, ""
        return 0.0, np.nan, 0.0, False, "induced_drag_singularity"
    phi_deg = abs(rad_to_deg(heel_rad))
    lo, hi = table_domain(tables, "effective_span", "heel_deg")
    if phi_deg > hi + 1e-12:
        return 0.0, np.nan, np.nan, False, f"heel_above_table_range: effective_span phi={phi_deg:.5g} > {hi:.5g}"
    coeff = bounded_table_row(tables, "effective_span", "heel_deg", max(phi_deg, lo))
    Fn = froude_number(V_B, hydro.LWL, constants)
    root_chord = chord_at_span(centerboard.contour, np.min(centerboard.contour[:, 1]))[2]
    tip_chord = chord_at_span(centerboard.contour, np.max(centerboard.contour[:, 1]))[2]
    TR = safe_div(tip_chord, root_chord, default=1.0)
    tc_over_T = safe_div(hydro.TC, total_draft_m, default=0.0)
    TE_over_T = (
        coeff["A1"] * tc_over_T
        + coeff["A2"] * tc_over_T**2
        + coeff["A3"] * safe_div(hydro.BWL, hydro.TC, default=0.0)
        + coeff["A4"] * TR
    ) * (coeff["B0"] + coeff["B1"] * Fn)
    TE = float(TE_over_T * total_draft_m)
    if not np.isfinite(TE) or TE <= 0.0:
        return 0.0, TE, np.nan, False, f"invalid_effective_span: T_eff={TE}"
    q_water = 0.5 * constants.RHO_WATER * V_B**2
    return float(F_side_abs**2 / (np.pi * q_water * TE**2)), TE, float(q_water), True, ""

def resistance_breakdown(V_B: float, heel_rad: float, F_side: float, state: BoatGeometryState) -> ResistanceBreakdown:
    """Compute resistance with signed local deltas and explicit hydro-model labels."""

    hydro = state.hydro
    tables = state.delft_tables
    RvC, viscous_warnings = hull_viscous_resistance(V_B, hydro, state)
    RvD = orc_appendage_viscous(V_B, state.centerboard)
    RvR = orc_appendage_viscous(V_B, state.rudder)
    raw_r, Rr, hull_ok, hull_ext, hull_ext_table = bare_hull_residuary(V_B, hydro, tables)
    Rrk, keel_ok, keel_ext, keel_ext_table = centerboard_residuary(V_B, hydro, state.centerboard, state.T_max_m, tables)
    if state.settings.wetted_surface_mode == "STL direct" and state.settings.geometry_input_mode == "stl":
        _, dRf = direct_hull_wetted_surface_delta(V_B, heel_rad, state)
        heel_f_ok, heel_f_ext, heel_f_table = True, False, ""
    else:
        _, dRf, heel_f_ok, heel_f_ext, heel_f_table = wetted_surface_heel_delta(V_B, heel_rad, hydro, tables)
    dRr, heel_h_ok, heel_h_ext, heel_h_table = heeled_hull_residuary_delta(V_B, heel_rad, hydro, tables)
    dRrk, heel_k_ok, heel_k_ext, heel_k_table = heeled_keel_residuary_delta(V_B, heel_rad, hydro, state.centerboard, state.T_max_m)
    RI, T_eff, q_water, eff_ok, eff_reason = induced_hydrodynamic_drag(V_B, F_side, heel_rad, hydro, state.centerboard, state.T_max_m, tables)
    RRU = rail_under_drag(heel_rad, Rr, Rrk)
    Raw = 0.0
    total = sum(value for value in [RvC, RvD, RvR, Rr, Rrk, dRf, dRr, dRrk, RI, RRU, Raw] if np.isfinite(value))
    status_pairs = [
        ("hull_residuary", hull_ok, "" if hull_ok else hull_ext_table),
        ("appendage_residuary", keel_ok, "" if keel_ok else keel_ext_table),
        ("heel_friction", heel_f_ok, "" if heel_f_ok else heel_f_table),
        ("heel_hull_residuary", heel_h_ok, "" if heel_h_ok else heel_h_table),
        ("heel_appendage_residuary", heel_k_ok, "" if heel_k_ok else heel_k_table),
        ("effective_span", eff_ok, eff_reason),
    ]
    invalid_reasons = [reason or name for name, ok, reason in status_pairs if not ok]
    ext_tables = [name for name in [hull_ext_table if hull_ext else "", keel_ext_table if keel_ext else "", heel_f_table if heel_f_ext else "", heel_h_table if heel_h_ext else "", heel_k_table if heel_k_ext else ""] if name]
    non_positive_total = bool(total <= 0.0)
    if non_positive_total:
        invalid_reasons.append("Computed total resistance is non-positive; check resistance components.")
    table_valid = all(ok for _, ok, _ in status_pairs) and not non_positive_total
    warning_parts = list(dict.fromkeys(list(hydro.delft_warnings) + viscous_warnings + invalid_reasons))
    return ResistanceBreakdown(
        R_hull_viscous_N=RvC,
        R_centerboard_viscous_N=RvD,
        R_rudder_viscous_N=RvR,
        R_hull_residuary_raw_N=raw_r,
        R_hull_residuary_N=Rr,
        R_centerboard_residuary_N=Rrk,
        R_heel_friction_delta_N=dRf,
        R_heel_hull_residuary_delta_N=dRr,
        R_heel_keel_residuary_delta_N=dRrk,
        R_induced_N=RI,
        R_rail_under_N=RRU,
        R_total_N=float(total),
        T_eff_m=T_eff,
        q_water_Pa=q_water,
        hull_residuary_valid=hull_ok,
        appendage_residuary_valid=keel_ok,
        heel_friction_valid=heel_f_ok,
        heel_hull_residuary_valid=heel_h_ok,
        heel_appendage_residuary_valid=heel_k_ok,
        effective_span_valid=eff_ok,
        used_low_fn_physical_extension=bool(ext_tables),
        low_fn_extension_tables=", ".join(ext_tables),
        delft_valid=table_valid,
        delft_geometry_valid=bool(hydro.delft_valid),
        table_domain_valid=table_valid,
        invalid_reason="; ".join(invalid_reasons),
        hydro_model=getattr(state.settings, "hydro_model", "oceanvpp_delft"),
        induced_drag_model="orc_simplified_effective_span",
        appendage_lift_drag_model="CL_assumed_zero",
        R_added_waves_N=Raw,
        warning_message="; ".join(part for part in warning_parts if part),
        delft_warnings=list(dict.fromkeys(list(hydro.delft_warnings) + viscous_warnings)),
    )

def solver_speed_bounds(state: BoatGeometryState) -> tuple[float, float]:
    """Use the most restrictive common upper Froude bound, with explicit fallback warnings."""

    required = ["bare_hull", "keel", "heeled_residuary"]
    hi_candidates: list[float] = []
    missing: list[str] = []
    for key in required:
        try:
            _, hi = fn_domain(state.delft_tables, key)
            hi_candidates.append(float(hi))
        except Exception:
            missing.append(key)
    fallback_hi = 0.80
    if missing:
        warnings.warn(f"solver_speed_bounds: missing table(s) {', '.join(missing)}; using fallback Fn<={fallback_hi:.2f}.")
        hi_candidates.extend([fallback_hi] * len(missing))
    if not hi_candidates:
        hi_candidates = [fallback_hi]
    hi_fn = min(hi_candidates)
    scale = np.sqrt(C.G * max(state.hydro.LWL, 1e-9))
    return 0.0, float(hi_fn * scale)

RESULT_COLUMNS = [
    "TWS", "TWS_unit", "TWS_kn", "TWS_mps", "TWA_deg", "AWA_deg", "AWS", "AWS_mps", "AWS_kn",
    "boat_speed_mps", "boat_speed_kn", "boat_speed_knots", "heel_deg", "Fn", "best_alpha_deg",
    "best_flat", "best_reef", "aero_model", "sailset", "reference_area_mode", "flat", "reef",
    "alpha_wing_deg", "wing_orientation_deg", "alpha_mode", "alpha_candidate_count",
    "alpha_valid_candidate_count", "flat_reef_candidate_count", "flat_reef_valid_candidate_count",
    "CL_main", "CD_main", "CL_jib", "CD_jib", "CL_spin", "CD_spin", "CL_lift_weighted",
    "CDp_weighted", "CDi", "CD0", "CD_total", "A_ref", "A_main", "A_ref_m2", "A_main_m2", "A_jib", "A_fore", "A_spi",
    "zCoE_sailplan_m", "coefficient_source", "interpolation_method",
    "Cl_wing", "Cd_wing", "Cm_wing", "CD_profile", "CD_induced",
    "Re_wing", "Re_lower", "Re_upper", "Re_low_used", "Re_high_used", "Re_reference",
    "Re_domain_min", "Re_domain_max", "alpha_domain_min_deg", "alpha_domain_max_deg",
    "alpha_min_valid_deg", "alpha_max_valid_deg", "profile_naca", "polar_file", "naca_valid",
    "naca_warning", "naca_cache_file_reads", "naca_cache_hits", "F_drive_N", "F_side_N",
    "F_side_aero_N", "F_side_signed_N", "F_side_hydro_assumed_N", "R_viscous_hull_N",
    "R_viscous_appendages_N", "R_hull_viscous_N", "R_centerboard_viscous_N", "R_rudder_viscous_N",
    "R_residuary_hull_N", "R_residuary_appendages_N", "R_hull_residuary_N", "R_centerboard_residuary_N",
    "R_heel_delta_N", "R_heel_friction_delta_N", "R_heel_hull_residuary_delta_N",
    "R_heel_keel_residuary_delta_N", "R_induced_N", "induced_drag_model", "appendage_lift_drag_model", "T_eff_m", "q_water_Pa",
    "R_rail_under_N", "R_added_waves_N", "R_total_N", "hull_residuary_valid",
    "appendage_residuary_valid", "heel_friction_valid", "heel_hull_residuary_valid",
    "heel_appendage_residuary_valid", "effective_span_valid", "used_low_fn_physical_extension",
    "low_fn_extension_tables", "z_CE_m", "z_WL_m", "z_CLR_m", "T_max_m", "M_aero_Nm",
    "M_aero_about_WL_Nm", "M_hydroheel_Nm", "M_heel_Nm", "M_heel_total_Nm", "heeling_moment_model", "M_hydrostatic_Nm",
    "RMV_Nm", "M_right_Nm", "M_right_total_Nm", "force_residual_N", "moment_residual_Nm",
    "hydro_model", "solver_method", "geometry_input_mode", "mass_mode", "delft_valid", "delft_geometry_valid",
    "table_domain_valid", "hydrostatic_geometry_valid", "hydrostatic_boolean_union_used",
    "mesh_overlap_detected", "estimated_overlap_volume_m3", "solver_success", "validity_flag",
    "failure_reason_code", "invalid_reason", "warning_message", "solver_message", "solver_iterations",
]

def row_from_solution(TWS_kn: float, TWS_mps: float, TWA_deg: float, solution: dict[str, Any], TWS_unit: str = "knots", TWS_display: float | None = None) -> dict[str, Any]:
    """Build one comparison-friendly result row."""

    eq: EquilibriumState | None = solution.get("eq")
    row = {col: np.nan for col in RESULT_COLUMNS}
    row.update({
        "TWS": float(TWS_display if TWS_display is not None else TWS_kn),
        "TWS_unit": TWS_unit,
        "TWS_kn": TWS_kn,
        "TWS_mps": TWS_mps,
        "TWA_deg": TWA_deg,
    })
    if eq is None:
        warning = solution.get("invalid_reason", solution.get("message", "solveur indisponible"))
        row.update({
            "delft_valid": False,
            "table_domain_valid": False,
            "naca_valid": False,
            "solver_success": False,
            "validity_flag": "invalid",
            "failure_reason_code": solution.get("failure_reason_code", ""),
            "invalid_reason": warning,
            "warning_message": warning,
            "solver_message": solution.get("message", "solveur indisponible"),
            "solver_iterations": solution.get("solver_iterations", np.nan),
        })
        return row
    aero = eq.aero
    rb = eq.resistance
    success = bool(solution.get("success", False))
    display_speed = solution["boat_speed_mps"] if success else np.nan
    display_heel = solution["heel_rad"] if success else np.nan
    state_for_row = solution.get("state", None)
    manual_for_row = getattr(state_for_row, "manual_input", None)
    reference_area_mode_value = getattr(getattr(manual_for_row, "sailset", None), "reference_area_mode", np.nan)
    heeling_moment_model = "pedro_hardcoded" if is_pedro_aero_model(aero.aero_model) and bool(getattr(getattr(state_for_row, "settings", None), "pedro_hardcoded_hydroheel", False)) else "argo_geometric_CLR"
    warning_message = "; ".join(dict.fromkeys(part for part in [rb.warning_message, aero.naca_warning, getattr(getattr(state_for_row, "settings", None), "setup_warning_message", ""), solution.get("invalid_reason", ""), eq.invalid_reason, "pedro_hardcoded reproduces OceanVPP/Pedro behaviour; argo_geometric_CLR is more physically explicit for ARGO-VPP."] if part))
    row.update({
        "AWA_deg": aero.AWA_deg,
        "AWS": aero.AWS_mps,
        "AWS_mps": aero.AWS_mps,
        "AWS_kn": aero.AWS_mps / C.KNOT_TO_MPS,
        "boat_speed_mps": display_speed,
        "boat_speed_kn": display_speed / C.KNOT_TO_MPS if np.isfinite(display_speed) else np.nan,
        "boat_speed_knots": display_speed / C.KNOT_TO_MPS if np.isfinite(display_speed) else np.nan,
        "heel_deg": rad_to_deg(display_heel) if np.isfinite(display_heel) else np.nan,
        "Fn": eq.Fn,
        "best_alpha_deg": aero.alpha_wing_deg,
        "best_flat": solution.get("flat", aero.flat),
        "best_reef": solution.get("reef", aero.reef),
        "aero_model": aero.aero_model,
        "sailset": aero.sailset,
        "reference_area_mode": reference_area_mode_value,
        "flat": aero.flat,
        "reef": aero.reef,
        "alpha_wing_deg": aero.alpha_wing_deg,
        "wing_orientation_deg": aero.wing_orientation_deg,
        "alpha_mode": solution.get("alpha_mode", np.nan),
        "alpha_candidate_count": solution.get("alpha_candidate_count", 1),
        "alpha_valid_candidate_count": solution.get("alpha_valid_candidate_count", 1 if success else 0),
        "flat_reef_candidate_count": solution.get("flat_reef_candidate_count", np.nan),
        "flat_reef_valid_candidate_count": solution.get("flat_reef_valid_candidate_count", np.nan),
        "CL_main": aero.CL_main,
        "CD_main": aero.CD_main,
        "CL_jib": aero.CL_jib,
        "CD_jib": aero.CD_jib,
        "CL_spin": aero.CL_spin,
        "CD_spin": aero.CD_spin,
        "CL_lift_weighted": aero.CL_lift_weighted,
        "CDp_weighted": aero.CDp_weighted,
        "CDi": aero.CDi,
        "CD0": aero.CD0,
        "CD_total": aero.CD_total,
        "A_ref": aero.A_ref,
        "A_main": aero.A_main,
        "A_ref_m2": aero.A_ref,
        "A_main_m2": aero.A_main,
        "A_jib": aero.A_jib,
        "A_fore": aero.A_fore,
        "A_spi": aero.A_spi,
        "zCoE_sailplan_m": aero.zCoE_sailplan_m,
        "coefficient_source": aero.coefficient_source,
        "interpolation_method": aero.interpolation_method,
        "Cl_wing": aero.CL,
        "Cd_wing": aero.CD,
        "Cm_wing": aero.CM,
        "CD_profile": aero.CD_profile,
        "CD_induced": aero.CD_induced,
        "Re_wing": aero.Re,
        "Re_lower": aero.Re_lower,
        "Re_upper": aero.Re_upper,
        "Re_low_used": aero.Re_low_used,
        "Re_high_used": aero.Re_high_used,
        "Re_reference": aero.Re_reference,
        "Re_domain_min": aero.Re_domain_min,
        "Re_domain_max": aero.Re_domain_max,
        "alpha_domain_min_deg": aero.alpha_domain_min_deg,
        "alpha_domain_max_deg": aero.alpha_domain_max_deg,
        "alpha_min_valid_deg": aero.alpha_min_valid_deg,
        "alpha_max_valid_deg": aero.alpha_max_valid_deg,
        "profile_naca": aero.profile_naca,
        "polar_file": aero.polar_file,
        "naca_valid": aero.naca_valid,
        "naca_warning": aero.naca_warning,
        "naca_cache_file_reads": aero.naca_cache_file_reads,
        "naca_cache_hits": aero.naca_cache_hits,
        "F_drive_N": aero.F_drive_N,
        "F_side_N": aero.F_side_N,
        "F_side_aero_N": aero.F_side_N,
        "F_side_signed_N": aero.F_side_signed_N,
        "F_side_hydro_assumed_N": aero.F_side_N,
        "R_viscous_hull_N": rb.R_hull_viscous_N,
        "R_viscous_appendages_N": rb.R_centerboard_viscous_N + rb.R_rudder_viscous_N,
        "R_hull_viscous_N": rb.R_hull_viscous_N,
        "R_centerboard_viscous_N": rb.R_centerboard_viscous_N,
        "R_rudder_viscous_N": rb.R_rudder_viscous_N,
        "R_residuary_hull_N": rb.R_hull_residuary_N,
        "R_residuary_appendages_N": rb.R_centerboard_residuary_N,
        "R_hull_residuary_N": rb.R_hull_residuary_N,
        "R_centerboard_residuary_N": rb.R_centerboard_residuary_N,
        "R_heel_delta_N": rb.R_heel_friction_delta_N + rb.R_heel_hull_residuary_delta_N + rb.R_heel_keel_residuary_delta_N,
        "R_heel_friction_delta_N": rb.R_heel_friction_delta_N,
        "R_heel_hull_residuary_delta_N": rb.R_heel_hull_residuary_delta_N,
        "R_heel_keel_residuary_delta_N": rb.R_heel_keel_residuary_delta_N,
        "R_induced_N": rb.R_induced_N,
        "induced_drag_model": rb.induced_drag_model,
        "appendage_lift_drag_model": rb.appendage_lift_drag_model,
        "T_eff_m": rb.T_eff_m,
        "q_water_Pa": rb.q_water_Pa,
        "R_rail_under_N": rb.R_rail_under_N,
        "R_added_waves_N": rb.R_added_waves_N,
        "R_total_N": rb.R_total_N,
        "hull_residuary_valid": rb.hull_residuary_valid,
        "appendage_residuary_valid": rb.appendage_residuary_valid,
        "heel_friction_valid": rb.heel_friction_valid,
        "heel_hull_residuary_valid": rb.heel_hull_residuary_valid,
        "heel_appendage_residuary_valid": rb.heel_appendage_residuary_valid,
        "effective_span_valid": rb.effective_span_valid,
        "used_low_fn_physical_extension": rb.used_low_fn_physical_extension,
        "low_fn_extension_tables": rb.low_fn_extension_tables,
        "z_CE_m": eq.z_CE_m,
        "z_WL_m": eq.z_WL_m,
        "z_CLR_m": eq.z_CLR_m,
        "T_max_m": solution.get("T_max_m", np.nan),
        "M_aero_Nm": aero.M_aero_Nm,
        "M_aero_about_WL_Nm": eq.M_aero_about_WL_Nm,
        "M_hydroheel_Nm": eq.M_hydroheel_Nm,
        "M_heel_Nm": eq.M_heel_total_Nm,
        "M_heel_total_Nm": eq.M_heel_total_Nm,
        "heeling_moment_model": heeling_moment_model,
        "M_hydrostatic_Nm": eq.M_hydrostatic_Nm,
        "RMV_Nm": eq.RMV_Nm,
        "M_right_Nm": eq.M_right_total_Nm,
        "M_right_total_Nm": eq.M_right_total_Nm,
        "force_residual_N": eq.force_residual_N,
        "moment_residual_Nm": eq.moment_residual_Nm,
        "solver_iterations": solution.get("solver_iterations", np.nan),
        "hydro_model": rb.hydro_model,
        "solver_method": solution.get("solver_method", getattr(getattr(state_for_row, "settings", None), "solver_method", np.nan)),
        "geometry_input_mode": getattr(solution.get("state", None).settings, "geometry_input_mode", np.nan) if solution.get("state", None) is not None else np.nan,
        "mass_mode": getattr(solution.get("state", None).settings, "mass_mode", np.nan) if solution.get("state", None) is not None else np.nan,
        "delft_valid": rb.delft_valid,
        "delft_geometry_valid": rb.delft_geometry_valid,
        "table_domain_valid": eq.table_domain_valid,
        "hydrostatic_geometry_valid": eq.hydrostatic_geometry_valid,
        "hydrostatic_boolean_union_used": getattr(solution.get("state", None), "hydrostatic_boolean_union_used", np.nan),
        "mesh_overlap_detected": getattr(solution.get("state", None), "mesh_overlap_detected", np.nan),
        "estimated_overlap_volume_m3": getattr(solution.get("state", None), "estimated_overlap_volume_m3", np.nan),
        "solver_success": success,
        "validity_flag": "valid" if success else "invalid",
        "failure_reason_code": solution.get("failure_reason_code", ""),
        "invalid_reason": "" if success else solution.get("invalid_reason", eq.invalid_reason),
        "warning_message": warning_message,
        "solver_message": solution["message"],
    })
    return row

def solve_vpp_grid(tws_knots: list[float], state: BoatGeometryState, twa_values_deg: list[float] | np.ndarray | None = None, tws_unit: str = "knots", progress_callback: Callable[[int, int, float, float, bool], None] | None = None) -> pd.DataFrame:
    """Solve the 2-DOF VPP grid. TWS may be supplied in knots or m/s."""

    rows: list[dict[str, Any]] = []
    previous_by_twa: dict[float, np.ndarray] = {}
    twa_values = np.asarray(twa_values_deg, dtype=float) if twa_values_deg is not None else analysis_angle_values(TWA_RANGE_MIN_DEG, TWA_RANGE_MAX_DEG, TWA_RANGE_STEPS)
    total_conditions = len(tws_knots) * len(twa_values)
    completed_conditions = 0
    for tws_value in tws_knots:
        TWS = tws_to_mps(float(tws_value), tws_unit)
        TWS_kn = TWS / C.KNOT_TO_MPS
        x0 = None
        for TWA in twa_values:
            seed = previous_by_twa.get(float(TWA), x0)
            solution = solve_condition_with_alpha_mode(TWS, float(TWA), state, seed)
            solution["T_max_m"] = state.T_max_m
            solution["state"] = state
            rows.append(row_from_solution(float(TWS_kn), TWS, float(TWA), solution, TWS_unit=tws_unit, TWS_display=float(tws_value)))
            if solution.get("success"):
                x0 = np.array([solution["boat_speed_mps"], solution["heel_rad"]])
                previous_by_twa[float(TWA)] = x0
            completed_conditions += 1
            if progress_callback is not None:
                progress_callback(completed_conditions, total_conditions, float(tws_value), float(TWA), bool(solution.get("success")))
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)
