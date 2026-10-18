from __future__ import annotations
from .argo_vpp_common import Any, MU_AIR_DEFAULT, MU_WATER_DEFAULT, RHO_AIR_DEFAULT, RHO_WATER_DEFAULT, np, pd
from .argo_vpp_aerodynamics import interp_coeffs_for_re_alpha
from .argo_vpp_geometry import reynolds_number
from .argo_vpp_hydrodynamics import calc_hydro_drag
from .argo_vpp_models import SimulationCase

def normalize_cap_for_half_polar(CAP_deg: float) -> float:
    cap = float(CAP_deg) % 360.0
    if cap > 180.0:
        cap = 360.0 - cap
    return cap

def project_real_wind_forces_on_boat_axis(F_portance: float, F_trainee: float, CAP_deg: float) -> tuple[float, str]:
    cap = normalize_cap_for_half_polar(CAP_deg)
    if np.isclose(cap, 90.0, atol=1e-9):
        return float(F_portance), "CAP = 90 deg"
    if cap < 90.0:
        f_prop = F_portance * np.cos(np.radians(90.0 - cap)) - F_trainee * np.cos(np.radians(cap))
        return float(f_prop), "0 deg <= CAP < 90 deg"
    f_prop = F_portance * np.cos(np.radians(cap - 90.0)) + F_trainee * np.cos(np.radians(180.0 - cap))
    return float(f_prop), "90 deg < CAP <= 180 deg"

def project_real_wind_side_force(F_portance: float, F_trainee: float, CAP_deg: float) -> tuple[float, str]:
    cap = normalize_cap_for_half_polar(CAP_deg)
    if np.isclose(cap, 90.0, atol=1e-9):
        return float(F_trainee), "CAP = 90 deg"
    if cap < 90.0:
        f_aero = F_portance * np.cos(np.radians(cap)) + F_trainee * np.cos(np.radians(90.0 - cap))
        return float(f_aero), "0 deg <= CAP < 90 deg"
    magnitude = F_portance * np.cos(np.radians(180.0 - cap)) + F_trainee * np.cos(np.radians(cap - 90.0))
    return float(-magnitude), "90 deg < CAP <= 180 deg"

def calc_aero_simple_true_wind(
    V_TW: float,
    CAP_deg: float,
    alpha_trim_deg: float,
    polar_tables: dict[float, pd.DataFrame],
    A_sail: float,
    c_moy: float,
    rho_air: float = RHO_AIR_DEFAULT,
    mu_air: float = MU_AIR_DEFAULT,
) -> dict[str, Any]:
    re_value = reynolds_number(rho_air, V_TW, c_moy, mu_air)
    try:
        coeffs = interp_coeffs_for_re_alpha(polar_tables, re_value, alpha_trim_deg)
        cl, cd = float(coeffs["Cl"]), float(coeffs["Cd"])
    except Exception:
        cl, cd = 0.0, 0.0
    q_dyn = 0.5 * rho_air * float(V_TW) ** 2
    lift = cl * q_dyn * A_sail
    drag = cd * q_dyn * A_sail
    f_prop, projection_case = project_real_wind_forces_on_boat_axis(lift, drag, CAP_deg)
    f_lat, lateral_case = project_real_wind_side_force(lift, drag, CAP_deg)
    return {
        "F_prop": float(f_prop),
        "F_aero_lat": float(f_lat),
        "F_aero_lat_abs": float(abs(f_lat)),
        "L_aero": float(lift),
        "D_aero": float(drag),
        "Cl": cl,
        "Cd": cd,
        "Re": re_value,
        "projection_case": projection_case,
        "lateral_case": lateral_case,
    }

def solve_speed_from_propulsion(F_prop: float, S_wetted: float, CR_friction: float, Rcoque_coeff: float, rho_water: float) -> float:
    C_R_total = float(CR_friction) + float(Rcoque_coeff)
    denom = 0.5 * float(rho_water) * float(S_wetted) * C_R_total
    if F_prop <= 0.0 or denom <= 0.0:
        return 0.0
    return float(np.sqrt(float(F_prop) / denom))

def solve_equilibrium_sail_true_wind(
    V_TW: float,
    CAP_deg: float,
    polar_tables: dict[float, pd.DataFrame],
    case: SimulationCase,
) -> dict[str, Any]:
    aero = calc_aero_simple_true_wind(V_TW, CAP_deg, case.alpha_trim, polar_tables, case.A_sail, case.c_moy, case.rho_air, case.mu_air)
    V_B = solve_speed_from_propulsion(aero["F_prop"], case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water)
    return {"V_B": V_B, "converged": True, "aero": aero, "R_hydro": calc_hydro_drag(V_B, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water), "mode": "voile"}

def calc_spy_equilibrium_true_wind(V_TW: float, CAP_deg: float, case: SimulationCase) -> dict[str, Any] | None:
    cap = normalize_cap_for_half_polar(CAP_deg)
    if cap < 90.0:
        return None
    factor = abs(np.sin(np.radians(90.0 - cap)))
    if factor < 0.01:
        return None
    F_spy = case.Cx_spy * 0.5 * case.rho_air * float(V_TW) ** 2 * case.S_spy
    F_prop_spy = F_spy * factor
    V_B = solve_speed_from_propulsion(F_prop_spy, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water)
    return {"V_B": V_B, "mode": "spy", "F_prop_spy": F_prop_spy, "converged": True}

def calc_derive_hydro_forces(
    V_B: float,
    psi_deg: float,
    derive_tables: dict[float, pd.DataFrame],
    S_derive: float,
    c_derive: float,
    rho_water: float = RHO_WATER_DEFAULT,
    mu_water: float = MU_WATER_DEFAULT,
) -> dict[str, float]:
    V_B = float(V_B)
    psi_deg = abs(float(psi_deg))
    if V_B <= 0.0 or S_derive <= 0.0 or c_derive <= 0.0:
        return {"V_derive": 0.0, "V_lateral": 0.0, "Re_derive": 0.0, "Cl_derive": 0.0, "Cd_derive": 0.0, "L_derive": 0.0, "D_derive": 0.0, "F_hydro_derive": 0.0}
    psi_rad = np.radians(psi_deg)
    cos_psi = max(float(np.cos(psi_rad)), 1e-9)
    V_lateral = V_B * np.tan(psi_rad)
    V_derive = max(float(np.sqrt(V_B**2 + V_lateral**2)), float(V_B / cos_psi))
    re_derive = reynolds_number(rho_water, V_derive, c_derive, mu_water)
    try:
        coeffs = interp_coeffs_for_re_alpha(derive_tables, re_derive, psi_deg)
        cl, cd = float(coeffs["Cl"]), float(coeffs["Cd"])
    except Exception:
        cl, cd = 0.0, 0.0
    q_dyn = 0.5 * rho_water * V_derive**2
    lift = cl * q_dyn * S_derive
    drag = cd * q_dyn * S_derive
    f_hydro = drag * np.sin(psi_rad) + lift * np.cos(psi_rad)
    return {"V_derive": float(V_derive), "V_lateral": float(V_lateral), "Re_derive": float(re_derive), "Cl_derive": cl, "Cd_derive": cd, "L_derive": float(lift), "D_derive": float(drag), "F_hydro_derive": float(f_hydro)}

def solve_derive_equilibrium(
    F_aero_lat: float,
    V_B: float,
    derive_tables: dict[float, pd.DataFrame],
    S_derive: float,
    c_derive: float,
    rho_water: float = RHO_WATER_DEFAULT,
    mu_water: float = MU_WATER_DEFAULT,
    psi_max_deg: float = 20.0,
    tolerance: float = 0.5,
    max_iterations: int = 60,
) -> dict[str, Any]:
    target = abs(float(F_aero_lat))
    sign = 1.0 if F_aero_lat >= 0.0 else -1.0
    if target <= tolerance or V_B <= 0.0 or S_derive <= 0.0:
        forces_zero = calc_derive_hydro_forces(V_B, 0.0, derive_tables, S_derive, c_derive, rho_water, mu_water)
        return {"psi_deg": 0.0, "converged": True, "reason": "target zero", "iterations": 0, "derive_residual": forces_zero["F_hydro_derive"] - target, **forces_zero}
    psi_min = 0.0
    psi_max = float(psi_max_deg)
    forces_max = calc_derive_hydro_forces(V_B, psi_max, derive_tables, S_derive, c_derive, rho_water, mu_water)
    if forces_max["F_hydro_derive"] < target:
        return {"psi_deg": sign * psi_max, "converged": False, "reason": "psi max reached", "iterations": 0, "derive_residual": forces_max["F_hydro_derive"] - target, **forces_max}
    forces_mid: dict[str, float] = {}
    converged = False
    psi_mid = 0.0
    iterations = 0
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        psi_mid = 0.5 * (psi_min + psi_max)
        forces_mid = calc_derive_hydro_forces(V_B, psi_mid, derive_tables, S_derive, c_derive, rho_water, mu_water)
        error = forces_mid["F_hydro_derive"] - target
        if abs(error) < tolerance:
            converged = True
            break
        if error < 0.0:
            psi_min = psi_mid
        else:
            psi_max = psi_mid
    derive_residual = forces_mid["F_hydro_derive"] - target
    return {"psi_deg": float(sign * psi_mid), "converged": converged, "reason": "ok" if converged else "max_iterations", "iterations": iterations, "derive_residual": derive_residual, **forces_mid}

def compute_aero_application_height(case: SimulationCase) -> float:
    if case.h_aero_mode == "Manual":
        return max(float(case.h_aero_manual), 1e-6)
    return max(0.5 * float(case.wing_height), 1e-6)

def solve_heel_equilibrium(
    F_aero_lat: float,
    F_derive: float,
    h_aero: float,
    h_derive: float,
    m_quille_kg: float,
    hauteur_quille: float,
    theta_max_deg: float = 45.0,
    g: float = 9.81,
    tolerance: float = 0.05,
    max_iterations: int = 60,
) -> dict[str, Any]:
    F_aero_lat = abs(float(F_aero_lat))
    F_derive = abs(float(F_derive))
    M_quille_coeff = max(float(m_quille_kg), 0.0) * g * max(float(hauteur_quille), 0.0)

    def moments(theta_deg: float) -> tuple[float, float, float, float]:
        theta_rad = np.radians(theta_deg)
        M_aero = F_aero_lat * np.cos(theta_rad) * max(float(h_aero), 0.0)
        M_derive = F_derive * np.cos(theta_rad) * max(float(h_derive), 0.0)
        M_quille = M_quille_coeff * np.sin(theta_rad)
        return float(M_aero - M_derive - M_quille), float(M_aero), float(M_derive), float(M_quille)

    residual_0, M_a0, M_d0, M_q0 = moments(0.0)
    if theta_max_deg <= 0.0 or residual_0 <= 0.0:
        return {"theta_deg": 0.0, "converged": True, "reason": "stable at theta=0", "iterations": 0, "M_aero": M_a0, "M_derive": M_d0, "M_quille": M_q0, "heel_residual": residual_0}
    residual_max, M_am, M_dm, M_qm = moments(theta_max_deg)
    if residual_max > 0.0:
        return {"theta_deg": theta_max_deg, "converged": False, "reason": "theta max reached", "iterations": 0, "M_aero": M_am, "M_derive": M_dm, "M_quille": M_qm, "heel_residual": residual_max}
    lo, hi = 0.0, theta_max_deg
    theta_mid = 0.0
    values = moments(theta_mid)
    converged = False
    iterations = 0
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        theta_mid = 0.5 * (lo + hi)
        values = moments(theta_mid)
        if abs(values[0]) < tolerance:
            converged = True
            break
        if values[0] > 0.0:
            lo = theta_mid
        else:
            hi = theta_mid
    residual, M_aero, M_derive, M_quille = values
    return {"theta_deg": float(theta_mid), "converged": converged, "reason": "ok" if converged else "max_iterations", "iterations": iterations, "M_aero": M_aero, "M_derive": M_derive, "M_quille": M_quille, "heel_residual": residual}

def apply_heel_correction_to_equilibrium(
    eq: dict[str, Any],
    aero: dict[str, Any],
    derive_tables: dict[float, pd.DataFrame],
    case: SimulationCase,
    h_aero: float,
    n_iter: int = 4,
) -> dict[str, Any]:
    V_B_current = float(eq.get("V_B", 0.0))
    S_derive = max(float(case.S_derive), 1e-9)
    hauteur_quille = max(float(case.hauteur_quille), 1e-6)
    c_derive_eff = max(S_derive / hauteur_quille, 1e-6)
    h_derive = 0.5 * hauteur_quille
    F_aero_lat = float(aero.get("F_aero_lat", 0.0))
    F_prop_base = float(eq.get("F_prop_spy", aero.get("F_prop", 0.0)))
    derive = {}
    heel = {}
    F_prop_gite = F_prop_base
    for _ in range(int(n_iter)):
        derive = solve_derive_equilibrium(F_aero_lat, V_B_current, derive_tables, S_derive, c_derive_eff, case.rho_water, case.mu_water, case.psi_max, case.tolerance)
        heel = solve_heel_equilibrium(abs(F_aero_lat), derive["F_hydro_derive"], h_aero, h_derive, case.m_quille, hauteur_quille, case.theta_max)
        F_prop_gite = max(0.0, F_prop_base * np.cos(np.radians(float(heel["theta_deg"]))))
        V_next = solve_speed_from_propulsion(F_prop_gite, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water)
        if abs(V_next - V_B_current) < 1e-4:
            V_B_current = V_next
            break
        V_B_current = V_next
    return {"V_B_raw": float(eq.get("V_B", 0.0)), "V_B_gite": float(V_B_current), "F_prop_base": F_prop_base, "F_prop_gite": F_prop_gite, "R_hydro_gite": calc_hydro_drag(V_B_current, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water), "derive": derive, "heel": heel, "c_derive_eff": c_derive_eff, "h_derive": h_derive}

def solve_equilibrium_hybrid_true_wind(CAP_deg: float, polar_tables: dict[float, pd.DataFrame], case: SimulationCase) -> dict[str, Any]:
    eq_sail = solve_equilibrium_sail_true_wind(case.V_TW, CAP_deg, polar_tables, case)
    eq_spy = calc_spy_equilibrium_true_wind(case.V_TW, CAP_deg, case)
    V_sail = float(eq_sail.get("V_B", 0.0))
    V_spy = float(eq_spy.get("V_B", 0.0)) if eq_spy else 0.0
    if eq_spy and V_spy > V_sail:
        aero = calc_aero_simple_true_wind(case.V_TW, CAP_deg, case.alpha_trim, polar_tables, case.A_sail, case.c_moy, case.rho_air, case.mu_air)
        eq = {**eq_spy, "aero": aero, "R_hydro": calc_hydro_drag(V_spy, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.rho_water)}
    else:
        eq = eq_sail
    eq["V_B_voile"] = V_sail
    eq["V_B_spy"] = V_spy
    return eq

def compute_polar_true_wind(case: SimulationCase, polar_tables: dict[float, pd.DataFrame], derive_tables: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    h_aero = compute_aero_application_height(case)
    n_calc = (int(case.n_cap) // 2) + 1
    for cap in np.linspace(0.0, 180.0, n_calc):
        eq = solve_equilibrium_hybrid_true_wind(float(cap), polar_tables, case)
        aero = eq.get("aero", {})
        heel_pack = apply_heel_correction_to_equilibrium(eq, aero, derive_tables, case, h_aero)
        derive = heel_pack.get("derive", {})
        heel = heel_pack.get("heel", {})
        V_B = float(heel_pack.get("V_B_gite", eq.get("V_B", 0.0)))
        cap_rad = np.radians(float(cap))
        vmg_signed = V_B * np.cos(cap_rad)
        psi_signed = float(derive.get("psi_deg", 0.0))
        v_lateral_signed = float(np.sign(psi_signed) * float(derive.get("V_lateral", 0.0)))
        rows.append(
            {
                "CAP": float(cap),
                "TWA": float(cap),
                "V_B": V_B,
                "V_B_raw": float(heel_pack.get("V_B_raw", eq.get("V_B", 0.0))),
                "V_B_voile": float(eq.get("V_B_voile", 0.0)),
                "V_B_spy": float(eq.get("V_B_spy", 0.0)),
                "mode": eq.get("mode", "voile"),
                "F_prop": float(aero.get("F_prop", eq.get("F_prop_spy", 0.0))),
                "F_aero_lat": float(aero.get("F_aero_lat", 0.0)),
                "F_aero_lat_abs": float(aero.get("F_aero_lat_abs", 0.0)),
                "Cl": float(aero.get("Cl", 0.0)),
                "Cd": float(aero.get("Cd", 0.0)),
                "Re": float(aero.get("Re", 0.0)),
                "psi_deg": psi_signed,
                "theta_deg": float(heel.get("theta_deg", 0.0)),
                "V_lateral": v_lateral_signed,
                "V_derive": float(derive.get("V_derive", 0.0)),
                "F_hydro_derive": float(derive.get("F_hydro_derive", 0.0)),
                "M_aero": float(heel.get("M_aero", 0.0)),
                "M_derive": float(heel.get("M_derive", 0.0)),
                "M_quille": float(heel.get("M_quille", 0.0)),
                "h_derive": float(heel_pack.get("h_derive", 0.0)),
                "c_derive_eff": float(heel_pack.get("c_derive_eff", 0.0)),
                "L_derive": float(derive.get("L_derive", 0.0)),
                "D_derive": float(derive.get("D_derive", 0.0)),
                "Cl_derive": float(derive.get("Cl_derive", 0.0)),
                "Cd_derive": float(derive.get("Cd_derive", 0.0)),
                "Re_derive": float(derive.get("Re_derive", 0.0)),
                "derive_residual": float(derive.get("derive_residual", 0.0)),
                "heel_residual": float(heel.get("heel_residual", 0.0)),
                "derive_iterations": int(derive.get("iterations", 0)),
                "heel_iterations": int(heel.get("iterations", 0)),
                "derive_converged": bool(derive.get("converged", True)),
                "derive_reason": derive.get("reason", ""),
                "heel_converged": bool(heel.get("converged", True)),
                "converged": bool(eq.get("converged", True)),
                "VMG": float(abs(vmg_signed)),
                "VMG_signed": float(vmg_signed),
                "VMG_pres": float(max(vmg_signed, 0.0)),
                "VMG_portant": float(max(-vmg_signed, 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame()
    df_first = pd.DataFrame(rows)
    sym_rows = []
    for _, row in df_first.iterrows():
        cap_orig = float(row["CAP"])
        if 0.0 < cap_orig < 180.0:
            row_sym = row.copy()
            row_sym["CAP"] = 360.0 - cap_orig
            row_sym["TWA"] = row_sym["CAP"]
            cap_sym_rad = np.radians(float(row_sym["CAP"]))
            vmg_signed_sym = float(row["V_B"]) * np.cos(cap_sym_rad)
            row_sym["VMG_signed"] = float(vmg_signed_sym)
            row_sym["VMG"] = float(abs(vmg_signed_sym))
            row_sym["VMG_pres"] = float(max(vmg_signed_sym, 0.0))
            row_sym["VMG_portant"] = float(max(-vmg_signed_sym, 0.0))
            row_sym["F_aero_lat"] = -float(row["F_aero_lat"])
            row_sym["V_lateral"] = -float(row["V_lateral"])
            row_sym["psi_deg"] = -float(row["psi_deg"])
            sym_rows.append(row_sym)
    if sym_rows:
        return pd.concat([df_first, pd.DataFrame(sym_rows)], ignore_index=True).sort_values("CAP").reset_index(drop=True)
    return df_first.sort_values("CAP").reset_index(drop=True)
