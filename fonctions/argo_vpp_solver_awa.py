from __future__ import annotations
from .argo_vpp_common import Any, MU_AIR_DEFAULT, RHO_AIR_DEFAULT, np, pd
from .argo_vpp_aerodynamics import interp_coeffs_for_re_alpha
from .argo_vpp_geometry import reynolds_number
from .argo_vpp_hydrodynamics import calc_hydro_drag
from .argo_vpp_models import SimulationCase

def calc_apparent_wind_awa(V_TW: float, V_B: float, TWA_deg: float) -> dict[str, float]:
    V_TW = float(V_TW)
    V_B = float(V_B)
    TWA_deg = float(TWA_deg)
    V_TW_x = 0.0
    V_TW_y = V_TW
    angle_rot_rad = np.radians(90.0 - TWA_deg)
    V_B_x = V_B * np.cos(angle_rot_rad)
    V_B_y = V_B * np.sin(angle_rot_rad)
    V_AW_x = V_TW_x + V_B_x
    V_AW_y = V_TW_y + V_B_y
    V_AW = np.sqrt(V_AW_x**2 + V_AW_y**2)
    beta_AW_rad = 0.0 if abs(V_AW_y) < 1e-9 else np.arctan(V_AW_x / V_AW_y)
    return {"V_AW": float(V_AW), "beta_AW": float(np.degrees(beta_AW_rad))}

def calc_aero_simple_awa(
    V_AW: float,
    beta_AW_deg: float,
    alpha_trim_deg: float,
    polar_tables: dict[float, pd.DataFrame],
    A_sail: float,
    c_moy: float,
    TWA_deg: float,
    rho_air: float = RHO_AIR_DEFAULT,
    mu_air: float = MU_AIR_DEFAULT,
) -> dict[str, float]:
    alpha_eff_deg = float(alpha_trim_deg)
    re_value = reynolds_number(rho_air, V_AW, c_moy, mu_air)
    try:
        coeffs = interp_coeffs_for_re_alpha(polar_tables, re_value, alpha_eff_deg)
        cl, cd = float(coeffs["Cl"]), float(coeffs["Cd"])
    except Exception:
        cl, cd = 0.0, 0.0
    q_dyn = 0.5 * rho_air * float(V_AW) ** 2
    lift = cl * q_dyn * A_sail
    drag = cd * q_dyn * A_sail
    angle1 = alpha_trim_deg + beta_AW_deg + 90.0 - TWA_deg
    angle2 = TWA_deg - alpha_trim_deg - beta_AW_deg
    f_prop = lift * np.cos(np.radians(angle1)) - drag * np.cos(np.radians(angle2))
    return {"F_prop": float(f_prop), "Cl": cl, "Cd": cd, "Re": re_value, "L_aero": lift, "D_aero": drag}

def solve_equilibrium_awa(
    V_TW: float,
    TWA_deg: float,
    polar_tables: dict[float, pd.DataFrame],
    alpha_trim_deg: float,
    A_sail: float,
    c_moy: float,
    rho_air: float,
    rho_water: float,
    S_wetted: float,
    CR_friction: float,
    Rcoque_coeff: float,
    max_iterations: int = 50,
    tolerance: float = 0.5,
    V_B_max_factor: float = 2.0,
) -> dict[str, Any]:
    V_B_min = max(0.1, float(V_TW) * 0.05)
    V_B_max = float(V_TW) * float(V_B_max_factor)
    converged = False
    aero_info = {}
    aw_info = {}
    r_hydro = 0.0
    error = 0.0
    for it in range(int(max_iterations)):
        V_B_mid = 0.5 * (V_B_min + V_B_max)
        aw_info = calc_apparent_wind_awa(V_TW, V_B_mid, TWA_deg)
        aero_info = calc_aero_simple_awa(
            aw_info["V_AW"], aw_info["beta_AW"], alpha_trim_deg, polar_tables, A_sail, c_moy, TWA_deg, rho_air
        )
        r_hydro = calc_hydro_drag(V_B_mid, S_wetted, CR_friction, Rcoque_coeff, rho_water)
        error = aero_info["F_prop"] - r_hydro
        if abs(error) < tolerance:
            converged = True
            return {
                "V_B": float(V_B_mid),
                "converged": True,
                "iterations": it + 1,
                "aw": aw_info,
                "aero": aero_info,
                "R_hydro": r_hydro,
                "error": error,
                "mode": "voile",
            }
        if error > 0.0:
            V_B_min = V_B_mid
        else:
            V_B_max = V_B_mid
    V_B_solution = 0.5 * (V_B_min + V_B_max)
    return {
        "V_B": float(V_B_solution),
        "converged": converged,
        "iterations": int(max_iterations),
        "aw": aw_info,
        "aero": aero_info,
        "R_hydro": r_hydro,
        "error": error,
        "mode": "voile",
    }

def calc_spy_equilibrium_awa(
    V_TW: float,
    TWA_deg: float,
    rho_air: float,
    rho_water: float,
    S_wetted: float,
    CR_friction: float,
    Rcoque_coeff: float,
    Cx_spy: float = 1.0,
    S_spy: float = 1.0,
) -> dict[str, Any] | None:
    if TWA_deg < 90.0 or TWA_deg > 270.0:
        return None
    abs_sin_90_cap = abs(np.sin(np.radians(90.0 - TWA_deg)))
    if abs_sin_90_cap < 0.01:
        return None
    V_B_min = 0.1
    V_B_max = float(V_TW) * 2.0
    for it in range(50):
        V_B_mid = 0.5 * (V_B_min + V_B_max)
        F_spy = Cx_spy * 0.5 * rho_air * V_B_mid**2 * S_spy
        F_prop_spy = F_spy * abs_sin_90_cap
        R_hydro = calc_hydro_drag(V_B_mid, S_wetted, CR_friction, Rcoque_coeff, rho_water)
        error = F_prop_spy - R_hydro
        if abs(error) < 0.5:
            return {"V_B": float(V_B_mid), "mode": "spy", "converged": True, "iterations": it + 1}
        if error > 0.0:
            V_B_min = V_B_mid
        else:
            V_B_max = V_B_mid
    return {"V_B": float(0.5 * (V_B_min + V_B_max)), "mode": "spy", "converged": False, "iterations": 50}

def solve_equilibrium_hybrid_awa(
    V_TW: float,
    TWA_deg: float,
    polar_tables: dict[float, pd.DataFrame],
    case: SimulationCase,
) -> dict[str, Any]:
    eq_sail = solve_equilibrium_awa(
        V_TW,
        TWA_deg,
        polar_tables,
        case.alpha_trim,
        case.A_sail,
        case.c_moy,
        case.rho_air,
        case.rho_water,
        case.S_wetted,
        case.CR_friction,
        case.Rcoque_coeff,
        case.max_iterations,
        case.tolerance,
    )
    eq_spy = calc_spy_equilibrium_awa(
        V_TW, TWA_deg, case.rho_air, case.rho_water, case.S_wetted, case.CR_friction, case.Rcoque_coeff, case.Cx_spy, case.S_spy
    )
    V_B_sail = eq_sail["V_B"] if eq_sail else 0.0
    V_B_spy = eq_spy["V_B"] if eq_spy else 0.0
    eq_final = dict(eq_spy) if eq_spy and V_B_spy > V_B_sail else dict(eq_sail)
    eq_final["V_B_voile"] = V_B_sail
    eq_final["V_B_spy"] = V_B_spy
    if "aw" not in eq_final:
        eq_final["aw"] = calc_apparent_wind_awa(V_TW, eq_final["V_B"], TWA_deg)
    return eq_final

def compute_polar_awa(case: SimulationCase, polar_tables: dict[float, pd.DataFrame]) -> pd.DataFrame:
    n_calc = int(case.n_twa // 2) + 1
    rows: list[dict[str, Any]] = []
    for twa in np.linspace(0, 180, n_calc):
        eq = solve_equilibrium_hybrid_awa(case.V_TW, float(twa), polar_tables, case)
        aw = eq.get("aw", {})
        aero = eq.get("aero", {})
        V_B = float(eq.get("V_B", 0.0))
        rows.append(
            {
                "TWA": float(twa),
                "V_B": V_B,
                "V_B_voile": float(eq.get("V_B_voile", 0.0)),
                "V_B_spy": float(eq.get("V_B_spy", 0.0)),
                "mode": eq.get("mode", "voile"),
                "V_AW": float(aw.get("V_AW", 0.0)),
                "beta_AW": float(aw.get("beta_AW", 0.0)),
                "Cl": float(aero.get("Cl", 0.0)),
                "Cd": float(aero.get("Cd", 0.0)),
                "F_prop": float(aero.get("F_prop", 0.0)),
                "R_hydro": float(eq.get("R_hydro", 0.0)),
                "VMG": V_B * abs(np.cos(np.radians(twa))),
                "converged": bool(eq.get("converged", True)),
                "iterations": int(eq.get("iterations", 0)),
            }
        )
    df_first = pd.DataFrame(rows)
    sym_rows = []
    for _, row in df_first.iterrows():
        twa = float(row["TWA"])
        if 0.0 < twa < 180.0:
            row_sym = row.copy()
            row_sym["TWA"] = 360.0 - twa
            row_sym["VMG"] = float(row["V_B"]) * abs(np.cos(np.radians(row_sym["TWA"])))
            sym_rows.append(row_sym)
    return pd.concat([df_first, pd.DataFrame(sym_rows)], ignore_index=True).sort_values("TWA").reset_index(drop=True)
