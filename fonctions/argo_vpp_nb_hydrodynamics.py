from __future__ import annotations

import numpy as np
import pandas as pd
from .argo_vpp_nb_appendages import (
    chord_at_span,
)
from .argo_vpp_nb_mesh import (
    immersed_mesh_for_waterline,
    surface_area_centroid,
)
from .argo_vpp_nb_models import (
    BoatGeometryState,
    C,
    FoilGeometry,
    HullHydrostatics,
    PhysicalConstants,
    deg_to_rad,
    rad_to_deg,
    safe_div,
)
from .argo_vpp_nb_stability import (
    find_waterline_for_target_volume,
    rotate_mesh_about_x,
)
from .argo_vpp_nb_tables import (
    bounded_table_row,
    fn_domain,
    table_domain,
)

__all__ = ['_bare_hull_formula', '_centerboard_formula', '_component_invalid', '_scale_low_fn', 'bare_hull_residuary', 'centerboard_residuary', 'direct_hull_wetted_surface_delta', 'froude_number', 'ittc_friction_resistance', 'orc_appendage_cf0', 'orc_appendage_viscous', 'rail_under_drag', 'wetted_surface_heel_delta']

def froude_number(V_B: float, LWL: float, constants: PhysicalConstants = C) -> float:
    """Boat Froude number recalculated for every speed."""

    return safe_div(V_B, np.sqrt(constants.G * max(LWL, 1e-9)), default=0.0)

def ittc_friction_resistance(V_B: float, length_m: float, wetted_area_m2: float, constants: PhysicalConstants = C) -> float:
    """Delft/OceanVPP hull viscous resistance."""

    if V_B <= 1e-9 or wetted_area_m2 <= 0:
        return 0.0
    Re = 0.7 * V_B * length_m / constants.NU_WATER
    if Re <= 1.0:
        return 0.0
    Cf = 0.075 / (np.log10(Re) - 2.0) ** 2
    return float(0.5 * constants.RHO_WATER * V_B**2 * wetted_area_m2 * Cf)

def orc_appendage_cf0(reynolds: float) -> float:
    """ORC 2026 appendage base friction coefficient."""

    if reynolds <= 1.0:
        return 0.0
    log_re = np.log10(max(float(reynolds), 10.0))
    cf0 = (
        0.0000853 * log_re**4
        - 0.0025252 * log_re**3
        + 0.0278513 * log_re**2
        - 0.1363492 * log_re
        + 0.2539752
    )
    return max(float(cf0), 0.0)

def orc_appendage_viscous(V_B: float, foil: FoilGeometry, constants: PhysicalConstants = C, bands: int = 5, CL_appendage: float = 0.0) -> float:
    """Appendage strip drag with ORC-style friction and optional lift drag.

    The 2-DOF model has no local appendage lift distribution yet, so callers
    use CL_appendage=0 by default. The lift-dependent terms remain available
    for a documented future side-force allocation.
    """

    if V_B <= 1e-9:
        return 0.0
    df = foil.chord_sections.dropna()
    if df.empty:
        return 0.0
    s_min, s_max = float(df["span"].min()), float(df["span"].max())
    edges = np.linspace(s_min, s_max, bands + 1)
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        le, te, chord = chord_at_span(foil.contour, mid)
        if chord <= 0:
            continue
        span_width = b - a
        band_area = 2.0 * chord * span_width
        tc = min(float(foil.thickness_ratio), 0.18)
        Re = max(V_B * chord / constants.NU_WATER, 10.0)
        cf0 = orc_appendage_cf0(Re)
        kf = 1.75 * tc + 30.0 * tc**4
        cf_form_corrected = cf0 * (1.0 + kf)
        cd_total_section = cf_form_corrected + 0.0016 * float(CL_appendage) + 0.0032 * float(CL_appendage) ** 2
        total += cd_total_section * band_area * 0.5 * constants.RHO_WATER * V_B**2
    return float(max(total, 0.0))

def direct_hull_wetted_surface_delta(V_B: float, heel_rad: float, state: BoatGeometryState) -> tuple[float, float]:
    """Compute heeled hull wetted-surface delta by directly recutting the STL."""

    heel_key = round(abs(rad_to_deg(heel_rad)), 4)
    cached = state.wetted_surface_cache.get(heel_key)
    if cached is None:
        rotated = rotate_mesh_about_x(state.hull_mesh, state.masses.center_of_gravity, deg_to_rad(heel_key))
        z_min = float(rotated.bounds[0, 2]) - 0.05 * max(rotated.extents[2], 1e-6)
        z_max = float(rotated.bounds[1, 2]) + 0.05 * max(rotated.extents[2], 1e-6)
        z_wl, _immersed = find_waterline_for_target_volume(rotated, state.hydro.volume_C, (z_min, z_max), tol=state.settings.waterline_tol)
        _, _, wet_triangles = immersed_mesh_for_waterline(rotated, z_wl, cap=True)
        S_WC, _ = surface_area_centroid(wet_triangles)
        cached = (float(S_WC), float(z_wl))
        state.wetted_surface_cache[heel_key] = cached
    S_WC, _ = cached
    R_delta = ittc_friction_resistance(V_B, state.hydro.LWL, S_WC) - ittc_friction_resistance(V_B, state.hydro.LWL, state.hydro.S_C)
    R_delta = float(R_delta) if np.isfinite(R_delta) else 0.0
    return float(S_WC), R_delta

def rail_under_drag(heel_rad: float, R_r: float, R_rk: float) -> float:
    """Rail-under drag for heel above 30 degrees."""

    phi = abs(rad_to_deg(heel_rad))
    if phi <= 30.0:
        return 0.0
    return float(0.0004 * (R_r + R_rk) * (phi - 30.0) ** 2)

def _component_invalid(reason: str) -> tuple[float, bool, bool, str]:
    return 0.0, False, False, reason

def _scale_low_fn(value_at_min: float, Fn: float, Fn_min: float) -> float:
    return float(value_at_min * max(Fn, 0.0) / max(Fn_min, 1e-12))

def _bare_hull_formula(coeff: dict[str, float], hydro: HullHydrostatics, constants: PhysicalConstants = C) -> float:
    # DSYHS IV-1 uses LCB and LCF distances from FPP in metres; formulas below use fractions of LWL.
    lcf_ratio = safe_div(hydro.LCB_fpp, hydro.LCF_fpp, default=0.0)
    lcb_from_fpp_fraction = safe_div(hydro.LCB_fpp, hydro.LWL, default=0.0)
    term1 = (
        coeff["a1"] * lcb_from_fpp_fraction
        + coeff["a2"] * hydro.C_P
        + coeff["a3"] * safe_div(hydro.volume_C ** (2 / 3), hydro.A_WP, default=0.0)
        + coeff["a4"] * safe_div(hydro.BWL, hydro.LWL, default=0.0)
    ) * safe_div(hydro.volume_C ** (1 / 3), hydro.LWL, default=0.0)
    term2 = (
        coeff["a5"] * lcf_ratio
        + coeff["a6"] * safe_div(hydro.BWL, hydro.TC, default=0.0)
        + coeff["a7"] * hydro.C_M
    ) * safe_div(hydro.volume_C ** (1 / 3), hydro.LWL, default=0.0)
    P_r = coeff["a0"] + term1 + term2
    # Table IV-1 coefficients are already dimensionless values of Rr/(rho*g*volume).
    return float(constants.RHO_WATER * constants.G * hydro.volume_C * P_r)

def bare_hull_residuary(V_B: float, hydro: HullHydrostatics, tables: dict[str, pd.DataFrame], constants: PhysicalConstants = C) -> tuple[float, float, bool, bool, str]:
    """Delft bare-hull residuary resistance with controlled low-Fn physical extension."""

    Fn = froude_number(V_B, hydro.LWL, constants)
    lo, hi = fn_domain(tables, "bare_hull")
    if Fn <= 1e-12:
        return 0.0, 0.0, True, False, ""
    if Fn > hi + 1e-12:
        return 0.0, 0.0, False, False, f"delft_fn_above_max: bare_hull Fn={Fn:.5g} > {hi:.5g}"
    if Fn < lo - 1e-12:
        coeff = bounded_table_row(tables, "bare_hull", "Fn", lo)
        raw_min = _bare_hull_formula(coeff, hydro, constants)
        raw = _scale_low_fn(raw_min, Fn, lo)
        return raw, float(max(raw, 0.0)), True, True, "bare_hull"
    coeff = bounded_table_row(tables, "bare_hull", "Fn", Fn)
    raw = _bare_hull_formula(coeff, hydro, constants)
    return raw, float(max(raw, 0.0)), True, False, ""

def _centerboard_formula(coeff: dict[str, float], hydro: HullHydrostatics, centerboard: FoilGeometry, total_draft_m: float, constants: PhysicalConstants = C) -> float:
    volume_k = max(centerboard.volume_m3, 1e-12)
    z_bottom_hull = hydro.waterline_z - hydro.TC
    z_cb = float(centerboard.volume_centroid[2])
    Z_CBk = max(0.0, z_bottom_hull - z_cb)
    poly = (
        coeff["A0"]
        + coeff["A1"] * safe_div(total_draft_m, hydro.BWL, default=0.0)
        + coeff["A2"] * safe_div(hydro.TC + Z_CBk, volume_k ** (1 / 3), default=0.0)
        + coeff["A3"] * safe_div(hydro.volume_C, volume_k, default=0.0)
    )
    return float(max(volume_k * constants.RHO_WATER * constants.G * poly, 0.0))

def centerboard_residuary(V_B: float, hydro: HullHydrostatics, centerboard: FoilGeometry, total_draft_m: float, tables: dict[str, pd.DataFrame], constants: PhysicalConstants = C) -> tuple[float, bool, bool, str]:
    """Delft appendage residuary resistance with controlled low-Fn physical extension."""

    Fn = froude_number(V_B, hydro.LWL, constants)
    lo, hi = fn_domain(tables, "keel")
    if Fn <= 1e-12:
        return 0.0, True, False, ""
    if Fn > hi + 1e-12:
        return 0.0, False, False, f"delft_fn_above_max: keel Fn={Fn:.5g} > {hi:.5g}"
    if Fn < lo - 1e-12:
        value_min = _centerboard_formula(bounded_table_row(tables, "keel", "Fn", lo), hydro, centerboard, total_draft_m, constants)
        return _scale_low_fn(value_min, Fn, lo), True, True, "keel"
    return _centerboard_formula(bounded_table_row(tables, "keel", "Fn", Fn), hydro, centerboard, total_draft_m, constants), True, False, ""

def wetted_surface_heel_delta(V_B: float, heel_rad: float, hydro: HullHydrostatics, tables: dict[str, pd.DataFrame]) -> tuple[float, float, bool, bool, str]:
    """Delft wetted-surface correction; phi=0 is explicitly zero and 0..first angle is linear."""

    phi_deg = abs(rad_to_deg(heel_rad))
    lo, hi = table_domain(tables, "wetted_surface_heel", "heel_deg")
    if phi_deg <= 1e-12:
        return float(hydro.S_C), 0.0, True, False, ""
    if phi_deg > hi + 1e-12:
        return float(hydro.S_C), 0.0, False, False, f"heel_above_table_range: wetted_surface_heel phi={phi_deg:.5g} > {hi:.5g}"
    coeff_phi = lo if phi_deg < lo - 1e-12 else phi_deg
    coeff = bounded_table_row(tables, "wetted_surface_heel", "heel_deg", coeff_phi)
    bt = safe_div(hydro.BWL, hydro.TC, default=0.0)
    S_at = hydro.S_C * (1.0 + 0.01 * (coeff["s0"] + coeff["s1"] * bt + coeff["s2"] * bt**2 + coeff["s3"] * hydro.C_M))
    S_at = max(S_at, 0.0)
    R_at = ittc_friction_resistance(V_B, hydro.LWL, S_at) - ittc_friction_resistance(V_B, hydro.LWL, hydro.S_C)
    if phi_deg < lo - 1e-12:
        f = phi_deg / max(lo, 1e-12)
        return float(hydro.S_C + (S_at - hydro.S_C) * f), float(R_at * f), True, True, "wetted_surface_heel"
    return float(S_at), float(R_at), True, False, ""
