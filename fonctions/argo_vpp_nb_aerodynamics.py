from __future__ import annotations

import numpy as np
from .argo_vpp_nb_models import (
    deg_to_rad,
)
from .argo_vpp_nb_tables import (
    finite_wing_from_profile,
)

__all__ = ['apparent_wind', 'apparent_wind_components', 'rigid_wing_coefficients']

def apparent_wind_components(TWS_mps: float, boat_speed_mps: float, TWA_deg: float, heel_rad: float = 0.0) -> dict[str, float]:
    """Return apparent wind vector with the transverse true-wind component reduced by cos(phi)."""

    twa = deg_to_rad(TWA_deg)
    V_Ax = TWS_mps * np.cos(twa) + boat_speed_mps
    V_Ay = TWS_mps * np.sin(twa) * np.cos(heel_rad)
    AWS = float(np.hypot(V_Ax, V_Ay))
    AWA = float(np.rad2deg(np.arctan2(V_Ay, V_Ax)))
    return {"V_Ax": float(V_Ax), "V_Ay": float(V_Ay), "AWS": AWS, "AWA_deg": AWA}

def apparent_wind(TWS_mps: float, boat_speed_mps: float, TWA_deg: float, heel_rad: float = 0.0) -> tuple[float, float]:
    """Return apparent wind speed and angle; at V_B=0 and phi=0, AWS=TWS and AWA=TWA."""

    aw = apparent_wind_components(TWS_mps, boat_speed_mps, TWA_deg, heel_rad)
    return aw["AWS"], aw["AWA_deg"]

def rigid_wing_coefficients(alpha_deg: float, aspect_ratio: float, oswald: float) -> tuple[float, float, float]:
    """Legacy analytic fallback retained only for explicit comparisons, not used by the VPP."""

    alpha_rad = deg_to_rad(alpha_deg)
    alpha_stall = deg_to_rad(15.0)
    alpha_eff = float(np.clip(alpha_rad, -alpha_stall, alpha_stall))
    cl_2d = 2.0 * np.pi * alpha_eff
    corrected = finite_wing_from_profile(cl_2d, 0.012, aspect_ratio, oswald)
    return float(corrected["CL"]), float(corrected["CD"]), 0.0
