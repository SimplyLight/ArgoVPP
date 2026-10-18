from __future__ import annotations
from .argo_vpp_common import math
from .argo_vpp_geometry import close_polygon, polygon_area

def finite_section_correction(cl_inf: float, cd_inf: float, aspect_ratio: float, e_oswald: float) -> dict[str, float]:
    ar = max(float(aspect_ratio), 1e-9)
    e = max(float(e_oswald), 1e-9)
    cl = float(cl_inf) / (1.0 + 2.0 / ar)
    cd_induced = cl**2 / (math.pi * e * ar)
    cd = float(cd_inf) + cd_induced
    finesse_angle = math.degrees(math.atan2(cd, cl)) if abs(cl) > 1e-12 else 90.0
    return {"Cl_finite": cl, "Cd_induced": cd_induced, "Cd_finite": cd, "finesse_angle": finesse_angle}

def compute_hydro_appendage_from_coeffs(
    name: str,
    points: list[tuple[float, float]],
    rho_water: float,
    velocity: float,
    cl_inf: float,
    cd_inf: float,
    cm_inf: float,
    e_oswald: float,
) -> dict[str, float | str]:
    pts = close_polygon(points)
    area = polygon_area(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(ys) - min(ys)
    chord = area / span if span > 1e-9 else 0.0
    ar = span**2 / area if area > 1e-12 else 0.0
    corr = finite_section_correction(cl_inf, cd_inf, ar, e_oswald)
    q_dyn = 0.5 * rho_water * velocity**2
    lift = corr["Cl_finite"] * q_dyn * area
    drag = corr["Cd_finite"] * q_dyn * area
    moment = cm_inf * q_dyn * area * max(chord, 1e-9)
    return {
        "name": name,
        "area": area,
        "span": span,
        "chord": chord,
        "AR": ar,
        **corr,
        "Lift": lift,
        "Drag": drag,
        "Moment": moment,
    }

def calc_hydro_drag(V_B: float, S_wetted: float, CR_friction: float, Rcoque_coeff: float, rho_water: float) -> float:
    q_dyn_water = 0.5 * float(rho_water) * float(V_B) ** 2
    C_R_total = float(CR_friction) + float(Rcoque_coeff)
    return float(C_R_total * q_dyn_water * float(S_wetted))
