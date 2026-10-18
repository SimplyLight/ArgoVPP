from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from .argo_vpp_nb_models import (
    C,
    FoilGeometry,
    GeometryInputs,
    MassProperties,
    PhysicalConstants,
    safe_div,
)

__all__ = ['component_mass', 'compute_mass_properties', 'thickness_m']

def thickness_m(mm: float) -> float:
    """Convert millimetres to metres."""

    return float(mm) * 1e-3

def component_mass(name: str, area_m2: float, thickness_mm: float, centroid: np.ndarray, constants: PhysicalConstants = C) -> dict[str, Any]:
    """Mass of a thin carbon skin component."""

    mass = constants.RHO_CARBON * thickness_m(thickness_mm) * float(area_m2)
    return {
        "component": name,
        "mass_kg": mass,
        "x_G_m": float(centroid[0]),
        "y_G_m": float(centroid[1]),
        "z_G_m": float(centroid[2]),
        "area_m2": float(area_m2),
    }

def compute_mass_properties(
    inputs: GeometryInputs,
    hull_wet_or_external_area_m2: float,
    hull_surface_centroid: np.ndarray,
    wing: FoilGeometry,
    centerboard: FoilGeometry,
    rudder: FoilGeometry,
    volume_total_m3: float,
    ballast_position: np.ndarray,
    constants: PhysicalConstants = C,
) -> MassProperties:
    """Compute component masses, automatic/manual ballast and global centre of gravity."""

    rows = [
        component_mass("coque", hull_wet_or_external_area_m2, inputs.hull_skin_thickness_mm, hull_surface_centroid, constants),
        component_mass("derive", centerboard.external_area_m2, inputs.centerboard_skin_thickness_mm, centerboard.surface_centroid, constants),
        component_mass("safran", rudder.external_area_m2, inputs.rudder_skin_thickness_mm, rudder.surface_centroid, constants),
        component_mass("aile", wing.external_area_m2, inputs.wing_skin_thickness_mm, wing.surface_centroid, constants),
    ]
    if inputs.equipment_mass_kg > 0:
        rows.append({
            "component": "equipements",
            "mass_kg": float(inputs.equipment_mass_kg),
            "x_G_m": float(inputs.equipment_x_m),
            "y_G_m": float(inputs.equipment_y_m),
            "z_G_m": float(inputs.equipment_z_m),
            "area_m2": np.nan,
        })
    displacement_mass = constants.RHO_WATER * float(volume_total_m3)
    structural_mass = float(sum(row["mass_kg"] for row in rows))
    if inputs.ballast_mode == "automatique":
        ballast_mass = displacement_mass - structural_mass
        if ballast_mass < -1e-9:
            raise ValueError(
                "Ballast automatique negatif: masses structurelles incompatibles avec le tirant d'eau impose. "
                f"m_structure={structural_mass:.3f} kg, m_deplacement={displacement_mass:.3f} kg."
            )
        ballast_mass = max(0.0, ballast_mass)
    elif inputs.ballast_mode == "manuel":
        ballast_mass = float(inputs.manual_ballast_kg)
    else:
        raise ValueError("Mode ballast invalide.")
    rows.append({
        "component": "ballast ponctuel de quille",
        "mass_kg": ballast_mass,
        "x_G_m": float(ballast_position[0]),
        "y_G_m": float(ballast_position[1]),
        "z_G_m": float(ballast_position[2]),
        "area_m2": np.nan,
    })
    table = pd.DataFrame(rows)
    total_mass = float(table["mass_kg"].sum())
    if total_mass <= 0:
        raise ValueError("Masse totale nulle.")
    positions = table[["x_G_m", "y_G_m", "z_G_m"]].to_numpy(dtype=float)
    center = np.average(positions, axis=0, weights=table["mass_kg"].to_numpy(dtype=float))
    mass_error = total_mass - displacement_mass
    rel = safe_div(mass_error, displacement_mass, default=np.inf)
    if inputs.ballast_mode == "manuel" and abs(rel) > 0.01:
        raise ValueError(
            "Mode ballast manuel incoherent: ecart relatif masse/deplacement > 1 %. "
            f"m_total={total_mass:.3f} kg, m_requise={displacement_mass:.3f} kg, ecart={mass_error:.3f} kg."
        )
    return MassProperties(
        component_table=table,
        total_mass_kg=total_mass,
        center_of_gravity=center,
        ballast_mass_kg=ballast_mass,
        ballast_position=np.asarray(ballast_position, dtype=float),
        displacement_mass_kg=displacement_mass,
        mass_error_kg=mass_error,
        mass_error_rel=rel,
    )
