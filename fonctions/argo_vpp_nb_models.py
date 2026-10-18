from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import numpy as np
import pandas as pd
import trimesh

__all__ = ['AeroState', 'BoatGeometryState', 'C', 'CLR_DRAFT_FACTOR', 'EquilibriumState', 'FoilGeometry', 'GeometryInputs', 'HEEL_RANGE_MAX_DEG', 'HEEL_RANGE_MIN_DEG', 'HEEL_RANGE_STEPS', 'HullHydrostatics', 'MassProperties', 'PhysicalConstants', 'ResistanceBreakdown', 'SolverSettings', 'TWA_RANGE_MAX_DEG', 'TWA_RANGE_MIN_DEG', 'TWA_RANGE_STEPS', 'deg_to_rad', 'rad_to_deg', 'safe_div']

TWA_RANGE_MIN_DEG = 30.0

TWA_RANGE_MAX_DEG = 150.0

TWA_RANGE_STEPS = 5

HEEL_RANGE_MIN_DEG = 0.0

HEEL_RANGE_MAX_DEG = 60.0

HEEL_RANGE_STEPS = 10

@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants used by the VPP in SI units."""

    RHO_AIR: float = 1.225
    MU_AIR: float = 1.81e-5
    RHO_WATER: float = 1025.0
    NU_WATER: float = 1.19e-6
    G: float = 9.80665
    KNOT_TO_MPS: float = 0.514444
    RHO_CARBON: float = 1500.0

C = PhysicalConstants()

CLR_DRAFT_FACTOR = 0.43

@dataclass
class SolverSettings:
    """Numerical settings for geometry analysis and VPP solve."""

    force_tol_N: float = 0.5
    moment_tol_Nm: float = 0.5
    max_nfev: int = 180
    section_count: int = 80
    waterline_tol: float = 1e-8
    use_rmv: bool = True
    wetted_surface_mode: str = "Delft"
    out_of_domain_policy: str = "reject"
    solver_method: str = "least_squares_multistart"
    sail_coefficient_data_source: str = "naca"
    aero_model: str = "naca_wing"
    geometry_input_mode: str = "stl"
    hydro_model: str = "oceanvpp_delft"
    mass_mode: str = "design_ballast"
    use_wind_gradient: bool = False
    wind_reference_height_m: float = 10.0
    wind_roughness_z0_m: float = 0.005
    orc_LSM1_m: float | None = None
    pedro_hardcoded_hydroheel: bool = False
    setup_warning_message: str = ""

@dataclass
class GeometryInputs:
    """User controlled geometry and mass parameters."""

    stl_path: str = "Coques/Cas_Froude_0.45_x_0.9_rank01.stl"
    stl_unit: str = "auto"
    longitudinal_orientation: str = "x_min = poupe, x_max = proue"
    hull_draft_m: float = 0.08
    wing_contour_text: str = "0.2,0.0\n0.463,0.313\n0.463,0.713\n0.09,2.4\n-0.2,2.4\n-0.375,0.418\n-0.2,0.0\n0.0,0.0"
    centerboard_contour_text: str = "0.00,0.00\n0.20,0.00\n0.18,0.90\n0.04,0.95\n0.00,0.00"
    rudder_contour_text: str = "0.00,0.00\n0.10,0.00\n0.09,0.55\n0.02,0.60\n0.00,0.00"
    wing_naca: str = "NACA0012"
    centerboard_naca: str = "NACA0010"
    rudder_naca: str = "NACA0009"
    hull_skin_thickness_mm: float = 1.2
    centerboard_skin_thickness_mm: float = 1.0
    rudder_skin_thickness_mm: float = 0.9
    wing_skin_thickness_mm: float = 0.8
    equipment_mass_kg: float = 0.0
    equipment_x_m: float = 0.0
    equipment_y_m: float = 0.0
    equipment_z_m: float = 0.0
    ballast_mode: str = "automatique"
    manual_ballast_kg: float = 0.0
    fixed_alpha_deg: float = 6.0
    wing_alpha_mode: str = "manuel"
    alpha_min_deg: float = 0.0
    alpha_max_deg: float = 15.0
    alpha_step_deg: float = 1.0
    oswald_efficiency: float = 0.85
    delft_table_folder: str = "Keuning_coefficients_tables/VPP_coefficients_tables"
    polar_folder: str = "Profil NACA"

@dataclass
class FoilGeometry:
    """Geometry and mesh of a wing-like element."""

    name: str
    contour: np.ndarray
    naca: str
    planform_area_m2: float
    external_area_m2: float
    wetted_area_m2: float
    span_m: float
    mean_chord_m: float
    volume_m3: float
    volume_centroid: np.ndarray
    surface_centroid: np.ndarray
    aspect_ratio: float
    thickness_ratio: float
    root_position: np.ndarray
    mesh: trimesh.Trimesh
    chord_sections: pd.DataFrame

@dataclass
class HullHydrostatics:
    """Hydrostatic quantities extracted from the hull STL at upright waterline."""

    LWL: float
    BWL: float
    TC: float
    volume_C: float
    S_C: float
    A_WP: float
    LCB: float
    LCF: float
    AMS_CB: float
    C_M: float
    C_P: float
    x_FP: float
    LCB_fpp: float
    LCF_fpp: float
    LCB_pct_midship: float
    LCF_pct_midship: float
    waterline_z: float
    B_CB: float
    delft_valid: bool
    delft_warnings: list[str] = field(default_factory=list)

@dataclass
class MassProperties:
    """Mass and centre of gravity summary."""

    component_table: pd.DataFrame
    total_mass_kg: float
    center_of_gravity: np.ndarray
    ballast_mass_kg: float
    ballast_position: np.ndarray
    displacement_mass_kg: float
    mass_error_kg: float
    mass_error_rel: float

def deg_to_rad(value: float) -> float:
    return float(np.deg2rad(value))

def rad_to_deg(value: float) -> float:
    return float(np.rad2deg(value))

def safe_div(num: float, den: float, default: float = np.nan) -> float:
    """Return num/den and protect against zero or non-finite denominators."""

    if not np.isfinite(den) or abs(den) < 1e-12:
        return default
    return float(num / den)

@dataclass
class BoatGeometryState:
    """Fully analyzed geometry ready for VPP evaluation."""

    inputs: GeometryInputs
    settings: SolverSettings
    hull_mesh: trimesh.Trimesh
    immersed_hull_mesh: trimesh.Trimesh
    hydro: HullHydrostatics
    wing: FoilGeometry
    centerboard: FoilGeometry
    rudder: FoilGeometry
    masses: MassProperties
    total_displaced_volume_m3: float
    T_max_m: float
    z_min_submerged_m: float
    total_draft_m: float
    lateral_area_m2: float
    gz_curve: pd.DataFrame
    delft_tables: dict[str, pd.DataFrame]
    hydrostatic_boolean_union_used: bool = False
    mesh_overlap_detected: bool = False
    estimated_overlap_volume_m3: float = 0.0
    hydrostatic_geometry_valid: bool = True
    wetted_surface_cache: dict[float, tuple[float, float]] = field(default_factory=dict)

@dataclass
class AeroState:
    """Aerodynamic result of the rigid wing."""

    AWS_mps: float
    AWA_deg: float
    CL: float
    CD: float
    CM: float
    Re: float
    wing_orientation_deg: float
    F_drive_N: float
    F_side_N: float
    F_side_signed_N: float
    M_aero_Nm: float
    center_of_effort: np.ndarray
    profile_naca: str
    polar_file: str
    alpha_wing_deg: float
    alpha_domain_min_deg: float
    alpha_domain_max_deg: float
    alpha_min_valid_deg: float
    alpha_max_valid_deg: float
    Re_domain_min: float
    Re_domain_max: float
    Re_lower: float
    Re_upper: float
    Re_low_used: float
    Re_high_used: float
    Re_reference: float
    CD_profile: float
    CD_induced: float
    naca_valid: bool
    naca_warning: str
    naca_cache_file_reads: int
    naca_cache_hits: int
    z_CE_m: float
    z_WL_m: float
    z_CLR_m: float
    M_aero_about_WL_Nm: float
    aero_model: str = "naca_wing"
    sailset: str = ""
    flat: float = np.nan
    reef: float = np.nan
    CL_main: float = np.nan
    CD_main: float = np.nan
    CL_jib: float = np.nan
    CD_jib: float = np.nan
    CL_spin: float = np.nan
    CD_spin: float = np.nan
    CL_lift_weighted: float = np.nan
    CDp_weighted: float = np.nan
    CDi: float = np.nan
    CD0: float = np.nan
    CD_total: float = np.nan
    A_ref: float = np.nan
    A_main: float = np.nan
    A_jib: float = np.nan
    A_fore: float = np.nan
    A_spi: float = np.nan
    zCoE_sailplan_m: float = np.nan
    coefficient_source: str = ""
    interpolation_method: str = ""
    pedro_hydroheel_Nm: float = np.nan

@dataclass
class ResistanceBreakdown:
    """Hydrodynamic resistance components with per-table domain diagnostics."""

    R_hull_viscous_N: float
    R_centerboard_viscous_N: float
    R_rudder_viscous_N: float
    R_hull_residuary_raw_N: float
    R_hull_residuary_N: float
    R_centerboard_residuary_N: float
    R_heel_friction_delta_N: float
    R_heel_hull_residuary_delta_N: float
    R_heel_keel_residuary_delta_N: float
    R_induced_N: float
    R_rail_under_N: float
    R_total_N: float
    T_eff_m: float
    q_water_Pa: float
    hull_residuary_valid: bool
    appendage_residuary_valid: bool
    heel_friction_valid: bool
    heel_hull_residuary_valid: bool
    heel_appendage_residuary_valid: bool
    effective_span_valid: bool
    used_low_fn_physical_extension: bool
    low_fn_extension_tables: str
    delft_valid: bool
    delft_geometry_valid: bool
    table_domain_valid: bool
    invalid_reason: str
    hydro_model: str = "oceanvpp_delft"
    induced_drag_model: str = "orc_simplified_effective_span"
    appendage_lift_drag_model: str = "CL_assumed_zero"
    R_added_waves_N: float = 0.0
    warning_message: str = ""
    delft_warnings: list[str] = field(default_factory=list)

@dataclass
class EquilibriumState:
    """Full state at one VPP evaluation."""

    aero: AeroState
    resistance: ResistanceBreakdown
    M_hydroheel_Nm: float
    M_heel_total_Nm: float
    M_hydrostatic_Nm: float
    RMV_Nm: float
    M_right_total_Nm: float
    force_residual_N: float
    moment_residual_Nm: float
    Fn: float
    z_CE_m: float
    z_WL_m: float
    z_CLR_m: float
    M_aero_about_WL_Nm: float
    table_domain_valid: bool
    naca_valid: bool
    hydrostatic_geometry_valid: bool
    failure_reason_code: str
    invalid_reason: str
