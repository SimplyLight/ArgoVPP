from __future__ import annotations
from .argo_vpp_common import APP_DIR, Any, Path, np, pd, re
from .argo_vpp_geometry import reynolds_number
from .argo_vpp_models import SimulationCase

def available_profiles_from_folder(folder: Path = APP_DIR) -> list[str]:
    profiles: list[str] = []
    for file_path in sorted(folder.glob("NACA *.txt")):
        name = file_path.stem.replace(" ", "").upper()
        if name.startswith("NACA"):
            profiles.append(name)
    return profiles

def resolve_profile_file(profile_name: str, folder: Path = APP_DIR) -> Path:
    profile = str(profile_name).replace(" ", "").upper()
    if not profile.startswith("NACA"):
        profile = f"NACA{profile}"
    if len(profile) >= 8 and profile[4:].isdigit():
        candidate = folder / f"NACA {profile[4:]}.txt"
        if candidate.exists():
            return candidate
    for file_path in folder.glob("NACA *.txt"):
        if file_path.stem.replace(" ", "").upper() == profile:
            return file_path
    raise FileNotFoundError(f"Polar file not found for {profile}")

def load_polar_file(file_path: str | Path) -> dict[float, pd.DataFrame]:
    path = Path(file_path)
    re_data: dict[float, list[tuple[float, float, float, float]]] = {}
    current_re: float | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("Mach") and "Re" in line:
                match = re.search(r"Re\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
                if match:
                    current_re = float(match.group(1))
                    re_data.setdefault(current_re, [])
                continue
            if current_re is None:
                continue
            if line.startswith(("Nom", "Fini de surface", "alpha", "[", "Î", "Ã")):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                alpha, cl, cd, cm = map(float, parts[:4])
            except ValueError:
                continue
            re_data[current_re].append((alpha, cl, cd, cm))
    if not re_data:
        raise ValueError(f"No readable polar data in {path}")
    tables = {}
    for re_key, rows in re_data.items():
        df = pd.DataFrame(rows, columns=["alpha", "Cl", "Cd", "Cm"])
        tables[re_key] = df.sort_values("alpha").drop_duplicates("alpha", keep="last").reset_index(drop=True)
    return tables

def interp_coeffs_at_alpha(df_alpha: pd.DataFrame, alpha_target: float) -> dict[str, float]:
    x = df_alpha["alpha"].to_numpy(dtype=float)
    alpha = float(np.clip(alpha_target, np.nanmin(x), np.nanmax(x)))
    return {
        "Cl": float(np.interp(alpha, x, df_alpha["Cl"].to_numpy(dtype=float))),
        "Cd": float(np.interp(alpha, x, df_alpha["Cd"].to_numpy(dtype=float))),
        "Cm": float(np.interp(alpha, x, df_alpha["Cm"].to_numpy(dtype=float))),
        "alpha_used": alpha,
    }

def interp_coeffs_for_re_alpha(
    re_tables: dict[float, pd.DataFrame], re_target: float, alpha_target: float
) -> dict[str, Any]:
    re_sorted = sorted(re_tables.keys())
    re_t = float(re_target)
    if re_t <= re_sorted[0]:
        coeffs = interp_coeffs_at_alpha(re_tables[re_sorted[0]], alpha_target)
        coeffs["Re_used"] = re_sorted[0]
        return coeffs
    if re_t >= re_sorted[-1]:
        coeffs = interp_coeffs_at_alpha(re_tables[re_sorted[-1]], alpha_target)
        coeffs["Re_used"] = re_sorted[-1]
        return coeffs
    for idx in range(len(re_sorted) - 1):
        re0, re1 = re_sorted[idx], re_sorted[idx + 1]
        if re0 <= re_t <= re1:
            c0 = interp_coeffs_at_alpha(re_tables[re0], alpha_target)
            c1 = interp_coeffs_at_alpha(re_tables[re1], alpha_target)
            t = (re_t - re0) / (re1 - re0) if re1 != re0 else 0.0
            return {
                "Cl": float(c0["Cl"] + t * (c1["Cl"] - c0["Cl"])),
                "Cd": float(c0["Cd"] + t * (c1["Cd"] - c0["Cd"])),
                "Cm": float(c0["Cm"] + t * (c1["Cm"] - c0["Cm"])),
                "alpha_used": float(c0["alpha_used"] + t * (c1["alpha_used"] - c0["alpha_used"])),
                "Re_used": (re0, re1),
            }
    coeffs = interp_coeffs_at_alpha(re_tables[re_sorted[-1]], alpha_target)
    coeffs["Re_used"] = re_sorted[-1]
    return coeffs

def compute_aero_moments(
    bands: pd.DataFrame,
    polar_tables: dict[float, pd.DataFrame],
    alpha_deg: float,
    velocity: float,
    rho_air: float,
    mu_air: float,
    delta_x: float = 0.0,
) -> pd.DataFrame:
    rows = []
    for _, row in bands.iterrows():
        chord = max(float(row["c_moy_i"]), 1e-9)
        area = max(float(row["area_i"]), 0.0)
        re_val = reynolds_number(rho_air, velocity, chord, mu_air)
        coeffs = interp_coeffs_for_re_alpha(polar_tables, re_val, alpha_deg)
        q_dyn = 0.5 * rho_air * velocity**2
        lift = coeffs["Cl"] * q_dyn * area
        drag = coeffs["Cd"] * q_dyn * area
        pitch_moment = coeffs["Cm"] * q_dyn * area * chord
        arm = float(row["ym"]) + float(delta_x)
        torsion = pitch_moment + lift * arm
        rows.append(
            {
                "band": int(row["band"]),
                "ym": float(row["ym"]),
                "chord": chord,
                "area": area,
                "Re": re_val,
                "Cl": coeffs["Cl"],
                "Cd": coeffs["Cd"],
                "Cm": coeffs["Cm"],
                "Lift": lift,
                "Drag": drag,
                "arm": arm,
                "Mt": torsion,
            }
        )
    return pd.DataFrame(rows)

def plot_mt_dataset(
    bands: pd.DataFrame, polar_tables: dict[float, pd.DataFrame], case: SimulationCase
) -> pd.DataFrame:
    rows = []
    deltas = np.linspace(case.delta_x_min, case.delta_x_max, int(case.delta_x_steps))
    scenarios = [
        ("case_1", case.alpha_case_1, case.velocity_case_1),
        ("case_2", case.alpha_case_2, case.velocity_case_2),
        ("case_3", case.alpha_case_3, case.velocity_case_3),
    ]
    for name, alpha, velocity in scenarios:
        for delta_x in deltas:
            df = compute_aero_moments(bands, polar_tables, alpha, velocity, case.rho_air, case.mu_air, delta_x)
            rows.append(
                {
                    "scenario": name,
                    "alpha": alpha,
                    "velocity": velocity,
                    "DeltaX": float(delta_x),
                    "Mt_total": float(df["Mt"].sum()),
                    "Lift_total": float(df["Lift"].sum()),
                    "Drag_total": float(df["Drag"].sum()),
                }
            )
    return pd.DataFrame(rows)
