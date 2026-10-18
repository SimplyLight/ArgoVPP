from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import re
import warnings

__all__ = ['NACA_POLAR_CACHE_DIAGNOSTICS', 'PolarDomainError', 'TableDomainError', '_cached_load_polar_file', '_generic_polar_csv', '_physical_load_polar_file', '_resolve_naca_polar_file_cached', 'available_naca_profiles', 'bounded_interpolate_1d', 'bounded_table_row', 'bounded_table_row_low_zero', 'cached_naca_polar', 'default_delft_folder', 'finite_wing_from_profile', 'fn_domain', 'global_common_alpha_domain', 'interpolate_naca_polar', 'load_delft_tables', 'load_polar_file', 'naca4_coordinates', 'normalize_naca_code', 'normalize_polar_column_name', 'polar_domain', 'resolve_naca_polar_file', 'reynolds_bracket_and_alpha_domain', 'table_domain', 'table_row_at']

class TableDomainError(ValueError):
    """Raised when an empirical table is requested outside its documented domain."""

class PolarDomainError(TableDomainError):
    """Raised when a NACA polar is requested outside alpha/Re data."""

def bounded_interpolate_1d(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    x: float | np.ndarray,
    table_name: str = "table",
    x_name: str = "x",
) -> np.ndarray:
    """Interpolate only inside the real tabulated domain.

    The default policy is reject: no silent clipping and no free extrapolation.
    The error message carries the requested value and table bounds.
    """

    x_arr = np.asarray(x_nodes, dtype=float).reshape(-1)
    y_arr = np.asarray(y_nodes, dtype=float).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        raise TableDomainError(f"{table_name}: au moins deux points finis sont requis pour interpoler {x_name}.")
    df = pd.DataFrame({"x": x_arr, "y": y_arr}).sort_values("x")
    duplicate_count = int(df.duplicated("x").sum())
    if duplicate_count:
        warnings.warn(f"{table_name}: {duplicate_count} doublon(s) en {x_name}; valeurs moyennÃƒÆ’Ã‚Â©es.")
        df = df.groupby("x", as_index=False)["y"].mean()
    x_sorted = df["x"].to_numpy(dtype=float)
    y_sorted = df["y"].to_numpy(dtype=float)
    lo = float(x_sorted[0])
    hi = float(x_sorted[-1])
    x_eval = np.asarray(x, dtype=float)
    below = x_eval < lo - 1e-12
    above = x_eval > hi + 1e-12
    if np.any(below) or np.any(above):
        requested = x_eval if x_eval.ndim else float(x_eval)
        raise TableDomainError(f"{table_name}: {x_name}={requested} hors domaine [{lo}, {hi}].")
    return np.interp(x_eval, x_sorted, y_sorted)

def default_delft_folder() -> Path:
    """Return the first available Delft coefficient folder."""

    candidates = [
        Path("VPP_coefficients_tables"),
        Path("Keuning_coefficients_tables") / "VPP_coefficients_tables",
    ]
    for folder in candidates:
        if folder.exists():
            return folder
    return candidates[0]

def load_delft_tables(folder: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load Delft/OceanVPP coefficient tables from CSV files."""

    root = Path(folder) if folder else default_delft_folder()
    required = {
        "bare_hull": "IV_1_bare_hull_residuary.csv",
        "keel": "IV_2_keel_residuary.csv",
        "wetted_surface_heel": "IV_3_wetted_surface_vs_heel.csv",
        "heeled_residuary": "IV_4_heeled_residuary_delta.csv",
        "effective_span": "IV_5_effective_span.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for key, filename in required.items():
        path = root / filename
        if not path.exists():
            missing.append(str(path))
            continue
        tables[key] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Tables Delft manquantes:\n" + "\n".join(missing))
    return tables

def bounded_table_row(tables: dict[str, pd.DataFrame], key: str, x_col: str, x_value: float) -> dict[str, float]:
    """Interpolate every numeric coefficient of a table at x_value without extrapolation."""

    if key not in tables:
        raise TableDomainError(f"Table Delft absente: {key}")
    df = tables[key]
    if x_col not in df.columns:
        raise TableDomainError(f"{key}: colonne d'interpolation absente: {x_col}")
    x_nodes = df[x_col].to_numpy(dtype=float)
    result: dict[str, float] = {}
    for col in df.columns:
        if col == x_col:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        result[col] = float(bounded_interpolate_1d(x_nodes, values, x_value, table_name=key, x_name=x_col))
    return result

def table_row_at(tables: dict[str, pd.DataFrame], key: str, x_col: str, x_value: float) -> dict[str, float]:
    """Compatibility wrapper: table interpolation is bounded by default."""

    return bounded_table_row(tables, key, x_col, x_value)

def fn_domain(tables: dict[str, pd.DataFrame], key: str) -> tuple[float, float]:
    """Return Froude number domain of a table."""

    fn = tables[key]["Fn"].to_numpy(dtype=float)
    return float(np.nanmin(fn)), float(np.nanmax(fn))

def table_domain(tables: dict[str, pd.DataFrame], key: str, x_col: str) -> tuple[float, float]:
    """Return min/max domain of a tabulated variable."""

    values = tables[key][x_col].to_numpy(dtype=float)
    return float(np.nanmin(values)), float(np.nanmax(values))

def normalize_naca_code(code: str) -> str:
    """Normalize a NACA four digit code to the form NACA0012."""

    compact = str(code).strip().replace(" ", "").replace("_", "").replace("-", "").upper()
    if compact.startswith("NACA"):
        digits = compact[4:]
    else:
        digits = compact
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"Code NACA quatre chiffres invalide: {code!r}")
    return "NACA" + digits

def available_naca_profiles(folder: str | Path = "Profil NACA") -> list[tuple[str, str]]:
    """Return NACA profile files as ipywidgets Dropdown (label, value) options."""

    root = Path(folder)
    candidates = sorted(root.glob("*.txt")) if root.exists() else []
    if not candidates:
        candidates = sorted(Path(".").glob("NACA*.txt"))
    options: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            code = normalize_naca_code(path.stem)
        except ValueError:
            continue
        if code not in seen:
            seen.add(code)
            options.append((f"{code} - {path.name}", code))
    return options

def normalize_polar_column_name(name: str) -> str | None:
    """Normalize common polar column names to alpha_deg, CL, CD, CM or Re."""

    raw = str(name).strip().lower().replace("[", "").replace("]", "").replace("degree", "deg")
    raw = raw.replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "alpha": "alpha_deg",
        "alphadeg": "alpha_deg",
        "aoa": "alpha_deg",
        "angle": "alpha_deg",
        "a": "alpha_deg",
        "cl": "CL",
        "cz": "CL",
        "cd": "CD",
        "cx": "CD",
        "cm": "CM",
        "cm0.25": "CM",
        "cm025": "CM",
        "re": "Re",
        "reynolds": "Re",
    }
    return aliases.get(raw)

def naca4_coordinates(code: str, n: int = 120) -> tuple[np.ndarray, float]:
    """Generate closed normalized coordinates of a NACA four digit profile.

    Returns coordinates around the profile in chord units and the maximum thickness ratio.
    """

    code = normalize_naca_code(code)
    digits = code[4:]
    m = int(digits[0]) / 100.0
    p = int(digits[1]) / 10.0
    t = int(digits[2:]) / 100.0
    beta = np.linspace(0.0, np.pi, max(80, n))
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    yc = np.zeros_like(x)
    dyc = np.zeros_like(x)
    if m > 0.0 and p > 0.0:
        left = x < p
        yc[left] = m / p**2 * (2.0 * p * x[left] - x[left] ** 2)
        dyc[left] = 2.0 * m / p**2 * (p - x[left])
        yc[~left] = m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x[~left] - x[~left] ** 2)
        dyc[~left] = 2.0 * m / (1.0 - p) ** 2 * (p - x[~left])
    theta = np.arctan(dyc)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    upper = np.column_stack([xu[::-1], yu[::-1]])
    lower = np.column_stack([xl[1:], yl[1:]])
    coords = np.vstack([upper, lower])
    if np.linalg.norm(coords[0] - coords[-1]) > 1e-10:
        coords = np.vstack([coords, coords[0]])
    return coords, t

def _generic_polar_csv(path: Path) -> pd.DataFrame:
    """Fallback parser for simple CSV/space-separated polar files."""

    raw = pd.read_csv(path, sep=r"[\s,;]+", engine="python", comment="#")
    renamed: dict[str, str] = {}
    for col in raw.columns:
        norm = normalize_polar_column_name(col)
        if norm:
            renamed[col] = norm
    df = raw.rename(columns=renamed)
    if "Re" not in df.columns:
        df["Re"] = np.nan
    return df

def load_polar_file(path: str | Path) -> pd.DataFrame:
    """Load a NACA polar file with repeated Reynolds blocks.

    The project files contain 2D profile polars (Allongement = 0) with columns:
    alpha, Cz, Cx and Cm 0.25. These are normalized to alpha_deg, CL, CD, CM and Re.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float]] = []
    current_re: float | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match_re = re.search(r"Re\s*=\s*([0-9]+(?:[\.,][0-9]+)?)", line, flags=re.IGNORECASE)
        if match_re:
            current_re = float(match_re.group(1).replace(",", "."))
            continue
        if current_re is None or not re.match(r"^-?\d", line):
            continue
        cleaned = line.replace(",", ".")
        parts = re.split(r"[\s;]+", cleaned)
        if len(parts) < 4:
            continue
        try:
            alpha, cl, cd, cm = (float(parts[i]) for i in range(4))
        except ValueError:
            warnings.warn(f"{path.name}: ligne polaire non numerique ignoree: {raw_line!r}")
            continue
        rows.append({"Re": current_re, "alpha_deg": alpha, "CL": cl, "CD": cd, "CM": cm})

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = _generic_polar_csv(path)

    required = ["alpha_deg", "CL", "CD", "CM"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: colonnes polaires manquantes: {missing}")
    if "Re" not in df.columns or df["Re"].isna().all():
        raise ValueError(f"{path}: nombre de Reynolds absent; aucune dependance artificielle n'est creee.")

    for col in ["Re", "alpha_deg", "CL", "CD", "CM"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Re", "alpha_deg", "CL", "CD", "CM"])
    dropped = before - len(df)
    if dropped:
        warnings.warn(f"{path.name}: {dropped} ligne(s) non numerique(s) ignoree(s).")
    duplicate_count = int(df.duplicated(["Re", "alpha_deg"]).sum())
    if duplicate_count:
        warnings.warn(f"{path.name}: {duplicate_count} doublon(s) Re/alpha; coefficients moyennes.")
        df = df.groupby(["Re", "alpha_deg"], as_index=False)[["CL", "CD", "CM"]].mean()
    df = df.sort_values(["Re", "alpha_deg"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path}: aucune ligne polaire exploitable.")
    df.attrs["source_file"] = str(path)
    return df

def polar_domain(df: pd.DataFrame) -> dict[str, float]:
    """Return alpha and Reynolds domains from a polar dataframe."""

    return {
        "alpha_min": float(df["alpha_deg"].min()),
        "alpha_max": float(df["alpha_deg"].max()),
        "Re_min": float(df["Re"].min()),
        "Re_max": float(df["Re"].max()),
    }

def finite_wing_from_profile(CL_profile: float, CD_profile: float, aspect_ratio: float, oswald: float) -> dict[str, float]:
    """Convert 2D profile coefficients to finite-wing coefficients.

    The inspected files declare Allongement = 0, so they are treated as 2D profile polars.
    CL is reduced by a lifting-line slope correction. CD_profile is kept as profile drag and
    one induced drag term CD_i = CL_3D^2 / (pi e AR) is added exactly once.
    """

    ar = max(float(aspect_ratio), 0.1)
    e = max(float(oswald), 0.1)
    CL = float(CL_profile) / (1.0 + 2.0 / (ar * e))
    CD_induced = CL**2 / (np.pi * e * ar)
    return {
        "CL": CL,
        "CD": float(CD_profile) + float(CD_induced),
        "CD_profile": float(CD_profile),
        "CD_induced": float(CD_induced),
    }

NACA_POLAR_CACHE_DIAGNOSTICS = {
    "resolve_calls": 0,
    "load_calls": 0,
    "physical_file_reads": 0,
    "cache_hits": 0,
}

@lru_cache(maxsize=128)
def _resolve_naca_polar_file_cached(profile_name: str, folder_str: str) -> str:
    code = normalize_naca_code(profile_name)
    roots = [Path(folder_str), Path(".")]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.txt")):
            try:
                if normalize_naca_code(path.stem) == code:
                    return str(path)
            except ValueError:
                continue
    raise FileNotFoundError(f"Aucune polaire trouvee pour {code} dans {folder_str}. Le calcul aerodynamique est bloque.")

def resolve_naca_polar_file(profile_name: str, folder: str | Path = "Profil NACA") -> Path:
    """Cached resolution from selected NACA code to an existing polar file."""

    NACA_POLAR_CACHE_DIAGNOSTICS["resolve_calls"] += 1
    return Path(_resolve_naca_polar_file_cached(normalize_naca_code(profile_name), str(folder)))

_physical_load_polar_file = load_polar_file

@lru_cache(maxsize=64)
def _cached_load_polar_file(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    NACA_POLAR_CACHE_DIAGNOSTICS["physical_file_reads"] += 1
    return _physical_load_polar_file(path_str)

def cached_naca_polar(profile_name: str, folder: str | Path = "Profil NACA") -> tuple[Path, pd.DataFrame]:
    """Return cached cleaned polar data; cache key includes file path, mtime and size."""

    NACA_POLAR_CACHE_DIAGNOSTICS["load_calls"] += 1
    path = resolve_naca_polar_file(profile_name, folder)
    stat = path.stat()
    before = NACA_POLAR_CACHE_DIAGNOSTICS["physical_file_reads"]
    df = _cached_load_polar_file(str(path), int(stat.st_mtime_ns), int(stat.st_size)).copy()
    if NACA_POLAR_CACHE_DIAGNOSTICS["physical_file_reads"] == before:
        NACA_POLAR_CACHE_DIAGNOSTICS["cache_hits"] += 1
    df.attrs["source_file"] = str(path)
    return path, df

def reynolds_bracket_and_alpha_domain(df: pd.DataFrame, Re: float, table_name: str = "NACA") -> dict[str, float]:
    """Return the Reynolds bracket and common alpha interval valid for interpolation."""

    re_values = np.sort(df["Re"].dropna().unique().astype(float))
    if len(re_values) == 0:
        raise PolarDomainError(f"{table_name}: aucune valeur de Reynolds exploitable.")
    Re_actual = float(Re)
    re_min = float(re_values[0])
    re_max = float(re_values[-1])
    if len(re_values) == 1:
        re_low = re_high = re_min
    elif Re_actual < re_min - 1e-12 or Re_actual > re_max + 1e-12:
        raise PolarDomainError(f"naca_reynolds_out_of_range: {table_name}: Re={Re_actual:.4g} hors domaine [{re_min:.4g}, {re_max:.4g}].")
    elif np.any(np.isclose(re_values, Re_actual, rtol=0.0, atol=1e-9)):
        re_low = re_high = float(re_values[np.argmin(np.abs(re_values - Re_actual))])
    else:
        re_low = float(re_values[re_values <= Re_actual][-1])
        re_high = float(re_values[re_values >= Re_actual][0])

    def alpha_bounds_for_re(re_value: float) -> tuple[float, float]:
        sub = df[df["Re"] == re_value]
        return float(sub["alpha_deg"].min()), float(sub["alpha_deg"].max())

    low_min, low_max = alpha_bounds_for_re(re_low)
    high_min, high_max = alpha_bounds_for_re(re_high)
    alpha_min_valid = max(low_min, high_min)
    alpha_max_valid = min(low_max, high_max)
    if alpha_min_valid > alpha_max_valid:
        raise PolarDomainError(f"{table_name}: domaines alpha disjoints entre Re={re_low:.0f} et Re={re_high:.0f}.")
    return {
        "Re_low_used": re_low,
        "Re_high_used": re_high,
        "alpha_min_valid_deg": alpha_min_valid,
        "alpha_max_valid_deg": alpha_max_valid,
        "Re_min": re_min,
        "Re_max": re_max,
    }

def interpolate_naca_polar(df: pd.DataFrame, alpha_deg: float, Re: float, table_name: str = "NACA") -> dict[str, Any]:
    """Interpolate CL/CD/CM using the common alpha domain of the Reynolds bracket."""

    alpha = float(alpha_deg)
    bracket = reynolds_bracket_and_alpha_domain(df, Re, table_name)
    if alpha < bracket["alpha_min_valid_deg"] - 1e-12 or alpha > bracket["alpha_max_valid_deg"] + 1e-12:
        raise PolarDomainError(
            f"naca_alpha_out_of_range: {table_name}: alpha={alpha:.4g} deg hors domaine commun "
            f"[{bracket['alpha_min_valid_deg']:.4g}, {bracket['alpha_max_valid_deg']:.4g}] deg."
        )
    re_low = bracket["Re_low_used"]
    re_high = bracket["Re_high_used"]

    def at_re(re_value: float) -> dict[str, float]:
        subset = df[df["Re"] == re_value]
        return {
            col: float(bounded_interpolate_1d(subset["alpha_deg"].to_numpy(), subset[col].to_numpy(), alpha, table_name, "alpha_deg"))
            for col in ["CL", "CD", "CM"]
        }

    low = at_re(re_low)
    high = at_re(re_high)
    if re_low == re_high:
        coeffs = low
        re_reference = re_low
    else:
        weight = (float(Re) - re_low) / (re_high - re_low)
        coeffs = {col: float(low[col] + weight * (high[col] - low[col])) for col in ["CL", "CD", "CM"]}
        re_reference = np.nan
    coeffs.update({
        "Re_lower": re_low,
        "Re_upper": re_high,
        "Re_low_used": re_low,
        "Re_high_used": re_high,
        "Re_reference": re_reference,
        "Re_warning": "",
        "alpha_min": bracket["alpha_min_valid_deg"],
        "alpha_max": bracket["alpha_max_valid_deg"],
        "alpha_min_valid_deg": bracket["alpha_min_valid_deg"],
        "alpha_max_valid_deg": bracket["alpha_max_valid_deg"],
        "Re_min": bracket["Re_min"],
        "Re_max": bracket["Re_max"],
    })
    return coeffs

def global_common_alpha_domain(df: pd.DataFrame) -> tuple[float, float]:
    """Intersection of alpha domains across all Reynolds curves, used for alpha sweep setup."""

    mins = []
    maxs = []
    for _, group in df.groupby("Re"):
        mins.append(float(group["alpha_deg"].min()))
        maxs.append(float(group["alpha_deg"].max()))
    return max(mins), min(maxs)

def bounded_table_row_low_zero(
    tables: dict[str, pd.DataFrame],
    key: str,
    x_col: str,
    x_value: float,
) -> tuple[dict[str, float], bool, str]:
    """Interpolate a coefficient table with an explicit physical zero below the first tabulated point.

    This helper is only appropriate for tables whose final resistance/correction is known to
    vanish at zero speed or zero heel. It never extends above the maximum tabulated point.
    """

    df = tables[key]
    lo, hi = table_domain(tables, key, x_col)
    x = float(x_value)
    if x < -1e-12:
        raise TableDomainError(f"{key}: {x_col}={x} negatif.")
    if x > hi + 1e-12:
        raise TableDomainError(f"delft_fn_above_max: {key}: {x_col}={x:.5g} hors domaine superieur [{lo}, {hi}].")
    if x < lo - 1e-12:
        return bounded_table_row(tables, key, x_col, lo), True, key
    return bounded_table_row(tables, key, x_col, x), False, ""
