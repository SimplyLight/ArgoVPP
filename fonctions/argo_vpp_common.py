from __future__ import annotations

from __future__ import annotations

import json

import math

import re

import struct

import sys

import traceback

from dataclasses import asdict, dataclass, field

from pathlib import Path

from typing import Any, Callable

import numpy as np

import pandas as pd

APP_DIR = Path(__file__).resolve().parent.parent

NOTEBOOK_PATH = APP_DIR / "moments_aile.ipynb"

STARTUP_LOG_PATH = APP_DIR / "argo_vpp_startup_error.log"

GEOMETRY_LOG_PATH = APP_DIR / "Résultats geometrie" / "argo_vpp_geometry.log"

RHO_AIR_DEFAULT = 1.225

MU_AIR_DEFAULT = 1.81e-5

RHO_WATER_DEFAULT = 1025.0

MU_WATER_DEFAULT = 1.05e-3

COORDINATE_CONVENTION = {
    "x": "longitudinal hull axis",
    "y": "transverse axis, center plane y=0",
    "z": "vertical positive upward",
}

NOTEBOOK_SHA256_REFERENCE = "32E7723D3D287A74550D344E2450B4D4A3194C7C3F39001EB25D47447DBAE0C6"

DEFAULT_SAIL_CONTOUR = "0.00,0.00\n0.52,0.00\n0.62,0.70\n0.18,1.20\n0.00,0.00"

def append_geometry_log(message: str) -> None:
    try:
        GEOMETRY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GEOMETRY_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")
    except Exception:
        pass

TRACEABILITY_ROWS = [
    {
        "cell": 1,
        "role": "Imports and utilities",
        "imports": "numpy, matplotlib, ipywidgets, IPython.display, pandas",
        "constants": "-",
        "functions": "-",
        "inputs": "-",
        "globals": "-",
        "widgets": "-",
        "numeric_results": "-",
        "tables": "-",
        "plots": "-",
        "pyqt_location": "Imports, constants, pandas table model",
        "status": "migrated",
    },
    {
        "cell": 2,
        "role": "Interactive sail contour input",
        "imports": "numpy, matplotlib, ipywidgets",
        "constants": "-",
        "functions": "parse_point, close_polygon, intersections_with_line_y",
        "inputs": "textarea points, point x/y buttons",
        "globals": "contour points",
        "widgets": "Textarea, FloatText, Button, Output",
        "numeric_results": "closed polygon, area",
        "tables": "point list",
        "plots": "sail contour",
        "pyqt_location": "Sail geometry group, Geometry tab, 3D sail actor",
        "status": "migrated",
    },
    {
        "cell": 3,
        "role": "Mean chord by spanwise subdivisions",
        "imports": "-",
        "constants": "-",
        "functions": "compute_cmoy_from_contour",
        "inputs": "contour, number of bands",
        "globals": "c_moy_default",
        "widgets": "-",
        "numeric_results": "local chords, c_moy",
        "tables": "band table",
        "plots": "subdivisions and chords",
        "pyqt_location": "Geometry tab and Data tab",
        "status": "migrated",
    },
    {
        "cell": 4,
        "role": "Reynolds min/max",
        "imports": "ipywidgets",
        "constants": "-",
        "functions": "_calc_re",
        "inputs": "rho, mu, vmin, vmax, c_moy",
        "globals": "Re_min, Re_max",
        "widgets": "FloatText, Button, Output",
        "numeric_results": "Re_min, Re_max",
        "tables": "-",
        "plots": "-",
        "pyqt_location": "Top metrics and Aerodynamics tab",
        "status": "migrated",
    },
    {
        "cell": 5,
        "role": "Aerodynamic moments and Mt versus DeltaX",
        "imports": "re, pathlib, pandas, numpy, matplotlib",
        "constants": "N_USED_FIXED, rho/mu defaults",
        "functions": "polar parsing/interpolation, compute_aero_moments",
        "inputs": "profile, alpha/V cases, DeltaX",
        "globals": "moment results",
        "widgets": "Dropdown, FloatText, Button, Output, HTML",
        "numeric_results": "Cl, Cd, Cm, forces, arms, moments",
        "tables": "bands and coefficients",
        "plots": "Mt(DeltaX)",
        "pyqt_location": "Aerodynamics and Moments tabs",
        "status": "migrated",
    },
    {
        "cell": 6,
        "role": "Hydro appendages and finite section correction",
        "imports": "pandas, ipywidgets, numpy, matplotlib",
        "constants": "N_HYDRO_FIXED",
        "functions": "compute_hydro_appendage_from_coeffs, finite correction",
        "inputs": "rudder/keel contours, rho, speeds, alpha, Oswald",
        "globals": "appendage results",
        "widgets": "Textarea, FloatText, Button, Output",
        "numeric_results": "AR, Cl finite, Cd induced, finesse angle",
        "tables": "appendage summary",
        "plots": "appendage contours",
        "pyqt_location": "Hydrodynamics tab and appendage controls",
        "status": "migrated",
    },
    {
        "cell": 7,
        "role": "AWA polar with sail/SPY hybrid",
        "imports": "numpy, pandas, matplotlib, ipywidgets",
        "constants": "rho defaults, bisection tolerances",
        "functions": "calc_aero_simple_awa, solve_equilibrium_awa, compute_polar_awa",
        "inputs": "TWA, profile, trim, sail, hull resistance, SPY",
        "globals": "polaire_results_7",
        "widgets": "Dropdown, FloatText, IntSlider, Button",
        "numeric_results": "V_B, V_AW, beta, VMG, convergence",
        "tables": "AWA polar table",
        "plots": "polar, V_B, sail/SPY comparison, VMG",
        "pyqt_location": "AWA Polar tab",
        "status": "migrated and renamed",
    },
    {
        "cell": 8,
        "role": "True wind CAP/TWA polar with leeway and heel",
        "imports": "numpy, pandas, matplotlib, ipywidgets",
        "constants": "rho/mu defaults, theta/psi limits",
        "functions": "calc_aero_simple_true_wind, solve_derive_equilibrium, solve_heel_equilibrium",
        "inputs": "CAP, profile, hydro, keel, heel, SPY",
        "globals": "polaire_results_8",
        "widgets": "Dropdown, FloatText, IntSlider, Button",
        "numeric_results": "V_B raw/corrected, leeway, heel, moments, VMG",
        "tables": "CAP/TWA polar table",
        "plots": "8 CAP/TWA subplots: rose, speed, lateral force, heel, leeway, drift speed, VMG, heel moments",
        "pyqt_location": "CAP/TWA Polar tab with horizontal sub-tabs",
        "status": "migrated and renamed",
    },
]

RENAMED_FUNCTIONS = {
    "cell7.calc_aero_simple": "calc_aero_simple_awa",
    "cell8.calc_aero_simple": "calc_aero_simple_true_wind",
    "cell7.calc_hydro_drag": "calc_hydro_drag",
    "cell8.calc_hydro_drag": "calc_hydro_drag",
    "cell7.solve_equilibrium_7": "solve_equilibrium_awa",
    "cell8.solve_equilibrium_7": "solve_equilibrium_sail_true_wind",
    "cell7.calc_spy_equilibrium": "calc_spy_equilibrium_awa",
    "cell8.calc_spy_equilibrium": "calc_spy_equilibrium_true_wind",
    "cell7.solve_equilibrium_hybrid": "solve_equilibrium_hybrid_awa",
    "cell8.solve_equilibrium_hybrid": "solve_equilibrium_hybrid_true_wind",
    "cell7.compute_polaire_7": "compute_polar_awa",
    "cell8.compute_polaire_7": "compute_polar_true_wind",
    "cell7._on_polaire7": "MainWindow.recalculate / render_awa_tab",
    "cell8._on_polaire7": "MainWindow.recalculate / render_true_wind_tab",
}



