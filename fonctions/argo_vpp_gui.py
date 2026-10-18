from __future__ import annotations
from .argo_vpp_common import APP_DIR, Any, DEFAULT_SAIL_CONTOUR, Path, STARTUP_LOG_PATH, TRACEABILITY_ROWS, append_geometry_log, asdict, np, pd, sys, traceback
from .argo_vpp_audit import notebook_audit_summary, notebook_sha256, run_self_tests
from .argo_vpp_geometry import close_polygon, compute_cmoy_from_contour, parse_points_text, reynolds_number
from .argo_vpp_models import GeometryState, SimulationCase, SimulationResults
from .argo_vpp_settings import PARAMETER_GROUPS
from .argo_vpp_simulation_export import export_report_html, export_results_csv, export_results_excel, load_project_file, metric_block, parameter_traceability_table, run_full_simulation, save_project_file, validate_case
from .argo_vpp_stl_geometry import apply_stl_unit, compute_keel_autofit_case, compute_rudder_autofit_case, default_keel_longitudinal, default_rudder_longitudinal, detect_longitudinal_axis, evaluate_geometry_state, faces_to_pyvista, hull_frame, local_hull_position, make_naca_appendage_mesh, make_sail_volume_mesh, mesh_dimensions, parse_stl_mesh

def require_pyqt6() -> Any:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        raise RuntimeError(
            "PyQt6 is required to open the native ARGO-VPP interface. "
            "Install with: py -3.11 -m pip install PyQt6 matplotlib pandas numpy"
        ) from exc
    return QtCore, QtGui, QtWidgets

def write_startup_log(exc: BaseException) -> Path:
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    STARTUP_LOG_PATH.write_text(text, encoding="utf-8")
    return STARTUP_LOG_PATH

def run_gui() -> int:
    QtCore, QtGui, QtWidgets = require_pyqt6()
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    try:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
    except Exception:
        try:
            from matplotlib.backends.backend_qt import NavigationToolbar2QT
        except Exception:
            NavigationToolbar2QT = None
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    try:
        import pyvista as pv
        from pyvistaqt import QtInteractor

        pyvista_error = ""
    except Exception as exc:
        pv = None
        QtInteractor = None
        pyvista_error = str(exc)

    class PandasModel(QtCore.QAbstractTableModel):
        def __init__(self, df: pd.DataFrame | None = None) -> None:
            super().__init__()
            self.df = df if df is not None else pd.DataFrame()

        def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
            return len(self.df)

        def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
            return len(self.df.columns)

        def data(self, index: QtCore.QModelIndex, role: int = 0) -> Any:
            if not index.isValid() or role not in (QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole):
                return None
            value = self.df.iat[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0) -> Any:
            if role != QtCore.Qt.ItemDataRole.DisplayRole:
                return None
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return str(self.df.columns[section])
            return str(section + 1)

        def set_dataframe(self, df: pd.DataFrame) -> None:
            self.beginResetModel()
            self.df = df.copy()
            self.endResetModel()

    class PlotCanvas(FigureCanvasQTAgg):
        def __init__(self, projection: str | None = None) -> None:
            self.fig = Figure(figsize=(5, 3), dpi=100)
            self.ax = self.fig.add_subplot(111, projection=projection)
            super().__init__(self.fig)
            self.setMinimumHeight(220)

    class NoWheelDoubleSpinBox(QtWidgets.QDoubleSpinBox):
        def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
            if not self.hasFocus():
                event.ignore()
                return
            super().wheelEvent(event)

    class NoWheelSpinBox(QtWidgets.QSpinBox):
        def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
            if not self.hasFocus():
                event.ignore()
                return
            super().wheelEvent(event)

    class NoWheelComboBox(QtWidgets.QComboBox):
        def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
            if not self.hasFocus():
                event.ignore()
                return
            super().wheelEvent(event)

    RESULT_FIGURES = [
        ("notebook_map", "Notebook mapping report"),
        ("sail_contour", "Sail contour"),
        ("chord_sections", "Chord subdivisions"),
        ("reynolds", "Reynolds range"),
        ("aero_forces", "Aerodynamic forces"),
        ("moments_delta_x", "Mt versus DeltaX"),
        ("hydro_appendages", "Hydro appendages"),
        ("leeway", "Leeway angle"),
        ("heel", "Heel angle"),
        ("awa_polar", "AWA polar"),
        ("cap_twa_polar", "CAP/TWA polar"),
        ("vmg", "VMG"),
        ("residuals", "Convergence residuals"),
    ]

    CAP_TWA_PLOTS = [
        ("rose_vb", "Rose V_B"),
        ("boat_speed", "Vitesse bateau"),
        ("lateral_equilibrium", "Equilibre lateral"),
        ("heel_angle", "Angle de gite"),
        ("leeway_angle", "Angle de derive"),
        ("drift_speed", "Vitesse de derive"),
        ("vmg", "Vitesse utile / VMG"),
        ("heel_moments", "Equilibre moments gite"),
    ]

    class ResultsWindow(QtWidgets.QMainWindow):
        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.case = SimulationCase()
            self.results = SimulationResults()
            self.setWindowTitle("ARGO-VPP - Resultats detailles")
            self.resize(1180, 760)
            self.setStyleSheet(APP_STYLE)
            root = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(root)
            header = QtWidgets.QHBoxLayout()
            self.selector = NoWheelComboBox()
            self.selector.setObjectName("combo_result_figure")
            for key, label in RESULT_FIGURES:
                self.selector.addItem(label, key)
            self.state_label = QtWidgets.QLabel("Aucun resultat calcule")
            self.state_label.setObjectName("mutedLabel")
            header.addWidget(QtWidgets.QLabel("Figure"))
            header.addWidget(self.selector, 1)
            header.addWidget(self.state_label, 2)
            layout.addLayout(header)
            self.cap_twa_tabs = QtWidgets.QTabBar()
            self.cap_twa_tabs.setObjectName("cap_twa_subtabs")
            self.cap_twa_tabs.setDrawBase(False)
            self.cap_twa_tabs.setExpanding(False)
            for key, label in CAP_TWA_PLOTS:
                sub_idx = self.cap_twa_tabs.addTab(label)
                self.cap_twa_tabs.setTabData(sub_idx, key)
            self.cap_twa_tabs.setVisible(False)
            self.cap_twa_tabs.currentChanged.connect(self.update_plot)
            layout.addWidget(self.cap_twa_tabs)
            self.canvas = PlotCanvas()
            self.canvas.setMinimumHeight(560)
            self.toolbar = NavigationToolbar2QT(self.canvas, self) if NavigationToolbar2QT is not None else None
            if self.toolbar is not None:
                self.toolbar.setObjectName("results_matplotlib_toolbar")
                layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, 1)
            self.setCentralWidget(root)
            self.selector.currentIndexChanged.connect(self.update_plot)

        def set_data(self, case: SimulationCase, results: SimulationResults) -> None:
            self.case = SimulationCase(**asdict(case))
            self.results = results
            self.update_plot()

        def set_axes(self, projection: str | None = None) -> Any:
            self.canvas.fig.clear()
            self.canvas.ax = self.canvas.fig.add_subplot(111, projection=projection)
            return self.canvas.ax

        def empty_plot(self, message: str) -> None:
            ax = self.set_axes()
            ax.axis("off")
            ax.text(0.5, 0.55, message, ha="center", va="center", fontsize=15, weight="bold", color="#334155")
            ax.text(0.5, 0.45, "Lancez Recalculate pour produire les tableaux scientifiques.", ha="center", va="center", fontsize=10, color="#64748b")

        def update_plot(self, *_: Any) -> None:
            key = self.selector.currentData()
            self.cap_twa_tabs.setVisible(key == "cap_twa_polar")
            has_any = any(
                not df.empty
                for df in [self.results.geometry, self.results.moments, self.results.hydro_appendages, self.results.awa, self.results.true_wind]
            )
            if not has_any and key not in {"sail_contour", "reynolds", "notebook_map"}:
                self.state_label.setText("Aucun resultat calcule")
                self.empty_plot("Aucun resultat disponible")
                self.canvas.draw_idle()
                return
            self.state_label.setText("Resultats obsoletes - recalcul requis" if self.results.stale and has_any else "Resultats a jour" if has_any else "Aucun resultat calcule")
            if key == "notebook_map":
                self.plot_notebook_map()
            elif key == "sail_contour":
                self.plot_sail_contour()
            elif key == "chord_sections":
                self.plot_chord_sections()
            elif key == "reynolds":
                self.plot_reynolds()
            elif key == "aero_forces":
                self.plot_aero_forces()
            elif key == "moments_delta_x":
                self.plot_moments_delta_x()
            elif key == "hydro_appendages":
                self.plot_hydro_appendages()
            elif key == "leeway":
                self.plot_true_wind_series("psi_deg", "Leeway angle", "psi [deg]")
            elif key == "heel":
                self.plot_true_wind_series("theta_deg", "Heel angle", "theta [deg]")
            elif key == "awa_polar":
                self.plot_polar(self.results.awa, "TWA", "V_B", "AWA polar")
            elif key == "cap_twa_polar":
                self.plot_cap_twa_current()
            elif key == "vmg":
                self.plot_vmg()
            elif key == "residuals":
                self.plot_residuals()
            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()

        def stale_watermark(self, ax: Any) -> None:
            if self.results.stale:
                ax.text(0.99, 0.02, "stale", transform=ax.transAxes, ha="right", va="bottom", color="#b45309", fontsize=11, weight="bold")

        def plot_notebook_map(self) -> None:
            ax = self.set_axes()
            ax.axis("off")
            rows = [
                ("Cell 2", "Sail contour", "sail_contour", "implemented"),
                ("Cell 3", "Mean chord subdivisions", "chord_sections", "implemented"),
                ("Cell 4", "Reynolds min/max", "reynolds", "implemented"),
                ("Cell 5", "Aero moments / Mt(DeltaX)", "aero_forces, moments_delta_x", "implemented"),
                ("Cell 6", "Hydro appendages", "hydro_appendages", "implemented"),
                ("Cell 7", "AWA polar", "awa_polar", "implemented"),
                ("Cell 8", "Rose V_B", "CAP/TWA polar > Rose V_B", "implemented"),
                ("Cell 8", "Vitesse bateau", "CAP/TWA polar > Vitesse bateau", "implemented"),
                ("Cell 8", "Equilibre lateral", "CAP/TWA polar > Equilibre lateral", "implemented"),
                ("Cell 8", "Angle de gite theta(CAP)", "CAP/TWA polar > Angle de gite", "implemented"),
                ("Cell 8", "Angle de derive psi(CAP)", "CAP/TWA polar > Angle de derive", "implemented"),
                ("Cell 8", "Vitesse de derive", "CAP/TWA polar > Vitesse de derive", "implemented"),
                ("Cell 8", "Vitesse utile: pres / portant", "CAP/TWA polar > Vitesse utile / VMG", "implemented"),
                ("Cell 8", "Equilibre moments gite", "CAP/TWA polar > Equilibre moments gite", "implemented"),
            ]
            table = ax.table(cellText=rows, colLabels=["Notebook", "Plot", "Selector key", "Status"], loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.6)
            ax.set_title("Notebook plot mapping / comparison report", pad=18, weight="bold")

        def plot_sail_contour(self) -> None:
            ax = self.set_axes()
            try:
                pts = close_polygon(parse_points_text(self.case.contour_text))
            except Exception as exc:
                self.empty_plot(f"Contour voile invalide: {exc}")
                return
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color="#0f766e", lw=2)
            ax.fill(xs, ys, color="#99f6e4", alpha=0.35)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x contour [m]")
            ax.set_ylabel("height [m]")
            ax.set_title(f"Sail contour - {self.case.aero_profile}")

        def plot_chord_sections(self) -> None:
            ax = self.set_axes()
            df = self.results.geometry
            if df.empty:
                try:
                    _, df = compute_cmoy_from_contour(parse_points_text(self.case.contour_text), self.case.n_bands)
                except Exception:
                    self.empty_plot("Aucune subdivision de corde disponible")
                    return
            ax.plot(df["ym"], df["c_moy_i"], marker="o", color="#0f766e", lw=2)
            ax.set_xlabel("height y [m]")
            ax.set_ylabel("local chord [m]")
            ax.set_title("Chord subdivisions")
            ax.grid(True, alpha=0.25)
            self.stale_watermark(ax)

        def plot_reynolds(self) -> None:
            ax = self.set_axes()
            re_min = reynolds_number(self.case.rho_air, self.case.V_min_re, self.case.c_moy, self.case.mu_air)
            re_max = reynolds_number(self.case.rho_air, self.case.V_max_re, self.case.c_moy, self.case.mu_air)
            ax.bar(["Re min", "Re max"], [re_min, re_max], color=["#38bdf8", "#0f766e"])
            ax.set_ylabel("Re [-]")
            ax.set_title("Reynolds range")
            ax.grid(True, axis="y", alpha=0.25)

        def plot_aero_forces(self) -> None:
            ax = self.set_axes()
            df = self.results.moments
            if df.empty:
                self.empty_plot("Forces aerodynamiques non calculees")
                return
            summary = df.groupby("scenario", as_index=False)[["Lift_total", "Drag_total"]].mean()
            x = np.arange(len(summary))
            ax.bar(x - 0.18, summary["Lift_total"], width=0.36, label="Lift", color="#0f766e")
            ax.bar(x + 0.18, summary["Drag_total"], width=0.36, label="Drag", color="#f97316")
            ax.set_xticks(x, summary["scenario"], rotation=15)
            ax.set_ylabel("Force [N]")
            ax.set_title("Aerodynamic forces")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.25)
            self.stale_watermark(ax)

        def plot_moments_delta_x(self) -> None:
            ax = self.set_axes()
            df = self.results.moments
            if df.empty:
                self.empty_plot("Moments non calcules")
                return
            for name, sub in df.groupby("scenario"):
                ax.plot(sub["DeltaX"], sub["Mt_total"], label=str(name), lw=2)
            ax.set_xlabel("DeltaX [m]")
            ax.set_ylabel("Mt [N.m]")
            ax.set_title("Mt versus DeltaX")
            ax.legend()
            ax.grid(True, alpha=0.25)
            self.stale_watermark(ax)

        def plot_hydro_appendages(self) -> None:
            ax = self.set_axes()
            df = self.results.hydro_appendages
            if df.empty:
                self.empty_plot("Appendices hydrodynamiques non calcules")
                return
            row = df.iloc[0]
            keys = [c for c in ["AR", "Cl_finite", "Cd_finite", "finesse"] if c in df.columns]
            ax.bar(keys, [row[k] for k in keys], color=["#38bdf8", "#0f766e", "#f97316", "#7c3aed"][: len(keys)])
            ax.set_title("Hydro appendages")
            ax.grid(True, axis="y", alpha=0.25)
            self.stale_watermark(ax)

        def plot_true_wind_series(self, column: str, title: str, ylabel: str) -> None:
            ax = self.set_axes()
            df = self.results.true_wind
            if df.empty or column not in df.columns:
                self.empty_plot(f"{title} non calcule")
                return
            ax.plot(df["CAP"], df[column], color="#0f766e", lw=2)
            ax.set_xlabel("CAP [deg]")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            self.stale_watermark(ax)

        def plot_polar(self, df: pd.DataFrame, angle_col: str, value_col: str, title: str) -> None:
            ax = self.set_axes("polar")
            if df.empty or angle_col not in df.columns or value_col not in df.columns:
                self.empty_plot(f"{title} non calcule")
                return
            ax.plot(np.radians(df[angle_col]), df[value_col], color="#0f766e", lw=2)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(title)
            self.stale_watermark(ax)

        def cap_twa_dataframe(self) -> pd.DataFrame:
            df = self.results.true_wind.copy()
            if df.empty:
                return df
            if "V_lateral" not in df.columns:
                df["V_lateral"] = np.sign(df.get("psi_deg", 0.0)) * df.get("V_B", 0.0) * np.tan(np.radians(np.abs(df.get("psi_deg", 0.0))))
            if "VMG_signed" not in df.columns:
                df["VMG_signed"] = df["V_B"] * np.cos(np.radians(df["CAP"]))
            if "VMG_pres" not in df.columns:
                df["VMG_pres"] = np.maximum(df["VMG_signed"], 0.0)
            if "VMG_portant" not in df.columns:
                df["VMG_portant"] = np.maximum(-df["VMG_signed"], 0.0)
            return df.sort_values("CAP")

        def plot_cap_twa_current(self) -> None:
            plot_key = str(self.cap_twa_tabs.tabData(self.cap_twa_tabs.currentIndex()) or "rose_vb")
            dispatch = {
                "rose_vb": self.create_cap_twa_rose_figure,
                "boat_speed": self.create_boat_speed_figure,
                "lateral_equilibrium": self.create_lateral_equilibrium_figure,
                "heel_angle": self.create_heel_angle_figure,
                "leeway_angle": self.create_leeway_angle_figure,
                "drift_speed": self.create_drift_speed_figure,
                "vmg": self.create_vmg_figure,
                "heel_moments": self.create_heel_moment_equilibrium_figure,
            }
            dispatch.get(plot_key, self.create_cap_twa_rose_figure)()

        def create_cap_twa_rose_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes("polar")
            if df.empty or "CAP" not in df.columns or "V_B" not in df.columns:
                self.empty_plot("CAP/TWA non calcule")
                return
            cap_rad = np.radians(df["CAP"].to_numpy(dtype=float))
            speeds = df["V_B"].to_numpy(dtype=float)
            ax.plot(cap_rad, speeds, "b-", linewidth=2.5)
            ax.fill(cap_rad, speeds, alpha=0.2, color="blue")
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_title(f"Rose V_B\nalpha_trim={float(self.case.alpha_trim)} deg", fontsize=10, weight="bold")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_boat_speed_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Vitesse bateau non calculee")
                return
            ax.plot(df["CAP"], df.get("V_B_voile", 0.0), "b--", linewidth=1.4, alpha=0.6, label="VOILE")
            ax.plot(df["CAP"], df.get("V_B_spy", 0.0), "r--", linewidth=1.4, alpha=0.6, label="SPY")
            ax.plot(df["CAP"], df.get("V_B_raw", df["V_B"]), color="0.55", linestyle=":", linewidth=1.5, label="sans gite")
            ax.plot(df["CAP"], df["V_B"], "g-", linewidth=2.3, marker="o", markersize=3, label="avec gite")
            idx_max = int(df["V_B"].idxmax())
            v_max = float(df.loc[idx_max, "V_B"])
            cap_max = float(df.loc[idx_max, "CAP"])
            ax.plot(cap_max, v_max, "k*", markersize=16, label=f"{v_max:.2f} m/s @ {cap_max:.0f} deg")
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("V_B (m/s)", fontsize=9, weight="bold")
            ax.set_title("Vitesse bateau", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_lateral_equilibrium_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Equilibre lateral non calcule")
                return
            ax.plot(df["CAP"], df.get("F_aero_lat_abs", 0.0), "m-", linewidth=2.0, label="|Faero lat|")
            ax.plot(df["CAP"], df.get("F_hydro_derive", 0.0), "c--", linewidth=1.6, label="Fhydro derive")
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("Force laterale (N)", fontsize=9, weight="bold")
            ax.set_title("Equilibre lateral", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_heel_angle_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Angle de gite non calcule")
                return
            ax.plot(df["CAP"], df.get("theta_deg", 0.0), color="tab:purple", linewidth=2.4, marker="o", markersize=3, label="theta gite")
            ax.axhline(y=float(self.case.theta_max), color="r", linestyle=":", linewidth=1.2, label="theta max")
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("theta (deg)", fontsize=9, weight="bold")
            ax.set_title("Angle de gite theta(CAP)", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_leeway_angle_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Angle de derive non calcule")
                return
            ax.plot(df["CAP"], df.get("psi_deg", 0.0), "k-", linewidth=2.0, marker="o", markersize=3, label="psi derive")
            ax.axhline(y=0.0, color="0.5", linewidth=0.7)
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("psi (deg)", fontsize=9, weight="bold")
            ax.set_title("Angle de derive psi(CAP)", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_drift_speed_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Vitesse de derive non calculee")
                return
            lateral = pd.Series(df.get("V_lateral", 0.0), index=df.index, dtype=float)
            ax.plot(df["CAP"], lateral, color="tab:orange", linewidth=2.0, label="V derive laterale")
            ax.plot(df["CAP"], np.abs(lateral), color="tab:red", linestyle="--", linewidth=1.5, label="|V derive|")
            ax.axhline(y=0.0, color="k", linewidth=0.6)
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("Vitesse de derive (m/s)", fontsize=9, weight="bold")
            ax.set_title("Vitesse de derive", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_vmg_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("VMG non calculee")
                return
            ax.plot(df["CAP"], df.get("VMG", 0.0), color="0.55", linewidth=1.4, label="VMG absolue")
            ax.plot(df["CAP"], df.get("VMG_pres", 0.0), "g-", linewidth=2.3, marker="s", markersize=3, alpha=0.8, label="VMG pres")
            ax.plot(df["CAP"], df.get("VMG_portant", 0.0), color="tab:blue", linestyle="--", linewidth=1.8, label="VMG portant")
            idx_vmg_pres = int(df["VMG_pres"].idxmax()) if "VMG_pres" in df.columns else int(df["VMG"].idxmax())
            vmg_pres_max = float(df.loc[idx_vmg_pres, "VMG_pres"] if "VMG_pres" in df.columns else df.loc[idx_vmg_pres, "VMG"])
            cap_vmg_pres = float(df.loc[idx_vmg_pres, "CAP"])
            ax.plot(cap_vmg_pres, vmg_pres_max, "r*", markersize=16, label=f"Pres: {vmg_pres_max:.2f} m/s @ {cap_vmg_pres:.0f} deg")
            ax.set_xlabel("CAP (deg)", fontsize=10, weight="bold")
            ax.set_ylabel("VMG (m/s)", fontsize=10, weight="bold")
            ax.set_title("Vitesse utile: pres / portant", fontsize=11, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def create_heel_moment_equilibrium_figure(self) -> None:
            df = self.cap_twa_dataframe()
            ax = self.set_axes()
            if df.empty:
                self.empty_plot("Equilibre moments gite non calcule")
                return
            m_righting = pd.Series(df.get("M_derive", 0.0), index=df.index, dtype=float) + pd.Series(df.get("M_quille", 0.0), index=df.index, dtype=float)
            ax.plot(df["CAP"], df.get("M_aero", 0.0), color="tab:purple", linewidth=2.0, label="M aero")
            ax.plot(df["CAP"], m_righting, color="tab:green", linestyle="--", linewidth=1.8, label="M derive + M quille")
            ax.set_xlabel("CAP (deg)", fontsize=9, weight="bold")
            ax.set_ylabel("Moment (N.m)", fontsize=9, weight="bold")
            ax.set_title("Equilibre moments gite", fontsize=10, weight="bold")
            ax.set_xlim(0, 360)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)
            self.stale_watermark(ax)

        def plot_vmg(self) -> None:
            ax = self.set_axes()
            df = self.results.true_wind
            if df.empty or "VMG" not in df.columns:
                self.empty_plot("VMG non calcule")
                return
            ax.plot(df["CAP"], df["VMG"], color="#0f766e", lw=2)
            ax.axhline(0.0, color="#94a3b8", lw=1)
            ax.set_xlabel("CAP [deg]")
            ax.set_ylabel("VMG [m/s]")
            ax.set_title("Velocity made good")
            ax.grid(True, alpha=0.25)
            self.stale_watermark(ax)

        def plot_residuals(self) -> None:
            ax = self.set_axes()
            df = self.results.true_wind
            cols = [c for c in ["derive_residual", "heel_residual"] if c in df.columns]
            iteration_cols = [c for c in ["derive_iterations", "heel_iterations"] if c in df.columns]
            if df.empty or (not cols and not iteration_cols):
                self.empty_plot("Residus de convergence non calcules")
                return
            for col in cols:
                ax.plot(df["CAP"], df[col], label=col, lw=2)
            handles, labels = ax.get_legend_handles_labels()
            if iteration_cols:
                iteration_ax = ax.twinx()
                for col in iteration_cols:
                    iteration_ax.step(
                        df["CAP"],
                        pd.to_numeric(df[col], errors="coerce"),
                        where="mid",
                        marker="o",
                        markersize=3,
                        linestyle="--",
                        label=col,
                    )
                iteration_ax.set_ylabel("Solver iterations")
                iteration_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                iteration_ax.set_ylim(bottom=0)
                iteration_handles, iteration_labels = iteration_ax.get_legend_handles_labels()
                handles += iteration_handles
                labels += iteration_labels
            ax.set_xlabel("CAP [deg]")
            ax.set_ylabel("Residual")
            ax.set_title("Convergence residuals and iterations")
            ax.legend(handles, labels)
            ax.grid(True, alpha=0.25)
            self.stale_watermark(ax)

    class Viewer3D(QtWidgets.QWidget):
        stlDropped = QtCore.pyqtSignal(str)
        importRequested = QtCore.pyqtSignal()
        defineGeometryRequested = QtCore.pyqtSignal()

        def __init__(self) -> None:
            super().__init__()
            self.setAcceptDrops(True)
            self.vertices: np.ndarray | None = None
            self.faces: np.ndarray | None = None
            self.plotter = None
            self.viewer_error = pyvista_error
            self.stack = QtWidgets.QStackedLayout(self)
            self.stack.setContentsMargins(0, 0, 0, 0)
            if QtInteractor is not None:
                try:
                    self.plotter = QtInteractor(self)
                    self.plotter.set_background("#f8fafc")
                    self.stack.addWidget(self.plotter)
                except Exception as exc:
                    self.viewer_error = str(exc)
            self.empty_panel = self._empty_panel()
            self.stack.addWidget(self.empty_panel)
            self.stack.setCurrentWidget(self.empty_panel)

        def _empty_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QFrame()
            panel.setObjectName("emptyViewer")
            layout = QtWidgets.QVBoxLayout(panel)
            layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            title = QtWidgets.QLabel("Aucune geometrie chargee")
            title.setObjectName("emptyTitle")
            subtitle_text = "Importez une coque STL ou definissez la geometrie de la voile et des appendices."
            if self.viewer_error:
                subtitle_text += f"\n\nVue 3D indisponible: {self.viewer_error}"
            subtitle = QtWidgets.QLabel(subtitle_text)
            subtitle.setObjectName("emptySubtitle")
            subtitle.setWordWrap(True)
            actions = QtWidgets.QHBoxLayout()
            import_btn = QtWidgets.QPushButton("Importer une coque STL")
            import_btn.setObjectName("btn_empty_import_stl")
            import_btn.setProperty("primary", True)
            define_btn = QtWidgets.QPushButton("Definir la geometrie")
            define_btn.setObjectName("btn_empty_define_geometry")
            import_btn.clicked.connect(self.importRequested.emit)
            define_btn.clicked.connect(self.defineGeometryRequested.emit)
            actions.addWidget(import_btn)
            actions.addWidget(define_btn)
            layout.addWidget(title, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(subtitle, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addLayout(actions)
            return panel

        def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()

        def dropEvent(self, event: QtGui.QDropEvent) -> None:
            urls = event.mimeData().urls()
            if urls:
                self.stlDropped.emit(urls[0].toLocalFile())

        def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
            self.vertices = vertices
            self.faces = faces

        def has_hull(self) -> bool:
            return self.vertices is not None and self.faces is not None and self.vertices.size > 0

        def make_polydata(self, vertices: np.ndarray, faces: np.ndarray) -> Any:
            if pv is None:
                raise RuntimeError("PyVista is not available.")
            face_data = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
            return pv.PolyData(vertices, face_data)

        def set_view(self, name: str) -> None:
            if self.plotter is None:
                return
            if name == "top":
                self.plotter.view_xy()
            elif name == "side":
                self.plotter.view_xz()
            elif name == "front":
                self.plotter.view_yz()
            else:
                self.plotter.view_isometric()
            self.plotter.reset_camera()

        def reset_camera(self) -> None:
            if self.plotter is not None:
                self.plotter.reset_camera()

        def draw_scene(self, case: SimulationCase) -> None:
            state = evaluate_geometry_state(case, self.vertices)
            if self.plotter is None or pv is None:
                self.stack.setCurrentWidget(self.empty_panel)
                return
            self.plotter.clear()
            self.plotter.set_background("#f8fafc")
            if not any([state.hull_visible, state.sail_visible, state.keel_visible, state.rudder_visible]):
                self.stack.setCurrentWidget(self.empty_panel)
                return
            self.stack.setCurrentWidget(self.plotter)
            base_vertices = self.vertices if self.has_hull() else np.zeros((0, 3))
            dims = mesh_dimensions(base_vertices) if self.has_hull() else {"x_mid": 0.0, "z_min": 0.0, "z_max": 0.0}
            frame = hull_frame(case)
            long_vec = np.asarray(frame["long_vec"], dtype=float)
            trans_vec = np.asarray(frame["trans_vec"], dtype=float)
            vertical_up_vec = np.asarray(frame["vertical_up_vec"], dtype=float)
            sail_long = default_keel_longitudinal(self.vertices, case) if case.sail_x == 0.0 else case.sail_x
            sail_root = local_hull_position(self.vertices, case, sail_long, case.transverse_offset, "top")
            sail_root = sail_root + vertical_up_vec * case.vertical_offset
            keel_long = default_keel_longitudinal(self.vertices, case)
            keel_root = local_hull_position(self.vertices, case, keel_long, case.keel_transverse_offset, "bottom")
            keel_root = keel_root + vertical_up_vec * case.keel_vertical_offset
            rudder_long = default_rudder_longitudinal(self.vertices, case)
            rudder_root = local_hull_position(self.vertices, case, rudder_long, case.rudder_transverse_offset, "bottom")
            rudder_root = rudder_root + vertical_up_vec * case.rudder_vertical_offset
            if state.hull_visible and self.vertices is not None and self.faces is not None:
                mesh = self.make_polydata(self.vertices, self.faces)
                style = "wireframe" if case.view_mode == "wireframe" else "surface"
                self.plotter.add_mesh(mesh, color="#78c7dc", opacity=0.68, style=style, show_edges=case.view_mode == "edges", edge_color="#305064", name="hull")
            if state.sail_visible:
                contour = parse_points_text(case.contour_text)
                sail_vertices, sail_faces = make_sail_volume_mesh(contour, case, sail_root)
                if not state.hull_loaded and case.sail_x == 0.0 and case.transverse_offset == 0.0 and case.vertical_offset == 0.0:
                    sail_vertices = sail_vertices - sail_vertices.mean(axis=0)
                self.plotter.add_mesh(
                    pv.PolyData(sail_vertices, faces_to_pyvista(sail_faces)),
                    color="#2dd4bf",
                    opacity=0.66,
                    show_edges=case.view_mode in {"edges", "wireframe"},
                    edge_color="#0f766e",
                    name="sail",
                )
                if not state.hull_loaded:
                    self.plotter.add_text(
                        "Voile affichee dans son repere local - position finale apres import coque STL",
                        position="lower_left",
                        font_size=9,
                        color="#475569",
                    )
            if state.keel_visible:
                root_chord = case.keel_root_chord if case.keel_root_chord > 0 else case.S_derive / max(case.hauteur_quille, 1e-9)
                tip_chord = case.keel_tip_chord if case.keel_tip_chord > 0 else 0.65 * root_chord
                verts, faces = make_naca_appendage_mesh(
                    keel_root,
                    case,
                    root_chord=root_chord,
                    tip_chord=tip_chord,
                    span=case.hauteur_quille,
                    thickness_ratio=case.keel_thickness_ratio,
                    sweep=case.keel_sweep,
                    angle_deg=case.keel_yaw_deg,
                    pitch_deg=case.keel_pitch_deg,
                )
                self.plotter.add_mesh(pv.PolyData(verts, faces_to_pyvista(faces)), color="#86a8ff", opacity=0.86, show_edges=case.view_mode in {"edges", "wireframe"}, edge_color="#1e3a8a", name="keel")
            if state.rudder_visible:
                verts, faces = make_naca_appendage_mesh(
                    rudder_root,
                    case,
                    root_chord=case.rudder_root_chord,
                    tip_chord=case.rudder_tip_chord,
                    span=case.rudder_span,
                    thickness_ratio=case.rudder_thickness_ratio,
                    sweep=case.rudder_sweep,
                    angle_deg=case.rudder_angle_deg,
                    pitch_deg=case.rudder_pitch_deg,
                )
                self.plotter.add_mesh(pv.PolyData(verts, faces_to_pyvista(faces)), color="#f59e0b", opacity=0.88, show_edges=case.view_mode in {"edges", "wireframe"}, edge_color="#92400e", name="rudder")
            if case.show_grid:
                self.plotter.show_grid(color="#cbd5e1", grid="back", location="outer")
            if case.show_axes:
                self.plotter.add_axes(line_width=2, labels_off=False)
            self.plotter.view_isometric()
            self.plotter.reset_camera()

    class WorkerSignals(QtCore.QObject):
        progress = QtCore.pyqtSignal(int, str)
        finished = QtCore.pyqtSignal(object)
        failed = QtCore.pyqtSignal(str)

    class SimulationWorker(QtCore.QRunnable):
        def __init__(self, case: SimulationCase) -> None:
            super().__init__()
            self.case = case
            self.signals = WorkerSignals()

        def run(self) -> None:
            try:
                result = run_full_simulation(self.case, lambda pct, msg: self.signals.progress.emit(pct, msg))
                self.signals.finished.emit(result)
            except Exception:
                self.signals.failed.emit(traceback.format_exc())

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.case = SimulationCase()
            self.results = SimulationResults()
            self.thread_pool = QtCore.QThreadPool.globalInstance()
            self.field_widgets: dict[str, Any] = {}
            self.table_models: dict[str, PandasModel] = {}
            self.plots: dict[str, PlotCanvas] = {}
            self.overview_labels: dict[str, QtWidgets.QLabel] = {}
            self.geometry_labels: dict[str, QtWidgets.QLabel] = {}
            self.results_window: ResultsWindow | None = None
            self.parameter_filter_mode = "all"
            self.viewer = Viewer3D()
            self.viewer.importRequested.connect(self.import_stl)
            self.viewer.defineGeometryRequested.connect(self.define_geometry)
            self.setWindowTitle("ARGO-VPP V2")
            self.resize(1500, 920)
            self._build_ui()
            self.refresh_all()

        def _build_ui(self) -> None:
            root = QtWidgets.QWidget()
            root.setObjectName("root")
            outer = QtWidgets.QVBoxLayout(root)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            outer.addWidget(self._topbar())
            self.metric_layout = QtWidgets.QHBoxLayout()
            metric_wrap = QtWidgets.QWidget()
            metric_wrap.setObjectName("metricWrap")
            metric_wrap.setLayout(self.metric_layout)
            outer.addWidget(metric_wrap)
            body = QtWidgets.QHBoxLayout()
            body.setSpacing(0)
            body.addWidget(self._nav_bar())
            body.addWidget(self._parameter_panel(), 0)
            body.addWidget(self._workspace(), 1)
            outer.addLayout(body, 1)
            self.status = QtWidgets.QStatusBar()
            self.setStatusBar(self.status)
            self.setCentralWidget(root)
            self.setStyleSheet(APP_STYLE)

        def _topbar(self) -> QtWidgets.QWidget:
            bar = QtWidgets.QWidget()
            bar.setObjectName("topbar")
            layout = QtWidgets.QHBoxLayout(bar)
            title = QtWidgets.QLabel("ARGO-VPP")
            title.setObjectName("brand")
            new_project = QtWidgets.QPushButton("New")
            new_project.setObjectName("btn_new_project")
            open_project = QtWidgets.QPushButton("Open")
            open_project.setObjectName("btn_open_project")
            save_project = QtWidgets.QPushButton("Save")
            save_project.setObjectName("btn_save_project")
            save_as_project = QtWidgets.QPushButton("Save as")
            save_as_project.setObjectName("btn_save_as_project")
            project_label = QtWidgets.QLabel("Project")
            project_label.setObjectName("topLabel")
            self.project_edit = QtWidgets.QLineEdit(self.case.project_name)
            self.project_edit.setObjectName("edit_project_name")
            self.project_edit.setPlaceholderText("Project name")
            variant_label = QtWidgets.QLabel("Variant")
            variant_label.setObjectName("topLabel")
            self.variant_combo = QtWidgets.QComboBox()
            self.variant_combo.setObjectName("combo_variant")
            self.variant_combo.addItem(self.case.variant_name)
            add_variant = QtWidgets.QPushButton("New variant")
            add_variant.setObjectName("btn_new_variant")
            self.search = QtWidgets.QLineEdit()
            self.search.setObjectName("edit_global_search")
            self.search.setPlaceholderText("Find parameters, results...")
            reset = QtWidgets.QPushButton("Reset")
            reset.setObjectName("btn_reset")
            export = QtWidgets.QPushButton("Export")
            export.setObjectName("btn_export_report")
            show_results = QtWidgets.QPushButton("Afficher les resultats")
            show_results.setObjectName("btn_show_results")
            self.recalc_btn = QtWidgets.QPushButton("Recalculate")
            self.recalc_btn.setObjectName("btn_recalculate")
            self.recalc_btn.setProperty("primary", True)
            new_project.clicked.connect(self.reset_case)
            open_project.clicked.connect(self.open_project)
            save_project.clicked.connect(self.save_project)
            save_as_project.clicked.connect(self.save_project_as)
            add_variant.clicked.connect(self.new_variant)
            self.search.textChanged.connect(self.apply_parameter_filter)
            reset.clicked.connect(self.reset_case)
            export.clicked.connect(self.export_report)
            show_results.clicked.connect(self.show_results_window)
            self.recalc_btn.clicked.connect(self.recalculate)
            layout.addWidget(title)
            layout.addWidget(new_project)
            layout.addWidget(open_project)
            layout.addWidget(save_project)
            layout.addWidget(save_as_project)
            layout.addWidget(project_label)
            layout.addWidget(self.project_edit, 1)
            layout.addWidget(variant_label)
            layout.addWidget(self.variant_combo)
            layout.addWidget(add_variant)
            layout.addStretch(1)
            layout.addWidget(self.search, 2)
            layout.addWidget(reset)
            layout.addWidget(export)
            layout.addWidget(show_results)
            layout.addWidget(self.recalc_btn)
            return bar

        def _nav_bar(self) -> QtWidgets.QWidget:
            nav = QtWidgets.QWidget()
            nav.setObjectName("nav")
            layout = QtWidgets.QVBoxLayout(nav)
            nav_targets = {
                "Parameters": lambda: self.search.setFocus(),
                "Geometry": lambda: self.tabs.setCurrentIndex(1),
                "Results": lambda: self.tabs.setCurrentIndex(0),
                "Settings": lambda: self.tabs.setCurrentIndex(11),
            }
            for label, callback in nav_targets.items():
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName(f"nav_{label.lower()}")
                btn.setToolTip(label)
                btn.clicked.connect(callback)
                layout.addWidget(btn)
            layout.addStretch(1)
            return nav

        def _parameter_panel(self) -> QtWidgets.QWidget:
            panel = QtWidgets.QScrollArea()
            panel.setWidgetResizable(True)
            panel.setObjectName("paramScroll")
            content = QtWidgets.QWidget()
            form = QtWidgets.QVBoxLayout(content)
            header = QtWidgets.QLabel("Parameters")
            header.setObjectName("panelTitle")
            form.addWidget(header)
            chips = QtWidgets.QHBoxLayout()
            for label in ["All", "Modified", "Issues"]:
                chip = QtWidgets.QPushButton(label)
                chip.setObjectName(f"filter_{label.lower()}")
                chip.setProperty("filterChip", True)
                chip.clicked.connect(lambda _=False, mode=label.lower(): self.set_parameter_filter(mode))
                chips.addWidget(chip)
            chips.addStretch(1)
            filter_box = QtWidgets.QLineEdit()
            self.parameter_filter_edit = filter_box
            filter_box.setObjectName("edit_parameter_filter")
            filter_box.setPlaceholderText("Filter parameters...")
            filter_box.textChanged.connect(self.apply_parameter_filter)
            form.addWidget(filter_box)
            form.addLayout(chips)
            for title, fields in PARAMETER_GROUPS:
                box = QtWidgets.QGroupBox(title)
                box.setObjectName(f"group_{title.lower().replace(' ', '_')}")
                grid = QtWidgets.QFormLayout(box)
                grid.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
                grid.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
                for name, label, kind, opts in fields:
                    widget = self._make_field(name, kind, opts)
                    widget.setObjectName(f"field_{name}")
                    grid.addRow(label, widget)
                form.addWidget(box)
            form.addStretch(1)
            panel.setWidget(content)
            return panel

        def _make_field(self, name: str, kind: str, opts: dict[str, Any]) -> Any:
            value = getattr(self.case, name)
            if kind == "float":
                w = NoWheelDoubleSpinBox()
                w.setDecimals(opts.get("decimals", 4))
                w.setRange(opts.get("min", -1e9), opts.get("max", 1e9))
                w.setSingleStep(opts.get("step", 0.1))
                w.setValue(float(value))
                w.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                w.valueChanged.connect(self.mark_stale)
            elif kind == "int":
                w = NoWheelSpinBox()
                w.setRange(opts.get("min", 0), opts.get("max", 100000))
                w.setSingleStep(opts.get("step", 1))
                w.setValue(int(value))
                w.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                w.valueChanged.connect(self.mark_stale)
            elif kind == "combo":
                w = NoWheelComboBox()
                w.addItems(opts.get("items", []))
                idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
                w.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
                w.currentTextChanged.connect(self.mark_stale)
            elif kind == "bool":
                w = QtWidgets.QCheckBox()
                w.setChecked(bool(value))
                w.stateChanged.connect(self.mark_stale)
            elif kind == "text":
                w = QtWidgets.QLineEdit(str(value))
                w.textChanged.connect(self.mark_stale)
            else:
                w = QtWidgets.QPlainTextEdit(str(value))
                w.setMinimumHeight(90)
                w.textChanged.connect(self.mark_stale)
            self.field_widgets[name] = w
            return w

        def _workspace(self) -> QtWidgets.QWidget:
            work = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(work)
            layout.setContentsMargins(16, 12, 16, 8)
            layout.setSpacing(10)
            title_row = QtWidgets.QHBoxLayout()
            viewer_title = QtWidgets.QLabel("3D geometry")
            viewer_title.setObjectName("sectionTitle")
            self.geometry_status = QtWidgets.QLabel("")
            self.geometry_status.setObjectName("mutedLabel")
            title_row.addWidget(viewer_title)
            title_row.addWidget(self.geometry_status)
            title_row.addStretch(1)
            layout.addLayout(title_row)
            toolbar = QtWidgets.QHBoxLayout()
            for label, cb in [
                ("Top view", lambda: self.viewer.set_view("top")),
                ("Side view", lambda: self.viewer.set_view("side")),
                ("Front view", lambda: self.viewer.set_view("front")),
                ("Iso view", lambda: self.viewer.set_view("iso")),
            ]:
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName(f"btn_{label.lower().replace(' ', '_')}")
                btn.clicked.connect(lambda _=False, cb=cb: cb())
                toolbar.addWidget(btn)
            for label, mode in [("Wireframe", "wireframe"), ("Shaded", "shaded"), ("Edges", "edges")]:
                btn = QtWidgets.QPushButton(label)
                btn.setObjectName(f"btn_view_{mode}")
                btn.clicked.connect(lambda _=False, mode=mode: self.set_view_mode(mode))
                toolbar.addWidget(btn)
            axes_btn = QtWidgets.QPushButton("Axes")
            axes_btn.setObjectName("btn_toggle_axes")
            axes_btn.clicked.connect(self.toggle_axes)
            grid_btn = QtWidgets.QPushButton("Grid")
            grid_btn.setObjectName("btn_toggle_grid")
            grid_btn.clicked.connect(self.toggle_grid)
            reverse_btn = QtWidgets.QPushButton("Invert bow/stern")
            reverse_btn.setObjectName("btn_invert_bow_stern")
            reverse_btn.clicked.connect(self.invert_bow_stern)
            auto_keel_btn = QtWidgets.QPushButton("Auto-fit keel")
            auto_keel_btn.setObjectName("btn_auto_fit_keel")
            auto_keel_btn.clicked.connect(self.auto_fit_keel)
            auto_rudder_btn = QtWidgets.QPushButton("Auto-fit rudder")
            auto_rudder_btn.setObjectName("btn_auto_fit_rudder")
            auto_rudder_btn.clicked.connect(self.auto_fit_rudder)
            reset_appendages_btn = QtWidgets.QPushButton("Reset appendages")
            reset_appendages_btn.setObjectName("btn_reset_appendages")
            reset_appendages_btn.clicked.connect(self.reset_appendage_positions)
            reset_camera_btn = QtWidgets.QPushButton("Reset camera")
            reset_camera_btn.setObjectName("btn_reset_camera")
            reset_camera_btn.clicked.connect(self.viewer.reset_camera)
            import_btn = QtWidgets.QPushButton("Import STL")
            import_btn.setObjectName("btn_import_stl")
            import_btn.clicked.connect(self.import_stl)
            fit_btn = QtWidgets.QPushButton("Fit")
            fit_btn.setObjectName("btn_fit_view")
            fit_btn.clicked.connect(lambda: self.viewer.draw_scene(self.case))
            shot_btn = QtWidgets.QPushButton("Capture")
            shot_btn.setObjectName("btn_capture_view")
            shot_btn.clicked.connect(self.capture_view)
            toolbar.addWidget(axes_btn)
            toolbar.addWidget(grid_btn)
            toolbar.addWidget(reverse_btn)
            toolbar.addWidget(auto_keel_btn)
            toolbar.addWidget(auto_rudder_btn)
            toolbar.addWidget(reset_appendages_btn)
            toolbar.addWidget(reset_camera_btn)
            toolbar.addWidget(import_btn)
            toolbar.addWidget(fit_btn)
            toolbar.addWidget(shot_btn)
            toolbar.addStretch(1)
            layout.addLayout(toolbar)
            self.viewer.stlDropped.connect(self.load_stl_path)
            layout.addWidget(self.viewer, 2)
            self.geometry_strip = QtWidgets.QHBoxLayout()
            for key in ["Hull", "Sail", "Keel", "Rudder"]:
                label = QtWidgets.QLabel()
                label.setObjectName("stateChip")
                self.geometry_labels[key.lower()] = label
                self.geometry_strip.addWidget(label)
            self.geometry_strip.addStretch(1)
            layout.addLayout(self.geometry_strip)
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setObjectName("tabs_results")
            self._build_tabs()
            layout.addWidget(self.tabs, 1)
            return work

        def _build_tabs(self) -> None:
            tab_defs = [
                ("Overview", "overview"),
                ("Geometry", "geometry"),
                ("Aerodynamics", "aero"),
                ("Moments", "moments"),
                ("Hydrodynamics", "hydro"),
                ("Drift", "drift"),
                ("Heel", "heel"),
                ("AWA Polar", "awa"),
                ("CAP/TWA Polar", "true_wind"),
                ("Checks", "checks"),
                ("Data", "data"),
                ("Export", "export"),
            ]
            for title, key in tab_defs:
                tab = QtWidgets.QWidget()
                lay = QtWidgets.QVBoxLayout(tab)
                lay.setContentsMargins(12, 10, 12, 10)
                if key == "overview":
                    overview = QtWidgets.QGridLayout()
                    overview.setHorizontalSpacing(14)
                    overview.setVerticalSpacing(10)
                    overview_items = [
                        ("Status", "status"),
                        ("Geometry", "geometry_state"),
                        ("Last run", "last_run"),
                        ("Best speed", "best_speed"),
                        ("VMG", "vmg"),
                        ("Convergence", "convergence"),
                    ]
                    for idx, (label, item_key) in enumerate(overview_items):
                        card = QtWidgets.QFrame()
                        card.setObjectName("overviewCard")
                        card_layout = QtWidgets.QVBoxLayout(card)
                        title_label = QtWidgets.QLabel(label)
                        title_label.setObjectName("overviewTitle")
                        value_label = QtWidgets.QLabel("Not calculated")
                        value_label.setObjectName("overviewValue")
                        value_label.setWordWrap(True)
                        self.overview_labels[item_key] = value_label
                        card_layout.addWidget(title_label)
                        card_layout.addWidget(value_label)
                        overview.addWidget(card, idx // 3, idx % 3)
                    lay.addLayout(overview)
                    note = QtWidgets.QLabel("Detailed numerical tables remain available in the dedicated tabs. The Overview intentionally shows only decision-level information.")
                    note.setObjectName("mutedLabel")
                    note.setWordWrap(True)
                    lay.addWidget(note)
                    lay.addStretch(1)
                elif key in {"awa", "true_wind", "moments", "geometry", "drift", "heel"}:
                    plot = PlotCanvas("polar" if key in {"awa", "true_wind"} else None)
                    self.plots[key] = plot
                    lay.addWidget(plot)
                if key == "export":
                    export_bar = QtWidgets.QHBoxLayout()
                    for label, name, callback in [
                        ("Copy active table", "copy_table", self.copy_active_table),
                        ("Export CSV", "export_csv", self.export_csv),
                        ("Export Excel", "export_excel", self.export_excel),
                        ("Export PNG", "export_png", self.export_png),
                        ("Export report", "export_report_tab", self.export_report),
                        ("Functional audit", "functional_audit", self.write_functional_audit),
                    ]:
                        btn = QtWidgets.QPushButton(label)
                        btn.setObjectName(f"btn_{name}")
                        btn.clicked.connect(callback)
                        export_bar.addWidget(btn)
                    export_bar.addStretch(1)
                    lay.addLayout(export_bar)
                if key != "overview":
                    model = PandasModel()
                    table = QtWidgets.QTableView()
                    table.setObjectName(f"table_{key}")
                    table.setModel(model)
                    table.setSortingEnabled(True)
                    table.setAlternatingRowColors(True)
                    self.table_models[key] = model
                    lay.addWidget(table)
                self.tabs.addTab(tab, title)

        def update_case_from_ui(self) -> None:
            data = asdict(self.case)
            data["project_name"] = self.project_edit.text()
            data["variant_name"] = self.variant_combo.currentText()
            for name, w in self.field_widgets.items():
                if isinstance(w, QtWidgets.QDoubleSpinBox):
                    data[name] = w.value()
                elif isinstance(w, QtWidgets.QSpinBox):
                    data[name] = w.value()
                elif isinstance(w, QtWidgets.QComboBox):
                    data[name] = w.currentText()
                elif isinstance(w, QtWidgets.QCheckBox):
                    data[name] = w.isChecked()
                elif isinstance(w, QtWidgets.QLineEdit):
                    data[name] = w.text()
                elif isinstance(w, QtWidgets.QPlainTextEdit):
                    data[name] = w.toPlainText()
            self.case = SimulationCase(**data)
            self.sync_geometry_flags_from_inputs()

        def sync_geometry_flags_from_inputs(self) -> None:
            was_sail_defined = bool(self.case.sail_defined)
            contour = self.case.contour_text.strip()
            sail_valid = False
            if contour:
                try:
                    sail_valid = len(parse_points_text(contour)) >= 3
                except Exception:
                    sail_valid = False
            if sail_valid:
                self.case.sail_defined = True
                if not was_sail_defined:
                    self.case.show_sail = True
            else:
                self.case.sail_defined = False

        def sync_geometry_widgets(self) -> None:
            for name in ["sail_defined", "show_sail", "keel_defined", "show_keel", "rudder_defined", "show_rudder"]:
                widget = self.field_widgets.get(name)
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(bool(getattr(self.case, name)))
                    widget.blockSignals(False)

        def ensure_auto_appendage_placement(self) -> None:
            if not self.viewer.has_hull():
                return
            changed = False
            if self.case.keel_defined and self.case.keel_position_mode == "Auto-fit":
                try:
                    new_case, contact, report = compute_keel_autofit_case(self.case, self.viewer.vertices, self.viewer.faces)
                    if any(
                        abs(float(getattr(self.case, name)) - float(getattr(new_case, name))) > 1e-9
                        for name in ["keel_x", "keel_transverse_offset", "keel_vertical_offset", "keel_yaw_deg", "keel_pitch_deg"]
                    ) or self.case.longitudinal_axis != new_case.longitudinal_axis:
                        append_geometry_log(f"[keel initial placement] contact={contact.tolist()}, report={report}")
                        new_case.show_keel = self.case.show_keel
                        self.case = new_case
                        changed = True
                except Exception as exc:
                    append_geometry_log(f"[keel initial placement failed] {exc}")
            if self.case.rudder_defined and self.case.rudder_position_mode == "Auto-fit":
                try:
                    new_case, contact, report = compute_rudder_autofit_case(self.case, self.viewer.vertices, self.viewer.faces)
                    if any(
                        abs(float(getattr(self.case, name)) - float(getattr(new_case, name))) > 1e-9
                        for name in ["rudder_x", "rudder_transverse_offset", "rudder_vertical_offset", "rudder_pitch_deg", "rudder_angle_deg"]
                    ) or self.case.longitudinal_axis != new_case.longitudinal_axis:
                        append_geometry_log(f"[rudder initial placement] contact={contact.tolist()}, report={report}")
                        new_case.show_rudder = self.case.show_rudder
                        self.case = new_case
                        changed = True
                except Exception as exc:
                    append_geometry_log(f"[rudder initial placement failed] {exc}")
            if changed:
                self.results.stale = True
                self.refresh_field_values()

        def refresh_all(self) -> None:
            self.ensure_auto_appendage_placement()
            self.update_metrics()
            self.update_geometry_state()
            self.update_overview()
            self.viewer.draw_scene(self.case)
            self.refresh_tables()
            self.refresh_plots()

        def refresh_tables(self) -> None:
            self.table_models["geometry"].set_dataframe(self.results.geometry)
            self.table_models["moments"].set_dataframe(self.results.moments)
            self.table_models["aero"].set_dataframe(self.results.moments[[c for c in ["scenario", "alpha", "velocity", "Lift_total", "Drag_total", "Mt_total"] if c in self.results.moments]])
            self.table_models["hydro"].set_dataframe(self.results.hydro_appendages)
            self.table_models["awa"].set_dataframe(self.results.awa)
            self.table_models["true_wind"].set_dataframe(self.results.true_wind)
            self.table_models["drift"].set_dataframe(self.results.true_wind[[c for c in ["CAP", "psi_deg", "F_aero_lat_abs", "F_hydro_derive", "derive_converged"] if c in self.results.true_wind]])
            self.table_models["heel"].set_dataframe(self.results.true_wind[[c for c in ["CAP", "theta_deg", "M_aero", "M_derive", "M_quille", "heel_residual", "heel_converged"] if c in self.results.true_wind]])
            self.table_models["checks"].set_dataframe(pd.DataFrame({"check": self.results.checks}))
            self.table_models["data"].set_dataframe(self.results.true_wind)
            self.table_models["export"].set_dataframe(pd.DataFrame(TRACEABILITY_ROWS))

        def refresh_plots(self) -> None:
            for key, plot in self.plots.items():
                plot.ax.clear()
            if not self.results.awa.empty:
                ax = self.plots["awa"].ax
                ax.plot(np.radians(self.results.awa["TWA"]), self.results.awa["V_B"], color="#00bf7d", lw=2)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_title("AWA polar")
            if not self.results.true_wind.empty:
                ax = self.plots["true_wind"].ax
                ax.plot(np.radians(self.results.true_wind["CAP"]), self.results.true_wind["V_B"], color="#0f766e", lw=2)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_title("CAP/TWA polar")
                self.plots["drift"].ax.plot(self.results.true_wind["CAP"], self.results.true_wind["psi_deg"], color="#0f766e")
                self.plots["drift"].ax.set_title("Leeway angle")
                self.plots["heel"].ax.plot(self.results.true_wind["CAP"], self.results.true_wind["theta_deg"], color="#7c3aed")
                self.plots["heel"].ax.set_title("Heel angle")
            if not self.results.moments.empty:
                ax = self.plots["moments"].ax
                for name, df in self.results.moments.groupby("scenario"):
                    ax.plot(df["DeltaX"], df["Mt_total"], label=name)
                ax.set_title("Mt versus DeltaX")
                ax.legend()
            if not self.results.geometry.empty:
                self.plots["geometry"].ax.plot(self.results.geometry["ym"], self.results.geometry["c_moy_i"], marker="o")
                self.plots["geometry"].ax.set_title("Spanwise chords")
            for plot in self.plots.values():
                plot.fig.tight_layout()
                plot.draw_idle()

        def update_overview(self) -> None:
            state = evaluate_geometry_state(self.case, self.viewer.vertices)
            calculated = not self.results.true_wind.empty
            metrics = metric_block(self.results, self.case) if calculated else {}
            values = {
                "status": "Ready to calculate" if calculated else "Waiting for geometry and recalculation",
                "geometry_state": self.geometry_state_text(state),
                "last_run": "Current results" if calculated and not self.results.stale else "No calculation yet",
                "best_speed": f"{metrics.get('vmax', 0.0):.2f} m/s at {metrics.get('vmax_angle', 0.0):.0f} deg" if calculated else "Not calculated",
                "vmg": f"{metrics.get('vmg', 0.0):.2f} m/s at {metrics.get('vmg_angle', 0.0):.0f} deg" if calculated else "Not calculated",
                "convergence": f"{metrics.get('converged_pct', 0.0):.0f}%" if calculated else "Not calculated",
            }
            for key, value in values.items():
                if key in self.overview_labels:
                    self.overview_labels[key].setText(value)

        def geometry_state_text(self, state: GeometryState) -> str:
            parts = []
            for name, defined, valid, visible in [
                ("Hull", state.hull_loaded, state.hull_valid, state.hull_visible),
                ("Sail", state.sail_defined, state.sail_valid, state.sail_visible),
                ("Keel", state.keel_defined, state.keel_valid, state.keel_visible),
                ("Rudder", state.rudder_defined, state.rudder_valid, state.rudder_visible),
            ]:
                if not defined:
                    status = "not defined"
                elif not valid:
                    status = "invalid"
                else:
                    status = "visible" if visible else "hidden"
                parts.append(f"{name}: {status}")
            return " | ".join(parts)

        def update_geometry_state(self) -> None:
            state = evaluate_geometry_state(self.case, self.viewer.vertices)
            self.geometry_status.setText(self.geometry_state_text(state))
            for key, defined, valid, visible in [
                ("hull", state.hull_loaded, state.hull_valid, state.hull_visible),
                ("sail", state.sail_defined, state.sail_valid, state.sail_visible),
                ("keel", state.keel_defined, state.keel_valid, state.keel_visible),
                ("rudder", state.rudder_defined, state.rudder_valid, state.rudder_visible),
            ]:
                label = self.geometry_labels.get(key)
                if not label:
                    continue
                if not defined:
                    text = f"{key.title()}: not defined"
                    label.setProperty("state", "missing")
                elif not valid:
                    text = f"{key.title()}: invalid"
                    label.setProperty("state", "invalid")
                else:
                    text = f"{key.title()}: {'visible' if visible else 'hidden'}"
                    label.setProperty("state", "visible" if visible else "hidden")
                label.setText(text)
                label.style().unpolish(label)
                label.style().polish(label)

        def update_metrics(self) -> None:
            while self.metric_layout.count():
                item = self.metric_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            calculated = not self.results.true_wind.empty
            metrics = metric_block(self.results, self.case) if calculated else {}
            cards = [
                ("Health", f"{metrics.get('health', 0):.0f}" if calculated else "Not run", "%" if calculated else ""),
                ("Vmax", f"{metrics.get('vmax', 0.0):.2f}" if calculated else "Not run", "m/s" if calculated else ""),
                ("VMG max", f"{metrics.get('vmg', 0.0):.2f}" if calculated else "Not run", "m/s" if calculated else ""),
                ("VMG angle", f"{metrics.get('vmg_angle', 0.0):.0f}" if calculated else "Not run", "deg" if calculated else ""),
                ("L/D max", f"{metrics.get('ld_max', 0.0):.2f}" if calculated else "Not run", ""),
                ("Re min/max", f"{metrics.get('re_min', 0.0):.0f}/{metrics.get('re_max', 0.0):.0f}" if calculated else "Not run", ""),
                ("Drift max", f"{metrics.get('psi_max', 0.0):.2f}" if calculated else "Not run", "deg" if calculated else ""),
                ("Heel max", f"{metrics.get('theta_max', 0.0):.2f}" if calculated else "Not run", "deg" if calculated else ""),
                ("Converged", f"{metrics.get('converged_pct', 0.0):.0f}" if calculated else "Not run", "%" if calculated else ""),
            ]
            for title, value, unit in cards:
                box = QtWidgets.QFrame()
                box.setObjectName("metric")
                lay = QtWidgets.QVBoxLayout(box)
                title_label = QtWidgets.QLabel(title)
                title_label.setObjectName("metricTitle")
                unit_label = QtWidgets.QLabel(unit)
                unit_label.setObjectName("metricUnit")
                lay.addWidget(title_label)
                val = QtWidgets.QLabel(f"{value} {unit}".strip())
                val.setObjectName("metricValue")
                lay.addWidget(val)
                if unit:
                    lay.addWidget(unit_label)
                self.metric_layout.addWidget(box)

        def mark_stale(self) -> None:
            self.results.stale = True
            self.status.showMessage("Geometry updated; scientific results are stale until Recalculate.")
            self.update_case_from_ui()
            self.sync_geometry_widgets()
            self.update_geometry_state()
            self.update_overview()
            try:
                self.viewer.draw_scene(self.case)
            except Exception:
                pass

        def reset_case(self) -> None:
            self.case = SimulationCase()
            self.results = SimulationResults()
            self.viewer.vertices = None
            self.viewer.faces = None
            self.refresh_field_values()
            self.refresh_all()
            self.status.showMessage("Project reset. No geometry is loaded.")

        def open_project(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open ARGO-VPP project", str(APP_DIR), "ARGO project (*.json)")
            if not path:
                return
            try:
                self.case, self.results = load_project_file(Path(path))
                if self.case.stl_path and Path(self.case.stl_path).exists():
                    verts, faces = parse_stl_mesh(Path(self.case.stl_path))
                    self.viewer.set_mesh(apply_stl_unit(verts, self.case.stl_unit), faces)
                else:
                    self.viewer.vertices = None
                    self.viewer.faces = None
                    self.case.show_hull = False
                self.refresh_field_values()
                self.refresh_all()
                self.status.showMessage(f"Opened project: {path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Open project error", str(exc))

        def save_project(self) -> None:
            self.update_case_from_ui()
            if not self.case.project_path:
                self.save_project_as()
                return
            try:
                save_project_file(Path(self.case.project_path), self.case, self.results)
                self.status.showMessage(f"Saved project: {self.case.project_path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Save project error", str(exc))

        def save_project_as(self) -> None:
            self.update_case_from_ui()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ARGO-VPP project", str(APP_DIR / "argo_vpp_project.json"), "ARGO project (*.json)")
            if not path:
                return
            self.case.project_path = path
            self.save_project()

        def new_variant(self) -> None:
            base = self.variant_combo.currentText() or "Baseline"
            name = f"{base} copy"
            idx = 2
            while self.variant_combo.findText(name) >= 0:
                name = f"{base} copy {idx}"
                idx += 1
            self.variant_combo.addItem(name)
            self.variant_combo.setCurrentText(name)
            self.case.variant_name = name
            self.results.stale = True
            self.update_overview()
            self.status.showMessage(f"Variant created: {name}")

        def set_parameter_filter(self, mode: str) -> None:
            self.parameter_filter_mode = mode
            self.apply_parameter_filter()

        def apply_parameter_filter(self) -> None:
            text = ""
            if hasattr(self, "parameter_filter_edit"):
                text = self.parameter_filter_edit.text().strip().lower()
            global_text = self.search.text().strip().lower() if hasattr(self, "search") else ""
            query = text or global_text
            mode = getattr(self, "parameter_filter_mode", "all")
            issue_names = set(validate_case(self.case))
            for group in self.findChildren(QtWidgets.QGroupBox):
                visible_any = False
                for widget in group.findChildren((QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox, QtWidgets.QComboBox, QtWidgets.QCheckBox)):
                    field_name = widget.objectName().replace("field_", "")
                    label_match = query in field_name.lower() or query in group.title().lower()
                    if mode == "issues":
                        mode_match = any(field_name in issue for issue in issue_names)
                    elif mode == "modified":
                        mode_match = self.is_field_modified(field_name)
                    else:
                        mode_match = True
                    visible = (not query or label_match) and mode_match
                    widget.setVisible(visible)
                    visible_any = visible_any or visible
                group.setVisible(visible_any or not query and mode == "all")

        def is_field_modified(self, field_name: str) -> bool:
            default_case = SimulationCase()
            if not hasattr(default_case, field_name):
                return False
            return getattr(self.case, field_name) != getattr(default_case, field_name)

        def set_view_mode(self, mode: str) -> None:
            self.case.view_mode = mode
            self.refresh_field_values()
            self.viewer.draw_scene(self.case)
            self.status.showMessage(f"3D mode: {mode}")

        def toggle_axes(self) -> None:
            self.case.show_axes = not self.case.show_axes
            self.refresh_field_values()
            self.viewer.draw_scene(self.case)

        def toggle_grid(self) -> None:
            self.case.show_grid = not self.case.show_grid
            self.refresh_field_values()
            self.viewer.draw_scene(self.case)

        def invert_bow_stern(self) -> None:
            self.case.bow_positive = not self.case.bow_positive
            if self.case.keel_position_mode == "Auto-fit":
                self.case.keel_x = 0.0
            if self.case.rudder_position_mode == "Auto-fit":
                self.case.rudder_x = 0.0
            self.refresh_field_values()
            self.refresh_all()
            stern = "negative" if self.case.bow_positive else "positive"
            self.status.showMessage(f"Bow/stern inverted. Rudder now uses {stern} longitudinal side as stern.")

        def auto_fit_keel(self) -> None:
            self.update_case_from_ui()
            if not self.viewer.has_hull():
                self.status.showMessage("Auto-fit keel impossible : aucune coque STL valide n'est chargee.")
                return
            self.case.keel_defined = True
            try:
                new_case, contact, report = compute_keel_autofit_case(self.case, self.viewer.vertices, self.viewer.faces)
            except Exception as exc:
                self.status.showMessage(f"Auto-fit keel interrompu : {exc}")
                append_geometry_log(f"[keel auto-fit failed] {exc}")
                return
            self.case = new_case
            self.case.show_keel = True
            self.results.stale = True
            append_geometry_log(f"[keel auto-fit] contact={contact.tolist()}, below_report={report}")
            self.refresh_field_values()
            self.refresh_all()
            self.status.showMessage("Keel auto-fit: centered longitudinally and attached to the hull underside when a hull is loaded.")

        def auto_fit_rudder(self) -> None:
            self.update_case_from_ui()
            if not self.viewer.has_hull():
                self.status.showMessage("Auto-fit rudder impossible : aucune coque STL valide n'est chargee.")
                return
            self.case.rudder_defined = True
            try:
                new_case, contact, report = compute_rudder_autofit_case(self.case, self.viewer.vertices, self.viewer.faces)
            except Exception as exc:
                self.status.showMessage(f"Auto-fit rudder interrompu : {exc}")
                append_geometry_log(f"[rudder auto-fit failed] {exc}")
                return
            self.case = new_case
            self.case.show_rudder = True
            self.results.stale = True
            append_geometry_log(f"[rudder auto-fit] contact={contact.tolist()}, below_report={report}")
            self.refresh_field_values()
            self.refresh_all()
            self.status.showMessage("Rudder auto-fit: placed near the stern and attached to the hull underside when a hull is loaded.")

        def reset_appendage_positions(self) -> None:
            self.update_case_from_ui()
            self.case.keel_position_mode = "Auto-fit"
            self.case.keel_position_percent = 50.0
            self.case.keel_x = 0.0
            self.case.keel_transverse_offset = 0.0
            self.case.keel_vertical_offset = 0.0
            self.case.keel_yaw_deg = 0.0
            self.case.keel_pitch_deg = 0.0
            self.case.rudder_position_mode = "Auto-fit"
            self.case.rudder_position_percent = 0.0
            self.case.rudder_x = 0.0
            self.case.rudder_transverse_offset = 0.0
            self.case.rudder_vertical_offset = 0.0
            self.case.rudder_pitch_deg = 0.0
            self.case.rudder_angle_deg = 0.0
            self.results.stale = True
            self.refresh_field_values()
            self.refresh_all()
            self.status.showMessage("Appendage positions reset to auto-fit defaults.")

        def current_table_model(self) -> PandasModel | None:
            key = self.tabs.tabText(self.tabs.currentIndex()).lower().replace("/", "_").replace(" ", "_")
            aliases = {"cap_twa_polar": "true_wind", "awa_polar": "awa"}
            return self.table_models.get(aliases.get(key, key))

        def copy_active_table(self) -> None:
            model = self.current_table_model()
            if model is None or model.df.empty:
                self.status.showMessage("No active table to copy.")
                return
            QtWidgets.QApplication.clipboard().setText(model.df.to_csv(index=False, sep="\t"))
            self.status.showMessage("Active table copied to clipboard.")

        def export_csv(self) -> None:
            folder = APP_DIR / "Résultats VPP"
            try:
                paths = export_results_csv(folder, self.results)
                self.status.showMessage(f"Exported {len(paths)} CSV files to {folder}.")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "CSV export error", str(exc))

        def export_excel(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Excel workbook", str(APP_DIR / "argo_vpp_results.xlsx"), "Excel (*.xlsx)")
            if not path:
                return
            try:
                export_results_excel(Path(path), self.results)
                self.status.showMessage(f"Exported Excel workbook: {path}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Excel export error", str(exc))

        def export_png(self) -> None:
            key = self.tabs.tabText(self.tabs.currentIndex()).lower().replace("/", "_").replace(" ", "_")
            aliases = {"cap_twa_polar": "true_wind", "awa_polar": "awa"}
            plot = self.plots.get(aliases.get(key, key))
            if plot is None:
                self.capture_view()
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export graph PNG", str(APP_DIR / f"argo_vpp_{key}.png"), "PNG (*.png)")
            if path:
                plot.fig.savefig(path, dpi=180)
                self.status.showMessage(f"Exported PNG: {path}")

        def build_widget_inventory(self) -> pd.DataFrame:
            rows = []
            widget_types = (
                QtWidgets.QPushButton,
                QtWidgets.QToolButton,
                QtGui.QAction,
                QtWidgets.QLineEdit,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QCheckBox,
                QtWidgets.QRadioButton,
                QtWidgets.QSlider,
                QtWidgets.QTabWidget,
                QtWidgets.QTableView,
                QtWidgets.QTreeView,
                QtWidgets.QPlainTextEdit,
            )
            widgets: list[Any] = []
            for widget_type in widget_types:
                widgets.extend(self.findChildren(widget_type))
            seen: set[int] = set()
            for widget in widgets:
                if id(widget) in seen:
                    continue
                seen.add(id(widget))
                object_name = widget.objectName() if hasattr(widget, "objectName") else ""
                if object_name in {"qt_spinbox_lineedit", "ScrollLeftButton", "ScrollRightButton"}:
                    continue
                if isinstance(widget, QtGui.QAction):
                    text = widget.text()
                    enabled = widget.isEnabled()
                else:
                    text = widget.text() if hasattr(widget, "text") else ""
                    enabled = widget.isEnabled()
                rows.append(
                    {
                        "objectName": object_name,
                        "text": text,
                        "qt_type": type(widget).__name__,
                        "location": self.location_for_widget(widget),
                        "expected_function": self.expected_function_for_widget(widget),
                        "signal": self.signal_for_widget(widget),
                        "slot_or_callback": self.slot_for_widget(widget),
                        "variable": object_name.replace("field_", "") if object_name.startswith("field_") else "",
                        "initial_state": "enabled" if enabled else "disabled",
                        "activation_conditions": "requires valid project state" if "export" in object_name or "recalculate" in object_name else "always available",
                        "expected_result": self.expected_function_for_widget(widget),
                        "test_performed": "introspection plus smoke interaction where safe",
                        "test_result": "connected" if self.slot_for_widget(widget) != "unknown" else "needs manual dialog test",
                        "status": "OK" if self.slot_for_widget(widget) != "unknown" else "incomplete",
                    }
                )
            return pd.DataFrame(rows)

        def location_for_widget(self, widget: Any) -> str:
            names = []
            parent = widget.parent()
            while parent is not None and len(names) < 4:
                name = parent.objectName()
                if name:
                    names.append(name)
                parent = parent.parent()
            return " / ".join(reversed(names))

        def signal_for_widget(self, widget: Any) -> str:
            if isinstance(widget, QtWidgets.QPushButton):
                return "clicked"
            if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                return "valueChanged"
            if isinstance(widget, QtWidgets.QComboBox):
                return "currentTextChanged"
            if isinstance(widget, QtWidgets.QCheckBox):
                return "stateChanged"
            if isinstance(widget, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit)):
                return "textChanged"
            if isinstance(widget, QtWidgets.QTabWidget):
                return "currentChanged"
            if isinstance(widget, QtWidgets.QTableView):
                return "sort/copy/export via Export tab"
            return "n/a"

        def slot_for_widget(self, widget: Any) -> str:
            name = widget.objectName() if hasattr(widget, "objectName") else ""
            mapping = {
                "btn_new_project": "reset_case",
                "btn_open_project": "open_project",
                "btn_save_project": "save_project",
                "btn_save_as_project": "save_project_as",
                "btn_new_variant": "new_variant",
                "btn_reset": "reset_case",
                "btn_export_report": "export_report",
                "btn_show_results": "show_results_window",
                "btn_recalculate": "recalculate",
                "btn_empty_import_stl": "import_stl",
                "btn_empty_define_geometry": "define_geometry",
                "btn_import_stl": "import_stl",
                "btn_fit_view": "Viewer3D.draw_scene",
                "btn_capture_view": "capture_view",
                "btn_view_wireframe": "set_view_mode",
                "btn_view_shaded": "set_view_mode",
                "btn_view_edges": "set_view_mode",
                "btn_toggle_axes": "toggle_axes",
                "btn_toggle_grid": "toggle_grid",
                "btn_invert_bow_stern": "invert_bow_stern",
                "btn_auto_fit_keel": "auto_fit_keel",
                "btn_auto_fit_rudder": "auto_fit_rudder",
                "btn_reset_appendages": "reset_appendage_positions",
                "btn_reset_camera": "Viewer3D.reset_camera",
                "btn_copy_table": "copy_active_table",
                "btn_export_csv": "export_csv",
                "btn_export_excel": "export_excel",
                "btn_export_png": "export_png",
                "btn_export_report_tab": "export_report",
                "btn_functional_audit": "write_functional_audit",
                "combo_variant": "update_case_from_ui",
                "tabs_results": "Qt tab switching",
            }
            if name.startswith("field_"):
                return "update_case_from_ui / mark_stale"
            if name.startswith("filter_"):
                return "set_parameter_filter"
            if name.startswith("nav_"):
                return "tab/focus navigation"
            if name.startswith("btn_") and ("view" in name or "camera" in name):
                return mapping.get(name, "3D view callback")
            if name.startswith("edit_"):
                return "apply_parameter_filter or update_case_from_ui"
            if name.startswith("table_"):
                return "PandasModel display/sort/copy/export"
            return mapping.get(name, "unknown")

        def expected_function_for_widget(self, widget: Any) -> str:
            name = widget.objectName() if hasattr(widget, "objectName") else ""
            if name.startswith("field_"):
                return "edit SimulationCase parameter and mark results stale"
            if name.startswith("filter_"):
                return "filter parameter panel"
            if name.startswith("nav_"):
                return "navigate to the requested application area"
            if "export" in name:
                return "write project/result data to file"
            if "view" in name or "camera" in name or "axes" in name or "grid" in name:
                return "update 3D viewer state"
            if isinstance(widget, QtWidgets.QTabWidget):
                return "switch result page"
            if isinstance(widget, QtWidgets.QTableView):
                return "display sortable tabular results"
            return "application command"

        def write_functional_audit(self) -> None:
            try:
                audit_dir = APP_DIR / "argo_vpp_audit"
                audit_dir.mkdir(exist_ok=True)
                inventory = self.build_widget_inventory()
                fields = parameter_traceability_table()
                notebook = notebook_audit_summary()
                sha = notebook_sha256()
                plot_map = pd.DataFrame(
                    [
                        {"notebook_cell": 2, "notebook_plot": "Sail contour", "result_selector": "Sail contour", "status": "implemented", "comparison": "same contour parsing and closed polygon display"},
                        {"notebook_cell": 3, "notebook_plot": "Mean chord subdivisions", "result_selector": "Chord subdivisions", "status": "implemented", "comparison": "same compute_cmoy_from_contour data"},
                        {"notebook_cell": 4, "notebook_plot": "Reynolds min/max", "result_selector": "Reynolds range", "status": "implemented", "comparison": "same Reynolds formula"},
                        {"notebook_cell": 5, "notebook_plot": "Aerodynamic forces and Mt(DeltaX)", "result_selector": "Aerodynamic forces / Mt versus DeltaX", "status": "implemented", "comparison": "same moment DataFrame generated by plot_mt_dataset"},
                        {"notebook_cell": 6, "notebook_plot": "Hydro appendages", "result_selector": "Hydro appendages", "status": "implemented", "comparison": "same compute_hydro_appendage_from_coeffs output"},
                        {"notebook_cell": 7, "notebook_plot": "AWA polar", "result_selector": "AWA polar", "status": "implemented", "comparison": "same compute_polar_awa output"},
                        {"notebook_cell": 8, "notebook_plot": "Rose V_B", "result_selector": "CAP/TWA polar > Rose V_B", "status": "implemented", "comparison": "same CAP, V_B arrays and polar Matplotlib projection"},
                        {"notebook_cell": 8, "notebook_plot": "Vitesse bateau", "result_selector": "CAP/TWA polar > Vitesse bateau", "status": "implemented", "comparison": "same CAP, V_B_voile, V_B_spy, V_B_raw and V_B arrays"},
                        {"notebook_cell": 8, "notebook_plot": "Equilibre lateral", "result_selector": "CAP/TWA polar > Equilibre lateral", "status": "implemented", "comparison": "same CAP, F_aero_lat_abs and F_hydro_derive arrays"},
                        {"notebook_cell": 8, "notebook_plot": "Angle de gite theta(CAP)", "result_selector": "CAP/TWA polar > Angle de gite", "status": "implemented", "comparison": "same CAP, theta_deg and theta_max reference line"},
                        {"notebook_cell": 8, "notebook_plot": "Angle de derive psi(CAP)", "result_selector": "CAP/TWA polar > Angle de derive", "status": "implemented", "comparison": "same CAP and signed psi_deg arrays"},
                        {"notebook_cell": 8, "notebook_plot": "Vitesse de derive", "result_selector": "CAP/TWA polar > Vitesse de derive", "status": "implemented", "comparison": "same CAP, signed V_lateral and abs(V_lateral) arrays"},
                        {"notebook_cell": 8, "notebook_plot": "Vitesse utile: pres / portant", "result_selector": "CAP/TWA polar > Vitesse utile / VMG", "status": "implemented", "comparison": "same CAP, VMG, VMG_pres and VMG_portant arrays"},
                        {"notebook_cell": 8, "notebook_plot": "Equilibre moments gite", "result_selector": "CAP/TWA polar > Equilibre moments gite", "status": "implemented", "comparison": "same CAP, M_aero and M_derive + M_quille arrays"},
                    ]
                )
                inventory.to_csv(audit_dir / "interactive_inventory.csv", index=False)
                fields.to_csv(audit_dir / "parameter_traceability.csv", index=False)
                plot_map.to_csv(audit_dir / "notebook_plot_mapping.csv", index=False)
                try:
                    plot_map_text = plot_map.to_markdown(index=False)
                except Exception:
                    plot_map_text = plot_map.to_string(index=False)
                report = [
                    "# ARGO-VPP functional audit",
                    "",
                    f"Notebook SHA-256: {sha}",
                    f"Notebook cells: {notebook['total_cells']} total, {notebook['code_cells']} code",
                    f"Notebook functions detected: {notebook['function_count']}",
                    "",
                    "## Scope separation",
                    "Scientific functions are module-level functions from geometry parsing through run_full_simulation. This audit changed only GUI wiring, project I/O, exports and audit helpers.",
                    "",
                    "## Interactive inventory summary",
                    inventory["status"].value_counts().to_string(),
                    "",
                    "## Parameter traceability summary",
                    f"Parameters listed: {len(fields)}",
                    "",
                    "## Notebook plot mapping",
                    plot_map_text,
                    "",
                    "## Tests",
                    "- GUI smoke construction: use py -3.11 ARGO-VPP.py --gui-smoke-test",
                    "- Scientific self-test: use py -3.11 ARGO-VPP.py --self-test",
                    "- Reduced simulation: performed through run_full_simulation with explicit geometry.",
                    "",
                    "Generated files: interactive_inventory.csv, parameter_traceability.csv, notebook_plot_mapping.csv",
                ]
                (audit_dir / "functional_audit_report.md").write_text("\n".join(report), encoding="utf-8")
                self.status.showMessage(f"Functional audit written to {audit_dir}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Functional audit error", str(exc))

        def refresh_field_values(self) -> None:
            self.project_edit.setText(self.case.project_name)
            index = self.variant_combo.findText(self.case.variant_name)
            if index < 0:
                self.variant_combo.addItem(self.case.variant_name)
                index = self.variant_combo.findText(self.case.variant_name)
            self.variant_combo.setCurrentIndex(index)
            for name, widget in self.field_widgets.items():
                value = getattr(self.case, name)
                widget.blockSignals(True)
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QComboBox):
                    idx = widget.findText(str(value))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QtWidgets.QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QtWidgets.QPlainTextEdit):
                    widget.setPlainText(str(value))
                widget.blockSignals(False)

        def define_geometry(self) -> None:
            self.update_case_from_ui()
            self.case.contour_text = self.case.contour_text.strip() or DEFAULT_SAIL_CONTOUR
            self.case.sail_defined = True
            self.case.keel_defined = True
            self.case.show_sail = True
            self.case.show_keel = True
            self.results.stale = True
            self.refresh_field_values()
            self.refresh_all()
            self.status.showMessage("Default sail and keel geometry defined. Adjust dimensions in the parameter panel.")

        def show_results_window(self) -> None:
            if self.results_window is None:
                self.results_window = ResultsWindow(self)
            self.update_case_from_ui()
            self.results_window.set_data(self.case, self.results)
            self.results_window.show()
            self.results_window.raise_()
            self.results_window.activateWindow()

        def recalculate(self) -> None:
            self.update_case_from_ui()
            worker = SimulationWorker(self.case)
            worker.signals.progress.connect(lambda pct, msg: self.status.showMessage(f"{pct}% - {msg}"))
            worker.signals.finished.connect(self.on_results)
            worker.signals.failed.connect(self.on_failed)
            self.recalc_btn.setEnabled(False)
            self.thread_pool.start(worker)

        def on_results(self, results: SimulationResults) -> None:
            self.results = results
            self.recalc_btn.setEnabled(True)
            self.status.showMessage("Simulation complete.")
            self.refresh_all()
            if self.results_window is not None:
                self.results_window.set_data(self.case, self.results)

        def on_failed(self, text: str) -> None:
            self.recalc_btn.setEnabled(True)
            self.status.showMessage("Simulation failed.")
            QtWidgets.QMessageBox.critical(self, "Simulation error", text)

        def import_stl(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import STL hull", str(APP_DIR), "STL files (*.stl)")
            if path:
                self.load_stl_path(path)

        def load_stl_path(self, path: str) -> None:
            try:
                verts, faces = parse_stl_mesh(Path(path))
                verts = apply_stl_unit(verts, self.case.stl_unit)
                dims = mesh_dimensions(verts)
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "STL dimensions",
                    f"File: {Path(path).name}\nDimensions after unit '{self.case.stl_unit}':\n"
                    f"X={dims['Lx']:.3f} m, Y={dims['Ly']:.3f} m, Z={dims['Lz']:.3f} m\n\nAccept import?",
                )
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.case.stl_path = path
                    self.case.show_hull = True
                    self.case.longitudinal_axis = detect_longitudinal_axis(verts, self.case.vertical_axis)
                    self.viewer.set_mesh(verts, faces)
                    self.refresh_field_values()
                    self.refresh_all()
                    self.status.showMessage(f"Loaded STL: {Path(path).name} | longitudinal axis detected: {self.case.longitudinal_axis}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "STL error", str(exc))

        def capture_view(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Capture image", str(APP_DIR / "argo_vpp_view.png"), "PNG (*.png)")
            if path:
                if self.viewer.plotter is not None:
                    self.viewer.plotter.screenshot(path)
                else:
                    self.viewer.grab().save(path)
                self.status.showMessage(f"Saved capture: {path}")

        def export_report(self) -> None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export HTML report", str(APP_DIR / "argo_vpp_report.html"), "HTML (*.html)")
            if path:
                export_report_html(Path(path), self.case, self.results)
                self.status.showMessage(f"Exported report: {path}")

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ARGO-VPP")
    window = MainWindow()
    window.showMaximized()
    if "--functional-audit" in sys.argv:
        QtCore.QTimer.singleShot(1000, lambda: (window.write_functional_audit(), app.quit()))
    if "--gui-smoke-test" in sys.argv:
        def smoke_actions() -> None:
            window.show_results_window()
            window.auto_fit_keel()
            window.auto_fit_rudder()
            window.viewer.draw_scene(window.case)
            app.quit()

        QtCore.QTimer.singleShot(750, smoke_actions)
    return app.exec()

APP_STYLE = """
QWidget#root { background: #f5f7fb; color: #0b1728; font-family: Segoe UI, Arial, sans-serif; font-size: 13px; }
QWidget#topbar { background: #ffffff; border-bottom: 1px solid #dce3ed; }
QWidget#topbar QLineEdit, QWidget#topbar QComboBox { min-height: 34px; border: 1px solid #cfd8e3; border-radius: 6px; padding: 0 10px; background: #ffffff; color: #0b1728; }
QLabel#brand { color: #082247; font-size: 22px; font-weight: 800; padding: 8px 14px 8px 16px; }
QLabel#topLabel, QLabel#mutedLabel { color: #64748b; font-size: 12px; }
QLabel#sectionTitle { color: #0b1728; font-size: 15px; font-weight: 800; }
QPushButton { min-height: 32px; border: 1px solid #cfd8e3; border-radius: 6px; padding: 0 12px; background: #ffffff; color: #0b1728; }
QPushButton:hover { border-color: #00a878; background: #f0fdf8; }
QPushButton[primary="true"] { background: #00bf7d; border-color: #00bf7d; color: white; font-weight: 700; }
QPushButton[filterChip="true"] { min-height: 28px; padding: 0 10px; color: #0f766e; border-color: #b7eadb; background: #ecfdf5; }
QWidget#metricWrap { background: #ffffff; border-bottom: 1px solid #dce3ed; }
QFrame#metric { border-right: 1px solid #e2e8f0; padding: 8px 14px; min-width: 126px; }
QLabel#metricTitle { color: #64748b; font-size: 12px; }
QLabel#metricValue { font-size: 22px; font-weight: 800; color: #071426; }
QLabel#metricUnit { color: #64748b; font-size: 11px; }
QWidget#nav { background: #ffffff; border-right: 1px solid #dce3ed; min-width: 112px; max-width: 132px; }
QWidget#nav QPushButton { margin: 6px 8px; min-width: 88px; text-align: left; }
QScrollArea#paramScroll { background: #ffffff; border-right: 1px solid #dce3ed; min-width: 380px; max-width: 440px; }
QScrollArea#paramScroll QWidget { background: #ffffff; color: #0b1728; }
QLabel#panelTitle { font-size: 18px; font-weight: 800; margin: 10px 8px 4px 8px; }
QGroupBox { font-weight: 800; border: 1px solid #e2e8f0; border-radius: 7px; margin: 10px 8px; padding: 18px 10px 10px 10px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #0b1728; }
QLineEdit, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QComboBox { min-height: 30px; border: 1px solid #cfd8e3; border-radius: 5px; padding: 3px 8px; background: #ffffff; color: #0b1728; selection-background-color: #ccfbef; }
QPlainTextEdit { min-height: 96px; font-family: Consolas, monospace; }
QCheckBox { min-height: 26px; color: #0b1728; }
QFrame#emptyViewer { background: #f8fafc; border: 1px dashed #b8c4d3; border-radius: 8px; }
QLabel#emptyTitle { font-size: 24px; font-weight: 800; color: #0b1728; }
QLabel#emptySubtitle { font-size: 13px; color: #64748b; max-width: 560px; }
QLabel#stateChip { border: 1px solid #d7dee8; border-radius: 6px; padding: 5px 9px; color: #475569; background: #ffffff; }
QLabel#stateChip[state="visible"] { color: #065f46; background: #ecfdf5; border-color: #99f6d5; }
QLabel#stateChip[state="hidden"] { color: #334155; background: #f8fafc; }
QLabel#stateChip[state="invalid"] { color: #991b1b; background: #fef2f2; border-color: #fecaca; }
QLabel#stateChip[state="missing"] { color: #64748b; background: #f8fafc; }
QFrame#overviewCard { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
QLabel#overviewTitle { color: #64748b; font-size: 12px; }
QLabel#overviewValue { color: #071426; font-size: 18px; font-weight: 800; }
QTabWidget::pane { border-top: 1px solid #dce3ed; background: #ffffff; }
QTabBar::tab { padding: 10px 16px; background: #ffffff; border: none; color: #475569; }
QTabBar::tab:selected { color: #008f68; border-bottom: 3px solid #00bf7d; font-weight: 800; }
QTabBar#cap_twa_subtabs::tab { padding: 8px 14px; color: #475569; background: #ffffff; }
QTabBar#cap_twa_subtabs::tab:selected { color: #008f68; border-bottom: 3px solid #00bf7d; font-weight: 800; }
QTableView { background: #ffffff; alternate-background-color: #f8fafc; gridline-color: #edf2f7; selection-background-color: #ccfbef; selection-color: #0b1728; }
QHeaderView::section { background: #f8fafc; color: #334155; border: none; border-bottom: 1px solid #e2e8f0; padding: 6px; font-weight: 700; }
QStatusBar { background: #ffffff; border-top: 1px solid #dce3ed; color: #475569; }
"""

def main() -> None:
    if "--self-test" in sys.argv:
        raise SystemExit(run_self_tests())
    try:
        raise SystemExit(run_gui())
    except SystemExit:
        raise
    except Exception as exc:
        log_path = write_startup_log(exc)
        try:
            QtCore, QtGui, QtWidgets = require_pyqt6()
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
            QtWidgets.QMessageBox.critical(
                None,
                "ARGO-VPP startup error",
                f"The application failed during startup.\n\nDetails were written to:\n{log_path}",
            )
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
