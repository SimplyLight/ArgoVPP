"""Compatibility entry point for the modular ARGO-VPP application."""
from __future__ import annotations

from fonctions.argo_vpp_models import *  # noqa: F401,F403
from fonctions.argo_vpp_settings import *  # noqa: F401,F403
from fonctions.argo_vpp_geometry import *  # noqa: F401,F403
from fonctions.argo_vpp_aerodynamics import *  # noqa: F401,F403
from fonctions.argo_vpp_hydrodynamics import *  # noqa: F401,F403
from fonctions.argo_vpp_solver_awa import *  # noqa: F401,F403
from fonctions.argo_vpp_solver_true_wind import *  # noqa: F401,F403
from fonctions.argo_vpp_stl_geometry import *  # noqa: F401,F403
from fonctions.argo_vpp_simulation_export import *  # noqa: F401,F403
from fonctions.argo_vpp_audit import *  # noqa: F401,F403
from fonctions.argo_vpp_gui import main, run_gui  # noqa: F401


if __name__ == "__main__":
    main()
