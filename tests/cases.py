"""ICs, BCs and analytic comparisons — content, not tooling.

Two amrex-specific constraints the jax twin does not have, both measured:

1. BOUNDARY TAG NAMES. The driver maps its structured faces to model tag names
   via bc_sides, hardcoded to West/East/South/North (run_case.py:253). A model
   declaring core's usual left/right/top/bottom therefore matched nothing and,
   until 2026-07-20, silently resolved every side to tag index 0 — wrong BCs,
   green run. That fallback now aborts (ZoomyAmr.cpp tag_index), but the tags
   still have to BE these names, so bcs_for emits them.

   This is a genuine divergence from the jax suite's bcs_for, and it belongs in
   the blueprint compliance audit as data-structure/platform, not physics: the
   same four sides get the same four conditions, only the labels differ.

2. BCs MUST BE SYMPY. amrex emits boundary conditions as generated C++, so a BC
   built from numpy or a Python conditional raises at codegen time. ICs are
   different — they are SAMPLED in Python into .raw rasters (run_case.py:98-115)
   and so may be arbitrary callables, which is why IC.UserFunction ports over
   unchanged.
"""
import numpy as np

# The driver's face -> tag mapping, hardcoded at run_case.py:253.
TAGS_2D = ("West", "East", "South", "North")
TAGS_1D = ("West", "East")


def bcs_for(kind, dimension):
    """Boundary conditions on the amrex tag names.

    ``dimension`` follows the MODEL convention (SME/VAM count the vertical), so
    dimension=3 is two horizontal directions and dimension=2 is one.
    """
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, Extrapolation, FromModel, Periodic)

    tags = TAGS_2D if dimension == 3 else TAGS_1D

    if kind == "periodic":
        return BoundaryConditions([Periodic(tag=t) for t in tags])
    if kind == "wall":
        return BoundaryConditions(
            [FromModel(tag=t, definition="wall") for t in tags])
    if kind in ("extrapolation", "swashes"):
        # swashes: the analytic solutions are posed on an unbounded domain and
        # compared before any signal reaches the ends, so extrapolation is the
        # honest choice — a wall would reflect and contaminate the L1 error.
        return BoundaryConditions([Extrapolation(tag=t) for t in tags])
    raise ValueError(
        f"unknown BC kind {kind!r}. Add it here explicitly rather than letting "
        "it fall through to a default — a wrong-but-plausible BC is the one "
        "error a converged-looking result will not reveal.")


# ── initial conditions (sampled in Python, so arbitrary callables are fine) ──

def stoker_ic(x):
    """SWASHES wet dam break: h_l = 0.005, h_r = 0.001 at x = 5.

    THREE rows, not four: SWASHES is 1-D, and derived SME(level=0) at
    dimension=2 (one horizontal direction) has state [b, h, q_0]. A 4-row IC
    here would mismatch the state width — the kind of thing that surfaces as a
    confusing shape error deep inside the raster writer.
    """
    h = np.where(x[0] < 5.0, 0.005, 0.001)
    return np.array([np.zeros_like(h), h, np.zeros_like(h)])


def ritter_ic(x):
    """SWASHES dry dam break: h_l = 0.005 into a DRY bed at x = 5. 1-D, 3 rows."""
    h = np.where(x[0] < 5.0, 0.005, 0.0)
    return np.array([np.zeros_like(h), h, np.zeros_like(h)])


def lake_at_rest_ic(x):
    """Lake at rest over a bump: eta = const, u = 0.

    The topography gate. Mass conservation is BLIND to well-balancing, so a
    flat-bed suite cannot see a lost bed-slope treatment.
    """
    b = 0.2 * np.exp(-((x[0] - 5.0) ** 2))
    eta = 1.0
    return np.array([b, eta - b, np.zeros_like(b)])       # 1-D: [b, h, q_0]


def gaussian_pulse_2d(x):
    h = 1.0 + 0.1 * np.exp(-((x[0]) ** 2 + (x[1]) ** 2) / 0.05)
    z = np.zeros_like(h)
    return np.array([z, h, z, z])


IC_FOR = {"stoker_wet": stoker_ic, "ritter_dry": ritter_ic}


def ic_for(case):
    return IC_FOR[case]
