"""amrex test suite — fixtures and the march() equivalent.

Ported from the approved jax suite. The SHAPE is fixed and shared; only the
tech adapts, and the adaptation is concentrated HERE so the tests themselves
read the same as their jax twins.

The one real difference: jax marches in-process and returns arrays. amrex
GENERATES C++, COMPILES it, runs a binary, and writes plotfiles — so march()
below runs the case and reads the full state back out of the plotfile.

MUST RUN IN-CONTAINER. zoomy_core/zoomy_amrex are not importable on the host;
the containers have them (verified). Typical invocation:

    apptainer exec --nv --bind <ZOOMY>:/work \
        containers/zoomy_amrex/zoomy_amrex_gpu.sif \
        bash -lc 'cd /work/library/zoomy_amrex && pytest'
"""
import os
import pathlib
import sys

import numpy as np
import pytest

CFL_1D, CFL_2D = 0.9, 0.45          # user law — no augmentation, ever
ORDER_FLOOR = {1: 0.9, 2: 1.9}

_HERE = pathlib.Path(__file__).resolve().parent
_RUNS = _HERE / "_runs"


def pytest_addoption(parser):
    g = parser.getgroup("zoomy test tiers")
    g.addoption("--overwrite-results", action="store_true", default=False)
    g.addoption("--run-large", action="store_true", default=False)


@pytest.fixture
def overwrite(request):
    return (request.config.getoption("--overwrite-results")
            or os.environ.get("ZOOMY_OVERWRITE_RESULTS") == "1")


def _build_key_from(name, settings):
    """Build dir keyed on the EMITTED-CODE identity, not the output path.

    run_case puts its _build under the output dir (run_case.py:339), so a
    per-case output dir means a per-case AMReX compile+link. Cases that differ
    only in IC / mesh / t_end emit the SAME driver, so they share one build and
    pay the compile once. Measured: cold 38 s vs warm 8 s on CPU — the budget
    counts warm runs, and this is what keeps them warm.

    NOT parallel-safe: two tests sharing a key would race in one Exec dir.
    Run this suite serially (no pytest-xdist) until that is addressed.
    """
    return "{}_o{}_d{}".format(name,
                               settings.get("spatial_order", 1),
                               settings.get("dimension", 2))


def march(nsm, settings, *, key, n_steps=None, t_end=None, cfl=None):
    """Run one amrex case and return (Q, Qaux) for the FINAL state.

    Full state, both fields, never a hand-picked subset — the suite rule.
    Reads the plotfile directly via deliverable.read_levels, which yields a
    dense (ncomp, ny, nx) array; the VTK/HDF5 path is a detour we do not need.
    """
    from zoomy_amrex import run_case
    sys.path.insert(0, str(_HERE.parent))
    from deliverable import read_levels

    s = dict(settings)
    if cfl is not None:
        s["cfl"] = cfl
    if t_end is not None:
        s["time_end"] = t_end
    if n_steps is not None:
        # Exactly N steps. time_end must stay unreachable, or IT would stop the
        # run instead and the twin would silently march a different length.
        s["max_step"] = n_steps
        s.setdefault("time_end", 1.0e30)

    out = _RUNS / _build_key_from(key, s)
    # run_case._nsm returns an already-built (Numerical)SystemModel as-is, so
    # the test can construct Model -> SystemModel -> NSM explicitly (mandated:
    # the derivation chain stays visible) and hand the NSM straight over.
    run_case(nsm, s, str(out))

    plts = sorted((out / "_build" / "Exec").glob("plt_*"))
    assert plts, f"no plotfile written under {out} — the run produced nothing"
    levels, _ = read_levels(str(plts[-1]))
    arr = levels[0]["arr"]                      # (ncomp, ny, nx), level 0

    n_q = len(nsm.state)
    assert arr.shape[0] > n_q, (
        f"plotfile has {arr.shape[0]} components for {n_q} state rows — aux is "
        "missing. Q-only plotfiles were a regression fixed in 158b692; if this "
        "fires, the driver has lost aux output again.")
    return arr[:n_q], arr[n_q:]


def fit_order(sizes, errors):
    return float(-np.polyfit(np.log(sizes), np.log(errors), 1)[0])


def restrict(fine):      # conservative fine -> coarse, exact for cell averages
    return 0.5 * (fine[:, 0::2] + fine[:, 1::2])


def describe(nsm) -> str:
    """The MANDATED matrix print: sanity-check the operator slots BEFORE running.

    Standing mandate 2 — a missing hydrostatic_pressure entry is visible on
    sight, and a 30-second print beats a day of wrong conclusions.
    """
    try:
        return nsm.describe()
    except AttributeError:
        return repr(nsm)
