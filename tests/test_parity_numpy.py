"""amrex vs numpy on the SAME NumericalSystemModel — small tier.

amrex twin of the jax suite's test_numpy_jax_parity. numpy is the reference
implementation, so this is the one test that can catch amrex computing
something self-consistently wrong: every other test compares amrex against its
own blessed reference, which a uniformly wrong scheme would satisfy forever.

TOLERANCE POLICY: the assertion below is deliberately loose and the MEASURED
difference is printed and stored. The backend-suite protocol is that a
discrepancy against the reference implementation gets DISCUSSED, never silently
baselined into a passing tolerance. If this number is large, that is a finding
to report, not a constant to tune.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from tests import models, refs
from tests.cases import stoker_ic
from tests.conftest import CFL_1D, describe, march

N_CELLS = 100
DOMAIN = (0.0, 10.0)
T_END = 0.5


@pytest.mark.small
@pytest.mark.amrex
def test_amrex_numpy_parity(overwrite):
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    settings = {"dimension": 1, "domain": list(DOMAIN), "n_cells": N_CELLS,
                "spatial_order": 1, "output_snapshots": 1, "max_level": 0}
    t0 = time.perf_counter()
    Qa, Aa = march(nsm, settings, key="swe1d_swashes", t_end=T_END, cfl=CFL_1D)
    Qa = np.squeeze(Qa)                      # (n_q, 1, nx) -> (n_q, nx)

    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NumpySolver

    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_CELLS)
    Qn, An = NumpySolver(
        time_end=T_END,
        compute_dt=timestepping.adaptive(CFL=CFL_1D)).solve(
            mesh, nsm, write_output=False)
    Qn = np.asarray(Qn)[:, :N_CELLS]
    elapsed = time.perf_counter() - t0

    assert Qa.shape == Qn.shape, (
        f"grid mismatch: amrex {Qa.shape} vs numpy {Qn.shape} — the two are not "
        "discretising the same domain, so any agreement would be accidental")

    diff = np.abs(Qa - Qn)
    rel = diff.max() / max(np.abs(Qn).max(), 1e-300)
    print(f"[parity] amrex vs numpy: max|diff| {diff.max():.3e}, "
          f"relative {rel:.3e}")
    for r in range(Qa.shape[0]):
        print(f"[parity]   row {r}: max|diff| {diff[r].max():.3e}")

    assert np.isfinite(Qa).all() and np.isfinite(Qn).all()
    # Loose on purpose — see the tolerance policy above. A tight number here
    # would be a tuned constant, not a measurement.
    assert rel < 0.5, (
        f"amrex and numpy disagree by {rel:.3e} relative on the same NSM — "
        "that is a divergence to investigate, not a tolerance to widen")

    refs.check("parity_numpy", overwrite, Q=Qa, Qaux=Aa,
               diff=np.array([diff.max()]), rel=np.array([rel]))
    refs.check_time("parity_numpy", elapsed, overwrite)
