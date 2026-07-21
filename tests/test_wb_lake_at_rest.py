"""Lake at rest over topography — small tier, the WELL-BALANCING gate.

amrex twin of the jax suite's test_lake_at_rest_over_bump.

Why this test and not a mass check: mass conservation is BLIND to
well-balancing. A scheme can conserve mass to machine precision while tearing a
lake apart over a bump, so a flat-bed suite is structurally incapable of seeing
a lost bed-slope treatment. The assertions below are on the free surface and on
spurious currents, which is where the failure actually shows.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from tests import models, refs
from tests.cases import lake_at_rest_ic
from tests.conftest import CFL_1D, describe, march


@pytest.mark.small
@pytest.mark.amrex
def test_lake_at_rest_over_bump(overwrite):
    model = models.swe(dimension=2, bc="wall")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=lake_at_rest_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    # well_balanced turns on the driver's Audusse hydrostatic reconstruction.
    # That construct is frozen under the cid=53 side-by-side ruling and will
    # move to the emitted layer with the new march; the test asserts the
    # PROPERTY, so it survives that migration unchanged.
    settings = {"dimension": 1, "domain": [0.0, 10.0], "n_cells": 100,
                "spatial_order": 1, "output_snapshots": 1, "max_level": 0,
                "well_balanced": True}
    t0 = time.perf_counter()
    # key="swe1d", NOT a WB-specific key: well_balanced is a runtime deck flag,
    # not emitted code, so this shares the build with the other 1-D SWE tests.
    # Keying it separately cost a whole extra cold compile (34.6 s -> the suite
    # was paying 38 s twice) for byte-identical generated source.
    Q, Qaux = march(nsm, settings, key="swe1d", t_end=1.0, cfl=CFL_1D)
    elapsed = time.perf_counter() - t0

    b, h, q = Q[0], Q[1], Q[2]
    eta = b + h
    u = q / np.maximum(h, 1e-300)     # display only; not a scheme epsilon

    assert np.isfinite(Q).all()
    assert np.abs(eta - eta.mean()).max() < 1e-12, (
        f"lake tilted — WB lost. max|eta - mean| = "
        f"{np.abs(eta - eta.mean()).max():.3e}")
    assert np.abs(u).max() < 1e-12, (
        f"spurious currents over the bed: max|u| = {np.abs(u).max():.3e}")

    refs.check("wb_lake", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("wb_lake", elapsed, overwrite)
