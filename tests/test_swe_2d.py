"""2-D Gaussian pulse — small tier.

amrex twin of the jax suite's test_swe_2d_pulse. Two things only this test
covers: the 2-D emitted code path (a genuinely different Model.H — two momentum
rows, both NCP directions) and the CFL_2D = 0.45 law.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from tests import models, refs
from tests.cases import gaussian_pulse_2d
from tests.conftest import CFL_2D, describe, march


@pytest.mark.small
@pytest.mark.amrex
def test_swe_2d_pulse(overwrite):
    # dimension=3 is TWO horizontal directions: SME counts the vertical. The
    # opposite convention from the bespoke SWE class, and a genuine trap.
    model = models.swe(dimension=3, bc="extrapolation")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=gaussian_pulse_2d)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    settings = {"dimension": 2, "domain": [-1.0, 1.0, -1.0, 1.0],
                "n_cells": [32, 32], "spatial_order": 1,
                "output_snapshots": 1, "max_level": 0}
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, settings, key="swe2d", t_end=0.1, cfl=CFL_2D)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all()
    assert Q[1].min() > 0.0, "depth went non-positive"
    # The pulse spreads from the centre, so both momentum components must be
    # live. A 2-D run that only moves in x would pass a depth check while
    # silently having lost the y direction.
    assert np.abs(Q[2]).max() > 0.0, "x-momentum is zero"
    assert np.abs(Q[3]).max() > 0.0, "y-momentum is zero — 2-D path is not live"

    refs.check("swe_2d", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("swe_2d", elapsed, overwrite)
