"""SWASHES Ritter (dry dam break) — small tier.

amrex twin of the jax suite's test_ritter_dry. The dry front is the point: this
runs CAPLESS (cid=54 ruling), so positivity has to come from the scheme and the
model's desingularised hinv, never from a floor on h.
"""
import time

import numpy as np
import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from tests import models, refs
from tests.cases import ritter_ic
from tests.conftest import CFL_1D, describe, march


@pytest.mark.small
@pytest.mark.amrex
def test_ritter_dry(overwrite):
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=ritter_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    settings = {"dimension": 1, "domain": [0.0, 10.0], "n_cells": 100,
                "spatial_order": 1, "output_snapshots": 1, "max_level": 0}
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, settings, key="swe1d_swashes", t_end=1.0, cfl=CFL_1D)
    elapsed = time.perf_counter() - t0

    assert np.isfinite(Q).all(), "non-finite state on the dry front"
    # NO floor is permitted anywhere (standing mandate 3), so h >= 0 has to be
    # earned by the scheme. A negative depth here is a real defect, not
    # something to clamp away.
    assert Q[1].min() >= 0.0, "negative depth — and NO floor is permitted"
    assert Q[1].max() > 0.0, "the whole domain went dry — nothing was solved"

    refs.check("ritter_dry", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("ritter_dry", elapsed, overwrite)
