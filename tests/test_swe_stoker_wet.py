"""SWASHES Stoker (wet dam break) — small tier.

amrex twin of the jax suite's test_stoker_wet. Same model, same IC, same CFL
law; the only difference is that amrex generates, compiles and runs a binary,
which conftest.march() hides.
"""
import time

import pytest
import zoomy_core.model.initial_conditions as IC
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel

from tests import models, refs
from tests.cases import stoker_ic
from tests.conftest import CFL_1D, describe, march


@pytest.mark.small
@pytest.mark.amrex
def test_stoker_wet(overwrite):
    model = models.swe(dimension=2, bc="swashes")            # Model
    sm = SystemModel.from_model(model)                       # SystemModel
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(             # NumericalSystemModel
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    settings = {"dimension": 1, "domain": [0.0, 10.0], "n_cells": 100,
                "spatial_order": 1, "output_snapshots": 1, "max_level": 0}
    t0 = time.perf_counter()
    Q, Qaux = march(nsm, settings, key="swe1d_swashes", t_end=1.0, cfl=CFL_1D)
    elapsed = time.perf_counter() - t0

    import numpy as np
    assert np.isfinite(Q).all(), "non-finite state"
    assert Q[1].min() > 0.0, "depth went non-positive on a WET dam break"
    # The cap bug (core's wet_dry_eps default exceeding the SWASHES depths)
    # zeroed momentum everywhere while still converging-looking. Assert against
    # it directly: this case is capless by ruling, so momentum must be alive.
    assert np.abs(Q[2]).max() > 0.0, "momentum is zero — the cap bug is back"

    refs.check("stoker_wet", overwrite, Q=Q, Qaux=Qaux)
    refs.check_time("stoker_wet", elapsed, overwrite)
