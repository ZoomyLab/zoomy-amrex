"""2-rank MPI physics parity — small tier (user-directed addition).

amrex analogue of the jax suite's 2-device sharding test. The claim under test
is the only one worth making about a domain decomposition: the PHYSICS is
invariant to how the domain was split. Anything else — that it ran, that it
produced a file, that it did not crash — is liveness, and liveness passes for
a decomposition that quietly drops a halo exchange.

Same convention as everything else here: assert numbers, not survival.
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


@pytest.mark.small
@pytest.mark.amrex
def test_two_rank_physics_parity(overwrite):
    model = models.swe(dimension=2, bc="swashes")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(function=stoker_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1, limiter="minmod"))
    print(describe(nsm))

    base = {"dimension": 1, "domain": [0.0, 10.0], "n_cells": 128,
            "spatial_order": 1, "output_snapshots": 1, "max_level": 0}

    t0 = time.perf_counter()
    Q1, A1 = march(nsm, dict(base, mpi_ranks=1), key="swe1d_swashes",
                   t_end=0.5, cfl=CFL_1D)
    Q2, A2 = march(nsm, dict(base, mpi_ranks=2), key="swe1d_swashes",
                   t_end=0.5, cfl=CFL_1D)
    elapsed = time.perf_counter() - t0

    assert Q1.shape == Q2.shape, (
        f"1-rank {Q1.shape} vs 2-rank {Q2.shape} — the decomposition changed "
        "the recovered grid, so the comparison below would be meaningless")

    dq = np.abs(Q1 - Q2).max()
    da = np.abs(A1 - A2).max()
    print(f"[mpi] 1 rank vs 2 ranks: max|dQ| {dq:.3e}, max|dQaux| {da:.3e}")

    # A domain decomposition is a bookkeeping change, not a numerical one: the
    # same arithmetic happens in the same order on each cell. So this is held
    # to BIT-LEVEL agreement, not a tolerance. A nonzero difference means a
    # halo is stale or a reduction is order-dependent — both real defects.
    assert dq == 0.0, f"2-rank state differs from 1-rank by {dq:.3e}"
    assert da == 0.0, f"2-rank aux differs from 1-rank by {da:.3e}"

    refs.check("parallel_2rank", overwrite, Q=Q2, Qaux=A2)
    refs.check_time("parallel_2rank", elapsed, overwrite)
