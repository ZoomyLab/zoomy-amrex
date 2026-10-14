"""The printed march raises instead of ending silently (zoomy_core ANALYSIS 49, item 1): a step
that does not advance the time stops the loop and `check.march` fails the run, as does a
non-finite final state; the driver raises `FloatingPointError` with the runtime's message.
Builds the printed program (container): large.
"""
import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, PositiveNonconservativeHLL
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex

pytestmark = [pytest.mark.large, pytest.mark.amrex]


@pytest.mark.parametrize("dt", [None, 0.01])
def test_overflowing_state_raises(tmp_path, dt):
    """q = 1e300 in one cell: the adaptive march stops at t = 0 (an infinite wave speed), the
    constant-dt march ends with a non-finite state; both raise."""
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 1.0 + (x[0] < 0.5), 0.0]))
    nsm = NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeHLL)
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    Q0, _ = SymbolicHyperbolicSolverNumpy(time_end=0.05).setup(mesh, nsm)
    Q0 = np.array(Q0)
    Q0[2, 7] = 1e300
    with pytest.raises(FloatingPointError, match="non-finite state" if dt else
                       r"zoomy march stopped at t = 0 .*infinite wave speed"):
        SymbolicHyperbolicSolverAmrex(time_end=0.05, cfl=0.4, dt=dt).run(mesh, nsm, Q0, str(tmp_path))
