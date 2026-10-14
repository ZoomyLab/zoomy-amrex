"""Host-side: the amrex kernel writer refuses a derivative aux it does not compute (it would read 0):
it differentiates state rows only, so an aux function target or an `unknown` compound raises."""
import pytest

from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel.system_model import SystemModel

pytestmark = [pytest.mark.small]


@pytest.mark.parametrize("target", ["unknown", "function"])
def test_write_kernels_refuses_uncomputed_aux(tmp_path, target):
    from zoomy_amrex.symkernels import write_kernels
    nsm = NumericalSystemModel.from_system_model(SystemModel.from_model(SME(
        level=0, dimension=2, closures=[Newtonian(), ManningFriction(), StressFree()],
        boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                for t in ("left", "right")]))))
    nsm.aux_registry = list(nsm.aux_registry or []) + [dict(
        kind="derivative", name="h*zeta_x", row=0, atom=None, aux_symbol=None,
        target_name="h*zeta", multi_index=(1,), target_kind=target, function_row=0)]
    with pytest.raises(ValueError, match=rf"amrex: the aux 'h\*zeta_x' .*target_kind '{target}'"):
        write_kernels([("nsm", nsm)], "", tmp_path / "Kernels.H", *([1] if "amrex" == "amrex" else []))
    assert not (tmp_path / "Kernels.H").exists()
