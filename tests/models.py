"""Model definitions, cached.  Tests do the SystemModel and
NumericalSystemModel steps themselves — the chain stays visible.
NEVER no_cache(): model correctness is owned by the core goldens.

Ported from the approved jax suite. This module is deliberately
backend-INDEPENDENT: it constructs zoomy_core models only, so an amrex test and
a jax test that name the same model are provably comparing the same physics.
Anything amrex-specific (build, run, readback) lives in conftest.py, never here.

Standing mandate: SWE is the DERIVED SME(level=0) composition, never the
hand-built SWE class whose operators are stated by hand (swe.py:103-126).
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def swe(dimension: int, bc: str = "extrapolation"):
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import (
        ManningFriction, Newtonian, StressFree)
    from tests.cases import bcs_for
    return SME(level=0, dimension=dimension,
               closures=[Newtonian(), ManningFriction(), StressFree()],
               boundary_conditions=bcs_for(bc, dimension))


@lru_cache(maxsize=None)
def vam(level=1, dimension=2, bc="bump"):
    from zoomy_core.model.models import VAM
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from tests.cases import bcs_for
    return VAM(level=level, dimension=dimension,
               closures=[Newtonian(), StressFree()],
               boundary_conditions=bcs_for(bc, dimension))


@lru_cache(maxsize=None)
def mlsme(n_layers=2, level=1, bc="periodic"):
    from zoomy_core.model.models import MLSME
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from tests.cases import bcs_for
    return MLSME(level=level, n_layers=n_layers, dimension=2,
                 closures=[Newtonian(), StressFree()],
                 boundary_conditions=bcs_for(bc, 2))
