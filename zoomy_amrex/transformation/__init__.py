"""AMReX code-generation, owned by the zoomy_amrex backend.

Mirrors the jax precedent (``zoomy_jax/transformation/to_jax.py``): the backend
owns its printer rather than relying on ``zoomy_core``.  We reuse the AMReX
*syntax* mixin (``zoomy_core.transformation.to_amrex.AmrexCore``) and the shared
C++ body generator (``GenericCppBase``), but drive emission from a frozen
:class:`SystemModel` — the modern path foam took with ``FoamSystemModelPrinter``.
The old ``zoomy_core`` ``AmrexModel`` (Model-path) is bit-rotted against the
current model API and is not used.
"""

from zoomy_amrex.transformation.to_amrex import (
    AmrexSystemModelPrinter,
    AmrexNumericsPrinter,
    generate_headers,
)

__all__ = [
    "AmrexSystemModelPrinter",
    "AmrexNumericsPrinter",
    "generate_headers",
]
