"""``zoomy_amrex.solvers`` — param.Parameterized solver wrappers (gui REQ-131).

The GUI/CLI-facing entry point for the AMReX backend.  Two solver families, both
driving the existing ``run_case`` machinery (codegen → AMReX ``inputs`` deck →
make → run → VTK):

  * :class:`HyperbolicSolver` — explicit finite-volume march (SWE / SME) on the
    AmrCore driver.
  * :class:`SplitSolver` — N sequential system models + pressure projection
    (Chorin pressure-corrector; NOT VAM-specific), auto-selected for models that
    expose ``chorin_split``.

Usage (identical shape across all backends)::

    from zoomy_amrex.solvers import HyperbolicSolver
    solver = HyperbolicSolver(CFL=0.45, order=2)
    solver.solve(model, mesh, settings)      # -> VTK series in settings.output.directory

Design (gui REQ-131, user-approved 2026-07-13):
  * ``solve(model, mesh, settings)`` writes a VTK series into
    ``settings.output.directory`` and returns the ``.pvd`` path.
  * ``settings`` is STRUCTURED — ``settings.output.{directory, filename, snapshots}``
    is always honored; ``settings.time_end`` / ``settings.mesh.n_cells`` carry the
    run/grid properties.  Model-/scheme-specific things (reconstruction,
    eigenvalues, sources, IC, BC) come from the MODEL symbolically.
  * ``mesh`` is a path (``.msh`` — the structured domain is the mesh bbox) OR a
    descriptor ``{"domain": [...], "n_cells": [...]}``.  RESOLUTION always comes
    from the mesh/settings, never a solver param.
  * NO case knowledge lives here.  Case physics (e.g. a closed-basin wall) is a
    Wall BC on the MODEL, not a raster hack — the driver dispatches the model's
    face BCs (West/East/South/North).
"""
from __future__ import annotations
from pathlib import Path

import param

from .run_case import run_case


# ── structured-settings access (dict- or attribute-shaped) ──────────────────
def _get(obj, dotted, default=None):
    """Look up ``a.b.c`` through a structured ``settings`` that may mix dicts and
    attribute objects (param.Parameterized groups).  Returns ``default`` if any
    hop is missing/None."""
    cur = obj
    for key in dotted.split("."):
        if cur is None:
            return default
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return default if cur is None else cur


def _mesh_domain_ncells(mesh, settings):
    """Resolve ``mesh`` → (domain, n_cells, mesh_msh_path_or_None).

    ``mesh`` is either a descriptor ``{"domain", "n_cells"}`` (analytic-IC box:
    dambreak / radial) or a path to a ``.msh`` whose node bbox is the structured
    domain.  For a ``.msh`` the resolution is NOT in the file — it comes from
    ``settings.mesh.n_cells`` (or ``settings.n_cells``)."""
    if isinstance(mesh, dict):
        return mesh["domain"], mesh["n_cells"], None

    p = Path(mesh)
    if not p.exists():
        raise FileNotFoundError(f"solve: mesh path does not exist: {p}")
    import meshio
    m = meshio.read(str(p))
    xs, ys = m.points[:, 0], m.points[:, 1]
    domain = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
    nc = _get(settings, "mesh.n_cells") or _get(settings, "n_cells")
    if nc is None:
        raise ValueError(
            "solve: a .msh mesh needs a resolution — set settings['mesh']['n_cells']")
    return domain, nc, str(p.resolve())


class _BaseSolver(param.Parameterized):
    """Shared ``settings`` → legacy ``run_case`` dict translation."""

    def _legacy_settings(self, mesh, settings):
        domain, n_cells, mesh_msh = _mesh_domain_ncells(mesh, settings)
        nc = list(n_cells) if isinstance(n_cells, (list, tuple)) else [n_cells]
        dim = 1 if len(domain) == 2 else 2
        s = {
            "dimension": dim,
            "domain": list(domain),
            "n_cells": nc,
            "time_end": _get(settings, "time_end", 1.0),
            "output_snapshots": _get(settings, "output.snapshots", 10),
            "max_level": _get(settings, "max_level", 0),
        }
        if mesh_msh:
            s["mesh_msh"] = mesh_msh
        return s

    def _output_dir(self, settings):
        d = _get(settings, "output.directory")
        if d is None:
            raise ValueError("solve: settings['output']['directory'] is required")
        return d


class HyperbolicSolver(_BaseSolver):
    """Explicit finite-volume march for hyperbolic models (SWE / SME).

    The GUI auto-generates widgets from these bounded params."""

    CFL = param.Number(0.45, bounds=(0.0, 1.0), doc="Courant number")
    order = param.Integer(1, bounds=(1, 2), doc="spatial reconstruction order")
    well_balanced = param.Boolean(
        True, doc="Audusse hydrostatic reconstruction (lake-at-rest preserving)")

    def solve(self, model, mesh, settings, on_progress=None):
        s = self._legacy_settings(mesh, settings)
        s.update(cfl=self.CFL, spatial_order=self.order,
                 well_balanced=self.well_balanced)
        return run_case(model, s, self._output_dir(settings), on_progress=on_progress)


class SplitSolver(_BaseSolver):
    """Sequential system models + pressure projection (Chorin pressure-corrector).

    The split structure — predictor / pressure / corrector sub-models — comes
    from the MODEL (``model.chorin_split``); this wrapper only exposes the march
    and pressure-solve knobs."""

    cfl = param.Number(0.30, bounds=(0.0, 1.0), doc="Courant number")
    precond = param.Integer(
        3, bounds=(0, 4),
        doc="pressure preconditioner: 0 identity·1 Jacobi·2 MG·3 block-Jacobi·4 block-MG")
    dt = param.Number(
        None, allow_None=True, bounds=(0.0, None),
        doc="fixed timestep; None (default) = adaptive CFL dt. Pinning dt breaks "
            "the adapter feedback (dt collapse -> the dt-scaled elliptic operator "
            "-> P blows up), so it separates a dt-adapter artefact from a real "
            "formulation instability.")

    def solve(self, model, mesh, settings, on_progress=None):
        if not hasattr(model, "chorin_split"):
            raise TypeError(
                "SplitSolver needs a model with chorin_split (got "
                f"{type(model).__name__}); use HyperbolicSolver for hyperbolic models")
        s = self._legacy_settings(mesh, settings)
        # `precond` flows to the inputs deck (`precond.type`) via run_case's
        # Chorin path; `params` passes model parameters (g, rho, nu, lambda_s).
        s.update(cfl=self.cfl, params=_get(settings, "params", {}),
                 precond=self.precond)
        if self.dt is not None:
            s["dt"] = self.dt
        return run_case(model, s, self._output_dir(settings), on_progress=on_progress)
