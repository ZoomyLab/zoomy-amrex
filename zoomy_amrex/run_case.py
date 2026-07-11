"""``zoomy_amrex.run_case`` — importable structured-grid entry point + VTK output.

Runs a model on a structured AMReX grid and writes a **ParaView-ready VTK series**
(``.pvd`` + per-snapshot/-level ``.vtu``).  One call handles both solver families:

  * standard hyperbolic (SWE / SME) via the AmrCore driver (``main.cpp``);
  * non-hydrostatic VAM / ML-VAM via the Chorin pressure-projection driver
    (``chorin_main.cpp`` / ``chorin_amr.cpp``) — auto-selected when the model
    exposes ``chorin_split``.

The case scripts under ``thesis/cases/amrex`` are thin: build a model (+IC/BC),
give a ``settings`` dict (domain / n_cells / dimension / time_end / amr / …), call
``run_case`` → a ``.pvd`` for ParaView.  This is the same entry point the GUI/CLI
server adapter uses.  AMReX emits VTK only; a store-HDF5 (for the GUI viz) is
obtained from the shared ``zoomy_prepost.vtk_to_hdf5`` on this VTK if ever needed.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import importlib.util
import functools
from pathlib import Path

import numpy as np

from .transformation import generate_headers, write_chorin_headers

_ROOT = Path(__file__).resolve().parent.parent   # library/zoomy_amrex/ (build.py, Source/)
_SRC = _ROOT / "Source"
IDX_B, IDX_H = 0, 1                  # structured state layout: [b, h, ...]


@functools.lru_cache(maxsize=1)
def _bld():
    """Lazily import the sibling CLI ``build.py`` by path (not a package module)
    so its write_inputs / GNUMAKEFILE / chorin machinery is reused, not duplicated."""
    spec = importlib.util.spec_from_file_location("_zoomy_amrex_build", _ROOT / "build.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _nsm(model):
    """Promote a raw Model to a NumericalSystemModel (SWE/SME) or return a system
    model as-is.  For a Chorin model the caller uses ``model.system_model`` directly."""
    from zoomy_core.numerics.numerical_system_model import to_numerical_system_model
    if hasattr(model, "state") and not hasattr(model, "system_model"):
        return model
    return to_numerical_system_model(model)


# ── geometry / IC helpers ───────────────────────────────────────────────────
def _grid(settings):
    dim = int(settings.get("dimension", 2))
    dom = settings["domain"]; nc = settings["n_cells"]
    if dim == 1:
        nx = int(nc[0] if isinstance(nc, (list, tuple)) else nc); ny = 1
        x0, x1 = float(dom[0]), float(dom[1]); y0, y1 = 0.0, (x1 - x0) / nx
    else:
        nx, ny = (int(nc[0]), int(nc[1])) if isinstance(nc, (list, tuple)) else (int(nc), int(nc))
        x0, x1, y0, y1 = [float(v) for v in dom[:4]]
    return dim, {"nx": nx, "ny": ny, "prob_lo": (x0, y0), "prob_hi": (x1, y1)}


def _cell_centers(geom, dim):
    x0, x1 = geom["prob_lo"][0], geom["prob_hi"][0]; nx = geom["nx"]; dx = (x1 - x0) / nx
    xc = x0 + (np.arange(nx) + 0.5) * dx
    if dim == 1:
        return np.vstack([xc, np.zeros_like(xc)])
    y0, y1 = geom["prob_lo"][1], geom["prob_hi"][1]; ny = geom["ny"]; dy = (y1 - y0) / ny
    yc = y0 + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    return np.vstack([X.ravel(order="C"), Y.ravel(order="C")])


def _write_ic_rasters(model, sm, geom, dim, out_dir):
    """Evaluate the model's analytic IC on the cell centres and write bed/depth
    rasters (assumes zero initial momentum — dam breaks / still water / bumps)."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ic = getattr(model, "initial_conditions", None) or getattr(sm, "initial_conditions", None)
    if ic is None:
        return None, None
    Q = ic.apply(_cell_centers(geom, dim), np.zeros((len(sm.state), geom["nx"] * geom["ny"])))
    if len(sm.state) > IDX_H + 1 and np.abs(Q[IDX_H + 1:]).max() > 1e-12:
        print("run_case: WARNING non-zero initial momentum not carried through the "
              "bed/depth raster IC path")
    bed = out / "bed.raw"; dep = out / "depth.raw"
    Q[IDX_B].astype(np.float64).tofile(bed)
    Q[IDX_H].astype(np.float64).tofile(dep)
    # ABSOLUTE paths: the driver runs from Exec/, so a relative raster path would
    # miss (the SWE driver silently falls back to its default IC; the Chorin
    # driver aborts).  resolve() pins them regardless of the run cwd.
    return str(bed.resolve()), str(dep.resolve())


def gmsh_to_rasters(msh_path, out_dir, geom):
    """Project a gmsh ``.msh`` + its ``$NodeData`` (bed/depth) onto the structured
    grid — generalises ``malpasset.prepare_rasters`` (REQ-89).  Returns dem/release."""
    from .malpasset import prepare_rasters
    info = prepare_rasters(msh_path, out_dir, ncell_x=geom["nx"], ncell_y=geom["ny"])
    return str(Path(info["dem_file"]).resolve()), str(Path(info["release_file"]).resolve())


# ── VTK postprocessing (always run; ParaView-ready) ─────────────────────────
def _plotfiles_to_vtk(exec_dir, geom, dim, names, out_dir):
    """Convert the AMReX plotfiles to a ParaView ``.pvd`` + per-snapshot/-level
    ``.vtu`` (structured quad/line grids, cell data = state variables by name).
    Handles AMR: one ``.vtu`` per refinement level (loaded as parts in ParaView)."""
    import re, sys, meshio
    sys.path.insert(0, str(_ROOT))
    from deliverable import read_levels

    ex = Path(exec_dir)
    plts = sorted([p for p in ex.iterdir() if p.is_dir() and re.match(r"(plt|chk)_\d+$", p.name)],
                  key=lambda p: int(p.name.split("_")[-1]))
    if not plts:
        raise RuntimeError(f"run_case: no AMReX plotfiles in {ex}")
    vdir = Path(out_dir) / "vtk"; vdir.mkdir(parents=True, exist_ok=True)
    x0 = geom["prob_lo"][0]; y0 = geom["prob_lo"][1]
    dx0 = (geom["prob_hi"][0] - x0) / geom["nx"]
    dy0 = (geom["prob_hi"][1] - y0) / geom["ny"] if dim >= 2 else dx0

    pvd = ['<?xml version="1.0"?>',
           '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
           '  <Collection>']
    for it, p in enumerate(plts):
        levels, H = read_levels(str(p))
        hdr = (p / "Header").read_text().splitlines()
        t = float(hdr[2 + int(hdr[1]) + 1])
        for lev, ld in enumerate(levels):
            arr = ld["arr"]                     # (ncomp, ny, nx) over the level bbox
            ilo, jlo, nx, ny = ld["ext"]; ref = ld["ref"]
            dxl = dx0 / ref; dyl = dy0 / ref
            ox = x0 + ilo * dxl; oy = y0 + jlo * dyl
            # structured vertices + connectivity (j-major cells to match arr C-order)
            if dim == 1:
                pts = np.array([[ox + i * dxl, 0.0, 0.0] for i in range(nx + 1)])
                conn = np.array([[i, i + 1] for i in range(nx)])
                cells = [("line", conn)]
            else:
                pts = np.array([[ox + i * dxl, oy + j * dyl, 0.0]
                                for j in range(ny + 1) for i in range(nx + 1)])
                vid = lambda i, j: i + (nx + 1) * j
                conn = np.array([[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
                                 for j in range(ny) for i in range(nx)])
                cells = [("quad", conn)]
            cell_data = {names[c] if c < len(names) else f"var{c}": [arr[c].reshape(-1)]
                         for c in range(arr.shape[0])}
            vtu = vdir / (f"{p.name}_L{lev}.vtu")
            meshio.write_points_cells(str(vtu), pts, cells, cell_data=cell_data)
            pvd.append(f'    <DataSet timestep="{t}" part="{lev}" file="vtk/{vtu.name}"/>')
    pvd += ['  </Collection>', '</VTKFile>']
    pvd_path = Path(out_dir) / "simulation.pvd"
    pvd_path.write_text("\n".join(pvd))
    return str(pvd_path)


# ── build + run (standard SWE/SME  vs  Chorin VAM) ──────────────────────────
def _amrex_home():
    return os.environ.get("AMREX_HOME", "/opt/amrex")


def _run_swe(model, sm, settings, geom, dim, bdir, dem, rel):
    src = bdir / "Source"; ex = bdir / "Exec"
    for f in _SRC.iterdir():
        if f.suffix in (".cpp", ".H") or f.name == "Make.package":
            shutil.copy2(f, src / f.name)
    generate_headers(sm, src)
    (ex / "GNUmakefile").write_text(_bld().GNUMAKEFILE.format(
        amrex_home=_amrex_home(), dim=2, use_mpi="TRUE", use_cuda="FALSE", cuda_arch="89"))
    _bld().write_inputs(
        ex / "inputs", geom["nx"], 2,
        tend=settings.get("time_end", 0.1), order=settings.get("spatial_order", 1),
        plot_dt=settings.get("time_end", 0.1) / max(1, settings.get("output_snapshots", 10)),
        cfl=settings.get("cfl", 0.45), max_level=int(settings.get("max_level", 0)),
        geom=geom, dem_file=dem, release_file=rel)
    return ex


def _write_chorin_inputs(path, geom, dim, settings, dem, rel):
    """Chorin inputs.  Boundary conditions come from the MODEL (the driver maps the
    structured faces West/East/South/North to the model's same-named tags, no
    parameter).  So the case just puts BCs on the model — nothing BC-related here."""
    nx, ny = geom["nx"], geom["ny"]
    (gx0, gy0), (gx1, gy1) = geom["prob_lo"], geom["prob_hi"]
    tend = settings.get("time_end", 1.0)
    params = settings.get("params", {})
    max_level = int(settings.get("max_level", 0))
    lines = [f"amr.max_level     = {max_level}",
             f"amr.n_cell        = {nx} {ny}",
             f"amr.max_grid_size = 64",
             f"amr.blocking_factor = {8 if max_level>0 else 1}",
             f"amr.ref_ratio     = 2", "amr.regrid_int    = 2",
             f"geometry.prob_lo  = {gx0} {gy0}", f"geometry.prob_hi  = {gx1} {gy1}",
             "geometry.is_periodic = 0 0",
             f"init.dem_file     = {dem}", f"init.release_file = {rel}",
             "precond.type = 3",
              f"params.g = {params.get('g', 9.81)}", f"params.rho = {params.get('rho', 1.0)}",
              f"params.nu = {params.get('nu', 0.0)}", f"params.lambda_s = {params.get('lambda_s', 0.0)}",
              f"solver.time_end = {tend}", f"solver.cfl      = {settings.get('cfl', 0.3)}",
              f"output.plot_dt_interval = {tend / max(1, settings.get('output_snapshots', 10))}",
              "tagging.threshold = 0.02"]
    Path(path).write_text("\n".join(lines) + "\n")


def _run_chorin(model, sm, settings, geom, dim, bdir, dem, rel):
    import sympy as sp
    src = bdir / "Source"; ex = bdir / "Exec"
    amr = int(settings.get("max_level", 0)) > 0
    driver = "chorin_amr.cpp" if amr else "chorin_main.cpp"
    for name in (driver, "chorin_common.H", "init_solution.cpp", "constants.H"):
        shutil.copy2(_SRC / name, src / name)
    (src / "Make.package").write_text(
        _bld().CHORIN_AMR_MAKE_PACKAGE if amr else _bld().CHORIN_MAKE_PACKAGE)
    split = model.chorin_split(sp.Symbol("dt", positive=True), system_model=sm)
    write_chorin_headers(split, src)
    (ex / "GNUmakefile").write_text(_bld().GNUMAKEFILE.format(
        amrex_home=_amrex_home(), dim=2, use_mpi="TRUE", use_cuda="FALSE", cuda_arch="89"))
    _write_chorin_inputs(ex / "inputs", geom, dim, settings, dem, rel)
    return ex


def run_case(model, settings, output_dir, on_progress=None):
    """Run ``model`` on a structured AMReX grid built from ``settings``; write a
    ParaView ``.pvd`` (+ ``.vtu`` series) and return its path.

    settings: ``dimension`` (1|2), ``domain`` ([x0,x1] or [x0,x1,y0,y1]),
    ``n_cells`` (nx or [nx,ny]), ``time_end``, ``cfl``, ``spatial_order``,
    ``output_snapshots``, ``max_level`` (AMR; 0 = single level), optional
    ``mesh_msh`` (gmsh + $NodeData IC), and for Chorin models optional
    ``inflow`` / ``pin_all`` / ``params``.  The solver family (standard vs Chorin)
    is auto-selected from the model."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dim, geom = _grid(settings)
    is_chorin = hasattr(model, "chorin_split")
    sm = model.system_model if is_chorin else _nsm(model)
    names = [str(s) for s in sm.state]

    bdir = out / "_build"; src = bdir / "Source"; ex_dir = bdir / "Exec"
    src.mkdir(parents=True, exist_ok=True); ex_dir.mkdir(parents=True, exist_ok=True)

    msh = settings.get("mesh_msh")
    if msh and Path(msh).exists():
        dem, rel = gmsh_to_rasters(msh, str(bdir / "raster"), geom)
    else:
        dem, rel = _write_ic_rasters(model, sm, geom, dim, str(bdir / "raster"))

    ex = (_run_chorin if is_chorin else _run_swe)(model, sm, settings, geom, dim, bdir, dem, rel)

    n = os.cpu_count() or 4
    subprocess.run(["make", f"-j{n}"], cwd=ex, check=True)
    exe = next((p for p in ex.iterdir() if p.name.startswith("main") and os.access(p, os.X_OK)), None)
    if exe is None:
        raise RuntimeError("run_case: build produced no executable")
    subprocess.run([f"./{exe.name}", "inputs"], cwd=ex, check=True)
    if on_progress is not None:
        on_progress(-1, settings.get("time_end", 0.1), 0.0)

    return _plotfiles_to_vtk(ex, geom, dim, names, str(out))
