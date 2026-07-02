"""``zoomy_amrex.run_case`` — the importable structured-grid entry point (REQ-89).

The server's ``AmrexAdapter`` hands us a resolved ``model`` + a ``settings`` dict
(the shared folder-case format) and wants a zoomy-store HDF5 back, so one case
runs on every backend.  AMReX only takes STRUCTURED meshes, so this:

  (a) builds a structured grid from ``settings`` (domain / n_cells / dimension);
  (b) code-gens ``Model.H`` / ``Numerics.H`` / ``UserFunctions.H`` from the model
      via the backend printer (``generate_headers``);
  (c) applies the model's baked analytic IC by evaluating
      ``model.initial_conditions`` on the cell centres and writing the bed/depth
      rasters the driver loads (zero initial momentum — the common GUI case; a
      gmsh ``.msh`` + ``$NodeData`` case instead projects via ``gmsh_to_rasters``);
  (d) writes ``inputs`` + ``GNUmakefile``, compiles, runs the AMReX driver;
  (e) reads the AMReX plotfiles and writes a zoomy-store HDF5 (mesh + fields,
      matching ``zoomy_prepost.vtk_to_hdf5``) so the GUI viz works unchanged.

It wraps the existing ``build.py`` machinery — nothing model/case-specific here.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import importlib.util
import functools
from pathlib import Path

import numpy as np

from .transformation import generate_headers

_ROOT = Path(__file__).resolve().parent.parent   # library/zoomy_amrex/ (has build.py, Source/)
_SRC = _ROOT / "Source"
IDX_B, IDX_H = 0, 1                  # structured state layout: [b, h, ...]


@functools.lru_cache(maxsize=1)
def _bld():
    """Lazily import the sibling CLI ``build.py`` by path (it is not a package
    module) so its write_inputs / GNUMAKEFILE machinery is reused, not duplicated."""
    spec = importlib.util.spec_from_file_location("_zoomy_amrex_build", _ROOT / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nsm(model):
    """Promote a raw Model to a NumericalSystemModel (inherits wet/dry regs), or
    pass a SystemModel/NSM through."""
    from zoomy_core.numerics.numerical_system_model import to_numerical_system_model
    # already a (numerical) system model?
    if hasattr(model, "state") and not hasattr(model, "system_model"):
        return model
    return to_numerical_system_model(model)


def gmsh_to_rasters(msh_path, out_dir, geom, field_map=("B", "H")):
    """Project a gmsh ``.msh`` + its ``$NodeData`` (bed/depth) onto the structured
    grid — the generalisation of ``malpasset.prepare_rasters`` (REQ-89): reads the
    two named node-data fields, block-averages the triangulation onto the nx×ny
    cells, writes ``bed.raw`` / ``depth.raw``.  Returns the dem/release paths."""
    from .malpasset import prepare_rasters
    info = prepare_rasters(msh_path, out_dir, ncell_x=geom["nx"], ncell_y=geom["ny"])
    return info["dem_file"], info["release_file"]


def _cell_centers(geom, dim):
    x0, x1 = geom["prob_lo"][0], geom["prob_hi"][0]
    nx = geom["nx"]; dx = (x1 - x0) / nx
    xc = x0 + (np.arange(nx) + 0.5) * dx
    if dim == 1:
        return np.vstack([xc, np.zeros_like(xc)])                # (2, nx)
    y0, y1 = geom["prob_lo"][1], geom["prob_hi"][1]
    ny = geom["ny"]; dy = (y1 - y0) / ny
    yc = y0 + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="xy")                    # Y (j) major rows
    return np.vstack([X.ravel(order="C"), Y.ravel(order="C")])   # (2, nx*ny)


def _write_ic_rasters(model, sm, geom, dim, out_dir):
    """Evaluate the model's analytic IC on the cell centres and write bed/depth
    rasters (assumes zero initial momentum — dam breaks / still water / bumps).
    Row-major idx = i + nx*j, matching readRasterIntoComponent.  ``sm`` carries the
    state ordering; ``model`` carries the baked ``initial_conditions``."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ic = getattr(model, "initial_conditions", None) or getattr(sm, "initial_conditions", None)
    if ic is None:
        return None, None
    centers = _cell_centers(geom, dim)                            # (2, ncells)
    n_state = len(sm.state)
    Q = np.zeros((n_state, centers.shape[1]))
    Q = ic.apply(centers, Q)                                      # canonical core call
    mom = np.abs(Q[IDX_H + 1:]).max() if n_state > IDX_H + 1 else 0.0
    if mom > 1e-12:
        print(f"run_case: WARNING non-zero initial momentum ({mom:.3g}) not carried "
              "through the bed/depth raster IC path")
    bed = out / "bed.raw"; dep = out / "depth.raw"
    Q[IDX_B].astype(np.float64).tofile(bed)
    Q[IDX_H].astype(np.float64).tofile(dep)
    return str(bed), str(dep)


def _plotfiles_to_hdf5(build_exec_dir, geom, dim, out_hdf5):
    """Read the AMReX plotfiles (via deliverable.read_levels) and write the zoomy
    store HDF5 (mesh + fields/iteration_i/{time,Q}) — the same schema as
    zoomy_prepost.vtk_to_hdf5, so the GUI viz consumes it unchanged."""
    import h5py, re, sys
    sys.path.insert(0, str(_ROOT))
    from deliverable import read_plotfile                         # pure-python plt reader

    ex = Path(build_exec_dir)
    plts = sorted([p for p in ex.iterdir()
                   if p.is_dir() and re.match(r"(plt|chk)_\d+", p.name)])
    if not plts:
        raise RuntimeError(f"no AMReX plotfiles in {ex}")

    nx = geom["nx"]; ny = geom.get("ny", 1)
    x0, x1 = geom["prob_lo"][0], geom["prob_hi"][0]; dx = (x1 - x0) / nx
    if dim == 1:
        y0, dy, ny = 0.0, dx, 1
    else:
        y0 = geom["prob_lo"][1]; dy = (geom["prob_hi"][1] - y0) / ny

    # structured mesh: vertices + quad(2d)/line(1d) connectivity
    if dim == 1:
        verts = np.array([[x0 + i * dx for i in range(nx + 1)]])          # (1, nx+1)
        cells = np.array([[i for i in range(nx)], [i + 1 for i in range(nx)]])  # (2, nx)
        ztype = "line"
    else:
        gx, gy = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="xy")
        vx = (x0 + gx * dx).ravel(order="C"); vy = (y0 + gy * dy).ravel(order="C")
        verts = np.vstack([vx, vy])                                       # (2, (nx+1)(ny+1))
        def vid(i, j): return i + (nx + 1) * j
        quad = [[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
                for j in range(ny) for i in range(nx)]
        cells = np.array(quad).T                                          # (4, nx*ny)
        ztype = "quad"
    n_cells = nx * ny

    with h5py.File(out_hdf5, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("dimension", data=dim)
        g.create_dataset("type", data=ztype)
        g.create_dataset("n_cells", data=n_cells)
        g.create_dataset("n_inner_cells", data=n_cells)
        g.create_dataset("vertex_coordinates", data=verts)
        g.create_dataset("cell_vertices", data=cells)
        fields = f.create_group("fields")
        for it, p in enumerate(plts):
            arr = read_plotfile(str(p))[0]        # level-0 (ncomp, ny, nx) covers domain
            ncomp = arr.shape[0]
            Q = arr.reshape(ncomp, -1)            # (ncomp, n_cells), row-major i+nx*j
            hdr = (p / "Header").read_text().splitlines()
            t = float(hdr[2 + int(hdr[1]) + 1])   # time line (see deliverable._read_header)
            grp = fields.create_group(f"iteration_{it}")
            grp.create_dataset("time", data=t, dtype=float)
            grp.create_dataset("Q", data=Q)
    return out_hdf5


def run_case(model, settings, output_dir, on_progress=None):
    """Run ``model`` on a structured AMReX grid built from ``settings``; return the
    path to a zoomy-store HDF5 (mesh + fields) for the GUI/plotting stack.

    settings keys: ``dimension`` (1|2 horizontal), ``domain`` ([x0,x1] or
    [x0,x1,y0,y1]), ``n_cells`` (nx or [nx,ny]), ``time_end``, ``cfl``,
    ``spatial_order``, ``output_snapshots``, optional ``mesh_msh`` (gmsh .msh +
    $NodeData bed/IC to project instead of the analytic IC)."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dim = int(settings.get("dimension", 2))
    dom = settings["domain"]
    nc = settings["n_cells"]
    if dim == 1:
        nx = int(nc[0] if isinstance(nc, (list, tuple)) else nc); ny = 1
        x0, x1 = float(dom[0]), float(dom[1]); y0, y1 = 0.0, (x1 - x0) / nx
    else:
        nx, ny = (int(nc[0]), int(nc[1])) if isinstance(nc, (list, tuple)) else (int(nc), int(nc))
        x0, x1, y0, y1 = [float(v) for v in dom[:4]]
    geom = {"nx": nx, "ny": ny, "prob_lo": (x0, y0), "prob_hi": (x1, y1)}

    bdir = out / "_build"; src = bdir / "Source"; ex = bdir / "Exec"
    src.mkdir(parents=True, exist_ok=True); ex.mkdir(parents=True, exist_ok=True)
    for f in _SRC.iterdir():
        if f.suffix in (".cpp", ".H") or f.name == "Make.package":
            shutil.copy2(f, src / f.name)

    # code-gen device headers from the model
    sm = _nsm(model)
    generate_headers(sm, src)

    # IC: gmsh projection, or analytic-IC bed/depth rasters
    msh = settings.get("mesh_msh")
    if msh and Path(msh).exists():
        dem, rel = gmsh_to_rasters(msh, str(bdir / "raster"), geom)
    else:
        dem, rel = _write_ic_rasters(model, sm, geom, dim, str(bdir / "raster"))

    amrex_home = os.environ.get("AMREX_HOME", "/opt/amrex")
    (ex / "GNUmakefile").write_text(_bld().GNUMAKEFILE.format(
        amrex_home=amrex_home, dim=2, use_mpi="TRUE", use_cuda="FALSE", cuda_arch="89"))
    _bld().write_inputs(ex / "inputs", nx, 2,
                    tend=settings.get("time_end", 0.1),
                    order=settings.get("spatial_order", 1),
                    plot_dt=settings.get("time_end", 0.1) / max(1, settings.get("output_snapshots", 10)),
                    cfl=settings.get("cfl", 0.45),
                    geom=geom, dem_file=dem, release_file=rel)

    n = os.cpu_count() or 4
    subprocess.run(["make", f"-j{n}"], cwd=ex, check=True)
    exe = next((p for p in ex.iterdir() if p.name.startswith("main") and os.access(p, os.X_OK)), None)
    if exe is None:
        raise RuntimeError("run_case: build produced no executable")
    subprocess.run([f"./{exe.name}", "inputs"], cwd=ex, check=True)
    if on_progress is not None:
        on_progress(-1, settings.get("time_end", 0.1), 0.0)

    hdf5 = str(out / "simulation.h5")
    return _plotfiles_to_hdf5(ex, geom, dim, hdf5)
