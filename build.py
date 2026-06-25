#!/usr/bin/env python3
"""In-tree driver: SystemModel → AMReX headers → build → run.

The structural analogue of jax's ``solver.solve(...)`` and foam's
``create_model.py``: it consumes a frozen :class:`SystemModel`, generates the
``Model.H`` / ``Numerics.H`` / ``UserFunctions.H`` device code via the
backend-owned printer (``zoomy_amrex.transformation``), stages the hand-written
C++ driver next to them, writes a ``GNUmakefile`` + ``inputs``, and (optionally)
compiles and runs.

Run it INSIDE the zoomy_amrex container (it needs zoomy_core, AMReX at
``$AMREX_HOME``, and ``mpicxx``):

    apptainer exec --bind <ZOOMY>:/work <sif> \
        python3 /work/library/zoomy_amrex/build.py \
        --model SWE --dim 2 --ncell 100 --tend 0.1 --order 1 \
        --build-dir /tmp/run --make --run

This restores the previously broken / out-of-tree generation path (the
``zoomy_server`` adapter imported a non-existent module and emitted a zero-flux
``Numerics.H``).  Generation is reproducible; nothing is hand-placed.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Source"

GNUMAKEFILE = """AMREX_HOME ?= {amrex_home}
DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = {dim}

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
"""


def build_system_model(model_name, dim, level):
    """Return a frozen SystemModel for the requested model.

    Convention reminder: ``SWE.dimension`` counts *horizontal* dims (2 → 2-D),
    while ``SME.dimension`` counts the total incl. the vertical (3 → 2 horizontal).
    """
    from zoomy_core.model import models as M
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
    bcs = BoundaryConditions([Extrapolation(tag="left"), Extrapolation(tag="right")])
    if model_name == "SWE":
        return M.SWE(dimension=dim, boundary_conditions=bcs).system_model
    if model_name == "SME":
        # dim here is the TOTAL dimension; level adds moments q_1..q_level.
        return M.SME(level=level, dimension=dim, boundary_conditions=bcs).system_model
    raise SystemExit(f"unknown model {model_name!r} (try SWE or SME)")


def write_inputs(path, ncell, dim_mesh, tend, order, plot_dt, cfl=0.45):
    ncell_line = " ".join([str(ncell)] * 2 + (["1"] if dim_mesh == 3 else []))
    prob_hi = "1.0 1.0 1.0" if dim_mesh == 3 else "1.0 1.0"
    prob_lo = "0.0 0.0 0.0" if dim_mesh == 3 else "0.0 0.0"
    isper = "0 0 0" if dim_mesh == 3 else "0 0"
    path.write_text(f"""amr.max_level     = 0
amr.n_cell        = {ncell_line}
amr.max_grid_size = 64
amr.blocking_factor = 1
geometry.prob_lo  = {prob_lo}
geometry.prob_hi  = {prob_hi}
geometry.is_periodic = {isper}
output.identifier       = 0
output.plot_dt_interval = {plot_dt}
solver.time_end        = {tend}
solver.cfl             = {cfl}
solver.dtmin           = 1.e-7
solver.dtmax           = {tend}
solver.spatial_order   = {order}
solver.implicit_source = false
tagging.threshold      = 0.01
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SWE")
    ap.add_argument("--dim", type=int, default=2,
                    help="model dimension (SWE: horizontal=2; SME: total=3 for 2-D)")
    ap.add_argument("--level", type=int, default=0, help="SME moment level")
    ap.add_argument("--ncell", type=int, default=100)
    ap.add_argument("--tend", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--plot-dt", type=float, default=0.02)
    ap.add_argument("--dim-mesh", type=int, default=3, choices=(2, 3),
                    help="AMReX mesh DIM (3 with nz=1 matches the committed driver)")
    ap.add_argument("--build-dir", default="/tmp/zoomy_amrex_run")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    from zoomy_amrex.transformation import generate_headers

    bdir = Path(a.build_dir)
    src = bdir / "Source"
    ex = bdir / "Exec"
    src.mkdir(parents=True, exist_ok=True)
    ex.mkdir(parents=True, exist_ok=True)

    # stage hand-written C++ driver
    for f in SRC.iterdir():
        if f.suffix in (".cpp", ".H") or f.name == "Make.package":
            shutil.copy2(f, src / f.name)

    # generate device code from the SystemModel
    sm = build_system_model(a.model, a.dim, a.level)
    generate_headers(sm, src)
    print(f"generated headers for {a.model} (state={[str(s) for s in sm.state]})")

    # build files
    amrex_home = os.environ.get("AMREX_HOME", "/opt/amrex")
    (ex / "GNUmakefile").write_text(
        GNUMAKEFILE.format(amrex_home=amrex_home, dim=a.dim_mesh))
    write_inputs(ex / "inputs", a.ncell, a.dim_mesh, a.tend, a.order, a.plot_dt)

    if a.make:
        n = os.cpu_count() or 4
        subprocess.run(["make", f"-j{n}"], cwd=ex, check=True)
    if a.run:
        exe = next((p for p in ex.iterdir()
                    if p.name.startswith("main") and os.access(p, os.X_OK)), None)
        if exe is None:
            raise SystemExit("no executable; pass --make first")
        subprocess.run([f"./{exe.name}", "inputs"], cwd=ex, check=True)
        print(f"run complete; plotfiles in {ex}")


if __name__ == "__main__":
    main()
