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
USE_MPI      = {use_mpi}
USE_OMP      = FALSE
USE_CUDA     = {use_cuda}
CUDA_ARCH    = {cuda_arch}
COMP         = gnu
DIM          = {dim}

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/MLMG/Make.package
INCLUDE_LOCATIONS += $(AMREX_HOME)/Src/LinearSolvers
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
"""


def build_system_model(model_name, dim, level, bc="extrap"):
    """Return a NumericalSystemModel for the requested model.

    Convention reminder: ``SWE.dimension`` counts *horizontal* dims (2 → 2-D),
    while ``SME.dimension`` counts the total incl. the vertical (3 → 2 horizontal).

    ``bc``: "extrap" (zero-gradient on all sides) or "wall" (a reflective ``wall``
    tag + an ``outer`` extrapolation tag; pick per side in the inputs via
    ``bc.x_lo`` etc.).

    The Model's SystemModel is promoted through
    :func:`to_numerical_system_model`, the same front door the core
    numpy/jax/foam printers use.  For a depth-bearing transport system that runs
    ``desingularize_hinv()`` (Kurganov-Petrova ``1/h -> hinv`` so ``u = q·hinv``
    stays bounded as ``h -> 0``) and ``gate_eigenvalues_dry()`` (dry eigenvalue
    protection).  amrex thus INHERITS the wet/dry regularisation instead of
    relying on ad-hoc floors/caps in the C++ driver.
    """
    from zoomy_core.model import models as M
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, Extrapolation, Wall)
    from zoomy_core.numerics.numerical_system_model import (
        to_numerical_system_model)

    def _wall(mom_idx=None):
        # mom_idx: explicit momentum-reflection groups, needed for SME because
        # core's _state_momentum_groups mis-derives the moment-hierarchy slots
        # (reflects [h, q_x_0] -> negative ghost h -> NaN; core REQ filed). For
        # SWE the auto-derivation is correct, so leave it default.
        w = Wall(tag="wall", momentum_field_indices=mom_idx) if mom_idx else Wall(tag="wall")
        return BoundaryConditions([w, Extrapolation(tag="outer")])

    if model_name == "MalpassetSWE":
        # Canonical viscous wet/dry SWE (zoomy_core.model.models.malpasset):
        # state [b,h,hu,hv] + aux [hinv]. It carries ALL the wet/dry mechanisms
        # natively, so amrex inherits them through the generated code with no
        # driver-side floors/caps:
        #   1. eigenvalues gated for dry cells (conditional(h>eps, ., 0)),
        #   2. KP-desingularized 1/h via the hinv aux (flux/eigenvalues use it),
        #   3. update_variables caps |u|<=U_MAX and zeros dry-cell momentum.
        from zoomy_core.model.models.malpasset import MalpassetSWE
        bcs = _wall([[2, 3]]) if bc == "wall" else BoundaryConditions(
            [Extrapolation(tag="left"), Extrapolation(tag="right")])
        return MalpassetSWE(boundary_conditions=bcs).system_model
    if model_name == "SWE":
        bcs = _wall() if bc == "wall" else BoundaryConditions(
            [Extrapolation(tag="left"), Extrapolation(tag="right")])
        return to_numerical_system_model(
            M.SWE(dimension=dim, boundary_conditions=bcs))
    if model_name == "SME":
        # SME(level=L, dim=3) state = [b, h, q_x_0..q_x_L, q_y_0..q_y_L];
        # wall reflects each moment vector (q_x_k, q_y_k) -> groups [2+k, 3+L+k].
        if bc == "wall":
            groups = [[2 + k, 2 + (level + 1) + k] for k in range(level + 1)] \
                if dim == 3 else [[2 + k] for k in range(level + 1)]
            bcs = _wall(groups)
        else:
            bcs = BoundaryConditions([Extrapolation(tag="left"),
                                      Extrapolation(tag="right")])
        # dim is the TOTAL dimension incl. the vertical (dim=3 -> 2 horizontal).
        # The full closure set makes the moment source self-contained (the slip
        # stress is inlined): friction lives in `lambda_s`, viscosity in `nu`,
        # both settable via params.<name>.  level adds moments q_1..q_level.
        from zoomy_core.model.models.closures import (
            Newtonian, NavierSlip, StressFree)
        return to_numerical_system_model(
            M.SME(level=level, dimension=dim,
                  closures=[Newtonian(), NavierSlip(), StressFree()],
                  boundary_conditions=bcs))
    raise SystemExit(f"unknown model {model_name!r} (try SWE or SME)")


def write_inputs(path, ncell, dim_mesh, tend, order, plot_dt, cfl=0.45, bc="extrap",
                 implicit_source=False, implicit_global=False, friction=None, slip=None,
                 max_level=0, ref_ratio=2, geom=None, dem_file=None, release_file=None,
                 well_balanced=False, clamp_positivity=True, tag_b_max=None):
    if geom is not None:
        # Externally-supplied rectangular geometry (e.g. the Malpasset bbox).
        nx, ny = geom["nx"], geom["ny"]
        gx0, gy0 = geom["prob_lo"]; gx1, gy1 = geom["prob_hi"]
        ncell_line = f"{nx} {ny}" + (" 1" if dim_mesh == 3 else "")
        prob_lo = f"{gx0} {gy0}" + (" 0.0" if dim_mesh == 3 else "")
        prob_hi = f"{gx1} {gy1}" + (" 1.0" if dim_mesh == 3 else "")
    else:
        ncell_line = " ".join([str(ncell)] * 2 + (["1"] if dim_mesh == 3 else []))
        prob_hi = "1.0 1.0 1.0" if dim_mesh == 3 else "1.0 1.0"
        prob_lo = "0.0 0.0 0.0" if dim_mesh == 3 else "0.0 0.0"
    isper = "0 0 0" if dim_mesh == 3 else "0 0"
    bc_block = ""
    if bc == "wall":  # closed basin: every side reflective
        bc_block = ("bc.x_lo = wall\nbc.x_hi = wall\n"
                    "bc.y_lo = wall\nbc.y_hi = wall\n")
    fric_block = ""
    if friction is not None:
        fric_block += f"params.n_m = {friction}\n"
    if slip is not None:
        fric_block += f"params.lambda_s = {slip}\n"
    if dem_file is not None:
        fric_block += f"init.dem_file = {dem_file}\n"
    if release_file is not None:
        fric_block += f"init.release_file = {release_file}\n"
    # AMR: blocking_factor must divide n_cell in every mesh dimension. On DIM=3
    # with nz=1 only bf=1 is legal (and refinement in the degenerate z is moot),
    # so real adaptive refinement needs a DIM=2 mesh — then bf>=2 + ref_ratio
    # refine in x,y.
    bf = 2 if (max_level > 0 and dim_mesh == 2) else 1
    path.write_text(f"""amr.max_level     = {max_level}
amr.n_cell        = {ncell_line}
amr.max_grid_size = 64
amr.blocking_factor = {bf}
amr.ref_ratio     = {ref_ratio}
amr.regrid_int    = 2
geometry.prob_lo  = {prob_lo}
geometry.prob_hi  = {prob_hi}
geometry.is_periodic = {isper}
{bc_block}{fric_block}output.identifier       = 0
output.plot_dt_interval = {plot_dt}
solver.time_end        = {tend}
solver.cfl             = {cfl}
solver.dtmin           = 1.e-7
solver.dtmax           = {tend}
solver.spatial_order   = {order}
solver.implicit_source = {'true' if implicit_source else 'false'}
solver.implicit_global = {'true' if implicit_global else 'false'}
solver.well_balanced   = {'true' if well_balanced else 'false'}
solver.clamp_positivity = {'true' if clamp_positivity else 'false'}
tagging.threshold      = 0.02
{f'tagging.b_max          = {tag_b_max}' if tag_b_max is not None else ''}
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SWE")
    ap.add_argument("--dim", type=int, default=2,
                    help="model dimension (SWE: horizontal=2; SME: total=3 for 2-D)")
    ap.add_argument("--level", type=int, default=0, help="SME moment level")
    ap.add_argument("--ncell", type=int, default=100)
    ap.add_argument("--cfl", type=float, default=0.45)
    ap.add_argument("--no-clamp", action="store_true",
                    help="disable the (non-conservative) positivity clamp; with "
                         "closed walls + CFL<=0.5 this gives exact mass conservation")
    ap.add_argument("--tend", type=float, default=0.1)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--plot-dt", type=float, default=0.02)
    ap.add_argument("--dim-mesh", type=int, default=3, choices=(2, 3),
                    help="AMReX mesh DIM (3 with nz=1 matches the committed driver)")
    ap.add_argument("--bc", default="extrap", choices=("extrap", "wall"),
                    help="extrap (zero-gradient) or wall (closed reflective basin)")
    ap.add_argument("--implicit", action="store_true", help="implicit source")
    ap.add_argument("--well-balanced", action="store_true",
                    help="Audusse hydrostatic reconstruction (lake-at-rest "
                         "preserving; auto-on for --malpasset)")
    ap.add_argument("--implicit-global", action="store_true",
                    help="matrix-free JFNK source solve (nonlocal-capable)")
    ap.add_argument("--friction", type=float, default=None,
                    help="Manning n_m override (params.n_m)")
    ap.add_argument("--slip", type=float, default=None,
                    help="SME slip length lambda_s override (params.lambda_s)")
    ap.add_argument("--max-level", type=int, default=0, help="AMR levels (needs --dim-mesh 2)")
    ap.add_argument("--ref-ratio", type=int, default=2)
    ap.add_argument("--malpasset", metavar="MSH", default=None,
                    help="project the Malpasset triangular mesh onto a structured "
                         "grid: forces SWE/dim-mesh=2, sets the bbox geometry and "
                         "bed/depth rasters from the .msh node data")
    ap.add_argument("--ncell-x", type=int, default=180, help="structured nx (--malpasset)")
    ap.add_argument("--ncell-y", type=int, default=96, help="structured ny (--malpasset)")
    ap.add_argument("--wall-above", type=float, default=5.0,
                    help="exterior wall bed = reservoir surface + this (m)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="bbox padding (fraction of span) so the domain is fully "
                         "ringed by wall cells -> a closed, exactly-conserving basin")
    ap.add_argument("--build-dir", default="/tmp/zoomy_amrex_run")
    ap.add_argument("--gpu", action="store_true",
                    help="build for CUDA GPU (USE_CUDA=TRUE, single device: "
                         "USE_MPI=FALSE); needs nvcc + an --nv container")
    ap.add_argument("--cuda-arch", default="89",
                    help="CUDA compute capability (L40S/Ada=89)")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    from zoomy_amrex.transformation import generate_headers

    geom = dem_file = release_file = None
    if a.malpasset:
        from zoomy_amrex.malpasset import prepare_rasters
        a.model, a.dim, a.dim_mesh = "MalpassetSWE", 2, 2  # canonical wet/dry SWE
        a.well_balanced = True                     # real bathymetry needs WB
        a.bc = "wall"                              # closed basin
        # No positivity clamp: the model's update_variables (applied after each
        # update) caps momentum and keeps the wavespeed/dt bounded, and the model
        # operators are finite for any depth, so the run is stable without
        # clamping h -> mass is conserved to machine precision.
        a.no_clamp = True
        if a.pad == 0.0:
            a.pad = 0.04   # closed basin (domain ringed by wall) -> exact mass
        geom = prepare_rasters(a.malpasset, str(Path(a.build_dir) / "raster"),
                               ncell_x=a.ncell_x, ncell_y=a.ncell_y,
                               wall_above=a.wall_above, pad_frac=a.pad)
        dem_file, release_file = geom["dem_file"], geom["release_file"]
        print(f"malpasset raster: {geom['nx']}x{geom['ny']} "
              f"prob_lo={geom['prob_lo']} prob_hi={geom['prob_hi']} "
              f"dx={geom['dx']:.1f} dy={geom['dy']:.1f} "
              f"wet={100*geom['wet_frac']:.1f}% bed=[{geom['bed_min']:.1f},"
              f"{geom['bed_max']:.1f}] wall={geom['wall_bed']:.1f}")

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
    sm = build_system_model(a.model, a.dim, a.level, bc=a.bc)
    generate_headers(sm, src)
    print(f"generated headers for {a.model} (state={[str(s) for s in sm.state]}, bc={a.bc})")

    # build files
    amrex_home = os.environ.get("AMREX_HOME", "/opt/amrex")
    (ex / "GNUmakefile").write_text(
        GNUMAKEFILE.format(amrex_home=amrex_home, dim=a.dim_mesh,
                           use_mpi="FALSE" if a.gpu else "TRUE",
                           use_cuda="TRUE" if a.gpu else "FALSE",
                           cuda_arch=a.cuda_arch))
    write_inputs(ex / "inputs", a.ncell, a.dim_mesh, a.tend, a.order, a.plot_dt, bc=a.bc,
                 cfl=a.cfl, implicit_source=a.implicit, implicit_global=a.implicit_global,
                 friction=a.friction, slip=a.slip,
                 max_level=a.max_level, ref_ratio=a.ref_ratio,
                 geom=geom, dem_file=dem_file, release_file=release_file,
                 well_balanced=a.well_balanced, clamp_positivity=not a.no_clamp,
                 tag_b_max=(geom["wall_bed"] - 1.0) if geom is not None else None)

    if a.make:
        n = os.cpu_count() or 4
        make_cmd = ["make", f"-j{n}"]
        if a.gpu:
            # Pass CUDA_ARCH as a command-line make var (highest precedence) so
            # AMReX's auto-detect / AMREX_CUDA_ARCH override cannot clobber it.
            make_cmd.append(f"CUDA_ARCH={a.cuda_arch}")
        subprocess.run(make_cmd, cwd=ex, check=True)
    if a.run:
        exe = next((p for p in ex.iterdir()
                    if p.name.startswith("main") and os.access(p, os.X_OK)), None)
        if exe is None:
            raise SystemExit("no executable; pass --make first")
        subprocess.run([f"./{exe.name}", "inputs"], cwd=ex, check=True)
        print(f"run complete; plotfiles in {ex}")


if __name__ == "__main__":
    main()
