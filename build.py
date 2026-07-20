#!/usr/bin/env python3
"""Shared AMReX build helpers — the low-level pieces the solver wrappers reuse.

This module holds ONLY the reusable build primitives: the ``GNUmakefile``
template, the ``inputs``-deck writer (:func:`write_inputs`), and the Chorin
``Make.package`` snippets.  The public, GUI/CLI-facing entry point is
``zoomy_amrex.solvers`` (``HyperbolicSolver`` / ``SplitSolver``), which owns the
codegen → inputs → make → run → VTK pipeline via ``zoomy_amrex.run_case``.

There is deliberately NO case knowledge here and NO dev-CLI: the old
``--malpasset`` / ``--vam`` / ``--vam2d`` flags and their per-case raster/BC
helpers were dissolved (gui REQ-131).  A case is now a thin script that builds a
model (with IC/BC on the model) + a mesh + settings and calls
``solver.solve(model, mesh, settings)``.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Source"

GNUMAKEFILE = """AMREX_HOME ?= {amrex_home}
DEBUG        = FALSE
TINY_PROFILE = {tiny_profile}
USE_MPI      = {use_mpi}
USE_OMP      = FALSE
USE_CUDA     = {use_cuda}
CUDA_ARCH    = {cuda_arch}
COMP         = gnu
DIM          = {dim}

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

# -fmad=false: disable nvcc's automatic fused multiply-add contraction.
# The device eigensolver (Source/ZoomyEig.H) makes ACCEPT/REJECT decisions on
# floating-point comparisons -- QR deflation, the pivot-collapse test, the
# ||L R - I|| gate -- and FMA contraction changes those outcomes between the
# host and device builds.  A face that is a Roe matrix on CPU and a +inf refusal
# on GPU is not a rounding difference, it is a different scheme.
ifeq ($(USE_CUDA),TRUE)
  CXXFLAGS += -fmad=false
endif

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


def write_inputs(path, ncell, dim_mesh, tend, order, plot_dt, cfl=0.45, bc="extrap",
                 implicit_source=False, implicit_global=False, friction=None, slip=None,
                 max_level=0, ref_ratio=2, geom=None, dem_file=None, release_file=None,
                 well_balanced=False, clamp_positivity=True, tag_b_max=None,
                 bc_sides=None, state_rasters=None, positivity="none", dtmax=None,
                 max_step=None, dtmin=None):
    if geom is not None:
        # Externally-supplied rectangular geometry (e.g. the mesh bbox).
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
    if bc_sides:      # per-side model BC tags (structured faces West/East/South/North)
        for side in ("x_lo", "x_hi", "y_lo", "y_hi"):
            if side in bc_sides:
                bc_block += f"bc.{side} = {bc_sides[side]}\n"
    elif bc == "wall":  # closed basin: every side reflective
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
    if state_rasters:   # REQ-123: one raster per state row -> the driver loads all
        fric_block += "init.state_rasters = " + " ".join(state_rasters) + "\n"
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
{f'solver.max_step        = {max_step}' + chr(10) if max_step is not None else ''}solver.cfl             = {cfl}
solver.dtmin           = {'1.e-7' if dtmin is None else dtmin}
solver.dtmax           = {tend if dtmax is None else dtmax}
solver.spatial_order   = {order}
solver.implicit_source = {'true' if implicit_source else 'false'}
solver.implicit_global = {'true' if implicit_global else 'false'}
solver.well_balanced   = {'true' if well_balanced else 'false'}
solver.clamp_positivity = {'true' if clamp_positivity else 'false'}
solver.positivity      = {positivity}
tagging.threshold      = 0.02
{f'tagging.b_max          = {tag_b_max}' if tag_b_max is not None else ''}
""")


_CHORIN_HEADERS = """CEXE_headers += constants.H
CEXE_headers += chorin_common.H
CEXE_headers += ModelPred.H
CEXE_headers += ModelPress.H
CEXE_headers += ModelCorr.H
CEXE_headers += NumericsPred.H
CEXE_headers += UserFunctions.H
CEXE_headers += ZoomyEig.H
"""
CHORIN_MAKE_PACKAGE = ("CEXE_sources += chorin_main.cpp\n"
                       "CEXE_sources += init_solution.cpp\n\n" + _CHORIN_HEADERS)
CHORIN_AMR_MAKE_PACKAGE = ("CEXE_sources += chorin_amr.cpp\n"
                           "CEXE_sources += init_solution.cpp\n\n" + _CHORIN_HEADERS)
