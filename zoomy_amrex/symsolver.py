"""Driver of the symbolic solvers on AMReX: prints the C++ solver, builds it in the AMReX apptainer,
runs it on the mesh's uniform Cartesian grid and reads the result back.  Sits behind the C++ solver
template (`zoomy_core.symsolver.backend_cxx`) and in front of `Source/symsolver`.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import param

from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.mesh.geometry_xp import build_topology, compute_geometry
from zoomy_core.misc.options import OptionParam
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_core.numerics.boundary_closure import closure_kinds
from zoomy_core.symsolver.backend_cxx import RUNTIME_HEADER, CxxBackend, CxxChorinBackend
from zoomy_amrex.symkernels import write_kernels

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Source"
CACHE = Path(os.environ.get("ZOOMY_AMREX_SYMCACHE", ROOT / ".symcache"))
# the superrepo's own images (containers/zoomy_amrex, with preCICE), else the
# Zoomy checkout's
_OWN = ROOT.parent.parent / "containers" / "zoomy_amrex"
CONTAINERS = Path(os.environ.get("ZOOMY_AMREX_CONTAINERS",
                                 _OWN if (_OWN / "zoomy_amrex.sif").exists()
                                 else Path.home() / "git/Zoomy/containers/zoomy_amrex"))
SIF = {False: CONTAINERS / "zoomy_amrex.sif", True: CONTAINERS / "zoomy_amrex_gpu.sif"}
# hypre for the linear solves (static libHYPRE.a, built once per variant by
# containers/zoomy_amrex/build_hypre_cuda.sh into ~/.cache/zoomy)
HYPRE_VERSION = "3.2.0"
HYPRE_CACHE = Path(os.environ.get("ZOOMY_HYPRE_CACHE", Path.home() / ".cache" / "zoomy"))
HYPRE_SCRIPT = ROOT.parent.parent / "containers" / "zoomy_amrex" / "build_hypre_cuda.sh"


def _cpus(var, default):
    """The taskset CPU list: `$var`, else `default` when this process may run on one of its CPUs,
    else all the CPUs it may run on (a machine with fewer cores: taskset fails on absent CPUs).
    """
    if var in os.environ:
        return os.environ[var]
    lo, hi = map(int, default.split("-"))
    mine = sorted(os.sched_getaffinity(0))
    return default if any(lo <= c <= hi for c in mine) else ",".join(map(str, mine))


BUILD_CPUS = _cpus("ZOOMY_AMREX_BUILD_CPUS", "32-63")
RUN_CPUS = _cpus("ZOOMY_AMREX_RUN_CPUS", "16-31")
APP = ("symsolver/symAmrex.cpp", "symsolver/rt_amrex.H", "symsolver/amr_amrex.H",
       "UserFunctions.H", "ZoomyEig.H")


def _apptainer(gpu, script, binds=(), **kw):
    """Run the shell `script` in the CPU or GPU AMReX apptainer with the cache and `binds`
    mounted.
    """
    cmd = ["apptainer", "exec", "--cleanenv", *(["--nv"] if gpu else [])]
    for b in sorted({str(CACHE), *map(str, binds)}):
        cmd += ["--bind", b]
    return subprocess.run(cmd + [str(SIF[gpu]), "bash", "-c", script], **kw)


_AMREX_SOLVERS = [LinearSolver.hypre, LinearSolver.gmres]


class AmrexDriver(param.Parameterized):
    """Builds, launches and reads back the AMReX C++ solver of a `backend_cxx` template: build
    tree, Cartesian case with LSQ tables and inputs, optional AMR levels, `mpirun`.

    References:
        Zhang et al. 2019, https://doi.org/10.21105/joss.01370
        Berger, Oliger 1984, https://doi.org/10.1016/0021-9991(84)90073-1
        Berger, Colella 1989, https://doi.org/10.1016/0021-9991(89)90035-1
    """
    _gpu_boxes = 1          # boxes per GPU of the default max_grid_size
    ranks = param.Integer(default=1, bounds=(1, None), doc="MPI ranks (GPUs with gpu=True).")
    gpu = param.Boolean(default=False, doc="CUDA build (one GPU per rank).")
    max_grid_size = param.Integer(default=None, allow_None=True,
                                  doc="AMReX box size; None: about 2 boxes per rank.")
    tiny_profile = param.Boolean(default=False,
                                 doc="AMReX TinyProfiler build (a tree of its own).")
    build_only = param.Boolean(default=False,
                               doc="run() only builds (and caches) the binary.")
    linear_solver = OptionParam(
        LinearSolver, default=LinearSolver.hypre, members=_AMREX_SOLVERS, doc=(
        "Linear solves (linsolve): 'hypre' = BoomerAMG-preconditioned FlexGMRES on the assembled, "
        "cell-block-scaled operator; 'gmres' = amrex::GMRES with the cell-block Jacobi "
        "preconditioner."))
    amr = param.Dict(default=None, allow_None=True, doc=(
        "AMR (one dt for all levels; amr_amrex.H): max_level, ratio (2), "
        "regrid_int (0: the initial grids only), tag = (state name, 'value' | "
        "'gradient', threshold) or region = (lo, hi), reflux (True), "
        "blocking_factor (8), n_error_buf (2), grid_eff (0.7).  The CFL step "
        "from the cells' wave speeds."))

    def _amr_loop(self):
        """True when AMR is configured."""
        return self.amr is not None

    # ── build ──────────────────────────────────────────────────────────────
    def build(self, nsm, dimension, classes=None):
        """Print the solver, write the build tree and make it, cached by a hash of the sources.
        Called by `run`.

        Args:
            classes: boundary-closure classes of the case (`lsq_fit`); without, no closure in
                the derivative aux.
        Returns:
            Path of the binary.
        """
        self.source = self._print(nsm, dimension)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"kernels_{os.getpid()}_{threading.get_ident()}.H"   # per thread
        write_kernels(self._systems(nsm), self.source, tmp, dimension, classes)
        files = {"Solver.H": self.source, "Kernels.H": tmp.read_text()}
        tmp.unlink()
        for f in APP:
            files[Path(f).name] = (SRC / f).read_text()
        files[RUNTIME_HEADER.name] = RUNTIME_HEADER.read_text()
        cpl = bool(self.couple)
        hypre = hypre_prefix(self.gpu) if self.linear_solver == "hypre" else None
        make = _gnumakefile(self.gpu, self.tiny_profile, dimension, cpl, hypre)
        h = hashlib.sha256(make.encode())          # the build flags, then the sources
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
        binary = CACHE / "bin" / h.hexdigest()[:16] / "symAmrex.ex"
        if binary.exists():
            return binary
        tree = CACHE / (("gpu" if self.gpu else "cpu") + f"{dimension}d"
                        + ("_tp" if self.tiny_profile else "") + ("_cpl" if cpl else "")
                        + ("_hypre" if hypre else ""))
        (tree / "Source").mkdir(parents=True, exist_ok=True)
        (tree / "Exec").mkdir(parents=True, exist_ok=True)
        with open(tree / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if binary.exists():                     # built while this one waited
                return binary
            for k, v in files.items():
                p = tree / "Source" / k
                if not p.exists() or p.read_text() != v:
                    p.write_text(v)
            (tree / "Source" / "Make.package").write_text(
                "CEXE_sources += symAmrex.cpp\nCEXE_headers += rt_common.H rt_amrex.H amr_amrex.H "
                "Kernels.H Solver.H UserFunctions.H ZoomyEig.H\n")
            (tree / "Exec" / "GNUmakefile").write_text(make)
            r = _apptainer(self.gpu, f"cd {tree}/Exec && taskset -c {BUILD_CPUS} "
                           f"make -j32 > make.log 2>&1; rc=$?; tail -30 make.log; "
                           f"exit $rc", binds=[hypre] if hypre and hypre.exists() else (),
                           capture_output=True, text=True)
            exe = sorted((tree / "Exec").glob(f"main{dimension}d*.ex"), key=os.path.getmtime)
            if r.returncode != 0 or not exe:
                raise RuntimeError(f"symAmrex make failed ({tree}/Exec/make.log):\n"
                                   f"{r.stdout}\n{r.stderr}")
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exe[-1], binary.with_suffix(".tmp"))
            os.replace(binary.with_suffix(".tmp"), binary)     # never over a running binary
            (binary.parent / "Solver.H").write_text(files["Solver.H"])
            (binary.parent / "Kernels.H").write_text(files["Kernels.H"])
        return binary

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, mesh, model, Q0, workdir, Qaux0=None, fine=()):
        """March `Q0` (mesh order) to `time_end` and return `(time, Q)`; snapshots, wall time
        and (AMR) per-level states stay in `self.snapshots`, `self.wall`, `self.initial` and
        `self.levels`.

        Args:
            fine: AMR only, `[(mesh_l, Q0_l)]` the initial state of level l on its refined domain
                (else interpolated).
        Returns:
            `(time, Q)` with `Q` on level 0 (the composite averaged down); the binary if
                `build_only`.
        """
        nsm = self._prepare(mesh, model)
        if (self.amr or self.couple) and (self.restart_from is not None
                                          or self.dt_checkpoint is not None):
            raise NotImplementedError("restart / checkpoints with AMR or preCICE coupling")
        restart = None if self.restart_from is None else restart_dir(self.restart_from)
        grid = cartesian(mesh)
        dim = grid["dim"]
        branch, periodic = sides(mesh, nsm, dim)
        fit = self._lsq_fit(nsm, grid, branch)
        binary = self.build(nsm, dim, fit["classes"])
        if self.build_only:
            return binary
        case = Path(workdir).resolve()
        staged = None
        if restart is not None and case in (restart, *restart.parents):
            staged = Path(tempfile.mkdtemp(dir=case.parent))   # out of the case recreated here
            restart = Path(shutil.copytree(restart, staged / "restart"))
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        if restart is not None:
            shutil.copytree(restart, case / "restart")
        if staged is not None:
            shutil.rmtree(staged)
        n = grid["n"]
        to_grid = lambda Q: np.asarray(Q, dtype=np.float64)[:, grid["order"]]  # noqa: E731
        to_grid(Q0).tofile(case / "q0.bin")
        n_aux = len(nsm.aux_state)
        A0 = np.zeros((max(n_aux, 1), grid["nc"])) if Qaux0 is None else Qaux0
        to_grid(A0).tofile(case / "qaux0.bin")
        (case / "lsq.txt").write_text(lsq_table(fit, grid, periodic, _owner(nsm)))
        amr = self._amr_inputs(case, nsm, grid, periodic, branch, fine) if self.amr else ""
        if self.couple:
            c = self.couple
            amr += (f"zs.precice = {c['participant']} {c['config']} {c['mesh']}\n"
                    f"zs.precice_write = {' '.join(c['write'])}\n"
                    f"zs.precice_read = {' '.join(c['read'])}\n"
                    f"zs.precice_side = {side_of(mesh, c['tag'], dim)}\n")
        # boxes per rank: CPU ~8 (load balance across wet / dry / wall cells),
        # GPU `_gpu_boxes` (hyperbolic 1: one launch covers all boxes)
        per = self._gpu_boxes if self.gpu else 8
        # box side from the cell count, so an elongated domain still gets ~per boxes per rank
        mgs = self.max_grid_size or max(16, int(np.ceil(
            (np.prod(n) / (per * self.ranks)) ** (1.0 / dim) - 1e-9)))
        f = lambda v: " ".join(repr(float(x)) for x in v)                  # noqa: E731
        ckpt = None if self.checkpoint is None else Path(self.checkpoint).resolve()
        if ckpt is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
        (case / "inputs").write_text(
            f"zs.n_cell = {' '.join(map(str, n))}\nzs.prob_lo = {f(grid['lo'])}\n"
            f"zs.prob_hi = {f(grid['hi'])}\nzs.periodic = {' '.join(map(str, periodic))}\n"
            f"zs.max_grid_size = {mgs}\nzs.time_end = "
            f"{1e30 if self.time_end is None else self.time_end!r}\n"
            f"zs.branch = {' '.join(map(str, branch))}\n"
            f"zs.lsq = lsq.txt\nzs.q0 = q0.bin\nzs.qaux0 = qaux0.bin\nzs.out = out\n"
            f"zs.time_start = {float(self.time_start)!r}\n"
            + ("" if restart is None else "zs.restart = restart\n")
            + ("" if ckpt is None else f"zs.checkpoint = {ckpt}\n")
            + f"zs.linsolve = {self.linear_solver}\n"
            "amrex.the_arena_init_size = 1073741824\namrex.verbose = 0\n"
            "tiny_profiler.device_synchronize_around_region = 1\n" + amr)
        launch = (f"mpirun --oversubscribe --bind-to none -n {self.ranks} "
                  if self.ranks > 1 or self.gpu else "")
        cpus = "" if self.gpu else f"taskset -c {RUN_CPUS} "
        r = _apptainer(self.gpu, f"cd {case} && {cpus}{launch}{binary} inputs > log.symAmrex 2>&1",
                       binds=[case, binary.parent] + ([] if ckpt is None else [ckpt]),
                       capture_output=True, text=True)
        log = (case / "log.symAmrex").read_text()
        if r.returncode != 0:
            raise RuntimeError(f"symAmrex failed, see {case}/log.symAmrex:\n{log[-3000:]}\n{r.stderr}")
        self.log = log
        self.wall = next((float(ln.split()[-2]) for ln in log.splitlines()
                          if "march wall time" in ln), None)
        self.steps = next((int(ln.split()[-1]) for ln in log.splitlines()
                           if "loop iterations" in ln), None)
        if self.amr:
            self.levels = [_read(case / "out" / f"final_l{lv}.bin", g)[1]
                           for lv, g in enumerate(self._grids)
                           if (case / "out" / f"final_l{lv}.bin").exists()]
            self.snapshots = []
            self.initial = _read(case / "out" / "initial.bin", grid)[1]
            return _read(case / "out" / "final.bin", grid)
        snaps = sorted((case / "out").glob("[0-9]*.bin"), key=lambda f: int(f.stem))
        self.snapshots = [_read(f, grid) for f in snaps]
        return _read(case / "out" / "final.bin", grid)

    def _lsq_fit(self, nsm, grid, branch):
        """LSQ fits of `grid` with the boundary closure of every system the template
        differentiates.
        """
        return lsq_fit(nsm, [s for _, s in self._systems(nsm)[1:]], self._degree(nsm),
                       grid, branch)

    def _amr_inputs(self, case, nsm, grid, periodic, branch, fine):
        """Per-level LSQ tables and initial states; returns the AMR block of the inputs file."""
        a = {**dict(ratio=2, regrid_int=0, reflux=True, blocking_factor=8, n_error_buf=2,
                    grid_eff=0.7), **self.amr}
        r, dim = int(a["ratio"]), grid["dim"]
        self._grids, lsq, q0 = [], [], []
        for lv in range(int(a["max_level"]) + 1):
            n = [n * r**lv for n in grid["n"]]
            g = dict(grid, n=n, d=[d / r**lv for d in grid["d"]], nc=int(np.prod(n)),
                     order=np.arange(int(np.prod(n))))          # grid order
            if 0 < lv <= len(fine):
                g = cartesian(fine[lv - 1][0])
                np.asarray(fine[lv - 1][1], dtype=np.float64)[:, g["order"]].tofile(
                    case / f"q0_{lv}.bin")
            self._grids.append(g)
            (case / f"lsq_{lv}.txt").write_text(
                lsq_table(self._lsq_fit(nsm, g, branch), g, periodic, _owner(nsm)))
            lsq.append(f"lsq_{lv}.txt")
            q0.append("q0.bin" if lv == 0 else (f"q0_{lv}.bin" if lv <= len(fine) else "-"))
        names = [str(v) for v in nsm.state]
        tag = ""
        if "tag" in a:
            row, mode, thr = a["tag"]
            tag = f"zs.tag = {['value', 'gradient'].index(mode)} {names.index(row)} {float(thr)!r}\n"
        if "region" in a:
            tag = (f"zs.tag_lo = {' '.join(map(repr, map(float, a['region'][0])))}\n"
                   f"zs.tag_hi = {' '.join(map(repr, map(float, a['region'][1])))}\n")
        mgs = self.max_grid_size or 32
        f = lambda v: " ".join(repr(float(x)) for x in v)                  # noqa: E731
        return (f"zs.amr = 1\nzs.lsq = {' '.join(lsq)}\nzs.q0 = {' '.join(q0)}\n{tag}"
                f"zs.regrid_int = {int(a['regrid_int'])}\nzs.reflux = {int(bool(a['reflux']))}\n"
                f"zs.dt_write = {float(self.dt_write or 0.0)!r}\n"
                f"amr.n_cell = {' '.join(map(str, grid['n']))}\namr.max_level = {a['max_level']}\n"
                f"amr.ref_ratio = {r}\namr.blocking_factor = {a['blocking_factor']}\n"
                f"amr.max_grid_size = {mgs}\namr.n_error_buf = {a['n_error_buf']}\n"
                f"amr.grid_eff = {a['grid_eff']!r}\namr.n_proper = 2\n"
                f"geometry.coord_sys = 0\ngeometry.prob_lo = {f(grid['lo'])}\n"
                f"geometry.prob_hi = {f(grid['hi'])}\n"
                f"geometry.is_periodic = {' '.join(map(str, periodic))}\n")


class SymbolicHyperbolicSolverAmrex(AmrexDriver, CxxBackend):
    """The explicit hyperbolic solver template on AMReX; its linear solves (Newton-Krylov of the
    implicit diffusion, finite-difference Jacobians) default to amrex::GMRES.
    """
    linear_solver = OptionParam(LinearSolver, default=LinearSolver.gmres, members=_AMREX_SOLVERS)


class SymbolicChorinSolverAmrex(AmrexDriver, CxxChorinBackend):
    """The Chorin projection solver template on AMReX; the model is the triple ``(pred, press,
    corr)``.

    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """
    _gpu_boxes = 4          # hypre on 2 / 4 GPUs, L2P: 571 -> 445 / 392 ms per step


# ── the grid ────────────────────────────────────────────────────────────────
def cartesian(mesh):
    """The mesh's uniform Cartesian grid (1-3-D) as a dict: `lo, hi, n, d`, the grid order of the
    cells (x fastest) and each cell's grid index.  Anything but a full uniform grid raises
    `NotImplementedError`.
    """
    dim = int(mesh.dimension)
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:dim, :nc]
    vx = np.asarray(mesh.vertex_coordinates)[:dim]
    lo, hi, n, d = [], [], [], []
    for ax in range(dim):
        u = np.unique(np.round(cc[ax], 9))          # the distinct lines (count only)
        a, b = float(vx[ax].min()), float(vx[ax].max())       # the domain: its vertices
        h = (b - a) / len(u)
        if len(u) > 1 and not np.allclose(np.diff(u), h, rtol=1e-6, atol=0):
            raise NotImplementedError("zoomy_amrex symsolver: uniform Cartesian grids only")
        lo.append(a); hi.append(b)
        n.append(len(u)); d.append(h)
    if int(np.prod(n)) != nc:
        raise NotImplementedError("zoomy_amrex symsolver: a full Cartesian grid only")
    ij = [np.rint((cc[ax] - lo[ax]) / d[ax] - 0.5).astype(int) for ax in range(dim)]
    flat = np.zeros(nc, dtype=int)
    for ax in reversed(range(dim)):
        flat = flat * n[ax] + ij[ax]
    order = np.empty(nc, dtype=int)
    order[flat] = np.arange(nc)
    return dict(dim=dim, lo=lo, hi=hi, n=n, d=d, nc=nc, order=order, ij=ij)


def sides(mesh, nsm, dim):
    """BC branch of the x-lo, x-hi, y-lo, ... sides (by outward normal; -1 on a periodic side) and
    the periodic flag per direction.
    """
    from zoomy_core.model.boundary_conditions import Periodic
    bcs = getattr(nsm.sm, "_bc_source", None)
    periodic_tags = {b.tag for b in getattr(bcs, "boundary_conditions_list", [])
                     if isinstance(b, Periodic)}
    names = list(mesh.boundary_conditions_sorted_names)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)
    br = np.asarray(boundary_face_branches(mesh, bcs))
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)[:len(br)]
    normals = np.asarray(mesh.face_normals)[:dim, faces]
    branch, periodic = [], []
    for ax in range(dim):
        per = False
        for s in (-1, 1):
            hit = np.flatnonzero(np.isclose(normals[ax], s))
            tags = {names[nums[i]] for i in hit}
            if tags & periodic_tags:
                per = True
                branch.append(-1)
                continue
            vals = np.unique(br[hit])
            if len(vals) != 1:
                raise NotImplementedError(f"side {ax}{s}: BC branches {vals}, one per side only")
            branch.append(int(vals[0]))
        periodic.append(int(per))
    return branch + [-1] * (6 - len(branch)), periodic


def side_of(mesh, tag, dim):
    """The side (`2 * axis + upper`) of the boundary faces tagged `tag`."""
    names = list(mesh.boundary_conditions_sorted_names)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)
    hit = [i for i in range(len(nums)) if names[nums[i]] == tag]
    n = np.asarray(mesh.face_normals)[:dim, faces[hit]].mean(axis=1)
    ax = int(np.argmax(np.abs(n)))
    return 2 * ax + int(n[ax] > 0)


def _owner(nsm):
    """The name a system's closure classes are keyed by (`lsq_closure_class`)."""
    return getattr(nsm, "_symbolic_name", None) or "nsm"


def lsq_fit(nsm, systems, degree, grid, branch):
    """LSQ topology and geometry (`lsq_gradQ`; for degree >= 2 also the boundary closure
    `lsq_closure`) of a proxy mesh with `grid`'s spacing, its sides carrying the case's BC
    branches.  Closure classes are renumbered by the kinds of the six sides, so a class has the same
    id on every proxy (grid size, AMR level) and in `Kernels.H`.

    Args:
        systems: further systems the template differentiates besides `nsm`.
        branch: BC branch per side (`sides`; -1: periodic, no closure).
    Returns:
        dict `proxy, np, topo, geom, order, classes`; `classes` maps `(owner, kind)` (`kind`
        `"state"` or `"reconstruction"`) to the class per row.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    dim, n, lo, h = grid["dim"], grid["n"], grid["lo"], grid["d"]
    np_ = [min(n[ax], 2 * degree + 3) for ax in range(dim)]
    dom = [v for ax in range(dim) for v in (lo[ax], lo[ax] + np_[ax] * h[ax])]
    proxy = (LSQMesh.create_1d(domain=tuple(dom), n_inner_cells=np_[0]) if dim == 1 else
             LSQMesh.create_2d(domain=tuple(dom), nx=np_[0], ny=np_[1]) if dim == 2 else
             LSQMesh.create_3d(domain=tuple(dom), nx=np_[0], ny=np_[1], nz=np_[2]))
    bf = np.asarray(proxy.boundary_face_face_indices, dtype=int)
    normals = np.asarray(proxy.face_normals)[:dim, bf]
    axis = np.abs(normals).argmax(axis=0)
    side = 2 * axis + (normals[axis, np.arange(bf.size)] > 0)       # x-lo, x-hi, y-lo, ...
    kinds = (closure_kinds(nsm, systems, np.asarray(branch)[side], normals)
             if degree >= 2 else {})
    topo = build_topology(proxy, lsq_degree=degree, closure=kinds)
    geom = compute_geometry(topo, np.asarray(proxy.vertex_coordinates)[:dim])
    first = [int(np.flatnonzero(side == s)[0]) for s in range(2 * dim)]
    kind_of = {k: tuple(int(kinds[key][r, f]) for f in first)
               for key, cls in topo.lsq_closure_class.items()
               for r, k in enumerate(cls) if k >= 0}
    order = sorted(kind_of, key=kind_of.get)
    ident = {k: i for i, k in enumerate(order)}
    classes = {key: tuple(ident.get(k, -1) for k in cls)
               for key, cls in topo.lsq_closure_class.items()}
    return dict(proxy=proxy, np=np_, topo=topo, geom=geom, order=order, classes=classes)


def lsq_table(fit, grid, periodic, owner):
    """Text table of the LSQ stencils and weights per cell class (distance to each side, up to the
    stencil reach), read by `zs::Mesh::load`.  Each class has the base list and one list per
    boundary-closure class; a periodic axis uses the interior class, its seam samples being halo
    cells.

    Args:
        fit: result of `lsq_fit`.
        owner: system whose reconstruction classes (the O2 gradient) follow the monomials.
    Returns:
        The table as text.
    References:
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
    """
    dim, n, h = grid["dim"], grid["n"], grid["d"]
    np_, proxy, topo, geom = fit["np"], fit["proxy"], fit["topo"], fit["geom"]
    pg = cartesian(proxy)
    nc = pg["nc"]
    cc = np.asarray(proxy.cell_centers)[:dim]
    nbrs = np.asarray(topo.lsq_neighbors)
    bnbr = np.asarray(topo.lsq_boundary_face_neighbors)
    A = np.asarray(geom["lsq_gradQ"])
    T = np.asarray(geom["lsq_closure"])
    cl = topo.lsq_plan.get("closure")
    cl_cells = {int(c): i for i, c in enumerate(cl["cells"])} if cl is not None else {}
    scale = np.asarray(topo.lsq_scale_factors, dtype=float)
    mon = [tuple(int(x) for x in m) for m in topo.lsq_monomial_multi_index]
    fcen = np.asarray(proxy.face_centers)[:, :dim].T
    bf_face = np.asarray(proxy.boundary_face_face_indices, dtype=int)
    bf_cell = np.asarray(proxy.boundary_face_cells, dtype=int)
    d = np.asarray(h)

    def samples(c, W):      # W (R, n_mon) in lsq_gradQ's layout: stencil cells, boundary slots
        """Samples `(kind, offset, weights)` of cell `c` from the rows `W` of `lsq_gradQ`'s
        layout (kind 1: boundary ghost).
        """
        row, smp = 0, []
        for nb in nbrs[c]:
            w = scale * W[row]
            row += 1
            if nb == c or nb >= nc or not np.any(w):
                continue
            smp.append((0, *np.rint((cc[:, nb] - cc[:, c]) / d).astype(int), w))
        for bf in bnbr[c]:
            w = scale * W[row]
            row += 1
            if bf < 0 or not np.any(w):
                continue
            ghost = 2 * fcen[:, bf_face[bf]] - cc[:, bf_cell[bf]]
            smp.append((1, *np.rint((ghost - cc[:, c]) / d).astype(int), w))
        return smp

    def lists(c):           # the base list, then the closure classes' (None: the base)
        """The base sample list of cell `c`, then one per closure class (None: the base list
        applies).
        """
        out = [samples(c, A[c])]
        for k in fit["order"]:
            i = cl_cells.get(c)
            out.append(samples(c, T[k, i]) if i is not None and cl["active"][k, i] else None)
        return out
    allv = {c: lists(c) for c in range(nc)}
    R = max(max(abs(int(o)) for o in s[1:1 + dim]) for ls in allv.values()
            for smp in ls if smp for s in smp)
    for ax in range(dim):
        if np_[ax] < n[ax] and np_[ax] < 2 * R + 1:
            raise RuntimeError(f"LSQ reach {R}: proxy grid too small")

    def key(i, m, per):     # a periodic axis has no sides: the interior stencil,
        """Class of index `i` of `m` cells along an axis: distance to each side up to the reach
        `R`; a periodic axis has the interior class.
        """
        return (R, R) if per else (min(i, R), min(m - 1 - i, R))   # wrapped by the halo
    keys = [sorted({key(i, np_[ax], periodic[ax]) for i in range(np_[ax])})
            for ax in range(dim)]
    rep = {}                # each class: a proxy cell of it (the proxy is not periodic)
    for c in range(nc):
        kc = [key(pg["ij"][ax][c], np_[ax], False) for ax in range(dim)]
        if all(kc[ax] in keys[ax] for ax in range(dim)):
            rep.setdefault(tuple(keys[ax].index(kc[ax]) for ax in range(dim)), c)
    cls = [[keys[ax].index(key(i, n[ax], periodic[ax])) for i in range(n[ax])]
           for ax in range(dim)]
    recon = fit["classes"].get((owner, "reconstruction"), ())
    out = [f"{R} {dim} {' '.join(map(str, n))} {' '.join(str(len(k)) for k in keys)} "
           f"{len(mon)} {len(fit['order'])}",
           *(" ".join(map(str, cl_)) for cl_ in cls),
           " ".join(" ".join(map(str, m)) for m in mon),
           f"{len(recon)} {' '.join(map(str, recon))}".strip()]
    for k in np.ndindex(*[len(kk) for kk in keys]):
        for smp in allv[rep[tuple(int(x) for x in k)]]:
            if smp is None:
                out.append("-1")
                continue
            out.append(str(len(smp)))
            out += [f"{s[0]} {' '.join(str(int(o)) for o in s[1:1 + dim])} "
                    + " ".join(repr(float(x)) for x in s[-1]) for s in smp]
    return "\n".join(out) + "\n"


def restart_dir(path):
    """The checkpoint directory `chk<t>` (of `io.save_state`) at `path`: itself or its latest."""
    from zoomy_core.symsolver.restart import missing
    p = Path(path).resolve()

    def time_of(d):
        """The time in `d/Header`; None: no checkpoint."""
        try:
            fmt, t = (d / "Header").read_text().split()[:2]
        except (OSError, ValueError):
            return None
        return float(t) if fmt == "zoomy-restart-1" else None
    if not p.exists():
        raise missing(p)
    if time_of(p) is not None:
        return p
    found = [(time_of(d), d) for d in p.iterdir() if d.is_dir()] if p.is_dir() else []
    found = [(t, d) for t, d in found if t is not None]
    if not found:
        raise FileNotFoundError(f"restart_from {str(path)!r}: no AMReX restart state (a chk<t> "
                                "directory with a zoomy-restart-1 Header, or a directory holding "
                                "one)")
    return max(found)[1]


def _read(path, grid):
    """Read an output file of the app; returns `(time, Q)` with `Q` in mesh order."""
    raw = np.fromfile(path, dtype=np.float64)
    dim = grid["dim"]
    time, nq = raw[0], int(raw[1])
    Q = raw[2 + dim:].reshape(nq, -1)
    out = np.empty_like(Q)
    out[:, grid["order"]] = Q
    return float(time), out


def hypre_prefix(gpu):
    """The hypre install of the CPU or GPU build (`~/.cache/zoomy/hypre-{cpu,cuda}-3.2.0`),
    built first by `containers/zoomy_amrex/build_hypre_cuda.sh` when missing.

    References:
        Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
    """
    if os.environ.get("ZOOMY_HYPRE_DIR"):       # e.g. /opt/hypre baked into the image
        return Path(os.environ["ZOOMY_HYPRE_DIR"])
    variant = "cuda" if gpu else "cpu"
    prefix = HYPRE_CACHE / f"hypre-{variant}-{HYPRE_VERSION}"
    if (prefix / "lib" / "libHYPRE.a").exists():
        return prefix
    if not HYPRE_SCRIPT.exists():
        raise RuntimeError(f"no hypre in {prefix}: build it with build_hypre_cuda.sh {variant} "
                           "(Zoomy containers/zoomy_amrex), or linear_solver=LinearSolver.gmres")
    HYPRE_CACHE.mkdir(parents=True, exist_ok=True)
    with open(HYPRE_CACHE / f".hypre-{variant}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (prefix / "lib" / "libHYPRE.a").exists():
            r = subprocess.run(["bash", str(HYPRE_SCRIPT), variant], capture_output=True, text=True,
                               env=dict(os.environ, ZOOMY_AMREX_CONTAINERS=str(CONTAINERS),
                                        ZOOMY_HYPRE_CACHE=str(HYPRE_CACHE),
                                        ZOOMY_AMREX_BUILD_CPUS=BUILD_CPUS))
            if r.returncode != 0:
                raise RuntimeError(f"hypre build failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return prefix


def _gnumakefile(gpu, tiny_profile=False, dim=2, precice=False, hypre=None):
    """Text of the AMReX GNUmakefile (CPU or GPU, dimension, optional preCICE, optional hypre
    install `hypre`)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_zoomy_amrex_build", ROOT / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mk = mod.GNUMAKEFILE.format(amrex_home="/opt/amrex", dim=dim, use_mpi="TRUE",
                                use_cuda="TRUE" if gpu else "FALSE", cuda_arch="89",
                                tiny_profile="TRUE" if tiny_profile else "FALSE")
    if hypre is not None:   # static libHYPRE.a, our own init (AMReX 25.02's USE_HYPRE predates hypre 3)
        libs = "-lHYPRE" + (" -lcusparse -lcurand -lcublas -lcusolver" if gpu else "")
        mk = mk.replace("include ../Source/Make.package",
                        f"DEFINES += -DZS_HYPRE\nSYSTEM_INCLUDE_LOCATIONS += {hypre}/include\n"
                        f"LIBRARY_LOCATIONS += {hypre}/lib\nLIBRARIES += {libs}\n"
                        "include ../Source/Make.package", 1)
    if precice:                         # the container's libprecice (pkg-config)
        mk = mk.replace("include ../Source/Make.package",
                        "DEFINES += -DZS_PRECICE\nLIBRARIES += -lprecice\n"
                        "include ../Source/Make.package")
    return mk
