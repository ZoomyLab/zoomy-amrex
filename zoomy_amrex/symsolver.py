"""Driver: print the symbolic solvers for AMReX (C++), build, run.

    solver = SymbolicHyperbolicSolverAmrex(time_end=0.5, ranks=4)       # CPU, MPI
    solver = SymbolicHyperbolicSolverAmrex(time_end=0.5, gpu=True)      # CUDA
    time, Q = solver.run(mesh, nsm, Q0, workdir)        # Q in the mesh's order

The printed host program (``Solver.H``, :mod:`zoomy_core.symsolver.program`:
the same text foam runs), the device kernels (``Kernels.H``,
:mod:`zoomy_amrex.symkernels`) and the app (``Source/symsolver/symAmrex.cpp``
+ ``rt_amrex.H``) are compiled in the AMReX apptainer (GNU make, one build
tree per CPU/GPU, so AMReX itself compiles once), cached by a hash of the
sources.  The grid is the mesh's uniform Cartesian grid; the LSQ stencils
and weights are the zoomy mesh's own (``lsq_gradQ``), per cell class.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import param

from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.solver.constants import ELLIPTIC_RTOL, elliptic_constants
from zoomy_core.symsolver.program import chorin_program, hyperbolic_program
from zoomy_amrex.symkernels import write_kernels

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Source"
CACHE = Path(os.environ.get("ZOOMY_AMREX_SYMCACHE", ROOT / ".symcache"))
CONTAINERS = Path(os.environ.get("ZOOMY_AMREX_CONTAINERS",
                                 Path.home() / "git/Zoomy/containers/zoomy_amrex"))
SIF = {False: CONTAINERS / "zoomy_amrex.sif", True: CONTAINERS / "zoomy_amrex_gpu.sif"}
BUILD_CPUS = os.environ.get("ZOOMY_AMREX_BUILD_CPUS", "32-63")
RUN_CPUS = os.environ.get("ZOOMY_AMREX_RUN_CPUS", "16-31")
APP = ("symsolver/symAmrex.cpp", "symsolver/rt_amrex.H", "UserFunctions.H", "ZoomyEig.H")


def _apptainer(gpu, script, binds=(), **kw):
    cmd = ["apptainer", "exec", "--cleanenv", *(["--nv"] if gpu else [])]
    for b in sorted({str(CACHE), *map(str, binds)}):
        cmd += ["--bind", b]
    return subprocess.run(cmd + [str(SIF[gpu]), "bash", "-c", script], **kw)


class SymbolicHyperbolicSolverAmrex(param.Parameterized):
    time_end = param.Number(default=None, allow_None=True)
    cfl = param.Number(default=0.9, bounds=(0, None))
    dt = param.Number(default=None, allow_None=True,
                      doc="Constant time step; None = adaptive CFL step.")
    cfl_check_every = param.Integer(default=10, bounds=(0, None))
    cfl_wave_speed = param.Selector(default="faces", objects=["faces", "cells"])
    mood = param.Boolean(default=True,
                         doc="A-posteriori MOOD at order >= 2.")
    dt_write = param.Number(default=None, allow_None=True,
                            doc="Output interval (plotfiles); None = final state only.")
    imex = param.Boolean(default=False)
    implicit_tol = param.Number(default=1e-8, bounds=(0, None))
    implicit_maxiter = param.Integer(default=6, bounds=(1, None))
    gmres_tol = param.Number(default=1e-7, bounds=(0, None))
    gmres_restart = param.Integer(default=20, bounds=(1, None))
    gmres_maxiter = param.Integer(default=30, bounds=(1, None))
    ranks = param.Integer(default=1, bounds=(1, None), doc="MPI ranks (GPUs with gpu=True).")
    gpu = param.Boolean(default=False, doc="CUDA build (one GPU per rank).")
    max_grid_size = param.Integer(default=None, allow_None=True,
                                  doc="AMReX box size; None: about 2 boxes per rank.")
    build_only = param.Boolean(default=False,
                               doc="run() only builds (and caches) the binary.")

    # ── printing ───────────────────────────────────────────────────────────
    def _systems(self, nsm):
        return [("nsm", nsm)]

    def _mood(self, nsm):
        return bool(self.mood and nsm.reconstruction.order >= 2
                    and nsm.call.admissible is not None)

    def _linsolve(self, nsm):
        return dict(tol=self.gmres_tol, restart=self.gmres_restart,
                    maxiter=self.gmres_maxiter, accept=-1.0)

    def _kw(self, nsm, dimension):
        return dict(dimension=dimension, cfl=self.cfl, mood=self._mood(nsm),
                    dt_write=self.dt_write, imex=self.imex,
                    cfl_check_every=self.cfl_check_every,
                    wave_speed=self.cfl_wave_speed, dt_constant=self.dt,
                    implicit=dict(tol=self.implicit_tol, maxiter=self.implicit_maxiter),
                    linsolve=self._linsolve(nsm))

    def _print(self, nsm, dimension):
        return hyperbolic_program(nsm, **self._kw(nsm, dimension))

    def _degree(self, nsm):
        return nsm.resolved_lsq_degree()

    # ── build ──────────────────────────────────────────────────────────────
    def build(self, nsm, dimension):
        """Print, write the build tree, make (cached).  Returns the binary."""
        self.source = self._print(nsm, dimension)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"kernels_{os.getpid()}.H"
        write_kernels(self._systems(nsm), self.source, tmp, dimension)
        files = {"Solver.H": self.source, "Kernels.H": tmp.read_text()}
        tmp.unlink()
        for f in APP:
            files[Path(f).name] = (SRC / f).read_text()
        h = hashlib.sha256(str(self.gpu).encode())
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
        binary = CACHE / "bin" / h.hexdigest()[:16] / "symAmrex.ex"
        if binary.exists():
            return binary
        tree = CACHE / ("gpu" if self.gpu else "cpu")
        (tree / "Source").mkdir(parents=True, exist_ok=True)
        (tree / "Exec").mkdir(parents=True, exist_ok=True)
        with open(tree / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for k, v in files.items():
                p = tree / "Source" / k
                if not p.exists() or p.read_text() != v:
                    p.write_text(v)
            (tree / "Source" / "Make.package").write_text(
                "CEXE_sources += symAmrex.cpp\nCEXE_headers += rt_amrex.H Kernels.H "
                "Solver.H UserFunctions.H ZoomyEig.H\n")
            (tree / "Exec" / "GNUmakefile").write_text(_gnumakefile(self.gpu))
            r = _apptainer(self.gpu, f"cd {tree}/Exec && taskset -c {BUILD_CPUS} "
                           f"make -j32 > make.log 2>&1; rc=$?; tail -30 make.log; "
                           f"exit $rc", capture_output=True, text=True)
            exe = sorted((tree / "Exec").glob("main2d*.ex"), key=os.path.getmtime)
            if r.returncode != 0 or not exe:
                raise RuntimeError(f"symAmrex make failed ({tree}/Exec/make.log):\n"
                                   f"{r.stdout}\n{r.stderr}")
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exe[-1], binary)
            (binary.parent / "Solver.H").write_text(files["Solver.H"])
            (binary.parent / "Kernels.H").write_text(files["Kernels.H"])
        return binary

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, mesh, model, Q0, workdir, Qaux0=None):
        """March ``Q0`` (mesh order); returns ``(time, Q)`` and keeps the
        snapshots in ``self.snapshots`` [(time, Q)] and the wall time of the
        march in ``self.wall``."""
        nsm = (model if isinstance(model, NumericalSystemModel)
               else NumericalSystemModel.from_system_model(model))
        grid = cartesian(mesh)
        binary = self.build(nsm, 2)
        if self.build_only:
            return binary
        case = Path(workdir).resolve()
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        nx, ny = grid["n"]
        to_grid = lambda Q: np.asarray(Q, dtype=np.float64)[:, grid["order"]]  # noqa: E731
        to_grid(Q0).tofile(case / "q0.bin")
        n_aux = len(nsm.aux_state)
        A0 = np.zeros((max(n_aux, 1), grid["nc"])) if Qaux0 is None else Qaux0
        to_grid(A0).tofile(case / "qaux0.bin")
        (case / "lsq.txt").write_text(lsq_table(mesh, nsm, self._degree(nsm), grid))
        mgs = self.max_grid_size or max(8, int(np.ceil(max(nx, ny) / max(1, int(np.ceil(np.sqrt(2 * self.ranks)))))))
        (case / "inputs").write_text(
            f"zs.n_cell = {nx} {ny}\nzs.prob_lo = {grid['lo'][0]!r} {grid['lo'][1]!r}\n"
            f"zs.prob_hi = {grid['hi'][0]!r} {grid['hi'][1]!r}\n"
            f"zs.max_grid_size = {mgs}\nzs.time_end = "
            f"{1e30 if self.time_end is None else self.time_end!r}\n"
            f"zs.branch = {' '.join(map(str, branches(mesh, nsm)))}\n"
            f"zs.lsq = lsq.txt\nzs.q0 = q0.bin\nzs.qaux0 = qaux0.bin\nzs.out = out\n"
            "amrex.the_arena_init_size = 1073741824\namrex.verbose = 0\n")
        launch = (f"mpirun --oversubscribe --bind-to none -n {self.ranks} "
                  if self.ranks > 1 or self.gpu else "")
        cpus = "" if self.gpu else f"taskset -c {RUN_CPUS} "
        r = _apptainer(self.gpu, f"cd {case} && {cpus}{launch}{binary} inputs > log.symAmrex 2>&1",
                       binds=[case, binary.parent], capture_output=True, text=True)
        log = (case / "log.symAmrex").read_text()
        if r.returncode != 0:
            raise RuntimeError(f"symAmrex failed, see {case}/log.symAmrex:\n{log[-3000:]}\n{r.stderr}")
        self.log = log
        self.wall = next((float(ln.split()[-2]) for ln in log.splitlines()
                          if "march wall time" in ln), None)
        snaps = sorted((case / "out").glob("[0-9]*.bin"), key=lambda f: int(f.stem))
        self.snapshots = [_read(f, grid) for f in snaps]
        return _read(case / "out" / "final.bin", grid)


class SymbolicChorinSolverAmrex(SymbolicHyperbolicSolverAmrex):
    """The Chorin projection template on AMReX.

        time, Q = SymbolicChorinSolverAmrex(time_end=0.05).run(
            mesh, (pred, press, corr), Q0, workdir)"""
    time_order = param.Integer(default=1, bounds=(1, 2))
    pressure_tol = param.Number(default=ELLIPTIC_RTOL, bounds=(0, None))
    pressure_maxit = param.Integer(default=100, bounds=(1, None))
    pressure_restart = param.Integer(default=None, allow_None=True, bounds=(1, None))

    def _systems(self, nsm):
        return [("pred", nsm), ("press", self.press), ("corr", self.corr)]

    def _linsolve(self, nsm):
        n = len(self.press.equation_to_state_index) * self._nc
        ell = elliptic_constants(n_unknowns=n, tol=self.pressure_tol,
                                 restart=self.pressure_restart).values
        tol = float(ell["c_elliptic_tol"])
        return dict(tol=tol, restart=int(ell["c_elliptic_restart"]),
                    maxiter=self.pressure_maxit, accept=max(1e-6, 10.0 * tol))

    def _print(self, nsm, dimension):
        return chorin_program(nsm, self.press, self.corr, time_order=self.time_order,
                              **self._kw(nsm, dimension))

    def _degree(self, nsm):
        return max(s.resolved_lsq_degree() for s in (nsm, self.press, self.corr))

    def run(self, mesh, systems, Q0, workdir, Qaux0=None):
        named = []
        for sm in systems:
            named.append(sm if isinstance(sm, NumericalSystemModel)
                         else NumericalSystemModel.from_system_model(sm))
        pred, self.press, self.corr = named
        self._nc = int(mesh.n_inner_cells)
        return super().run(mesh, pred, Q0, workdir, Qaux0)


# ── the grid ────────────────────────────────────────────────────────────────
def cartesian(mesh):
    """The mesh's uniform 2-D grid: ``lo, hi, n, d``, the grid order of its
    cells (x fastest) and the cell index of every grid cell."""
    nc = int(mesh.n_inner_cells)
    cc = np.asarray(mesh.cell_centers)[:2, :nc]
    lo, hi, n, d = [], [], [], []
    for ax in range(2):
        u = np.unique(np.round(cc[ax], 9))
        h = float(np.mean(np.diff(u))) if len(u) > 1 else 1.0
        if len(u) > 1 and not np.allclose(np.diff(u), h, rtol=1e-6, atol=0):
            raise NotImplementedError("zoomy_amrex symsolver: uniform Cartesian grids only")
        lo.append(float(u.min() - h / 2)); hi.append(float(u.max() + h / 2))
        n.append(len(u)); d.append(h)
    if n[0] * n[1] != nc:
        raise NotImplementedError("zoomy_amrex symsolver: a full Cartesian grid only")
    ij = [np.rint((cc[ax] - lo[ax]) / d[ax] - 0.5).astype(int) for ax in range(2)]
    order = np.empty(nc, dtype=int)
    order[ij[1] * n[0] + ij[0]] = np.arange(nc)
    return dict(lo=lo, hi=hi, n=n, d=d, nc=nc, order=order, ij=ij)


def branches(mesh, nsm):
    """BC branch of the x-lo, x-hi, y-lo, y-hi sides (by outward normal)."""
    br = np.asarray(boundary_face_branches(mesh, getattr(nsm.sm, "_bc_source", None)))
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)[:len(br)]
    normals = np.asarray(mesh.face_normals)[:2, faces]
    out = []
    for ax in range(2):
        for s in (-1, 1):
            hit = np.flatnonzero(np.isclose(normals[ax], s))
            vals = np.unique(br[hit])
            if len(vals) != 1:
                raise NotImplementedError(f"side {ax}{s}: BC branches {vals}, one per side only")
            out.append(int(vals[0]))
    return out


def lsq_table(mesh, nsm, degree, grid):
    """The LSQ stencils and weights of the zoomy mesh per cell class, from a
    proxy grid of the same spacing (classes: the distance to each side, up to
    the stencil reach R).  Text read by ``zs::Mesh::load``."""
    n = grid["n"]
    np_ = [min(n[ax], 2 * degree + 3) for ax in range(2)]
    lo = grid["lo"]
    proxy = LSQMesh.create_2d(domain=(lo[0], lo[0] + np_[0] * grid["d"][0],
                                      lo[1], lo[1] + np_[1] * grid["d"][1]),
                              nx=np_[0], ny=np_[1])
    if degree > 1:
        proxy._build_lsq_stencil(degree)
    pg = cartesian(proxy)
    nc = pg["nc"]
    cc = np.asarray(proxy.cell_centers)[:2]
    nbrs = np.asarray(proxy.lsq_neighbors)
    bnbr = np.asarray(proxy.lsq_boundary_face_neighbors)
    A = np.asarray(proxy.lsq_gradQ)
    scale = np.asarray(proxy.lsq_scale_factors, dtype=float)
    mon = [tuple(int(x) for x in m) for m in np.asarray(proxy.lsq_monomial_multi_index)]
    fcen = np.asarray(proxy.face_centers)[:, :2].T
    bf_face = np.asarray(proxy.boundary_face_face_indices, dtype=int)
    bf_cell = np.asarray(proxy.boundary_face_cells, dtype=int)
    d = np.asarray(grid["d"])[:, None]
    samples = {}
    for c in range(nc):
        row, smp = 0, []
        for nb in nbrs[c]:
            w = scale * A[c, row]
            row += 1
            if nb == c or nb >= nc or not np.any(w):
                continue
            off = np.rint((cc[:, nb] - cc[:, c]) / d[:, 0]).astype(int)
            smp.append((0, *off, w))
        for bf in bnbr[c]:
            w = scale * A[c, row]
            row += 1
            if bf < 0 or not np.any(w):
                continue
            ghost = 2 * fcen[:, bf_face[bf]] - cc[:, bf_cell[bf]]
            off = np.rint((ghost - cc[:, c]) / d[:, 0]).astype(int)
            smp.append((1, *off, w))
        samples[c] = smp
    R = max(max(abs(s[1]), abs(s[2])) for smp in samples.values() for s in smp)
    for ax in range(2):
        if np_[ax] < n[ax] and np_[ax] < 2 * R + 1:
            raise RuntimeError(f"LSQ reach {R}: proxy grid too small")

    def key(i, m):
        return (min(i, R), min(m - 1 - i, R))
    kx = sorted({key(i, np_[0]) for i in range(np_[0])})
    ky = sorted({key(j, np_[1]) for j in range(np_[1])})
    rep = {}
    for c in range(nc):
        i, j = pg["ij"][0][c], pg["ij"][1][c]
        rep.setdefault((kx.index(key(i, np_[0])), ky.index(key(j, np_[1]))), c)
    cls_x = [kx.index(key(i, n[0])) for i in range(n[0])]
    cls_y = [ky.index(key(j, n[1])) for j in range(n[1])]
    out = [f"{R} {n[0]} {n[1]} {len(kx)} {len(ky)} {len(mon)}",
           " ".join(map(str, cls_x)), " ".join(map(str, cls_y)),
           " ".join(f"{m[0]} {m[1]}" for m in mon)]
    for a in range(len(kx)):
        for b in range(len(ky)):
            smp = samples[rep[(a, b)]]
            out.append(str(len(smp)))
            out += [f"{k} {di} {dj} " + " ".join(repr(float(x)) for x in w)
                    for k, di, dj, w in smp]
    return "\n".join(out) + "\n"


def _read(path, grid):
    raw = np.fromfile(path, dtype=np.float64)
    time, nq, nx, ny = raw[:4]
    Q = raw[4:].reshape(int(nq), int(ny) * int(nx))
    out = np.empty_like(Q)
    out[:, grid["order"]] = Q
    return float(time), out


def _gnumakefile(gpu):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_zoomy_amrex_build", ROOT / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GNUMAKEFILE.format(amrex_home="/opt/amrex", dim=2, use_mpi="TRUE",
                                  use_cuda="TRUE" if gpu else "FALSE", cuda_arch="89",
                                  tiny_profile="FALSE")
