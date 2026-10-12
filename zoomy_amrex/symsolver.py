"""Driver: print the symbolic solvers for AMReX (C++), build, run.

    solver = SymbolicHyperbolicSolverAmrex(time_end=0.5, ranks=4)       # CPU, MPI
    solver = SymbolicHyperbolicSolverAmrex(time_end=0.5, gpu=True)      # CUDA
    time, Q = solver.run(mesh, nsm, Q0, workdir)        # Q in the mesh's order

The printed host program (``Solver.H``, :mod:`zoomy_core.symsolver.backend_cxx`:
the same text foam runs), the device kernels (``Kernels.H``,
:mod:`zoomy_amrex.symkernels`) and the app (``Source/symsolver/symAmrex.cpp``
+ ``rt_amrex.H`` + zoomy_core's ``rt_common.H``) are compiled in the AMReX
apptainer (GNU make, one build tree per CPU/GPU, so AMReX itself compiles
once), cached by a hash of the sources.  The grid is the mesh's uniform
Cartesian grid; the LSQ stencils and weights are the zoomy mesh's own
(``lsq_gradQ``), per cell class.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
import param

from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.boundary_conditions import boundary_face_branches
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
BUILD_CPUS = os.environ.get("ZOOMY_AMREX_BUILD_CPUS", "32-63")
RUN_CPUS = os.environ.get("ZOOMY_AMREX_RUN_CPUS", "16-31")
APP = ("symsolver/symAmrex.cpp", "symsolver/rt_amrex.H", "symsolver/amr_amrex.H",
       "UserFunctions.H", "ZoomyEig.H")


def _apptainer(gpu, script, binds=(), **kw):
    cmd = ["apptainer", "exec", "--cleanenv", *(["--nv"] if gpu else [])]
    for b in sorted({str(CACHE), *map(str, binds)}):
        cmd += ["--bind", b]
    return subprocess.run(cmd + [str(SIF[gpu]), "bash", "-c", script], **kw)


class AmrexDriver(param.Parameterized):
    """The AMReX runtime of a C++ solver backend (:mod:`zoomy_core.symsolver.backend_cxx`):
    the build tree (GNU make in the container), the case (the Cartesian grid,
    LSQ tables, inputs, AMR levels), the launch (``mpirun``) and the read-back."""
    ranks = param.Integer(default=1, bounds=(1, None), doc="MPI ranks (GPUs with gpu=True).")
    gpu = param.Boolean(default=False, doc="CUDA build (one GPU per rank).")
    max_grid_size = param.Integer(default=None, allow_None=True,
                                  doc="AMReX box size; None: about 2 boxes per rank.")
    tiny_profile = param.Boolean(default=False,
                                 doc="AMReX TinyProfiler build (a tree of its own).")
    build_only = param.Boolean(default=False,
                               doc="run() only builds (and caches) the binary.")
    amr = param.Dict(default=None, allow_None=True, doc=(
        "AMR (one dt for all levels; amr_amrex.H): max_level, ratio (2), "
        "regrid_int (0: the initial grids only), tag = (state name, 'value' | "
        "'gradient', threshold) or region = (lo, hi), reflux (True), "
        "blocking_factor (8), n_error_buf (2), grid_eff (0.7).  The CFL step "
        "from the cells' wave speeds."))

    def _amr_loop(self):
        return self.amr is not None

    # ── build ──────────────────────────────────────────────────────────────
    def build(self, nsm, dimension):
        """Print, write the build tree, make (cached).  Returns the binary."""
        self.source = self._print(nsm, dimension)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"kernels_{os.getpid()}_{threading.get_ident()}.H"   # per thread
        write_kernels(self._systems(nsm), self.source, tmp, dimension)
        files = {"Solver.H": self.source, "Kernels.H": tmp.read_text()}
        tmp.unlink()
        for f in APP:
            files[Path(f).name] = (SRC / f).read_text()
        files[RUNTIME_HEADER.name] = RUNTIME_HEADER.read_text()
        cpl = bool(self.couple)
        h = hashlib.sha256(f"{self.gpu}{self.tiny_profile}{cpl}".encode())
        for k in sorted(files):
            h.update(k.encode() + files[k].encode())
        binary = CACHE / "bin" / h.hexdigest()[:16] / "symAmrex.ex"
        if binary.exists():
            return binary
        tree = CACHE / (("gpu" if self.gpu else "cpu") + f"{dimension}d"
                        + ("_tp" if self.tiny_profile else "") + ("_cpl" if cpl else ""))
        (tree / "Source").mkdir(parents=True, exist_ok=True)
        (tree / "Exec").mkdir(parents=True, exist_ok=True)
        with open(tree / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for k, v in files.items():
                p = tree / "Source" / k
                if not p.exists() or p.read_text() != v:
                    p.write_text(v)
            (tree / "Source" / "Make.package").write_text(
                "CEXE_sources += symAmrex.cpp\nCEXE_headers += rt_common.H rt_amrex.H amr_amrex.H "
                "Kernels.H Solver.H UserFunctions.H ZoomyEig.H\n")
            (tree / "Exec" / "GNUmakefile").write_text(
                _gnumakefile(self.gpu, self.tiny_profile, dimension, cpl))
            r = _apptainer(self.gpu, f"cd {tree}/Exec && taskset -c {BUILD_CPUS} "
                           f"make -j32 > make.log 2>&1; rc=$?; tail -30 make.log; "
                           f"exit $rc", capture_output=True, text=True)
            exe = sorted((tree / "Exec").glob(f"main{dimension}d*.ex"), key=os.path.getmtime)
            if r.returncode != 0 or not exe:
                raise RuntimeError(f"symAmrex make failed ({tree}/Exec/make.log):\n"
                                   f"{r.stdout}\n{r.stderr}")
            binary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exe[-1], binary)
            (binary.parent / "Solver.H").write_text(files["Solver.H"])
            (binary.parent / "Kernels.H").write_text(files["Kernels.H"])
        return binary

    # ── run ────────────────────────────────────────────────────────────────
    def run(self, mesh, model, Q0, workdir, Qaux0=None, fine=()):
        """March ``Q0`` (mesh order); returns ``(time, Q)`` and keeps the
        snapshots in ``self.snapshots`` [(time, Q)] and the wall time of the
        march in ``self.wall``.  AMR: ``fine`` = [(mesh_l, Q0_l)] the initial
        state of level l on its whole (refined) domain (else interpolated);
        ``Q`` is level 0 (the composite averaged down), ``self.initial`` its
        initial state, ``self.levels`` each level's final state (NaN where it
        has no cells; mesh_l order)."""
        nsm = self._prepare(mesh, model)
        grid = cartesian(mesh)
        dim = grid["dim"]
        binary = self.build(nsm, dim)
        if self.build_only:
            return binary
        case = Path(workdir).resolve()
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        n = grid["n"]
        to_grid = lambda Q: np.asarray(Q, dtype=np.float64)[:, grid["order"]]  # noqa: E731
        to_grid(Q0).tofile(case / "q0.bin")
        n_aux = len(nsm.aux_state)
        A0 = np.zeros((max(n_aux, 1), grid["nc"])) if Qaux0 is None else Qaux0
        to_grid(A0).tofile(case / "qaux0.bin")
        branch, periodic = sides(mesh, nsm, dim)
        (case / "lsq.txt").write_text(lsq_table(nsm, self._degree(nsm), grid, periodic))
        amr = self._amr_inputs(case, nsm, grid, periodic, fine) if self.amr else ""
        if self.couple:
            c = self.couple
            amr += (f"zs.precice = {c['participant']} {c['config']} {c['mesh']}\n"
                    f"zs.precice_write = {' '.join(c['write'])}\n"
                    f"zs.precice_read = {' '.join(c['read'])}\n"
                    f"zs.precice_side = {side_of(mesh, c['tag'], dim)}\n")
        # boxes per rank: CPU ~8 (load balance across wet / dry / wall cells),
        # GPU ~1 (one launch covers all boxes; fewer, larger boxes)
        per = 1 if self.gpu else 8
        mgs = self.max_grid_size or max(16, int(np.ceil(max(n) / np.ceil(
            (per * self.ranks) ** (1.0 / dim) - 1e-9))))
        f = lambda v: " ".join(repr(float(x)) for x in v)                  # noqa: E731
        (case / "inputs").write_text(
            f"zs.n_cell = {' '.join(map(str, n))}\nzs.prob_lo = {f(grid['lo'])}\n"
            f"zs.prob_hi = {f(grid['hi'])}\nzs.periodic = {' '.join(map(str, periodic))}\n"
            f"zs.max_grid_size = {mgs}\nzs.time_end = "
            f"{1e30 if self.time_end is None else self.time_end!r}\n"
            f"zs.branch = {' '.join(map(str, branch))}\n"
            f"zs.lsq = lsq.txt\nzs.q0 = q0.bin\nzs.qaux0 = qaux0.bin\nzs.out = out\n"
            "amrex.the_arena_init_size = 1073741824\namrex.verbose = 0\n"
            "tiny_profiler.device_synchronize_around_region = 1\n" + amr)
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

    def _amr_inputs(self, case, nsm, grid, periodic, fine):
        """Per-level LSQ tables and initial states, the AMR inputs."""
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
            (case / f"lsq_{lv}.txt").write_text(lsq_table(nsm, self._degree(nsm), g, periodic))
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
    """The explicit hyperbolic template on AMReX.

        time, Q = SymbolicHyperbolicSolverAmrex(time_end=0.5).run(
            mesh, nsm, Q0, workdir)"""


class SymbolicChorinSolverAmrex(AmrexDriver, CxxChorinBackend):
    """The Chorin projection template on AMReX.

        time, Q = SymbolicChorinSolverAmrex(time_end=0.05).run(
            mesh, (pred, press, corr), Q0, workdir)"""


# ── the grid ────────────────────────────────────────────────────────────────
def cartesian(mesh):
    """The mesh's uniform Cartesian grid (1-, 2- or 3-D): ``lo, hi, n, d``,
    the grid order of its cells (x fastest) and each cell's grid index."""
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
    """BC branch of the x-lo, x-hi, y-lo, ... sides (by outward normal; -1 on a
    periodic side) and the periodic flag per direction."""
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
    """The side (2 axis + upper) of the boundary faces tagged ``tag``."""
    names = list(mesh.boundary_conditions_sorted_names)
    nums = np.asarray(mesh.boundary_face_function_numbers, dtype=int)
    faces = np.asarray(mesh.boundary_face_face_indices, dtype=int)
    hit = [i for i in range(len(nums)) if names[nums[i]] == tag]
    n = np.asarray(mesh.face_normals)[:dim, faces[hit]].mean(axis=1)
    ax = int(np.argmax(np.abs(n)))
    return 2 * ax + int(n[ax] > 0)


def lsq_table(nsm, degree, grid, periodic):
    """The LSQ stencils and weights of the zoomy mesh per cell class, from a
    proxy grid of the same spacing (classes: the distance to each side, up to
    the stencil reach R; along a periodic axis the interior class, its samples
    across the seam the partner cells in the halo, as zoomy_core's wrapped
    stencils).  Text
    read by ``zs::Mesh::load``."""
    dim, n, lo, h = grid["dim"], grid["n"], grid["lo"], grid["d"]
    np_ = [min(n[ax], 2 * degree + 3) for ax in range(dim)]
    dom = [v for ax in range(dim) for v in (lo[ax], lo[ax] + np_[ax] * h[ax])]
    proxy = (LSQMesh.create_1d(domain=tuple(dom), n_inner_cells=np_[0]) if dim == 1 else
             LSQMesh.create_2d(domain=tuple(dom), nx=np_[0], ny=np_[1]) if dim == 2 else
             LSQMesh.create_3d(domain=tuple(dom), nx=np_[0], ny=np_[1], nz=np_[2]))
    if degree > 1:
        proxy._build_lsq_stencil(degree)
    pg = cartesian(proxy)
    nc = pg["nc"]
    cc = np.asarray(proxy.cell_centers)[:dim]
    nbrs = np.asarray(proxy.lsq_neighbors)
    bnbr = np.asarray(proxy.lsq_boundary_face_neighbors)
    A = np.asarray(proxy.lsq_gradQ)
    scale = np.asarray(proxy.lsq_scale_factors, dtype=float)
    mon = [tuple(int(x) for x in m) for m in np.asarray(proxy.lsq_monomial_multi_index)]
    fcen = np.asarray(proxy.face_centers)[:, :dim].T
    bf_face = np.asarray(proxy.boundary_face_face_indices, dtype=int)
    bf_cell = np.asarray(proxy.boundary_face_cells, dtype=int)
    d = np.asarray(h)
    samples = {}
    for c in range(nc):
        row, smp = 0, []
        for nb in nbrs[c]:
            w = scale * A[c, row]
            row += 1
            if nb == c or nb >= nc or not np.any(w):
                continue
            smp.append((0, *np.rint((cc[:, nb] - cc[:, c]) / d).astype(int), w))
        for bf in bnbr[c]:
            w = scale * A[c, row]
            row += 1
            if bf < 0 or not np.any(w):
                continue
            ghost = 2 * fcen[:, bf_face[bf]] - cc[:, bf_cell[bf]]
            smp.append((1, *np.rint((ghost - cc[:, c]) / d).astype(int), w))
        samples[c] = smp
    R = max(max(abs(int(o)) for o in s[1:1 + dim]) for smp in samples.values() for s in smp)
    for ax in range(dim):
        if np_[ax] < n[ax] and np_[ax] < 2 * R + 1:
            raise RuntimeError(f"LSQ reach {R}: proxy grid too small")

    def key(i, m, per):     # a periodic axis has no sides: the interior stencil,
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
    out = [f"{R} {dim} {' '.join(map(str, n))} {' '.join(str(len(k)) for k in keys)} {len(mon)}",
           *(" ".join(map(str, cl)) for cl in cls),
           " ".join(" ".join(map(str, m)) for m in mon)]
    for k in np.ndindex(*[len(kk) for kk in keys]):
        smp = samples[rep[tuple(int(x) for x in k)]]
        out.append(str(len(smp)))
        out += [f"{s[0]} {' '.join(str(int(o)) for o in s[1:1 + dim])} "
                + " ".join(repr(float(x)) for x in s[-1]) for s in smp]
    return "\n".join(out) + "\n"


def _read(path, grid):
    raw = np.fromfile(path, dtype=np.float64)
    dim = grid["dim"]
    time, nq = raw[0], int(raw[1])
    Q = raw[2 + dim:].reshape(nq, -1)
    out = np.empty_like(Q)
    out[:, grid["order"]] = Q
    return float(time), out


def _gnumakefile(gpu, tiny_profile=False, dim=2, precice=False):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_zoomy_amrex_build", ROOT / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mk = mod.GNUMAKEFILE.format(amrex_home="/opt/amrex", dim=dim, use_mpi="TRUE",
                                use_cuda="TRUE" if gpu else "FALSE", cuda_arch="89",
                                tiny_profile="TRUE" if tiny_profile else "FALSE")
    if precice:                         # the container's libprecice (pkg-config)
        mk = mk.replace("include ../Source/Make.package",
                        "DEFINES += -DZS_PRECICE\nLIBRARIES += -lprecice\n"
                        "include ../Source/Make.package")
    return mk
