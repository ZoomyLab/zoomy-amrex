# zoomy-amrex

AMReX backend for [Zoomy](https://github.com/ZoomyLab/Zoomy) — a block-structured
AMR finite-volume solver for the shallow-water / shallow-moment family, driven by
code generated from a `zoomy_core` `SystemModel`.

## Architecture (parity with the jax / foam backends)

```
Model  ──▶  SystemModel  ──▶  AMReX device code  ──▶  AmrCore FV solver
(zoomy_core)              (zoomy_amrex.transformation)   (Source/*.cpp,*.H)
```

* **`zoomy_amrex/transformation/`** — the backend-owned code printer (the jax
  backend owns its printer the same way). `AmrexSystemModelPrinter` emits
  `Model.H` from a frozen `SystemModel`; `Numerics.H` comes from `zoomy_core`'s
  `AmrexNumerics` (a Riemann scheme over the same SystemModel). All "placeholder"
  kernel functions (`conditional`, `clamp_*`, `eigensystem`,
  `compute_derivative`, …) are mapped in the printer's `c_functions`; the opaque
  ones resolve to hand-written helpers in the generated `UserFunctions.H`. The CFL
  wave speed is `Numerics::local_max_abs_eigenvalue` (a real `Max(|λ_i|)` kernel
  emitted by core's `AmrexNumerics`, Gershgorin row-sum when `eigenvalues=None`).
* **`Source/`** — the hand-written C++ driver: `ZoomyAmr.cpp` (AmrCore: multi-level
  AMR, subcycling, FillPatch, RK1/SSP-RK2), `make_rhs.H` (MUSCL/minmod
  reconstruction, path-conservative D± fluxes, a per-cell implicit-source Newton
  solve), consuming the generated `Model::` / `Numerics::` device functions. The
  driver is generic in `Model::n_dof_q` (works for any moment level).
* **`build.py`** — in-tree driver (the analogue of jax's `solver.solve` / foam's
  `create_model.py`): `SystemModel → headers → make → run`.
* **`deliverable.py`** — reproducible figure/GIF from the AMReX plotfiles.

## Quick start (inside the container)

```bash
SIF=.../zoomy_amrex_latest.sif          # AMReX at /opt/amrex, mpicxx, python3
apptainer exec --bind <ZOOMY>:/work $SIF bash -lc '
  export PYTHONPATH=/work/library/zoomy_core:/work/library/zoomy_amrex
  cd /work/library/zoomy_amrex
  python3 build.py --model SWE --dim 2 --ncell 100 --tend 0.1 --order 2 \
      --build-dir /tmp/run --make --run
  python3 deliverable.py /tmp/run/Exec --out /tmp/run/figures/dam_break
'
```

`--model SME --dim 3 --level N` generates and builds the N-moment shallow-moment
model (state grows with the level; the driver is dof-generic).

Or build the committed `Exec/` directly (headers in `Source/` are pre-generated
for 2-D SWE): `cd Exec && make -j && ./main3d.gnu.MPI.ex inputs`.

## Status

| capability | state |
|---|---|
| consume a `SystemModel` (code-gen) | ✅ restored & verified (printer in-tree) |
| real `Numerics.H` (Rusanov/NCP from core) | ✅ (previously a zero stub) |
| 2nd-order MUSCL + SSP-RK2 | ✅ runs |
| 2-D SWE dam break | ✅ verified (Ritter structure) |
| multiple moment levels (dof-generic) | ✅ code-gen + compile (SME level ≥ 1); full run needs `compute_derivative` aux |
| boundary conditions | ⚠ extrapolation only; full per-tag `Piecewise` dispatch is WIP |
| IMEX | ⚠ implicit-source Newton present; full IMEX-ARK is WIP |
| pressure splitting (Chorin) | ✗ not yet |

See the steward task files (`0025`–`0028`) for the remaining plan.
