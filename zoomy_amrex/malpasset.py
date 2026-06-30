#!/usr/bin/env python3
"""Project the Malpasset triangular benchmark onto a structured AMReX grid.

The benchmark ships as an unstructured triangular mesh
(``data/malpasset/geo_malpasset-small.msh``) carrying, as gmsh ``$NodeData``,
the survey bathymetry ``B`` and the reservoir-full initial condition
(``H`` depth, ``U``/``V`` velocity — both zero, water at rest).  AMReX is a
*structured* (rectangular) framework, so we:

1. take the axis-aligned bounding box of the triangulation as the rectangular
   ("quadrilateral") AMReX domain — the simplest way to embed the irregular
   catchment without embedded boundaries (EB);
2. interpolate ``B`` and ``H`` from the triangle vertices onto the structured
   cell centres with a linear *triangle* interpolant
   (``matplotlib.tri.LinearTriInterpolator``), which masks points that fall
   *outside* the actual (non-convex) mesh — so the real, irregular boundary is
   respected, not just the convex hull;
3. turn the masked exterior into a closed wall by raising the bed there above
   the reservoir surface and leaving it dry.

The result is two row-major ``float64`` ``.raw`` rasters (bed → state comp 0,
depth → state comp 1) laid out exactly as ``init_solution.cpp`` expects
(``idx = i + nx*j``), plus the geometry the driver needs.

Wet/dry on the structured grid is handled by the driver's ``h_min`` flux floor
+ negative-depth clamp (see ``Source/constants.H`` / ``make_rhs.H``); here we
only set the static bathymetry and the still-water IC.
"""
import os

import numpy as np


def _read_mesh(msh_path):
    import meshio
    m = meshio.read(msh_path)
    x = m.points[:, 0].astype(np.float64)
    y = m.points[:, 1].astype(np.float64)
    tris = None
    for c in m.cells:
        if c.type == "triangle":
            tris = c.data
            break
    if tris is None:
        raise SystemExit(f"no triangle cells in {msh_path}")
    pd = m.point_data
    if "B" not in pd or "H" not in pd:
        raise SystemExit(f"mesh {msh_path} lacks B/H node data (have {list(pd)})")
    return x, y, tris, pd["B"].astype(np.float64), pd["H"].astype(np.float64)


def prepare_rasters(msh_path, out_dir, ncell_x=180, ncell_y=96,
                    wall_above=5.0, pad_frac=0.0):
    """Write ``bed.raw`` + ``depth.raw`` and return the AMReX geometry dict.

    ncell_x/ncell_y  structured resolution (cell centres are sampled).
    wall_above       exterior wall bed = max reservoir surface + this (metres).
    pad_frac         optional bounding-box padding as a fraction of each span.
    """
    from matplotlib.tri import Triangulation, LinearTriInterpolator

    x, y, tris, B, H = _read_mesh(msh_path)
    os.makedirs(out_dir, exist_ok=True)

    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    if pad_frac:
        px, py = pad_frac * (x1 - x0), pad_frac * (y1 - y0)
        x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py

    dx = (x1 - x0) / ncell_x
    dy = (y1 - y0) / ncell_y
    xc = x0 + (np.arange(ncell_x) + 0.5) * dx
    yc = y0 + (np.arange(ncell_y) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc)          # (ny, nx): X[j,i]=xc[i], Y[j,i]=yc[j]

    tri = Triangulation(x, y, tris)
    Bg = LinearTriInterpolator(tri, B)(X, Y)   # masked outside the mesh
    Hg = LinearTriInterpolator(tri, H)(X, Y)
    outside = np.ma.getmaskarray(Bg)

    # reservoir free surface (bed+depth) over wet nodes -> the wall must clear it
    wet = H > 1e-6
    eta_max = float((B[wet] + H[wet]).max()) if wet.any() else float(B.max())
    wall_bed = eta_max + wall_above

    bed = np.where(outside, wall_bed, np.ma.filled(Bg, wall_bed)).astype(np.float64)
    depth = np.ma.filled(Hg, 0.0).astype(np.float64)
    depth = np.where(outside, 0.0, np.clip(depth, 0.0, None)).astype(np.float64)

    bed_path = os.path.join(out_dir, "bed.raw")
    dep_path = os.path.join(out_dir, "depth.raw")
    bed.ravel(order="C").tofile(bed_path)     # C-order -> offset i + nx*j
    depth.ravel(order="C").tofile(dep_path)

    info = dict(
        nx=ncell_x, ny=ncell_y,
        prob_lo=(x0, y0), prob_hi=(x1, y1),
        dx=dx, dy=dy,
        dem_file=bed_path, release_file=dep_path,
        eta_max=eta_max, wall_bed=wall_bed,
        wet_frac=float((depth > 1e-6).mean()),
        bed_min=float(bed.min()), bed_max=float(bed.max()),
        depth_max=float(depth.max()),
    )
    return info


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("msh")
    ap.add_argument("out_dir")
    ap.add_argument("--ncell-x", type=int, default=180)
    ap.add_argument("--ncell-y", type=int, default=96)
    ap.add_argument("--wall-above", type=float, default=5.0)
    a = ap.parse_args()
    info = prepare_rasters(a.msh, a.out_dir, a.ncell_x, a.ncell_y, a.wall_above)
    print(json.dumps(info, indent=2))
