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
                    wall_above=5.0, pad_frac=0.0, subsample=4, bed_agg="min"):
    """Write ``bed.raw`` + ``depth.raw`` and return the AMReX geometry dict.

    ncell_x/ncell_y  structured resolution.
    wall_above       exterior wall bed = max reservoir surface + this (metres).
    pad_frac         optional bounding-box padding as a fraction of each span.
    subsample        sub-grid factor per cell for the bathymetry aggregation.
    bed_agg          how to collapse the sub-grid bed to one cell value:
                     "min"  -> channel-preserving (thalweg): keep the lowest
                               point so the narrow gorge/breach stays open at
                               coarse resolution (a dam-break would otherwise be
                               sealed by cell-centre sampling on the rim);
                     "mean" -> the plain cell average.
                     The still-water *surface* (reservoir eta=100, sea eta=0) is
                     preserved, so depth = max(0, surface - bed).
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

    # Sample on a sub-grid (subsample per cell per axis), then aggregate per
    # cell. Sub-cell MIN of the bed keeps the thalweg/breach connected; the
    # surface is taken from the deepest wet sub-point so the reservoir/sea
    # lake-at-rest level is preserved.
    s = max(1, int(subsample))
    fx = x0 + (np.arange(ncell_x * s) + 0.5) * (dx / s)
    fy = y0 + (np.arange(ncell_y * s) + 0.5) * (dy / s)
    FX, FY = np.meshgrid(fx, fy)               # (ny*s, nx*s)

    tri = Triangulation(x, y, tris)
    Bf = LinearTriInterpolator(tri, B)(FX, FY)   # masked outside the mesh
    Hf = LinearTriInterpolator(tri, H)(FX, FY)
    out_f = np.ma.getmaskarray(Bf)
    Bf = np.ma.filled(Bf, np.nan)
    Hf = np.ma.filled(Hf, np.nan)
    eta_f = Bf + Hf                              # free surface at wet sub-points

    wet = H > 1e-6
    eta_max = float((B[wet] + H[wet]).max()) if wet.any() else float(B.max())
    wall_bed = eta_max + wall_above

    # reshape sub-grid -> (ny, s, nx, s) and reduce over the (s, s) sub-block
    def _blocks(a):
        return a.reshape(ncell_y, s, ncell_x, s).transpose(0, 2, 1, 3) \
                .reshape(ncell_y, ncell_x, s * s)
    Bb = _blocks(Bf); etab = _blocks(eta_f); Hb = _blocks(Hf)
    outb = _blocks(out_f.astype(float))

    inside_any = outb.min(axis=2) < 0.5          # cell has >=1 interior sub-point
    valid = ~np.isnan(Bb)
    wet_sub = np.nan_to_num(Hb) > 1e-6
    nvalid = valid.sum(axis=2)
    # a cell is wet only if most of its interior sub-points are wet -> avoids
    # over-filling reservoir/sea edge cells (which would otherwise inherit the
    # eta=100 surface over a single wet sub-point).
    cell_wet = inside_any & (nvalid > 0) & (wet_sub.sum(axis=2) >= 0.5 * nvalid)
    with np.errstate(invalid="ignore"):
        bed_in = np.nanmin(Bb, axis=2) if bed_agg == "min" else np.nanmean(Bb, axis=2)
        surf = np.nanmax(np.where(wet_sub, etab, np.nan), axis=2)  # wet surface
    bed = np.where(inside_any, np.where(np.isnan(bed_in), wall_bed, bed_in),
                   wall_bed).astype(np.float64)
    surf = np.where(np.isnan(surf), bed, surf)
    depth = np.clip(np.where(cell_wet, surf - bed, 0.0), 0.0, None).astype(np.float64)
    outside = ~inside_any

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
