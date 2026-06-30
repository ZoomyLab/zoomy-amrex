#!/usr/bin/env python3
"""Reproducible deliverable for the zoomy_amrex dam break.

Reads the AMReX plotfiles produced by ``main{2,3}d.gnu.MPI.ex`` and renders
(a) a water-depth heat-map + centreline profile PNG at the final time and
(b) a GIF over all snapshots.  Headless (Agg backend); needs only
numpy + matplotlib + Pillow, all present in the zoomy_amrex container.

Handles both 2-D and 3-D (nz=1) plotfiles and *multi-level* (AMR) output:
the level-0 field is shown as the heat-map, and every refined patch is
outlined so the adaptive mesh is visible.

Usage (inside the container):
    python3 deliverable.py <run_dir> [--out figures/dam_break]

where ``<run_dir>`` holds ``plt_*`` directories.  Variable order is
``[b, h, q_x_0, ...]`` (var0 = bed, var1 = depth).
"""
import argparse
import glob
import os
import re
import struct

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ── AMReX plotfile reader (2-D / 3-D, multi-level) ──────────────────────────
def _read_header(path):
    """Parse the plotfile Header: ncomp, names, finest_level, prob_lo/hi, refs."""
    with open(os.path.join(path, "Header")) as f:
        lines = f.read().splitlines()
    ncomp = int(lines[1])
    names = lines[2:2 + ncomp]
    i = 2 + ncomp
    dim = int(lines[i]); i += 1
    _time = float(lines[i]); i += 1
    finest_level = int(lines[i]); i += 1
    prob_lo = [float(v) for v in lines[i].split()]; i += 1
    prob_hi = [float(v) for v in lines[i].split()]; i += 1
    refs = [int(v) for v in lines[i].split()] if lines[i].split() and \
        lines[i].split()[0].lstrip("-").isdigit() else []
    return dict(ncomp=ncomp, names=names, dim=dim, finest_level=finest_level,
                prob_lo=prob_lo, prob_hi=prob_hi, refs=refs)


# Box format is dimension-agnostic: ((i,j[,k]) (i,j[,k]) (i,j[,k])).
_BOX = re.compile(r"\(\(([-\d,]+)\) \(([-\d,]+)\)")


def _pad3(coords):
    c = [int(v) for v in coords.split(",")]
    return c + [0] * (3 - len(c))


def _read_level(lvl_dir, ncomp):
    """Return (dense (ncomp,ny,nx) array, (ilo,jlo,nx,ny), boxes) for one level.

    boxes is a list of (lo_i, lo_j, hi_i, hi_j) in this level's index space.
    """
    with open(os.path.join(lvl_dir, "Cell_H")) as f:
        txt = f.read()
    raw_boxes = [(_pad3(a), _pad3(b)) for a, b in _BOX.findall(txt)]
    fods = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt)
    ilo = min(lo[0] for lo, _ in raw_boxes); jlo = min(lo[1] for lo, _ in raw_boxes)
    ihi = max(hi[0] for _, hi in raw_boxes); jhi = max(hi[1] for _, hi in raw_boxes)
    nx = ihi - ilo + 1; ny = jhi - jlo + 1
    out = np.full((ncomp, ny, nx), np.nan)
    boxes = []
    for (lo, hi), (fname, off) in zip(raw_boxes, fods):
        bx = hi[0] - lo[0] + 1; by = hi[1] - lo[1] + 1; bz = hi[2] - lo[2] + 1
        boxes.append((lo[0], lo[1], hi[0], hi[1]))
        with open(os.path.join(lvl_dir, fname), "rb") as fb:
            fb.seek(int(off))
            hdr = b""
            while not hdr.endswith(b"\n"):
                hdr += fb.read(1)
            count = bx * by * bz * ncomp
            data = np.array(struct.unpack("<%dd" % count, fb.read(count * 8)))
        data = data.reshape(ncomp, bz, by, bx)[:, 0, :, :]  # k=0 slice
        out[:, lo[1] - jlo:hi[1] - jlo + 1, lo[0] - ilo:hi[0] - ilo + 1] = data
    return out, (ilo, jlo, nx, ny), boxes


def read_plotfile(path):
    """Return (level0 array, names, prob extent, list of fine-patch rects).

    Fine-patch rects are (x0, y0, w, h) in physical coordinates, one per box
    on every refined level.
    """
    H = _read_header(path)
    arr, ext0, _ = _read_level(os.path.join(path, "Level_0"), H["ncomp"])
    plo, phi = H["prob_lo"], H["prob_hi"]
    Lx = phi[0] - plo[0]; Ly = phi[1] - plo[1]
    nx0, ny0 = ext0[2], ext0[3]
    rects = []
    cum_ref = 1
    for lev in range(1, H["finest_level"] + 1):
        cum_ref *= (H["refs"][lev - 1] if lev - 1 < len(H["refs"]) else 2)
        nlx, nly = nx0 * cum_ref, ny0 * cum_ref
        _, _, boxes = _read_level(os.path.join(path, f"Level_{lev}"), H["ncomp"])
        for (li, lj, hi, hj) in boxes:
            x0 = plo[0] + li / nlx * Lx
            y0 = plo[1] + lj / nly * Ly
            rects.append((x0, y0, (hi - li + 1) / nlx * Lx,
                          (hj - lj + 1) / nly * Ly))
    return arr, H["names"], (plo[0], phi[0], plo[1], phi[1]), rects


def read_levels(path):
    """Return (per-level dicts, header).

    Each level dict: arr (ncomp,ny,nx over its bounding box), ext (ilo,jlo,nx,ny)
    in that level's index space, boxes (list of li,lj,hi,hj), ref (cumulative
    refinement factor vs level 0).
    """
    H = _read_header(path)
    levels = []
    cum = 1
    for lev in range(H["finest_level"] + 1):
        if lev > 0:
            cum *= (H["refs"][lev - 1] if lev - 1 < len(H["refs"]) else 2)
        arr, ext, boxes = _read_level(os.path.join(path, f"Level_{lev}"), H["ncomp"])
        levels.append(dict(arr=arr, ext=ext, boxes=boxes, ref=cum))
    return levels, H


def composite_field(levels, comp_idx=1):
    """Paint every level onto one array at the finest resolution.

    Coarse cells are block-replicated; each finer level overwrites its patches,
    so the returned image shows the actual data the solver carries on the
    highest-resolution mesh available at each location.
    """
    nx0, ny0 = levels[0]["ext"][2], levels[0]["ext"][3]
    cum_f = levels[-1]["ref"]
    NX, NY = nx0 * cum_f, ny0 * cum_f
    comp = np.full((NY, NX), np.nan)
    for L in levels:
        f = cum_f // L["ref"]
        ilo, jlo, nx, ny = L["ext"]
        data = L["arr"][comp_idx]
        up = np.repeat(np.repeat(data, f, axis=0), f, axis=1)
        I0, J0 = ilo * f, jlo * f
        sub = comp[J0:J0 + ny * f, I0:I0 + nx * f]
        m = ~np.isnan(up)
        sub[m] = up[m]
    return comp


def _draw_mesh(ax, levels, ext, colors=("0.7", "k", "r", "m")):
    """Draw the real cell grid, each level only where it is the *finest* cell.

    A coarse cell that is covered by a finer level is skipped, so the result is
    a clean nested mesh: coarse cells in smooth regions, progressively finer
    cells where the solver refined (the gradients)."""
    plo_x, phi_x, plo_y, phi_y = ext
    Lx, Ly = phi_x - plo_x, phi_y - plo_y
    nx0, ny0 = levels[0]["ext"][2], levels[0]["ext"][3]
    for li, L in enumerate(levels):
        nlx, nly = nx0 * L["ref"], ny0 * L["ref"]
        dx, dy = Lx / nlx, Ly / nly
        covered = set()                     # this-level cells hidden by a finer level
        if li + 1 < len(levels):
            r = levels[li + 1]["ref"] // L["ref"]
            for (fi, fj, fhi, fhj) in levels[li + 1]["boxes"]:
                for ci in range(fi // r, fhi // r + 1):
                    for cj in range(fj // r, fhj // r + 1):
                        covered.add((ci, cj))
        c = colors[min(li, len(colors) - 1)]
        for (bi, bj, bhi, bhj) in L["boxes"]:
            for ci in range(bi, bhi + 1):
                for cj in range(bj, bhj + 1):
                    if (ci, cj) in covered:
                        continue
                    ax.add_patch(Rectangle((plo_x + ci * dx, plo_y + cj * dy),
                                           dx, dy, fill=False, edgecolor=c,
                                           lw=0.3, alpha=0.8))


def _frame_flood(ax, bed, h, ext, title, hmax, bed_lim, dry=0.01):
    """Flood map: greyscale bathymetry relief with the water depth (Blues)
    overlaid only where wet (dry cells transparent)."""
    ax.imshow(bed, origin="lower", extent=ext, cmap="Greys",
              vmin=bed_lim[0], vmax=bed_lim[1], aspect="auto")
    wet = np.ma.masked_less_equal(h, dry)
    im = ax.imshow(wet, origin="lower", extent=ext, cmap="Blues",
                   vmin=0, vmax=hmax, aspect="auto")
    ax.set_title(title); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    return im


def _frame_mesh(ax_h, ax_p, levels, ext, title, vlim):
    comp = composite_field(levels, comp_idx=1)
    NY, NX = comp.shape
    im = ax_h.imshow(comp, origin="lower", extent=ext,
                     vmin=vlim[0], vmax=vlim[1], cmap="viridis", aspect="auto")
    _draw_mesh(ax_h, levels, ext)
    ax_h.set_title(title); ax_h.set_xlabel("x"); ax_h.set_ylabel("y")
    x = ext[0] + (np.arange(NX) + 0.5) / NX * (ext[1] - ext[0])
    ax_p.plot(x, comp[NY // 2], "b-")
    ax_p.set_ylim(vlim[0] - 0.05, vlim[1] + 0.05)
    ax_p.set_xlabel("x"); ax_p.set_ylabel("h (finest level, centreline)")
    ax_p.grid(True, alpha=0.3)
    return im


# ── figures ─────────────────────────────────────────────────────────────────
def _frame(ax_h, ax_p, arr, ext, rects, title, vlim):
    h = arr[1]                      # var1 = water depth
    ny, nx = h.shape
    im = ax_h.imshow(h, origin="lower", extent=ext,
                     vmin=vlim[0], vmax=vlim[1], cmap="viridis", aspect="auto")
    for (x0, y0, w, hh) in rects:   # outline every refined patch
        ax_h.add_patch(Rectangle((x0, y0), w, hh, fill=False,
                                  edgecolor="red", lw=0.8))
    ax_h.set_title(title); ax_h.set_xlabel("x"); ax_h.set_ylabel("y")
    x = ext[0] + (np.arange(nx) + 0.5) / nx * (ext[1] - ext[0])
    ax_p.plot(x, h[ny // 2], "b-")
    ax_p.set_ylim(vlim[0] - 0.05, vlim[1] + 0.05)
    ax_p.set_xlabel("x"); ax_p.set_ylabel("h (centreline)")
    ax_p.grid(True, alpha=0.3)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="figures/dam_break")
    ap.add_argument("--mesh", action="store_true",
                    help="composite the field at the finest level and draw the "
                         "real cell grid (shows where the mesh refines)")
    ap.add_argument("--flood", action="store_true",
                    help="flood map: water depth (Blues) over greyscale "
                         "bathymetry, dry cells transparent (real-geometry cases)")
    a = ap.parse_args()

    plts = sorted(glob.glob(os.path.join(a.run_dir, "plt_*")))
    if not plts:
        raise SystemExit(f"no plt_* in {a.run_dir}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    if a.flood:
        from PIL import Image
        snaps = [read_plotfile(p) for p in plts]   # (arr, names, ext, rects)
        ext = snaps[0][2]
        beds = [s[0][0] for s in snaps]
        hs = [s[0][1] for s in snaps]
        hmax = max(np.nanmax(h) for h in hs)
        bed_lim = (min(np.nanmin(b) for b in beds), max(np.nanmax(b) for b in beds))
        frames = []
        for i, (b, h) in enumerate(zip(beds, hs)):
            fig, ax = plt.subplots(figsize=(9, 5))
            im = _frame_flood(ax, b, h, ext,
                              f"Malpasset SWE — water depth  (snapshot {i})",
                              hmax, bed_lim)
            if i == 0:
                fig.colorbar(im, ax=ax, label="depth h [m]")
            fig.tight_layout()
            if i == len(hs) - 1:
                fig.savefig(a.out + ".png", dpi=120)
            fig.canvas.draw()
            w, hh = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames.append(Image.fromarray(buf.reshape(hh, w, 4)[..., :3].copy()))
            plt.close(fig)
        frames[0].save(a.out + ".gif", save_all=True, append_images=frames[1:],
                       duration=200, loop=0)
        print("wrote", a.out + ".png and .gif")
        return

    if a.mesh:
        from PIL import Image
        lv_snaps = [read_levels(p)[0] for p in plts]
        ext = read_plotfile(plts[-1])[2]
        comps = [composite_field(lv) for lv in lv_snaps]
        vlim = (min(np.nanmin(c) for c in comps),
                max(np.nanmax(c) for c in comps))
        frames = []
        for i, lv in enumerate(lv_snaps):
            fig, (axh, axp) = plt.subplots(1, 2, figsize=(10, 4))
            nfine = sum(len(L["boxes"]) for L in lv[1:])
            _frame_mesh(axh, axp, lv, ext,
                        f"dam break — finest-level h + mesh "
                        f"(snapshot {i}, {len(lv)} levels)", vlim)
            fig.colorbar(axh.images[0], ax=axh, label="h")
            fig.tight_layout()
            fig.savefig(a.out + ".png", dpi=120) if i == len(lv_snaps) - 1 else None
            fig.canvas.draw()
            w, hh = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frames.append(Image.fromarray(buf.reshape(hh, w, 4)[..., :3].copy()))
            plt.close(fig)
        frames[0].save(a.out + ".gif", save_all=True, append_images=frames[1:],
                       duration=300, loop=0)
        print("wrote", a.out + ".png and .gif")
        return

    snaps = [read_plotfile(p) for p in plts]   # (arr, names, ext, rects) each
    hmin = min(s[0][1].min() for s in snaps)
    hmax = max(s[0][1].max() for s in snaps)
    vlim = (hmin, hmax)
    nlev = max(1 + bool(s[3]) for s in snaps)

    # final-time PNG
    arr, _, ext, rects = snaps[-1]
    fig, (axh, axp) = plt.subplots(1, 2, figsize=(10, 4))
    _frame(axh, axp, arr, ext, rects,
           f"dam break — h, final snapshot ({len(rects)} refined patches)", vlim)
    fig.colorbar(axh.images[0], ax=axh, label="h")
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    print("wrote", a.out + ".png")

    # GIF over all snapshots (via Pillow)
    from PIL import Image
    frames = []
    for i, (arr, _, ext, rects) in enumerate(snaps):
        fig, (axh, axp) = plt.subplots(1, 2, figsize=(10, 4))
        _frame(axh, axp, arr, ext, rects, f"dam break — snapshot {i}", vlim)
        fig.colorbar(axh.images[0], ax=axh, label="h")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(Image.fromarray(buf.reshape(h, w, 4)[..., :3].copy()))
        plt.close(fig)
    frames[0].save(a.out + ".gif", save_all=True, append_images=frames[1:],
                   duration=400, loop=0)
    print("wrote", a.out + ".gif")


if __name__ == "__main__":
    main()
