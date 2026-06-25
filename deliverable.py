#!/usr/bin/env python3
"""Reproducible deliverable for the zoomy_amrex SWE dam break.

Reads the single-level AMReX plotfiles produced by ``main3d.gnu.MPI.ex`` and
renders (a) a water-depth heat-map + centreline profile PNG at the final time
and (b) a GIF over all snapshots.  Headless (Agg backend); needs only
numpy + matplotlib + Pillow, all present in the zoomy_amrex container.

Usage (inside the container):
    python3 deliverable.py <run_dir> [--out figures/dam_break]

where ``<run_dir>`` holds ``plt_*`` directories.  Variable order is
``[b, h, hu, hv]`` (var0..var3).
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


# ── minimal single-level AMReX plotfile reader ──────────────────────────────
def _read_header(path):
    with open(os.path.join(path, "Header")) as f:
        lines = f.read().splitlines()
    ncomp = int(lines[1])
    names = lines[2:2 + ncomp]
    # domain corners are deeper in the header; we instead read geometry from
    # the Level_0 boxes + the problem-domain line.
    return ncomp, names


def _read_level0(path, ncomp):
    """Return a dense (ncomp, ny, nx) array for a single-level plotfile."""
    lvl = os.path.join(path, "Level_0")
    with open(os.path.join(lvl, "Cell_H")) as f:
        txt = f.read()
    # Boxes: lines like ((0,0,0) (51,51,0) (0,0,0))
    boxes = re.findall(r"\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)", txt)
    boxes = [tuple(map(int, b)) for b in boxes]
    # FabOnDisk: Cell_D_00000 <offset>
    fods = re.findall(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt)
    # global extent
    ilo = min(b[0] for b in boxes); jlo = min(b[1] for b in boxes)
    ihi = max(b[3] for b in boxes); jhi = max(b[4] for b in boxes)
    nx = ihi - ilo + 1; ny = jhi - jlo + 1
    out = np.full((ncomp, ny, nx), np.nan)
    for (lo_i, lo_j, lo_k, hi_i, hi_j, hi_k), (fname, off) in zip(boxes, fods):
        bx = hi_i - lo_i + 1; by = hi_j - lo_j + 1; bz = hi_k - lo_k + 1
        with open(os.path.join(lvl, fname), "rb") as fb:
            fb.seek(int(off))
            # FAB ASCII header terminated by newline, then raw doubles
            hdr = b""
            while not hdr.endswith(b"\n"):
                hdr += fb.read(1)
            count = bx * by * bz * ncomp
            raw = fb.read(count * 8)
            data = np.array(struct.unpack("<%dd" % count, raw))
        # FAB layout: comp-major, then z,y,x with x fastest
        data = data.reshape(ncomp, bz, by, bx)[:, 0, :, :]  # k=0 slice
        out[:, lo_j - jlo:hi_j - jlo + 1, lo_i - ilo:hi_i - ilo + 1] = data
    return out


def read_plotfile(path):
    ncomp, names = _read_header(path)
    arr = _read_level0(path, ncomp)
    return arr, names


# ── figures ─────────────────────────────────────────────────────────────────
def _frame(ax_h, ax_p, arr, title, vlim):
    h = arr[1]                      # var1 = water depth
    ny, nx = h.shape
    im = ax_h.imshow(h, origin="lower", extent=[0, 1, 0, 1],
                     vmin=vlim[0], vmax=vlim[1], cmap="viridis", aspect="auto")
    ax_h.set_title(title); ax_h.set_xlabel("x"); ax_h.set_ylabel("y")
    x = (np.arange(nx) + 0.5) / nx
    ax_p.plot(x, h[ny // 2], "b-")
    ax_p.set_ylim(vlim[0] - 0.05, vlim[1] + 0.05)
    ax_p.set_xlabel("x"); ax_p.set_ylabel("h (centreline)")
    ax_p.grid(True, alpha=0.3)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="figures/dam_break")
    a = ap.parse_args()

    plts = sorted(glob.glob(os.path.join(a.run_dir, "plt_*")))
    if not plts:
        raise SystemExit(f"no plt_* in {a.run_dir}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    fields = [read_plotfile(p)[0] for p in plts]
    hmin = min(f[1].min() for f in fields)
    hmax = max(f[1].max() for f in fields)
    vlim = (hmin, hmax)

    # final-time PNG
    fig, (axh, axp) = plt.subplots(1, 2, figsize=(10, 4))
    _frame(axh, axp, fields[-1], f"SWE dam break — h at final snapshot", vlim)
    fig.colorbar(axh.images[0], ax=axh, label="h")
    fig.tight_layout(); fig.savefig(a.out + ".png", dpi=110)
    print("wrote", a.out + ".png")

    # GIF over all snapshots (via Pillow)
    from PIL import Image
    frames = []
    for i, f in enumerate(fields):
        fig, (axh, axp) = plt.subplots(1, 2, figsize=(10, 4))
        _frame(axh, axp, f, f"SWE dam break — snapshot {i}", vlim)
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
