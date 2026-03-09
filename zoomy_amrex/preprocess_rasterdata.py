#!/usr/bin/env python3
"""
update_inputs_dot.py  inputs.txt  [--raster NAME PATH] ...

Example:
  python update_inputs_dot.py inputs.txt --raster dem dem.tif --raster release release.tif

* writes  <path>.raw (float64 row-major) for every provided raster
* patches / creates the following keys in dot-syntax format:
    init.<NAME>_file
    geometry.n_cell_x
    geometry.n_cell_y
    geometry.phy_bb_x0  ...  geometry.phy_bb_y1
"""

import argparse
import os
import re
from pathlib import Path
import rasterio
import numpy as np


# ---------------------------------------------------------------------- #
# helper: get raster size + bounds                                       #
# ---------------------------------------------------------------------- #
def raster_meta(fname: Path):
    with rasterio.open(fname) as ds:
        nx, ny = ds.width, ds.height
        T = ds.transform
        x0, y1 = T.c, T.f
        dx, dy = T.a, T.e  # dy negative in north-up images
        x1 = x0 + nx * dx
        y0 = y1 + ny * dy
    return nx, ny, x0, x1, y0, y1


# ---------------------------------------------------------------------- #
# main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Update inputs file and convert rasters.")
    ap.add_argument("inputs", help="existing dot-syntax inputs file")
    
    # Accept multiple rasters as tuples of (name, path)
    ap.add_argument("--raster", nargs=2, action="append", required=True,
                    metavar=("NAME", "PATH"),
                    help="Name of the parameter and path to the GeoTIFF (e.g., --raster dem dem.tif)")
    
    ns = ap.parse_args()
    preprocess(ns.inputs, ns.raster)


def preprocess(inputs_path, raster_list):
    updates = {}
    raw_names = []

    # ---------- 1. paths, raw files, and updates generation -------------
    for i, (name, path_str) in enumerate(raster_list):
        p = Path(path_str).resolve()
        raw_p = p.with_suffix(".raw")

        # write raw (float64 row-major)
        with rasterio.open(p) as ds:
            ds.read(1).astype(np.float64).tofile(raw_p)

        raw_names.append(raw_p.name)
        
        # Add the file key for this specific raster
        updates[f"init.{name}_file"] = f'"{raw_p}"'

        # Use the FIRST raster in the list to dictate the domain geometry
        if i == 0:
            nx, ny, x0, x1, y0, y1 = raster_meta(p)
            updates["geometry.n_cell_x"] = str(nx)
            updates["geometry.n_cell_y"] = str(ny)
            updates["geometry.phy_bb_x0"] = str(x0)
            updates["geometry.phy_bb_x1"] = str(x1)
            updates["geometry.phy_bb_y0"] = str(y0)
            updates["geometry.phy_bb_y1"] = str(y1)

    # ---------- 2. read original inputs file ---------------------------
    with open(inputs_path, "r") as f:
        lines = f.readlines()

    key_re = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=")

    # ---------- 3. replace existing keys -------------------------------
    remaining = updates.copy()
    for i, line in enumerate(lines):
        m = key_re.match(line)
        if not m:
            continue
        key = m.group(1)
        if key in remaining:
            # keep indentation / comment after value
            after_eq = line.split("=", 1)[1]
            comment = ""
            if "#" in after_eq:
                after_eq, comment = after_eq.split("#", 1)
                comment = " #" + comment.lstrip("#")
            
            lines[i] = f"{key} = {remaining[key]}{comment}".rstrip() + "\n"
            remaining.pop(key)

    # ---------- 4. append missing keys at the end ----------------------
    if remaining:
        lines.append("\n# Automatically added by update_inputs_dot.py\n")
        for k, v in remaining.items():
            lines.append(f"{k} = {v}\n")

    # ---------- 5. write result ----------------------------------------
    with open(inputs_path, "w") as f:
        f.writelines(lines)

    print(f"✓ raw files created: {', '.join(raw_names)}")
    print(f"✓ inputs updated → {os.path.abspath(inputs_path)}")


if __name__ == "__main__":
    main()