#pragma once

#include <AMReX_SmallMatrix.H>

inline constexpr amrex::Real eps = 1e-6;
inline constexpr int idx_h = 1; // Index of water height
inline constexpr int idx_b = 0; // Index of bathymetry

// Wet/dry desingularisation: the smallest depth used when the flux/eigenvalue
// kernels divide by h.  Dry cells (h≈0) would otherwise produce 1/0 -> NaN.
// Tiny enough to be invisible to fully-wet runs (h>=h_min always holds there).
inline constexpr amrex::Real h_min = 1e-8;

// A cell shallower than h_dry carries no flow: its momentum/moments are zeroed
// (so hu/h cannot blow up the wavespeed -> dt). The depth itself is preserved
// (mass conserving); only the flux/eigenvalue division uses the h_min floor.
inline constexpr amrex::Real h_dry = 1e-3;

// Velocity desingularisation ceiling: at a wet/dry front a thin newly-wetted
// cell can momentarily carry an enormous u = hu/h that throttles dt to dtmin.
// Cap |momentum|/h at u_max (m/s). Set well above any physical speed so it
// only clips the spurious spikes and is invisible to ordinary (wet) runs.
inline constexpr amrex::Real u_max = 50.0;