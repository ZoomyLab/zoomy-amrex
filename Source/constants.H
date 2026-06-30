#pragma once

#include <AMReX_SmallMatrix.H>

constexpr amrex::Real eps = 1e-6;
inline constexpr int idx_h = 1; // Index of water height
inline constexpr int idx_b = 0; // Index of bathymetry

// NOTE: the driver deliberately defines NO wet/dry constants (no h_min / h_dry /
// |u| ceiling).  All wet/dry regularisation is INHERITED from the model emitted
// by the printer (the NumericalSystemModel): the flux/eigenvalues use the KP-
// desingularised ``hinv`` aux, the eigenvalues are gated for dry cells and take
// sqrt(max(h, wet_dry_eps)), the friction source uses pow(max(h_floor, h)), and
// update_variables caps |u| and zeros dry-cell momentum.  The driver just
// consumes those operators.