#pragma once

#include <AMReX_SmallMatrix.H>

inline constexpr amrex::Real eps = 1e-6;
inline constexpr int idx_h = 1; // Index of water height
inline constexpr int idx_b = 0; // Index of bathymetry

// The wet/dry desingularisation of the PHYSICS (1/h -> hinv, dry eigenvalue
// gate, |u| cap) is inherited from the (Numerical)SystemModel.  The driver adds
// a thin NUMERICAL positivity/velocity safety net for the explicit FV update
// (the firedrake reference uses Zhang-Shu + dt-halving for the same purpose):
inline constexpr amrex::Real h_min = 1e-8;  // NaN guard floor (flux/dt/source)
inline constexpr amrex::Real h_dry = 1e-3;  // below this a cell carries no flow
inline constexpr amrex::Real u_max = 50.0;  // hard |u| ceiling (model caps ~30)