#pragma once
// amrex_user::max_wavespeed — eigenvalue-based wave speed for the Rusanov
// dissipation + CFL.  The generated Numerics kernels emit the symbolic
// max_wavespeed(*Q, *Qaux, *p, *n) as ONE flat variadic call; here we unpack
// it back into SmallMatrix args and take the spectral radius from
// Model::eigenvalues.  This avoids the base max(|args|) fallback (which
// wrongly includes the parameter vector — e.g. the slip length lambda_s).
#include "Model.H"
#include <AMReX_REAL.H>
#include <AMReX_SmallMatrix.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <cmath>

namespace amrex_user {

template <typename... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real max_wavespeed(Ts... args) noexcept
{
    constexpr int NQ = Model::n_dof_q;
    constexpr int NA = Model::n_dof_qaux;
    constexpr int NP = Model::n_parameters;
    constexpr int ND = Model::dimension;
    static_assert(sizeof...(args) == NQ + NA + NP + ND,
        "max_wavespeed: arg count must equal n_dof_q+n_dof_qaux+n_parameters+dimension");
    const amrex::Real packed[] = { static_cast<amrex::Real>(args)... };
    amrex::SmallMatrix<amrex::Real, NQ, 1> Q;
    amrex::SmallMatrix<amrex::Real, NA, 1> Qaux;
    amrex::SmallMatrix<amrex::Real, NP, 1> p;
    amrex::SmallMatrix<amrex::Real, ND, 1> n;
    int i = 0;
    for (int k = 0; k < NQ; ++k) Q(k, 0)    = packed[i++];
    for (int k = 0; k < NA; ++k) Qaux(k, 0) = packed[i++];
    for (int k = 0; k < NP; ++k) p(k, 0)    = packed[i++];
    for (int k = 0; k < ND; ++k) n(k, 0)    = packed[i++];
    auto ev = Model::eigenvalues(Q, Qaux, p, n);
    amrex::Real mx = 0.0;
    for (int k = 0; k < NQ; ++k) mx = amrex::max(mx, std::abs(ev(k, 0)));
    return mx;
}

}  // namespace amrex_user
