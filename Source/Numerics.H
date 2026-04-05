#pragma once
#include <AMReX_Array4.H>
#include <AMReX_Vector.H>
#include <AMReX_SmallMatrix.H>
#include "Model.H"

#include <vector>
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif


struct Numerics {
    using T = amrex::Real;
    static constexpr int n_dof_q = 4;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> numerical_flux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_plus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,3,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> numerical_fluctuations(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_plus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,3,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> local_max_abs_eigenvalue(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,3,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_0{};
v_arg_0(0, 0) = Q(0, 0);
v_arg_0(1, 0) = Q(1, 0);
v_arg_0(2, 0) = Q(2, 0);
v_arg_0(3, 0) = Q(3, 0);
    amrex::SmallMatrix<amrex::Real,0,1> v_arg_1{};
    amrex::SmallMatrix<amrex::Real,3,1> v_arg_2{};
v_arg_2(0, 0) = p(0, 0);
v_arg_2(1, 0) = p(1, 0);
v_arg_2(2, 0) = p(2, 0);
    amrex::SmallMatrix<amrex::Real,2,1> v_arg_3{};
v_arg_3(0, 0) = n(0, 0);
v_arg_3(1, 0) = n(1, 0);
    auto v_call_0 = Model::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_3);
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = amrex::max(std::abs(v_call_0(0, 0)), amrex::max(std::abs(v_call_0(1, 0)), amrex::max(std::abs(v_call_0(2, 0)), std::abs(v_call_0(3, 0)))));
    return res;
    }

};
