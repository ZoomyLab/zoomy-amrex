#pragma once
#include <AMReX_Array4.H>
#include <AMReX_Vector.H>
#include <AMReX_SmallMatrix.H>
#include <vector>
#include <string>
#include <algorithm>

#ifdef __CUDACC__
#define PORTABLE_FN __host__ __device__
#else
#define PORTABLE_FN
#endif

#include "UserFunctions.H"

struct Model {
    using T = amrex::Real;
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_eq   = 4;
    static constexpr int n_dof_qaux = 4;
    static constexpr int n_parameters = 5;
    static constexpr int dimension  = 1;
    static constexpr int n_dof_gradQ = 4;
    static constexpr bool has_diffusion = false;
    static constexpr int n_boundary_tags = 2;
    static const std::vector<std::string> get_boundary_tags() { return { "left", "right" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "rho", "nu", "lambda_s", "e_x" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 0.0, 0.0, 0.0 }; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> flux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = Q(2, 0);
    res(2, 0) = amrex::Math::powi<2>(Q(2, 0))*t0 + (1.0/3.0)*amrex::Math::powi<2>(Q(3, 0))*t0;
    res(3, 0) = 2*Q(2, 0)*Q(3, 0)*t0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> dflux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> hydrostatic_pressure(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> nonconservative_matrix(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = p(0, 0)*Q(1, 0);
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = -Q(2, 0)/Q(1, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> quasilinear_matrix(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<2>(Q(1, 0)));
    amrex::Real t1 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t2 = Q(2, 0)*t1;
    amrex::Real t3 = Q(3, 0)*t1;
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 1;
    res(7, 0) = 0;
    res(8, 0) = p(0, 0)*Q(1, 0);
    res(9, 0) = p(0, 0)*Q(1, 0) - amrex::Math::powi<2>(Q(2, 0))*t0 - 1.0/3.0*amrex::Math::powi<2>(Q(3, 0))*t0;
    res(10, 0) = 2*t2;
    res(11, 0) = (2.0/3.0)*t3;
    res(12, 0) = 0;
    res(13, 0) = -2*Q(2, 0)*Q(3, 0)*t0;
    res(14, 0) = 2*t3;
    res(15, 0) = t2;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> diffusive_flux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,4,1> const& gradQ,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> eigenvalues(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t1 = n(0, 0)*Q(2, 0);
    amrex::Real t2 = n(0, 0)*std::pow(p(0, 0)*amrex::Math::powi<3>(Q(1, 0)) + amrex::Math::powi<2>(Q(3, 0)), 1.0/2.0);
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = t0*t1;
    res(2, 0) = t0*(t1 + t2);
    res(3, 0) = t0*(t1 - t2);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> left_eigenvectors(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> right_eigenvectors(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> source(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t1 = (1.0 / amrex::Math::powi<1>(p(1, 0)));
    amrex::Real t2 = p(3, 0)*t0*t1;
    amrex::Real t3 = Q(3, 0)*t2;
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = p(4, 0)*p(0, 0)*Q(1, 0) - Q(2, 0)*t2 + t3;
    res(3, 0) = 3*p(3, 0)*Q(2, 0)*t0*t1 - 3*t3 - 12*p(2, 0)*Q(3, 0)/amrex::Math::powi<2>(Q(1, 0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> source_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<2>(Q(1, 0)));
    amrex::Real t1 = p(3, 0)/p(1, 0);
    amrex::Real t2 = t0*t1;
    amrex::Real t3 = Q(2, 0)*t2;
    amrex::Real t4 = Q(3, 0)*t2;
    amrex::Real t5 = t1/Q(1, 0);
    amrex::Real t6 = 3*t5;
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = p(4, 0)*p(0, 0) + t3 - t4;
    res(10, 0) = -t5;
    res(11, 0) = t5;
    res(12, 0) = 0;
    res(13, 0) = -3*t3 + 3*t4 + 24*p(2, 0)*Q(3, 0)/amrex::Math::powi<3>(Q(1, 0));
    res(14, 0) = t6;
    res(15, 0) = -12*p(2, 0)*t0 - t6;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> source_jacobian_wrt_aux_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> project_2d_to_3d(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> project_3d_to_2d(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> residual(
        amrex::Real const time,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const dX,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> interpolate(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = Q(1, 0);
    res(2, 0) = Q(2, 0);
    res(3, 0) = Q(3, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> initial_condition(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> initial_aux_condition(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> update_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = Q(1, 0);
    res(2, 0) = Q(2, 0);
    res(3, 0) = Q(3, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> update_aux_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = Qaux(0, 0);
    res(1, 0) = Qaux(1, 0);
    res(2, 0) = Qaux(2, 0);
    res(3, 0) = Qaux(3, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> update_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 1;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 1;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 1;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = 1;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> update_aux_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,1,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    if (bc_idx == 0 || bc_idx == 1) {
        amrex::SmallMatrix<amrex::Real,4,1> res;
        res(0, 0) = Q(0, 0);
        res(1, 0) = Q(1, 0);
        res(2, 0) = Q(2, 0);
        res(3, 0) = Q(3, 0);
        return res;
    }
    amrex::SmallMatrix<amrex::Real,4,1> default_res{};
    return default_res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> aux_boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,1,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = Qaux(0, 0);
    res(1, 0) = Qaux(1, 0);
    res(2, 0) = Qaux(2, 0);
    res(3, 0) = Qaux(3, 0);
    return res;
    }

};
