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


struct Model {
    using T = amrex::Real;
    static constexpr int n_dof_q    = 4;
    static constexpr int n_dof_qaux = 1;
    static constexpr int n_parameters = 9;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 4;
    static const std::vector<std::string> get_boundary_tags() { return { "E", "N", "S", "W" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "ey", "ez", "nu", "lamda", "rho", "c_slipmod", "C", "eps" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 0.0, 1.0, 1e-06, 0.001, 1000.0, 0.03333333333333333, 300.0, 1e-06 }; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> flux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(0, 0)*Q(1, 0);
    amrex::Real t1 = amrex::Math::powi<2>(Qaux(0, 0))*Q(1, 0);
    amrex::Real t2 = Q(2, 0)*Q(3, 0)*t1;
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = Q(2, 0)*t0;
    res(3, 0) = Q(3, 0)*t0;
    res(4, 0) = amrex::Math::powi<2>(Q(2, 0))*t1;
    res(5, 0) = t2;
    res(6, 0) = t2;
    res(7, 0) = amrex::Math::powi<2>(Q(3, 0))*t1;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> dflux(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
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
    static amrex::SmallMatrix<amrex::Real,8,1> hydrostatic_pressure(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0/2.0)*p(2, 0)*p(0, 0)*amrex::Math::powi<2>(Q(1, 0));
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = t0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = t0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,32,1> nonconservative_matrix(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,32,1> res;
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
    res(16, 0) = 0;
    res(17, 0) = 0;
    res(18, 0) = 0;
    res(19, 0) = 0;
    res(20, 0) = 0;
    res(21, 0) = 0;
    res(22, 0) = 0;
    res(23, 0) = 0;
    res(24, 0) = 0;
    res(25, 0) = 0;
    res(26, 0) = 0;
    res(27, 0) = 0;
    res(28, 0) = 0;
    res(29, 0) = 0;
    res(30, 0) = 0;
    res(31, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,32,1> quasilinear_matrix(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(0, 0)*Q(1, 0);
    amrex::Real t1 = p(2, 0)*p(0, 0)*Q(1, 0);
    amrex::Real t2 = amrex::Math::powi<2>(Qaux(0, 0));
    amrex::Real t3 = Q(3, 0)*t2;
    amrex::Real t4 = Q(2, 0)*t3;
    amrex::Real t5 = Q(1, 0)*Q(2, 0)*t2;
    amrex::Real t6 = Q(1, 0)*t3;
    amrex::SmallMatrix<amrex::Real,32,1> res;
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
    res(10, 0) = Qaux(0, 0)*Q(2, 0);
    res(11, 0) = Qaux(0, 0)*Q(3, 0);
    res(12, 0) = t0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = t0;
    res(16, 0) = 0;
    res(17, 0) = 0;
    res(18, 0) = amrex::Math::powi<2>(Q(2, 0))*t2 + t1;
    res(19, 0) = t4;
    res(20, 0) = 2*t5;
    res(21, 0) = t6;
    res(22, 0) = 0;
    res(23, 0) = t5;
    res(24, 0) = 0;
    res(25, 0) = 0;
    res(26, 0) = t4;
    res(27, 0) = amrex::Math::powi<2>(Q(3, 0))*t2 + t1;
    res(28, 0) = t6;
    res(29, 0) = 0;
    res(30, 0) = t5;
    res(31, 0) = 2*t6;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> eigenvalues(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = Q(1, 0) > p(8, 0);
    amrex::Real t1 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t2 = n(0, 0)*Q(2, 0) + n(1, 0)*Q(3, 0);
    amrex::Real t3 = std::pow(p(2, 0), 1.0/2.0)*std::pow(p(0, 0), 1.0/2.0)*std::pow(Q(1, 0), 3.0/2.0)*std::pow(amrex::Math::powi<2>(n(0, 0)) + amrex::Math::powi<2>(n(1, 0)), 1.0/2.0);
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = ((t0) ? (0) : (0));
    res(1, 0) = ((t0) ? (t1*t2) : (0));
    res(2, 0) = ((t0) ? (t1*(t2 - t3)) : (0));
    res(3, 0) = ((t0) ? (t1*(t2 + t3)) : (0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> left_eigenvectors(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
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
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
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
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> source_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
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
    static amrex::SmallMatrix<amrex::Real,4,1> source_jacobian_wrt_aux_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> project_2d_to_3d(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = Q(1, 0);
    res(2, 0) = Qaux(0, 0)*Q(2, 0);
    res(3, 0) = Qaux(0, 0)*Q(3, 0);
    res(4, 0) = 0;
    res(5, 0) = 9810.0*Q(1, 0)*(1 - X(2, 0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> project_3d_to_2d(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
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
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
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
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
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
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> initial_aux_condition(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> update_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = ((false) ? (0) : (Q(1, 0)));
    amrex::Real t1 = t0/(100.0*p(8, 0) + t0);
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = t0;
    res(2, 0) = Q(2, 0)*t1;
    res(3, 0) = Q(3, 0)*t1;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> update_aux_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = Q(1, 0)/(p(8, 0) + amrex::Math::powi<2>(Q(1, 0)));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,16,1> update_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = ((false) ? (0) : (1));
    amrex::Real t1 = ((false) ? (0) : (Q(1, 0)));
    amrex::Real t2 = 100.0*p(8, 0) + t1;
    amrex::Real t3 = (1.0 / amrex::Math::powi<1>(t2));
    amrex::Real t4 = t0*t3;
    amrex::Real t5 = t0*t1/amrex::Math::powi<2>(t2);
    amrex::Real t6 = t1*t3;
    amrex::SmallMatrix<amrex::Real,16,1> res;
    res(0, 0) = 1;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = t0;
    res(6, 0) = Q(2, 0)*t4 - Q(2, 0)*t5;
    res(7, 0) = Q(3, 0)*t4 - Q(3, 0)*t5;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = t6;
    res(11, 0) = 0;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = 0;
    res(15, 0) = t6;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> update_aux_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p) noexcept
    {
    amrex::Real t0 = amrex::Math::powi<2>(Q(1, 0));
    amrex::Real t1 = p(8, 0) + t0;
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = -2*t0/amrex::Math::powi<2>(t1) + (1.0 / amrex::Math::powi<1>(t1));
    res(2, 0) = 0;
    res(3, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,4,1> boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,2,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2 || bc_idx == 3) {
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
    static amrex::SmallMatrix<amrex::Real,1,1> aux_boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,2,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2 || bc_idx == 3) {
        amrex::SmallMatrix<amrex::Real,1,1> res;
        res(0, 0) = Qaux(0, 0);
        return res;
    }
    amrex::SmallMatrix<amrex::Real,1,1> default_res{};
    return default_res;
    }

};
