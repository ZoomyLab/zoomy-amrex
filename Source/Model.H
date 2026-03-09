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
    static constexpr int n_dof_q    = 6;
    static constexpr int n_dof_qaux = 2;
    static constexpr int n_parameters = 8;
    static constexpr int dimension  = 2;
    static constexpr int n_boundary_tags = 4;
    static const std::vector<std::string> get_boundary_tags() { return { "E", "N", "S", "W" }; }
    static const std::vector<std::string> parameter_names() { return { "g", "ez", "nu", "lamda", "rho", "c_slipmod", "C", "eps" }; }
    static const std::vector<T> default_parameters() { return { 9.81, 1.0, 1e-06, 0.001, 1000.0, 0.03333333333333333, 300.0, 1e-06 }; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,12,1> flux(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(0, 0)*Q(1, 0);
    amrex::Real t1 = amrex::Math::powi<2>(Qaux(0, 0))*Q(1, 0);
    amrex::Real t2 = (1.0/3.0)*t1;
    amrex::Real t3 = Q(2, 0)*t1;
    amrex::Real t4 = Q(3, 0)*Q(5, 0)*t2 + Q(4, 0)*t3;
    amrex::Real t5 = Q(4, 0)*t1;
    amrex::Real t6 = Q(3, 0)*t5 + Q(5, 0)*t3;
    amrex::SmallMatrix<amrex::Real,12,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = Q(2, 0)*t0;
    res(3, 0) = Q(4, 0)*t0;
    res(4, 0) = amrex::Math::powi<2>(Q(2, 0))*t1 + amrex::Math::powi<2>(Q(3, 0))*t2;
    res(5, 0) = t4;
    res(6, 0) = 2*Q(3, 0)*t3;
    res(7, 0) = t6;
    res(8, 0) = t4;
    res(9, 0) = amrex::Math::powi<2>(Q(4, 0))*t1 + amrex::Math::powi<2>(Q(5, 0))*t2;
    res(10, 0) = t6;
    res(11, 0) = 2*Q(5, 0)*t5;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,12,1> dflux(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,12,1> res;
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
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,12,1> hydrostatic_pressure(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0/2.0)*p(1, 0)*p(0, 0)*amrex::Math::powi<2>(Q(1, 0));
    amrex::SmallMatrix<amrex::Real,12,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = t0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = t0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,72,1> nonconservative_matrix(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = -Qaux(0, 0)*Q(2, 0);
    amrex::Real t1 = -Qaux(0, 0)*Q(4, 0);
    amrex::SmallMatrix<amrex::Real,72,1> res;
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
    res(32, 0) = 0;
    res(33, 0) = 0;
    res(34, 0) = 0;
    res(35, 0) = 0;
    res(36, 0) = 0;
    res(37, 0) = 0;
    res(38, 0) = 0;
    res(39, 0) = 0;
    res(40, 0) = 0;
    res(41, 0) = 0;
    res(42, 0) = t0;
    res(43, 0) = 0;
    res(44, 0) = 0;
    res(45, 0) = 0;
    res(46, 0) = 0;
    res(47, 0) = t0;
    res(48, 0) = 0;
    res(49, 0) = 0;
    res(50, 0) = 0;
    res(51, 0) = 0;
    res(52, 0) = 0;
    res(53, 0) = 0;
    res(54, 0) = 0;
    res(55, 0) = 0;
    res(56, 0) = 0;
    res(57, 0) = 0;
    res(58, 0) = 0;
    res(59, 0) = 0;
    res(60, 0) = 0;
    res(61, 0) = 0;
    res(62, 0) = 0;
    res(63, 0) = 0;
    res(64, 0) = 0;
    res(65, 0) = 0;
    res(66, 0) = t1;
    res(67, 0) = 0;
    res(68, 0) = 0;
    res(69, 0) = 0;
    res(70, 0) = 0;
    res(71, 0) = t1;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,72,1> quasilinear_matrix(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(0, 0)*Q(2, 0);
    amrex::Real t1 = Qaux(0, 0)*Q(4, 0);
    amrex::Real t2 = Qaux(0, 0)*Q(1, 0);
    amrex::Real t3 = p(1, 0)*p(0, 0)*Q(1, 0);
    amrex::Real t4 = amrex::Math::powi<2>(Qaux(0, 0));
    amrex::Real t5 = (1.0/3.0)*t4;
    amrex::Real t6 = Q(4, 0)*t4;
    amrex::Real t7 = Q(5, 0)*t5;
    amrex::Real t8 = Q(2, 0)*t6 + Q(3, 0)*t7;
    amrex::Real t9 = Q(2, 0)*t4;
    amrex::Real t10 = Q(1, 0)*t9;
    amrex::Real t11 = 2*t10;
    amrex::Real t12 = Q(1, 0)*t6;
    amrex::Real t13 = Q(1, 0)*t4;
    amrex::Real t14 = Q(3, 0)*t13;
    amrex::Real t15 = Q(1, 0)*t7;
    amrex::Real t16 = Q(1, 0)*Q(3, 0)*t5;
    amrex::Real t17 = Q(3, 0)*t6 + Q(5, 0)*t9;
    amrex::Real t18 = Q(5, 0)*t13;
    amrex::Real t19 = -t0;
    amrex::Real t20 = 2*t12;
    amrex::Real t21 = -t1;
    amrex::SmallMatrix<amrex::Real,72,1> res;
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
    res(14, 0) = t0;
    res(15, 0) = t1;
    res(16, 0) = t2;
    res(17, 0) = 0;
    res(18, 0) = 0;
    res(19, 0) = 0;
    res(20, 0) = 0;
    res(21, 0) = t2;
    res(22, 0) = 0;
    res(23, 0) = 0;
    res(24, 0) = 0;
    res(25, 0) = 0;
    res(26, 0) = amrex::Math::powi<2>(Q(2, 0))*t4 + amrex::Math::powi<2>(Q(3, 0))*t5 + t3;
    res(27, 0) = t8;
    res(28, 0) = t11;
    res(29, 0) = t12;
    res(30, 0) = (2.0/3.0)*t14;
    res(31, 0) = t15;
    res(32, 0) = 0;
    res(33, 0) = t10;
    res(34, 0) = 0;
    res(35, 0) = t16;
    res(36, 0) = 0;
    res(37, 0) = 0;
    res(38, 0) = 2*Q(3, 0)*t9;
    res(39, 0) = t17;
    res(40, 0) = 2*t14;
    res(41, 0) = t18;
    res(42, 0) = t11 + t19;
    res(43, 0) = t12;
    res(44, 0) = 0;
    res(45, 0) = t14;
    res(46, 0) = 0;
    res(47, 0) = t10 + t19;
    res(48, 0) = 0;
    res(49, 0) = 0;
    res(50, 0) = t8;
    res(51, 0) = amrex::Math::powi<2>(Q(4, 0))*t4 + amrex::Math::powi<2>(Q(5, 0))*t5 + t3;
    res(52, 0) = t12;
    res(53, 0) = 0;
    res(54, 0) = t15;
    res(55, 0) = 0;
    res(56, 0) = t10;
    res(57, 0) = t20;
    res(58, 0) = t16;
    res(59, 0) = (2.0/3.0)*t18;
    res(60, 0) = 0;
    res(61, 0) = 0;
    res(62, 0) = t17;
    res(63, 0) = 2*Q(5, 0)*t6;
    res(64, 0) = t18;
    res(65, 0) = 0;
    res(66, 0) = t12 + t21;
    res(67, 0) = 0;
    res(68, 0) = t14;
    res(69, 0) = 2*t18;
    res(70, 0) = t10;
    res(71, 0) = t20 + t21;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> eigenvalues(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = Q(1, 0) > p(7, 0);
    amrex::Real t1 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t2 = n(0, 0)*Q(2, 0) + n(1, 0)*Q(4, 0);
    amrex::Real t3 = n(0, 0)*Q(3, 0);
    amrex::Real t4 = n(1, 0)*Q(5, 0);
    amrex::Real t5 = amrex::Math::powi<2>(n(0, 0));
    amrex::Real t6 = amrex::Math::powi<2>(n(1, 0));
    amrex::Real t7 = p(1, 0)*p(0, 0)*amrex::Math::powi<3>(Q(1, 0));
    amrex::Real t8 = std::pow(amrex::Math::powi<2>(Q(3, 0))*t5 + amrex::Math::powi<2>(Q(5, 0))*t6 + 2*t3*t4 + t5*t7 + t6*t7, 1.0/2.0);
    amrex::Real t9 = (1.0/3.0)*std::pow(3, 1.0/2.0);
    amrex::Real t10 = t3*t9;
    amrex::Real t11 = t4*t9;
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = ((t0) ? (0) : (0));
    res(1, 0) = ((t0) ? (t1*t2) : (0));
    res(2, 0) = ((t0) ? (t1*(t2 - t8)) : (0));
    res(3, 0) = ((t0) ? (t1*(t2 + t8)) : (0));
    res(4, 0) = ((t0) ? (t1*(-t10 - t11 + t2)) : (0));
    res(5, 0) = ((t0) ? (t1*(t10 + t11 + t2)) : (0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,36,1> left_eigenvectors(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,36,1> res;
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
    res(32, 0) = 0;
    res(33, 0) = 0;
    res(34, 0) = 0;
    res(35, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,36,1> right_eigenvectors(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,36,1> res;
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
    res(32, 0) = 0;
    res(33, 0) = 0;
    res(34, 0) = 0;
    res(35, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> source(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = 1.0*Qaux(0, 0);
    amrex::Real t1 = Qaux(1, 0)/p(4, 0);
    amrex::Real t2 = t1*(Qaux(0, 0)*Q(2, 0) - Q(3, 0)*t0);
    amrex::Real t3 = t1*(Qaux(0, 0)*Q(4, 0) - Q(5, 0)*t0);
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = -2.0*t2;
    res(3, 0) = 6.0*t2;
    res(4, 0) = -2.0*t3;
    res(5, 0) = 6.0*t3;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,36,1> source_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(1, 0)*Qaux(0, 0)/p(4, 0);
    amrex::Real t1 = 2.0*t0;
    amrex::Real t2 = -t1;
    amrex::Real t3 = 6.0*t0;
    amrex::Real t4 = -t3;
    amrex::SmallMatrix<amrex::Real,36,1> res;
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
    res(14, 0) = t2;
    res(15, 0) = t3;
    res(16, 0) = 0;
    res(17, 0) = 0;
    res(18, 0) = 0;
    res(19, 0) = 0;
    res(20, 0) = t1;
    res(21, 0) = t4;
    res(22, 0) = 0;
    res(23, 0) = 0;
    res(24, 0) = 0;
    res(25, 0) = 0;
    res(26, 0) = 0;
    res(27, 0) = 0;
    res(28, 0) = t2;
    res(29, 0) = t3;
    res(30, 0) = 0;
    res(31, 0) = 0;
    res(32, 0) = 0;
    res(33, 0) = 0;
    res(34, 0) = t1;
    res(35, 0) = t4;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,12,1> source_jacobian_wrt_aux_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(p(4, 0)));
    amrex::Real t1 = Qaux(1, 0)*t0;
    amrex::Real t2 = t1*(Q(2, 0) - 1.0*Q(3, 0));
    amrex::Real t3 = t1*(Q(4, 0) - 1.0*Q(5, 0));
    amrex::Real t4 = 2.0*Qaux(0, 0);
    amrex::Real t5 = 6.0*Qaux(0, 0);
    amrex::SmallMatrix<amrex::Real,12,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = -2.0*t2;
    res(3, 0) = 6.0*t2;
    res(4, 0) = -2.0*t3;
    res(5, 0) = 6.0*t3;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = t0*(2.0*Qaux(0, 0)*Q(3, 0) - Q(2, 0)*t4);
    res(9, 0) = t0*(Q(2, 0)*t5 - Q(3, 0)*t5);
    res(10, 0) = t0*(2.0*Qaux(0, 0)*Q(5, 0) - Q(4, 0)*t4);
    res(11, 0) = t0*(Q(4, 0)*t5 - Q(5, 0)*t5);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> project_2d_to_3d(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = Qaux(0, 0)*(1 - 2*X(2, 0));
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = Q(1, 0);
    res(2, 0) = Qaux(0, 0)*Q(2, 0) + Q(3, 0)*t0;
    res(3, 0) = Qaux(0, 0)*Q(4, 0) + Q(5, 0)*t0;
    res(4, 0) = 0;
    res(5, 0) = 9810.0*Q(1, 0)*(1 - X(2, 0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> project_3d_to_2d(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
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
    static amrex::SmallMatrix<amrex::Real,6,1> residual(
        amrex::Real const time,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const dX,
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
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
    static amrex::SmallMatrix<amrex::Real,6,1> interpolate(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = Q(1, 0);
    res(2, 0) = Q(2, 0);
    res(3, 0) = Q(3, 0);
    res(4, 0) = Q(4, 0);
    res(5, 0) = Q(5, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> initial_condition(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
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
    static amrex::SmallMatrix<amrex::Real,2,1> initial_aux_condition(
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,2,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> update_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = ((false) ? (0) : (Q(1, 0)));
    amrex::Real t1 = t0/(100.0*p(7, 0) + t0);
    amrex::SmallMatrix<amrex::Real,6,1> res;
    res(0, 0) = Q(0, 0);
    res(1, 0) = t0;
    res(2, 0) = Q(2, 0)*t1;
    res(3, 0) = Q(3, 0)*t1;
    res(4, 0) = Q(4, 0)*t1;
    res(5, 0) = Q(5, 0)*t1;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,2,1> update_aux_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::SmallMatrix<amrex::Real,2,1> res;
    res(0, 0) = Q(1, 0)/(p(7, 0) + amrex::Math::powi<2>(Q(1, 0)));
    res(1, 0) = Qaux(1, 0);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,36,1> update_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = ((false) ? (0) : (1));
    amrex::Real t1 = ((false) ? (0) : (Q(1, 0)));
    amrex::Real t2 = 100.0*p(7, 0) + t1;
    amrex::Real t3 = (1.0 / amrex::Math::powi<1>(t2));
    amrex::Real t4 = t0*t3;
    amrex::Real t5 = t0*t1/amrex::Math::powi<2>(t2);
    amrex::Real t6 = t1*t3;
    amrex::SmallMatrix<amrex::Real,36,1> res;
    res(0, 0) = 1;
    res(1, 0) = 0;
    res(2, 0) = 0;
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = t0;
    res(8, 0) = Q(2, 0)*t4 - Q(2, 0)*t5;
    res(9, 0) = Q(3, 0)*t4 - Q(3, 0)*t5;
    res(10, 0) = Q(4, 0)*t4 - Q(4, 0)*t5;
    res(11, 0) = Q(5, 0)*t4 - Q(5, 0)*t5;
    res(12, 0) = 0;
    res(13, 0) = 0;
    res(14, 0) = t6;
    res(15, 0) = 0;
    res(16, 0) = 0;
    res(17, 0) = 0;
    res(18, 0) = 0;
    res(19, 0) = 0;
    res(20, 0) = 0;
    res(21, 0) = t6;
    res(22, 0) = 0;
    res(23, 0) = 0;
    res(24, 0) = 0;
    res(25, 0) = 0;
    res(26, 0) = 0;
    res(27, 0) = 0;
    res(28, 0) = t6;
    res(29, 0) = 0;
    res(30, 0) = 0;
    res(31, 0) = 0;
    res(32, 0) = 0;
    res(33, 0) = 0;
    res(34, 0) = 0;
    res(35, 0) = t6;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,12,1> update_aux_variables_jacobian_wrt_variables(
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,8,1> const& p) noexcept
    {
    amrex::Real t0 = amrex::Math::powi<2>(Q(1, 0));
    amrex::Real t1 = p(7, 0) + t0;
    amrex::SmallMatrix<amrex::Real,12,1> res;
    res(0, 0) = 0;
    res(1, 0) = 0;
    res(2, 0) = -2*t0/amrex::Math::powi<2>(t1) + (1.0 / amrex::Math::powi<1>(t1));
    res(3, 0) = 0;
    res(4, 0) = 0;
    res(5, 0) = 0;
    res(6, 0) = 0;
    res(7, 0) = 0;
    res(8, 0) = 0;
    res(9, 0) = 0;
    res(10, 0) = 0;
    res(11, 0) = 0;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,6,1> boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,2,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    if (bc_idx == 0) {
        amrex::SmallMatrix<amrex::Real,6,1> res;
        res(0, 0) = Q(0, 0);
        res(1, 0) = 0.5*std::cos(1.5700000000000001*time) + 5;
        res(2, 0) = Q(2, 0);
        res(3, 0) = Q(3, 0);
        res(4, 0) = Q(4, 0);
        res(5, 0) = Q(5, 0);
        return res;
    } else if (bc_idx == 1 || bc_idx == 2 || bc_idx == 3) {
        amrex::SmallMatrix<amrex::Real,6,1> res;
        res(0, 0) = Q(0, 0);
        res(1, 0) = Q(1, 0);
        res(2, 0) = Q(2, 0);
        res(3, 0) = Q(3, 0);
        res(4, 0) = Q(4, 0);
        res(5, 0) = Q(5, 0);
        return res;
    }
    amrex::SmallMatrix<amrex::Real,6,1> default_res{};
    return default_res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,2,1> aux_boundary_conditions(
        const int bc_idx,
        amrex::SmallMatrix<amrex::Real,6,1> const& Q,
        amrex::SmallMatrix<amrex::Real,2,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,2,1> const& n,
        amrex::SmallMatrix<amrex::Real,3,1> const& X,
        amrex::Real const time,
        amrex::Real const dX) noexcept
    {
    if (bc_idx == 0 || bc_idx == 1 || bc_idx == 2 || bc_idx == 3) {
        amrex::SmallMatrix<amrex::Real,2,1> res;
        res(0, 0) = Qaux(0, 0);
        res(1, 0) = Qaux(1, 0);
        return res;
    }
    amrex::SmallMatrix<amrex::Real,2,1> default_res{};
    return default_res;
    }

};
