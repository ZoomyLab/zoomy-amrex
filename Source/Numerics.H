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
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = amrex::max(Q_minus(0, 0), Q_plus(0, 0));
    amrex::Real t1 = -t0;
    amrex::Real t2 = amrex::max(0.0, Q_minus(0, 0) + Q_minus(1, 0) + t1);
    amrex::Real t3 = t2/amrex::max(Q_minus(1, 0), p(8, 0));
    amrex::Real t4 = amrex::max(0.0, Q_plus(0, 0) + Q_plus(1, 0) + t1);
    amrex::Real t5 = t4/amrex::max(Q_plus(1, 0), p(8, 0));
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_0{};
v_arg_0(0, 0) = t0;
v_arg_0(1, 0) = t2;
v_arg_0(2, 0) = Q_minus(2, 0)*t3;
v_arg_0(3, 0) = Q_minus(3, 0)*t3;
    amrex::SmallMatrix<amrex::Real,1,1> v_arg_1{};
v_arg_1(0, 0) = (1.0 / amrex::Math::powi<1>(p(8, 0) + t2));
    amrex::SmallMatrix<amrex::Real,9,1> v_arg_2{};
v_arg_2(0, 0) = p(0, 0);
v_arg_2(1, 0) = p(1, 0);
v_arg_2(2, 0) = p(2, 0);
v_arg_2(3, 0) = p(3, 0);
v_arg_2(4, 0) = p(4, 0);
v_arg_2(5, 0) = p(5, 0);
v_arg_2(6, 0) = p(6, 0);
v_arg_2(7, 0) = p(7, 0);
v_arg_2(8, 0) = p(8, 0);
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_3{};
v_arg_3(0, 0) = t0;
v_arg_3(1, 0) = t4;
v_arg_3(2, 0) = Q_plus(2, 0)*t5;
v_arg_3(3, 0) = Q_plus(3, 0)*t5;
    amrex::SmallMatrix<amrex::Real,1,1> v_arg_4{};
v_arg_4(0, 0) = (1.0 / amrex::Math::powi<1>(p(8, 0) + t4));
    amrex::SmallMatrix<amrex::Real,2,1> v_arg_5{};
v_arg_5(0, 0) = n(0, 0);
v_arg_5(1, 0) = n(1, 0);
    auto v_call_0 = Model::flux(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_1 = Model::hydrostatic_pressure(v_arg_0, v_arg_1, v_arg_2);
    auto v_call_2 = Model::flux(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_3 = Model::hydrostatic_pressure(v_arg_3, v_arg_4, v_arg_2);
    auto v_call_4 = Model::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_5);
    auto v_call_5 = Model::eigenvalues(v_arg_3, v_arg_4, v_arg_2, v_arg_5);
    amrex::Real t6 = 0.5*n(0, 0);
    amrex::Real t7 = 0.5*n(1, 0);
    amrex::Real t8 = -amrex::max(Q_minus(0, 0), Q_plus(0, 0));
    amrex::Real t9 = amrex::max(0.0, Q_minus(0, 0) + Q_minus(1, 0) + t8);
    amrex::Real t10 = amrex::max(0.0, Q_plus(0, 0) + Q_plus(1, 0) + t8);
    amrex::Real t11 = 0.5*amrex::max(std::abs(v_call_4(0, 0)), amrex::max(std::abs(v_call_4(1, 0)), amrex::max(std::abs(v_call_4(2, 0)), amrex::max(std::abs(v_call_4(3, 0)), amrex::max(std::abs(v_call_5(0, 0)), amrex::max(std::abs(v_call_5(1, 0)), amrex::max(std::abs(v_call_5(2, 0)), std::abs(v_call_5(3, 0)))))))));
    amrex::Real t12 = t9/amrex::max(Q_minus(1, 0), p(8, 0));
    amrex::Real t13 = (1.0 / amrex::Math::powi<1>(amrex::max(Q_plus(1, 0), p(8, 0))));
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = t6*(v_call_0(0, 0) + v_call_1(0, 0)) + t6*(v_call_2(0, 0) + v_call_3(0, 0)) + t7*(v_call_0(1, 0) + v_call_1(1, 0)) + t7*(v_call_2(1, 0) + v_call_3(1, 0));
    res(1, 0) = -t11*(t10 - t9) + t6*(v_call_0(2, 0) + v_call_1(2, 0)) + t6*(v_call_2(2, 0) + v_call_3(2, 0)) + t7*(v_call_0(3, 0) + v_call_1(3, 0)) + t7*(v_call_2(3, 0) + v_call_3(3, 0));
    res(2, 0) = -t11*(-Q_minus(2, 0)*t12 + Q_plus(2, 0)*t10*t13) + t6*(v_call_0(4, 0) + v_call_1(4, 0)) + t6*(v_call_2(4, 0) + v_call_3(4, 0)) + t7*(v_call_0(5, 0) + v_call_1(5, 0)) + t7*(v_call_2(5, 0) + v_call_3(5, 0));
    res(3, 0) = -t11*(-Q_minus(3, 0)*t12 + Q_plus(3, 0)*t10*t13) + t6*(v_call_0(6, 0) + v_call_1(6, 0)) + t6*(v_call_2(6, 0) + v_call_3(6, 0)) + t7*(v_call_0(7, 0) + v_call_1(7, 0)) + t7*(v_call_2(7, 0) + v_call_3(7, 0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> numerical_fluctuations(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_plus,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = amrex::max(Q_minus(0, 0), Q_plus(0, 0));
    amrex::Real t1 = -t0;
    amrex::Real t2 = amrex::max(0.0, Q_plus(0, 0) + Q_plus(1, 0) + t1);
    amrex::Real t3 = t2/amrex::max(Q_plus(1, 0), p(8, 0));
    amrex::Real t4 = amrex::max(0.0, Q_minus(0, 0) + Q_minus(1, 0) + t1);
    amrex::Real t5 = t4/amrex::max(Q_minus(1, 0), p(8, 0));
    amrex::SmallMatrix<amrex::Real,9,1> v_arg_0{};
v_arg_0(0, 0) = p(0, 0);
v_arg_0(1, 0) = p(1, 0);
v_arg_0(2, 0) = p(2, 0);
v_arg_0(3, 0) = p(3, 0);
v_arg_0(4, 0) = p(4, 0);
v_arg_0(5, 0) = p(5, 0);
v_arg_0(6, 0) = p(6, 0);
v_arg_0(7, 0) = p(7, 0);
v_arg_0(8, 0) = p(8, 0);
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_1{};
v_arg_1(0, 0) = t0;
v_arg_1(1, 0) = t2;
v_arg_1(2, 0) = Q_plus(2, 0)*t3;
v_arg_1(3, 0) = Q_plus(3, 0)*t3;
    amrex::SmallMatrix<amrex::Real,1,1> v_arg_2{};
v_arg_2(0, 0) = (1.0 / amrex::Math::powi<1>(p(8, 0) + t2));
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_3{};
v_arg_3(0, 0) = t0;
v_arg_3(1, 0) = t4;
v_arg_3(2, 0) = Q_minus(2, 0)*t5;
v_arg_3(3, 0) = Q_minus(3, 0)*t5;
    amrex::SmallMatrix<amrex::Real,1,1> v_arg_4{};
v_arg_4(0, 0) = (1.0 / amrex::Math::powi<1>(p(8, 0) + t4));
    auto v_call_0 = Model::hydrostatic_pressure(Q_plus, Qaux_plus, v_arg_0);
    auto v_call_1 = Model::hydrostatic_pressure(v_arg_1, v_arg_2, v_arg_0);
    auto v_call_2 = Model::hydrostatic_pressure(v_arg_3, v_arg_4, v_arg_0);
    auto v_call_3 = Model::hydrostatic_pressure(Q_minus, Qaux_minus, v_arg_0);
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = n(0, 0)*(-v_call_0(0, 0) + v_call_1(0, 0)) + n(1, 0)*(-v_call_0(1, 0) + v_call_1(1, 0));
    res(1, 0) = n(0, 0)*(-v_call_0(2, 0) + v_call_1(2, 0)) + n(1, 0)*(-v_call_0(3, 0) + v_call_1(3, 0));
    res(2, 0) = n(0, 0)*(-v_call_0(4, 0) + v_call_1(4, 0)) + n(1, 0)*(-v_call_0(5, 0) + v_call_1(5, 0));
    res(3, 0) = n(0, 0)*(-v_call_0(6, 0) + v_call_1(6, 0)) + n(1, 0)*(-v_call_0(7, 0) + v_call_1(7, 0));
    res(4, 0) = n(0, 0)*(-v_call_2(0, 0) + v_call_3(0, 0)) + n(1, 0)*(-v_call_2(1, 0) + v_call_3(1, 0));
    res(5, 0) = n(0, 0)*(-v_call_2(2, 0) + v_call_3(2, 0)) + n(1, 0)*(-v_call_2(3, 0) + v_call_3(3, 0));
    res(6, 0) = n(0, 0)*(-v_call_2(4, 0) + v_call_3(4, 0)) + n(1, 0)*(-v_call_2(5, 0) + v_call_3(5, 0));
    res(7, 0) = n(0, 0)*(-v_call_2(6, 0) + v_call_3(6, 0)) + n(1, 0)*(-v_call_2(7, 0) + v_call_3(7, 0));
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> local_max_abs_eigenvalue(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,1,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,9,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,4,1> v_arg_0{};
v_arg_0(0, 0) = Q(0, 0);
v_arg_0(1, 0) = Q(1, 0);
v_arg_0(2, 0) = Q(2, 0);
v_arg_0(3, 0) = Q(3, 0);
    amrex::SmallMatrix<amrex::Real,1,1> v_arg_1{};
v_arg_1(0, 0) = Qaux(0, 0);
    amrex::SmallMatrix<amrex::Real,9,1> v_arg_2{};
v_arg_2(0, 0) = p(0, 0);
v_arg_2(1, 0) = p(1, 0);
v_arg_2(2, 0) = p(2, 0);
v_arg_2(3, 0) = p(3, 0);
v_arg_2(4, 0) = p(4, 0);
v_arg_2(5, 0) = p(5, 0);
v_arg_2(6, 0) = p(6, 0);
v_arg_2(7, 0) = p(7, 0);
v_arg_2(8, 0) = p(8, 0);
    amrex::SmallMatrix<amrex::Real,2,1> v_arg_3{};
v_arg_3(0, 0) = n(0, 0);
v_arg_3(1, 0) = n(1, 0);
    auto v_call_0 = Model::eigenvalues(v_arg_0, v_arg_1, v_arg_2, v_arg_3);
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = amrex::max(std::abs(v_call_0(0, 0)), amrex::max(std::abs(v_call_0(1, 0)), amrex::max(std::abs(v_call_0(2, 0)), std::abs(v_call_0(3, 0)))));
    return res;
    }

};
