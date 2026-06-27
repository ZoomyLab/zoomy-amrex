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
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::Real t0 = 0.5*n(0, 0);
    amrex::Real t1 = (1.0/2.0)*p(0, 0);
    amrex::Real t2 = (1.0 / amrex::Math::powi<1>(Q_minus(1, 0)));
    amrex::Real t3 = (1.0 / amrex::Math::powi<1>(Q_plus(1, 0)));
    amrex::Real t4 = 1.0*n(0, 0);
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = Q_minus(2, 0)*t0 + Q_plus(2, 0)*t0;
    res(2, 0) = t0*(amrex::Math::powi<2>(Q_minus(1, 0))*t1 + amrex::Math::powi<2>(Q_minus(2, 0))*t2 + (1.0/3.0)*amrex::Math::powi<2>(Q_minus(3, 0))*t2) + t0*(amrex::Math::powi<2>(Q_plus(1, 0))*t1 + amrex::Math::powi<2>(Q_plus(2, 0))*t3 + (1.0/3.0)*amrex::Math::powi<2>(Q_plus(3, 0))*t3);
    res(3, 0) = Q_minus(2, 0)*Q_minus(3, 0)*t2*t4 + Q_plus(2, 0)*Q_plus(3, 0)*t3*t4;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> numerical_fluctuations(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_plus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(Q_minus(1, 0)));
    amrex::Real t1 = Q_minus(2, 0)*n(0, 0);
    amrex::Real t2 = (1.0 / amrex::Math::powi<1>(Q_plus(1, 0)));
    amrex::Real t3 = Q_plus(2, 0)*n(0, 0);
    amrex::Real t4 = n(0, 0)*std::pow(amrex::Math::powi<3>(Q_minus(1, 0))*p(0, 0) + amrex::Math::powi<2>(Q_minus(3, 0)), 1.0/2.0);
    amrex::Real t5 = n(0, 0)*std::pow(amrex::Math::powi<3>(Q_plus(1, 0))*p(0, 0) + amrex::Math::powi<2>(Q_plus(3, 0)), 1.0/2.0);
    amrex::Real t6 = 0.5*amrex::max(std::abs(t0*t1), amrex::max(std::abs(t2*t3), amrex::max(std::abs(t0*(t1 + t4)), amrex::max(std::abs(t2*(t3 + t5)), amrex::max(std::abs(t0*(t1 - t4)), std::abs(t2*(t3 - t5)))))));
    amrex::Real t7 = t6*(-Q_minus(1, 0) + Q_plus(1, 0));
    amrex::Real t8 = 0.1127016653792583*Q_minus(1, 0) + 0.8872983346207417*Q_plus(1, 0);
    amrex::Real t9 = p(0, 0)*n(0, 0);
    amrex::Real t10 = 0.27777777777777785*t9;
    amrex::Real t11 = 0.5*Q_minus(1, 0) + 0.5*Q_plus(1, 0);
    amrex::Real t12 = 0.8872983346207417*Q_minus(1, 0) + 0.1127016653792583*Q_plus(1, 0);
    amrex::Real t13 = 0.5*(-Q_minus(0, 0) + Q_plus(0, 0))*(t10*t12 + t10*t8 + 0.44444444444444442*t11*t9);
    amrex::Real t14 = t6*(-Q_minus(2, 0) + Q_plus(2, 0));
    amrex::Real t15 = -Q_minus(3, 0) + Q_plus(3, 0);
    amrex::Real t16 = 0.27777777777777785*n(0, 0);
    amrex::Real t17 = 0.5*t15*(-0.44444444444444442*n(0, 0)*(0.5*Q_minus(2, 0) + 0.5*Q_plus(2, 0))/t11 - t16*(0.1127016653792583*Q_minus(2, 0) + 0.8872983346207417*Q_plus(2, 0))/t8 - t16*(0.8872983346207417*Q_minus(2, 0) + 0.1127016653792583*Q_plus(2, 0))/t12);
    amrex::Real t18 = t15*t6;
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = 0;
    res(1, 0) = t7;
    res(2, 0) = t13 + t14;
    res(3, 0) = t17 + t18;
    res(4, 0) = 0;
    res(5, 0) = -t7;
    res(6, 0) = t13 - t14;
    res(7, 0) = t17 - t18;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> local_max_abs_eigenvalue(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,4,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,5,1> const& p,
        amrex::SmallMatrix<amrex::Real,1,1> const& n) noexcept
    {
    amrex::Real t0 = (1.0 / amrex::Math::powi<1>(Q(1, 0)));
    amrex::Real t1 = n(0, 0)*Q(2, 0);
    amrex::Real t2 = n(0, 0)*std::pow(p(0, 0)*amrex::Math::powi<3>(Q(1, 0)) + amrex::Math::powi<2>(Q(3, 0)), 1.0/2.0);
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = amrex::max(std::abs(t0*t1), amrex::max(std::abs(t0*(t1 + t2)), std::abs(t0*(t1 - t2))));
    return res;
    }

};
