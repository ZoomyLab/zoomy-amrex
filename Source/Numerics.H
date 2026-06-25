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
        amrex::SmallMatrix<amrex::Real,4,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = 0.5*n(0, 0);
    amrex::Real t1 = Q_minus(2, 0)*t0;
    amrex::Real t2 = 0.5*n(1, 0);
    amrex::Real t3 = Q_minus(3, 0)*t2;
    amrex::Real t4 = Q_plus(2, 0)*t0;
    amrex::Real t5 = Q_plus(3, 0)*t2;
    amrex::Real t6 = (1.0 / amrex::Math::powi<1>(Q_minus(1, 0)));
    amrex::Real t7 = (1.0 / amrex::Math::powi<1>(Q_plus(1, 0)));
    amrex::Real t8 = (1.0/2.0)*p(0, 0);
    amrex::Real t9 = amrex::Math::powi<2>(Q_minus(1, 0))*t8;
    amrex::Real t10 = amrex::Math::powi<2>(Q_plus(1, 0))*t8;
    amrex::SmallMatrix<amrex::Real,4,1> res;
    res(0, 0) = 0;
    res(1, 0) = t1 + t3 + t4 + t5;
    res(2, 0) = Q_minus(2, 0)*t3*t6 + Q_plus(2, 0)*t5*t7 + t0*(amrex::Math::powi<2>(Q_minus(2, 0))*t6 + t9) + t0*(amrex::Math::powi<2>(Q_plus(2, 0))*t7 + t10);
    res(3, 0) = Q_minus(3, 0)*t1*t6 + Q_plus(3, 0)*t4*t7 + t2*(amrex::Math::powi<2>(Q_minus(3, 0))*t6 + t9) + t2*(amrex::Math::powi<2>(Q_plus(3, 0))*t7 + t10);
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,8,1> numerical_fluctuations(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_minus,
        amrex::SmallMatrix<amrex::Real,4,1> const& Q_plus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_minus,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux_plus,
        amrex::SmallMatrix<amrex::Real,4,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::Real t0 = 0.5*amrex::max(amrex::max(amrex::Math::abs(Q_minus(0, 0)), amrex::max(amrex::Math::abs(Q_minus(1, 0)), amrex::max(amrex::Math::abs(Q_minus(2, 0)), amrex::max(amrex::Math::abs(Q_minus(3, 0)), amrex::max(amrex::Math::abs(p(0, 0)), amrex::max(amrex::Math::abs(p(1, 0)), amrex::max(amrex::Math::abs(p(2, 0)), amrex::max(amrex::Math::abs(p(3, 0)), amrex::max(amrex::Math::abs(n(0, 0)), amrex::Math::abs(n(1, 0))))))))))), amrex::max(amrex::Math::abs(Q_plus(0, 0)), amrex::max(amrex::Math::abs(Q_plus(1, 0)), amrex::max(amrex::Math::abs(Q_plus(2, 0)), amrex::max(amrex::Math::abs(Q_plus(3, 0)), amrex::max(amrex::Math::abs(p(0, 0)), amrex::max(amrex::Math::abs(p(1, 0)), amrex::max(amrex::Math::abs(p(2, 0)), amrex::max(amrex::Math::abs(p(3, 0)), amrex::max(amrex::Math::abs(n(0, 0)), amrex::Math::abs(n(1, 0))))))))))));
    amrex::Real t1 = t0*(-Q_minus(1, 0) + Q_plus(1, 0));
    amrex::Real t2 = t0*(-Q_minus(2, 0) + Q_plus(2, 0));
    amrex::Real t3 = 0.1127016653792583*Q_minus(1, 0) + 0.8872983346207417*Q_plus(1, 0);
    amrex::Real t4 = p(0, 0)*n(0, 0);
    amrex::Real t5 = 0.27777777777777785*t4;
    amrex::Real t6 = 0.22222222222222221*Q_minus(1, 0) + 0.22222222222222221*Q_plus(1, 0);
    amrex::Real t7 = 0.8872983346207417*Q_minus(1, 0) + 0.1127016653792583*Q_plus(1, 0);
    amrex::Real t8 = -0.5*Q_minus(0, 0) + 0.5*Q_plus(0, 0);
    amrex::Real t9 = t8*(t3*t5 + t4*t6 + t5*t7);
    amrex::Real t10 = t0*(-Q_minus(3, 0) + Q_plus(3, 0));
    amrex::Real t11 = p(0, 0)*n(1, 0);
    amrex::Real t12 = 0.27777777777777785*t11;
    amrex::Real t13 = t8*(t11*t6 + t12*t3 + t12*t7);
    amrex::SmallMatrix<amrex::Real,8,1> res;
    res(0, 0) = 0;
    res(1, 0) = t1;
    res(2, 0) = t2 + t9;
    res(3, 0) = t10 + t13;
    res(4, 0) = 0;
    res(5, 0) = -t1;
    res(6, 0) = -t2 + t9;
    res(7, 0) = -t10 + t13;
    return res;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::SmallMatrix<amrex::Real,1,1> local_max_abs_eigenvalue(
        amrex::SmallMatrix<amrex::Real,4,1> const& Q,
        amrex::SmallMatrix<amrex::Real,0,1> const& Qaux,
        amrex::SmallMatrix<amrex::Real,4,1> const& p,
        amrex::SmallMatrix<amrex::Real,2,1> const& n) noexcept
    {
    amrex::SmallMatrix<amrex::Real,1,1> res;
    res(0, 0) = amrex::max(amrex::Math::abs(Q(0, 0)), amrex::max(amrex::Math::abs(Q(1, 0)), amrex::max(amrex::Math::abs(Q(2, 0)), amrex::max(amrex::Math::abs(Q(3, 0)), amrex::max(amrex::Math::abs(p(0, 0)), amrex::max(amrex::Math::abs(p(1, 0)), amrex::max(amrex::Math::abs(p(2, 0)), amrex::max(amrex::Math::abs(p(3, 0)), amrex::max(amrex::Math::abs(n(0, 0)), amrex::Math::abs(n(1, 0)))))))))));
    return res;
    }

};
