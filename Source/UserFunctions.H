#pragma once
/// \file
/// AMReX device versions of the opaque kernel placeholders of zoomy_core.model.kernel_functions, called from the printed Chorin predictor.
/// A failure has no exception channel on device, so it is encoded in the value: lambda = +inf, R = L = I.

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include "ZoomyEig.H"

#ifndef NDEBUG
#  include <cassert>
#  define ZOOMY_EIG_TRAP(st) assert((st) == ::amrex_user::zoomy_eig::OK && \
        "zoomy eig kernel refused: non-finite / non-convergent / complex / singular")
#else
#  define ZOOMY_EIG_TRAP(st) ((void)(st))   // consume it: -Wunused-variable
#endif

namespace amrex_user {

/// idx-th eigenvalue of the row-major n x n matrix A_flat, ascending; a complex pair gives the signed modulus (the CFL bound stays finite).
template <typename I, typename... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eigenvalues(I idx_, Ts... a_flat_args) noexcept
{
    constexpr int M = sizeof...(Ts);
    constexpr int N = zoomy_eig::isqrt_ct(M);
    static_assert(N * N == M, "eigenvalues(): flat argument count is not a perfect square");
    static_assert(N >= 1 && N <= 16, "eigenvalues(): unsupported system size");

    const double A[M] = { static_cast<double>(a_flat_args)... };
    double out[N];
    const int st = zoomy_eig::values_core<N>(A, out);
    ZOOMY_EIG_TRAP(st);
    const int idx = static_cast<int>(idx_);
    if (idx < 0 || idx >= N) return ZOOMY_EIG_INF;
    return static_cast<amrex::Real>(out[idx]);
}

/// Upper bound on rho(A) of the row-major n x n A_flat: j squarings, each after dividing by the max row sum s, then w = max_i (|C| x)_i / x_i, x the row sums of |C| plus delta their max; returns s_0 sqrt(s_1 sqrt(... sqrt(w))), +inf for a non-finite entry (kernel_functions.gelfand_bound).
/// Ref: Horn, Johnson 2012, https://doi.org/10.1017/CBO9781139020411
template <typename J, typename D, typename... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real gelfand_bound(J j_, D delta_, Ts... a_flat_args) noexcept
{
    constexpr int M = sizeof...(Ts);
    constexpr int N = zoomy_eig::isqrt_ct(M);
    static_assert(N * N == M, "gelfand_bound(): flat argument count is not a perfect square");

    double C[M] = { static_cast<double>(a_flat_args)... };
    double T[M], x[N];
    for (int k = 0; k < M; ++k)
        if (!zoomy_eig::dfinite(C[k])) return ZOOMY_EIG_INF;
    const int j = static_cast<int>(j_) < 64 ? static_cast<int>(j_) : 64;
    double sc[64];
    for (int it = 0; it < j; ++it)
    {
        double s = 0.0;
        for (int i = 0; i < N; ++i)
        {
            double t = 0.0;
            for (int k = 0; k < N; ++k) t += zoomy_eig::dabs(C[i*N + k]);
            s = t > s ? t : s;
        }
        s = s > 0.0 ? s : 1.0;
        for (int k = 0; k < M; ++k) C[k] /= s;
        sc[it] = s;
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
            {
                double t = 0.0;
                for (int l = 0; l < N; ++l) t += C[i*N + l]*C[l*N + k];
                T[i*N + k] = t;
            }
        for (int k = 0; k < M; ++k) C[k] = T[k];
    }
    double xmax = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double t = 0.0;
        for (int k = 0; k < N; ++k) t += zoomy_eig::dabs(C[i*N + k]);
        x[i] = t;
        xmax = t > xmax ? t : xmax;
    }
    for (int i = 0; i < N; ++i) x[i] += static_cast<double>(delta_)*xmax;   // > 0: w continuous in A
    double w = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double y = 0.0;
        for (int k = 0; k < N; ++k) y += zoomy_eig::dabs(C[i*N + k])*x[k];
        const double q = x[i] > 0.0 ? y/x[i] : 0.0;
        w = q > w ? q : w;
    }
    double t = w;
    for (int it = j - 1; it >= 0; --it) t = sc[it]*std::sqrt(t);
    return static_cast<amrex::Real>(t);
}

/// idx-th entry of [lambda | R | L = R^-1] for the row-major n x n A_flat (R's columns the right eigenvectors), as |A| = R|Lambda|L for the Roe flux; +inf if no real eigenbasis exists.
template <typename I, typename... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real eigensystem(I idx_, Ts... a_flat_args) noexcept
{
    constexpr int M = sizeof...(Ts);
    constexpr int N = zoomy_eig::isqrt_ct(M);
    static_assert(N * N == M, "eigensystem(): flat argument count is not a perfect square");
    static_assert(N >= 1 && N <= 16, "eigensystem(): unsupported system size");

    const double A[M] = { static_cast<double>(a_flat_args)... };
    const int idx = static_cast<int>(idx_);
    if (idx < 0 || idx >= N + 2 * N * N) return ZOOMY_EIG_INF;

    // LAZY SPLIT on idx -- deliberately mirrors the canonical numpy kernel
    // (zoomy_core/fvm/userfunctions.py and the REQ-168 carry-patch
    // `_eigensystem_batched`), which splits for exactly the same reason:
    //
    //   idx <  n : an EIGENVALUE.  Route to `values_core`, which never forms R
    //              and therefore never asks whether R is invertible.  This is
    //              load-bearing, not an optimisation: the CFL wave speed reads
    //              ONLY this block, and VAM's Jacobian is genuinely DEFECTIVE
    //              wherever the moment is 0 -- at rest or in uniform flow,
    //              i.e. essentially every cell at t = 0.  Gating lambda on the
    //              invertibility of R made `local_max_abs_eigenvalue` return
    //              +inf everywhere and the solver sat at dt = 0 forever.  The
    //              spectrum itself is perfectly well defined there (measured
    //              192/192 OK on the emitted VAM L1 Jacobians).
    //   idx >= n : R or L.  Only a Roe flux asks.  Full decomposition, and the
    //              refusal contract applies in full.
    if (idx < N) {
        double out[N];
        const int st = zoomy_eig::values_core<N>(A, out);
        ZOOMY_EIG_TRAP(st);
        return static_cast<amrex::Real>(out[idx]);
    }

    double out[N + 2 * N * N];
    const int st = zoomy_eig::system_core<N>(A, out);
    ZOOMY_EIG_TRAP(st);
    return static_cast<amrex::Real>(out[idx]);
}

/// idx-th entry of A^-1 b by Gaussian elimination with partial pivoting, for the point-implicit source (A = I - dt J, b = S); a singular A gives +inf.
template <typename I, typename... Ts>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real solve(I idx_, Ts... args) noexcept
{
    constexpr int M = sizeof...(Ts);
    constexpr int N = zoomy_eig::isqrt_ct(M + 1) - 1;   // M = N^2 + N
    static_assert(N * N + N == M, "solve(): argument count is not n^2 + n");

    const double v[M] = { static_cast<double>(args)... };
    double A[N * N], b[N];
    for (int k = 0; k < N * N; ++k) A[k] = v[k];
    for (int i = 0; i < N; ++i) b[i] = v[N * N + i];

    bool ok = true;
    for (int k = 0; k < M; ++k) if (!zoomy_eig::dfinite(v[k])) ok = false;

    double anorm = 0.0;
    for (int i = 0; i < N && ok; ++i) {
        double rs = 0.0;
        for (int j = 0; j < N; ++j) rs += zoomy_eig::dabs(A[i * N + j]);
        if (rs > anorm) anorm = rs;
    }
    const double pivtol = 8.0 * N * zoomy_eig::EIG_EPS * anorm;

    for (int c = 0; c < N && ok; ++c) {
        int piv = c; double best = zoomy_eig::dabs(A[c * N + c]);
        for (int i = c + 1; i < N; ++i) {
            const double t = zoomy_eig::dabs(A[i * N + c]);
            if (t > best) { best = t; piv = i; }
        }
        if (best <= pivtol) { ok = false; break; }
        if (piv != c) {
            for (int j = 0; j < N; ++j) { double t = A[c * N + j]; A[c * N + j] = A[piv * N + j]; A[piv * N + j] = t; }
            double t = b[c]; b[c] = b[piv]; b[piv] = t;
        }
        for (int i = c + 1; i < N; ++i) {
            const double f = A[i * N + c] / A[c * N + c];
            if (f == 0.0) continue;
            for (int j = c; j < N; ++j) A[i * N + j] -= f * A[c * N + j];
            b[i] -= f * b[c];
        }
    }
    const int idx = static_cast<int>(idx_);
    if (!ok || idx < 0 || idx >= N) return ZOOMY_EIG_INF;
    for (int i = N - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < N; ++j) s -= A[i * N + j] * b[j];
        b[i] = s / A[i * N + i];
    }
    return static_cast<amrex::Real>(b[idx]);
}

/// Stub returning 0: never emitted on amrex, where the driver fills the derivative aux by finite differences.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T compute_derivative(T) noexcept { return T(0); }

}  // namespace amrex_user
