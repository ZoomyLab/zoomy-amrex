#pragma once
// AMReX implementations of the opaque "kernel" placeholder functions
// (zoomy_core.model.kernel_functions).  Unlike foam/dmplex (Eigen, host-only)
// and firedrake (LAPACK, host-only), these run on the DEVICE: the Chorin
// predictor calls them from inside an AMREX_GPU_DEVICE lambda.
//
// ── REFUSAL CONTRACT ──────────────────────────────────────────────────────
// There is no exception channel on device and no return code in the printer's
// call shape, so every failure mode is encoded in the VALUE:
//
//     lambda = +inf,  R = L = I
//
// for ALL of: non-finite input (MOOD candidate at a dry front, by design --
// REQ-168 addendum 1), QR non-convergence within the iteration budget, a
// COMPLEX conjugate pair (loss of hyperbolicity), and a numerically singular
// eigenbasis (defective Jacobian at wet/dry).  +inf makes the CFL bound clamp
// dt and MOOD flag the cell; R = L = I (not 0) keeps |A| = R|Lambda|L equal to
// inf*I instead of 0*inf = NaN, which would poison the whole flux row.
//
// ONE EXCEPTION: `eigenvalues` on a complex pair returns the SIGNED MODULUS
// sign(Re)*|lambda|, NOT +inf.  Its only consumer is max_i |lambda_i| -> the CFL
// bound, and the spectral radius is well defined on a complex spectrum;
// refusing there would freeze dt on every merely-weakly-hyperbolic state.
// `eigensystem` still refuses on complex, because R|Lambda|L has no
// real-arithmetic meaning there.
//
// ── RECORDED DIVERGENCES FROM foam / dmplex / numpy ───────────────────────
// (1) COMPLEX SPECTRUM.  foam/dmplex/numpy take real(V), which makes the two
//     conjugate columns identical, R exactly singular, and then fall back to a
//     Moore-Penrose pseudo-inverse for L.  The result is a finite,
//     plausible-looking |A| that is R|Lambda|L for NO eigenbasis.  amrex
//     refuses instead: a believable wrong number is worse than a loud +inf.
//     Consequence: on a non-hyperbolic state amrex and foam are on DIFFERENT
//     FORMULAS, not merely different round-off.  Any cross-backend byte
//     comparison must FIRST assert max|Im(lambda)| == 0.
// (2) DEFECTIVE R.  foam/dmplex/numpy pinv; firedrake sets L = I; amrex
//     returns R = L = I for the R/L blocks while still returning the TRUE
//     spectrum for the lambda block (see the lazy split in `eigensystem`).
//
// ── CONSEQUENCE OF (2), AND WHY ROE MUST NOT BE WIRED TO VAM ──────────────
// On a defective basis a Roe consumer reads a finite lambda block and R = L = I
// and therefore forms |A| = diag(|lambda|) -- Rusanov-like, NOT R|Lambda|L for
// A.  There is no way to make one kernel serve both consumers honestly here:
// the CFL path needs a finite lambda (or dt is 0 forever) and the Roe path
// needs an eigenbasis that does not exist.  The resolution is upstream, not in
// this file: VAM(1) at rest or in uniform flow has algebraic multiplicity 3 and
// geometric multiplicity 2 at lambda = u (a real Jordan block, cond(R) ~ 7e14,
// numpy concurs), so |A| = R|Lambda|L is UNDEFINED for it.  RUSANOV IS THE
// CORRECT SCHEME FOR VAM; do not set `riemann: NonconservativeRoe` on it.
// Under !NDEBUG the assert() fires on every such face so this is not silent.
//
// Free, and never to be byte-compared: eigenvalue ORDER and eigenvector
// SIGN/SCALE.  |A| = R|Lambda|L is invariant under R -> R*D.
//
// Under a debug build the refusal also trips assert() so it is visible in a
// test run rather than only as a stalled dt.

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

// ── eigenvalues(idx, *A_flat) ─────────────────────────────────────────────
// idx-th eigenvalue of the row-major n x n A_flat, n = sqrt(arity-1).
// Emitted by `local_max_abs_eigenvalue` / `numerical_fluctuations` /
// `numerical_face` in NumericsPred.H (VAM Chorin predictor).  The consumer only
// ever forms max_i |lambda_i|, so eigenvalue ORDER is free; we return them
// sorted ascending anyway, for reproducible debugging.  A complex pair yields
// sign(Re)*|lambda| here (see the exception in the contract note above), so
// max|.| is the true spectral radius and dt stays finite.
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

// ── eigensystem(idx, *A_flat) ─────────────────────────────────────────────
// idx-th component of [ lambda(n) | R(n*n) row-major | L = R^-1 (n*n) ].
// R's COLUMNS are the right eigenvectors (A R = R Lambda); L is an EXPLICIT
// inverse, never a transpose -- A is non-symmetric.  Consumed by
// NonconservativeRoe as |A| = R |Lambda| L.
//
// All n + 2n^2 component reads of one face recompute the decomposition; there
// is no thread_local cache on device.  That is CORRECT rather than merely slow
// because `system_core` is deterministic and lives in exactly one out-of-line
// body (ZOOMY_EIG_NOINLINE), so bitwise-identical A_flat args give
// bitwise-identical R, |Lambda| and L at every call site -- the one eigenbasis
// the contract demands.  It is also ~(n + 2n^2)x redundant: see the note in the
// proposal about lowering ONE decomposition per face in the printer instead.
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

// ── solve(idx, *A_flat, *b) ───────────────────────────────────────────────
// idx-th component of A^-1 b for the point-implicit source linearisation
// (A = I - dt J, b = S).  Arity 1 + n^2 + n.  Gaussian elimination with partial
// pivoting; a singular A refuses with +inf, same contract as above.
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

// ── compute_derivative ─── OPEN ITEM, task 0027 — NOT part of this change ──
// Non-local spatial-derivative aux.  Left EXACTLY as it was found: still a
// stub returning 0.  It is deliberately out of scope here.
//
// Why it is currently harmless: on amrex this call is never emitted.
// `_chorin_meta` (to_amrex.py) lowers the derivative aux into
// `static constexpr int deriv_aux[N][4]` and the C++ driver fills it by finite
// differences (chorin_common.H::fill_deriv_aux, called from :194).  Verified 0
// emitted call sites across VAM L0/L1/L2 (Chorin), SWE and SME.  So the stub is
// dead code today, not a latent silent-wrong-answer -- but it is a stub, and
// task 0027 owns turning it into either a real kernel or a loud refusal.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T compute_derivative(T) noexcept { return T(0); }

}  // namespace amrex_user
