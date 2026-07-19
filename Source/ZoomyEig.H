#pragma once
// ---------------------------------------------------------------------------
// Device-callable dense eigen kernels for the zoomy_amrex backend.
//
// Realises the zoomy_core opaque kernels `eigenvalues` and `eigensystem`
// (zoomy_core/model/kernel_functions.py) for a GENERAL real non-symmetric
// n x n matrix, entirely on the stack, with no allocation, no exceptions, no
// host-only library, and a hard iteration budget -- so the same code compiles
// and runs inside an AMREX_GPU_DEVICE lambda.
//
// Pipeline (the LAPACK dgeev path, minus the complex arithmetic):
//     balance (Parlett-Reinsch, radix 2, EXACT)      O(n^2) per sweep
//  -> Householder reduction to upper Hessenberg      (10/3) n^3
//  -> Francis double-shift QR -> real Schur form     ~10 n^3
//  -> [vectors] back-substitution on the quasi-triangular factor + back-transform
//  -> [vectors] Gauss-Jordan inversion of R with partial pivoting  2 n^3
//
// REFUSAL CONTRACT (see the long note in UserFunctions.H): every failure mode
// returns lambda = +inf and R = L = I. The kernel NEVER returns a plausible
// wrong number.
// ---------------------------------------------------------------------------

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>   // AMREX_FORCE_INLINE / AMREX_NO_INLINE
                               // (GpuQualifiers.H does NOT pull this in)

// ONE out-of-line body shared by every call site => bitwise-identical results
// for bitwise-identical inputs, which is what makes the printer's n + 2n^2
// separate component reads land on ONE eigenbasis without a thread_local cache
// (illegal in device code).
//
// MEASURED: AMREX_NO_INLINE is __attribute__((noinline)), which nvcc applies to
// the HOST compiler only -- device code was still inlined and specialised per
// call site, and 74 of the 78 components of one face then disagreed bitwise.
// CUDA's own __noinline__ is required for device code.
#if defined(__CUDACC__)
#  define ZOOMY_EIG_NOINLINE __noinline__
#else
#  define ZOOMY_EIG_NOINLINE AMREX_NO_INLINE
#endif

#define ZOOMY_EIG_INF (__builtin_huge_val())

namespace amrex_user {
namespace zoomy_eig {

enum Status : int {
    OK          = 0,
    NONFINITE   = 1,   // an input entry was NaN/Inf (MOOD candidate state)
    NOCONV      = 2,   // QR iteration budget exhausted
    COMPLEX_EIG = 3,   // 2x2 Schur block with complex pair -> not hyperbolic
    SINGULAR_R  = 4    // eigenbasis numerically singular (defective / degenerate)
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double dabs(double x) noexcept { return x < 0.0 ? -x : x; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool dfinite(double x) noexcept {
    // Hand-rolled: std::isfinite is not guaranteed device-callable everywhere,
    // and this form survives -ffinite-math-only being absent (AMReX default).
    return (x == x) && (x < ZOOMY_EIG_INF) && (x > -ZOOMY_EIG_INF);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double dsqrt(double x) noexcept { return x > 0.0 ? __builtin_sqrt(x) : 0.0; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double dmax(double a, double b) noexcept { return a > b ? a : b; }

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double dmin(double a, double b) noexcept { return a < b ? a : b; }

// |a + i b| without the intermediate overflow of sqrt(a*a + b*b).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double dhypot(double a, double b) noexcept
{
    a = dabs(a); b = dabs(b);
    const double hi = dmax(a, b), lo = dmin(a, b);
    if (hi == 0.0) return 0.0;
    const double t = lo / hi;
    return hi * dsqrt(1.0 + t * t);
}

constexpr double EIG_EPS    = 2.220446049250313e-16;
constexpr double EIG_SAFMIN = 2.2250738585072014e-308;

// ---------------------------------------------------------------------------
// Parlett-Reinsch balancing, scaling only (no permutation stage).
// Scales by powers of two, so D and D^-1 are EXACT in binary floating point and
// the back-transform R = D * X introduces no error at all.
// A <- D^-1 A D  (in place, row major),  scale[i] = D_ii.
// ---------------------------------------------------------------------------
template <int N>
AMREX_GPU_HOST_DEVICE
void balance(double* A, double* scale) noexcept
{
    for (int i = 0; i < N; ++i) scale[i] = 1.0;
    const double radix = 2.0;
    bool done = false;
    for (int sweep = 0; sweep < 64 && !done; ++sweep) {
        done = true;
        for (int i = 0; i < N; ++i) {
            double c = 0.0, r = 0.0;
            for (int j = 0; j < N; ++j) if (j != i) {
                c += dabs(A[j * N + i]);
                r += dabs(A[i * N + j]);
            }
            if (c == 0.0 || r == 0.0) continue;   // zero row/col (e.g. the VAM
                                                   // bathymetry row) -- skip
            double g = r / radix, f = 1.0;
            const double s = c + r;
            int guard = 0;
            while (c < g && ++guard < 1024) { f *= radix; c *= radix * radix; }
            g = r * radix;
            guard = 0;
            while (c > g && ++guard < 1024) { f /= radix; c /= radix * radix; }
            if ((c + r) / f < 0.95 * s) {
                done = false;
                const double ginv = 1.0 / f;
                scale[i] *= f;
                for (int j = 0; j < N; ++j) A[i * N + j] *= ginv;
                for (int j = 0; j < N; ++j) A[j * N + i] *= f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Householder reduction to upper Hessenberg.  H overwritten in place; when
// WantVec, V accumulates the orthogonal similarity (Schur vectors so far).
// ---------------------------------------------------------------------------
template <int N, bool WantVec>
AMREX_GPU_HOST_DEVICE
void hessenberg(double* H, double* V) noexcept
{
    double ort[N > 0 ? N : 1];
    for (int i = 0; i < N; ++i) ort[i] = 0.0;

    for (int m = 1; m <= N - 2; ++m) {
        double scale = 0.0;
        for (int i = m; i < N; ++i) scale += dabs(H[i * N + (m - 1)]);
        if (scale == 0.0) continue;

        double h = 0.0;
        for (int i = N - 1; i >= m; --i) {
            ort[i] = H[i * N + (m - 1)] / scale;
            h += ort[i] * ort[i];
        }
        double g = dsqrt(h);
        if (ort[m] > 0.0) g = -g;
        h -= ort[m] * g;
        ort[m] -= g;

        for (int j = m; j < N; ++j) {
            double f = 0.0;
            for (int i = N - 1; i >= m; --i) f += ort[i] * H[i * N + j];
            f /= h;
            for (int i = m; i < N; ++i) H[i * N + j] -= f * ort[i];
        }
        for (int i = 0; i < N; ++i) {
            double f = 0.0;
            for (int j = N - 1; j >= m; --j) f += ort[j] * H[i * N + j];
            f /= h;
            for (int j = m; j < N; ++j) H[i * N + j] -= f * ort[j];
        }
        ort[m] *= scale;
        H[m * N + (m - 1)] = scale * g;
    }

    if constexpr (WantVec) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) V[i * N + j] = (i == j) ? 1.0 : 0.0;
        for (int m = N - 2; m >= 1; --m) {
            if (H[m * N + (m - 1)] == 0.0) continue;
            for (int i = m + 1; i < N; ++i) ort[i] = H[i * N + (m - 1)];
            for (int j = m; j < N; ++j) {
                double g = 0.0;
                for (int i = m; i < N; ++i) g += ort[i] * V[i * N + j];
                g = (g / ort[m]) / H[m * N + (m - 1)];   // two divisions: no overflow
                for (int i = m; i < N; ++i) V[i * N + j] += g * ort[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Francis double-shift QR on the Hessenberg form -> real Schur form.
// d[] / e[] receive the real / imaginary parts of the spectrum.
// Returns OK or NOCONV.  Hard global iteration budget: no unbounded device loop.
// ---------------------------------------------------------------------------
template <int N, bool WantVec>
AMREX_GPU_HOST_DEVICE
int schur(double* H, double* V, double* d, double* e) noexcept
{
    const int nn = N;
    int nl = nn - 1;
    const int low = 0, high = nn - 1;
    const double eps = EIG_EPS;
    double exshift = 0.0;
    double p = 0, q = 0, r = 0, s = 0, z = 0, w, x, y;

    double norm = 0.0;
    for (int i = 0; i < nn; ++i)
        for (int j = (i - 1 > 0 ? i - 1 : 0); j < nn; ++j) norm += dabs(H[i * N + j]);

    int iter = 0;
    int budget = 0;
    const int BUDGET_MAX = 60 * N + 120;   // LAPACK uses 30 per eigenvalue

    while (nl >= low) {
        if (++budget > BUDGET_MAX) return NOCONV;

        // ---- small-subdiagonal / deflation test ----------------------------
        // NOTE (measured, do not "restore" the Ahues-Tisseur test without
        // re-measuring).  The design spec called for replacing this with
        // LAPACK dlahqr's Ahues-Tisseur criterion.  Both readings of that graft
        // were implemented and measured here, and BOTH are worse:
        //
        //  * as an OR-alternative ("deflate if plain OR A-T"): deflates on a
        //    subdiagonal that is NOT small, silently destroying the Schur
        //    relation A V = V T.  Eigenvalues stayed correct to 1e-12 while the
        //    leading eigenvector was wrong by O(1) -- ||R diag(lam) L - A||
        //    = 1.2e-1 on the VAM-shaped 6x6 in tests/test_userfunctions.cpp,
        //    which is what caught it.  A-T is a GUARD that makes deflation
        //    RARER, not a second way to deflate.
        //  * as LAPACK's actual nesting (plain test AND THEN A-T): correct, but
        //    it stalls catastrophically here -- 16610 / 30000 NOCONV at N=6,
        //    11116 / 30000 at N=4, versus 0 / 30000 and 2 / 30000 for the plain
        //    test below.  A-T refuses to deflate whenever the two diagonal
        //    entries are close (bb -> 0 drives the threshold to 0), which is
        //    exactly the clustered spectrum VAM sits in.
        //
        // The stall the graft was meant to cure is already cured by '<=' rather
        // than '<' (JAMA/EISPACK) plus the every-10 exceptional shift below:
        // measured 0 non-convergence in 30000 random 6x6 and 192/192 OK on the
        // real emitted VAM L1 Jacobians.
        const double smlnum = EIG_SAFMIN * (double(N) / eps);
        int l = nl;
        while (l > low) {
            const double hsub = dabs(H[l * N + (l - 1)]);
            // (a) unconditional underflow deflation (LAPACK dlahqr's first test).
            if (hsub <= smlnum) { H[l * N + (l - 1)] = 0.0; break; }
            // (b) the classic relative test.  '<=', not '<': with s == 0 AND
            // norm == 0 -- the all-zero Jacobian of a fully dry patch -- '<'
            // never deflates and the iteration burns its whole budget.
            s = dabs(H[(l - 1) * N + (l - 1)]) + dabs(H[l * N + l]);
            if (s == 0.0) s = norm;
            if (hsub <= eps * s) { H[l * N + (l - 1)] = 0.0; break; }
            --l;
        }

        if (l == nl) {                       // ---- one real root
            H[nl * N + nl] += exshift;
            d[nl] = H[nl * N + nl];
            e[nl] = 0.0;
            --nl; iter = 0;
        } else if (l == nl - 1) {            // ---- 2x2 block
            w = H[nl * N + (nl - 1)] * H[(nl - 1) * N + nl];
            p = (H[(nl - 1) * N + (nl - 1)] - H[nl * N + nl]) / 2.0;
            q = p * p + w;
            z = dsqrt(dabs(q));
            H[nl * N + nl] += exshift;
            H[(nl - 1) * N + (nl - 1)] += exshift;
            x = H[nl * N + nl];
            if (q >= 0.0) {                  // real pair
                z = (p >= 0.0) ? p + z : p - z;
                d[nl - 1] = x + z;
                d[nl] = d[nl - 1];
                if (z != 0.0) d[nl] = x - w / z;
                e[nl - 1] = 0.0; e[nl] = 0.0;
                x = H[nl * N + (nl - 1)];
                s = dabs(x) + dabs(z);
                p = x / s; q = z / s;
                r = dsqrt(p * p + q * q);
                p /= r; q /= r;
                for (int j = nl - 1; j < nn; ++j) {
                    z = H[(nl - 1) * N + j];
                    H[(nl - 1) * N + j] = q * z + p * H[nl * N + j];
                    H[nl * N + j] = q * H[nl * N + j] - p * z;
                }
                for (int i = 0; i <= nl; ++i) {
                    z = H[i * N + (nl - 1)];
                    H[i * N + (nl - 1)] = q * z + p * H[i * N + nl];
                    H[i * N + nl] = q * H[i * N + nl] - p * z;
                }
                if constexpr (WantVec) {
                    for (int i = low; i <= high; ++i) {
                        z = V[i * N + (nl - 1)];
                        V[i * N + (nl - 1)] = q * z + p * V[i * N + nl];
                        V[i * N + nl] = q * V[i * N + nl] - p * z;
                    }
                }
            } else {                          // complex conjugate pair
                d[nl - 1] = x + p; d[nl] = x + p;
                e[nl - 1] = z;     e[nl] = -z;
            }
            nl -= 2; iter = 0;
        } else {                             // ---- Francis double shift
            // A Francis step needs an active block of size >= 3; for N < 3
            // every deflation lands in the two branches above, so this arm is
            // dead.  `if constexpr` is not cosmetic: without it gcc still
            // instantiates the H[(m+2)*N + ...] indexing for N = 2 and warns
            // -Warray-bounds on a genuinely unreachable path.
            if constexpr (N < 3) { return NOCONV; } else {
            x = H[nl * N + nl];
            y = 0.0; w = 0.0;
            if (l < nl) {
                y = H[(nl - 1) * N + (nl - 1)];
                w = H[nl * N + (nl - 1)] * H[(nl - 1) * N + nl];
            }
            // Exceptional shift EVERY 10 iterations (LAPACK dlahqr's KEXSH),
            // not only at 10 and 20 as EISPACK/JAMA do.  Measured: with the
            // EISPACK rule a 6x6 wet/dry Jacobian with a 3-fold near-degenerate
            // cluster stagnates from iteration 21 on -- the QR step becomes an
            // exact no-op and nothing ever deflates again.
            if (iter > 0 && iter % 10 == 0) {  // exceptional shift
                exshift += x;
                for (int i = low; i <= nl; ++i) H[i * N + i] -= x;
                s = dabs(H[nl * N + (nl - 1)]) + dabs(H[(nl - 1) * N + (nl - 2)]);
                x = y = 0.75 * s;
                w = -0.4375 * s * s;
            }
            ++iter;
            // Budget must match the documented 60N+120. It was hardcoded to 100,
            // which is BELOW that for every N we emit (N=4 -> 360, N=6 -> 480,
            // N=8 -> 600), so the solver refused (NOCONV -> +inf) on 10-50% of
            // CLUSTERED / near-degenerate real spectra -- precisely the regime VAM
            // sits in.  Random test matrices never exposed it: they have
            // well-separated eigenvalues and converge in far fewer than 100 sweeps.
            // `iter` resets to 0 on every deflation, so this is a per-eigenvalue
            // budget, not a whole-matrix one.
            if (iter > 60 * N + 120) return NOCONV;

            s = (y - x) / 2.0;
            s = s * s + w;
            if (s > 0.0) {
                s = dsqrt(s);
                if (y < x) s = -s;
                s = x - w / ((y - x) / 2.0 + s);
                for (int i = low; i <= nl; ++i) H[i * N + i] -= s;
                exshift += s;
                x = y = w = 0.964;
            }

            int m = nl - 2;
            while (m >= l) {
                z = H[m * N + m];
                r = x - z;
                s = y - z;
                p = (r * s - w) / H[(m + 1) * N + m] + H[m * N + (m + 1)];
                q = H[(m + 1) * N + (m + 1)] - z - r - s;
                r = H[(m + 2) * N + (m + 1)];
                s = dabs(p) + dabs(q) + dabs(r);
                p /= s; q /= s; r /= s;
                if (m == l) break;
                if (dabs(H[m * N + (m - 1)]) * (dabs(q) + dabs(r)) <
                    eps * (dabs(p) * (dabs(H[(m - 1) * N + (m - 1)]) + dabs(z) +
                                      dabs(H[(m + 1) * N + (m + 1)])))) break;
                --m;
            }
            for (int i = m + 2; i <= nl; ++i) {
                H[i * N + (i - 2)] = 0.0;
                if (i > m + 2) H[i * N + (i - 3)] = 0.0;
            }

            for (int k = m; k <= nl - 1; ++k) {
                const bool notlast = (k != nl - 1);
                if (k != m) {
                    p = H[k * N + (k - 1)];
                    q = H[(k + 1) * N + (k - 1)];
                    r = notlast ? H[(k + 2) * N + (k - 1)] : 0.0;
                    x = dabs(p) + dabs(q) + dabs(r);
                    if (x == 0.0) continue;
                    p /= x; q /= x; r /= x;
                }
                s = dsqrt(p * p + q * q + r * r);
                if (p < 0.0) s = -s;
                if (s == 0.0) continue;
                if (k != m)        H[k * N + (k - 1)] = -s * x;
                else if (l != m)   H[k * N + (k - 1)] = -H[k * N + (k - 1)];
                p += s;
                x = p / s; y = q / s; z = r / s;
                q /= p; r /= p;
                for (int j = k; j < nn; ++j) {
                    p = H[k * N + j] + q * H[(k + 1) * N + j];
                    if (notlast) { p += r * H[(k + 2) * N + j]; H[(k + 2) * N + j] -= p * z; }
                    H[k * N + j] -= p * x;
                    H[(k + 1) * N + j] -= p * y;
                }
                const int ihi = (nl < k + 3) ? nl : k + 3;
                for (int i = 0; i <= ihi; ++i) {
                    p = x * H[i * N + k] + y * H[i * N + (k + 1)];
                    if (notlast) { p += z * H[i * N + (k + 2)]; H[i * N + (k + 2)] -= p * r; }
                    H[i * N + k] -= p;
                    H[i * N + (k + 1)] -= p * q;
                }
                if constexpr (WantVec) {
                    for (int i = low; i <= high; ++i) {
                        p = x * V[i * N + k] + y * V[i * N + (k + 1)];
                        if (notlast) { p += z * V[i * N + (k + 2)]; V[i * N + (k + 2)] -= p * r; }
                        V[i * N + k] -= p;
                        V[i * N + (k + 1)] -= p * q;
                    }
                }
            }
            }
        }
    }
    return OK;
}

// ---------------------------------------------------------------------------
// Right eigenvectors from the (purely upper-triangular, i.e. real-spectrum)
// Schur factor by back-substitution, then back-transform with the Schur
// vectors.  Only the REAL branch of dtrevc is implemented -- the complex branch
// is unreachable because the caller refuses a complex spectrum outright.
// On exit V's columns are the right eigenvectors of the BALANCED matrix.
// ---------------------------------------------------------------------------
template <int N>
AMREX_GPU_HOST_DEVICE
void eigvec_backsub(double* H, double* V, const double* d) noexcept
{
    const double eps = EIG_EPS;
    double norm = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = i; j < N; ++j) norm += dabs(H[i * N + j]);
    if (norm == 0.0) norm = 1.0;

    for (int nl = N - 1; nl >= 0; --nl) {
        const double p = d[nl];
        int l = nl;
        H[nl * N + nl] = 1.0;
        for (int i = nl - 1; i >= 0; --i) {
            const double w = H[i * N + i] - p;
            double r = 0.0;
            for (int j = l; j <= nl; ++j) r += H[i * N + j] * H[j * N + nl];
            l = i;
            H[i * N + nl] = (w != 0.0) ? -r / w : -r / (eps * norm);
            const double t = dabs(H[i * N + nl]);
            if ((eps * t) * t > 1.0)
                for (int j = i; j <= nl; ++j) H[j * N + nl] /= t;
        }
    }

    for (int j = N - 1; j >= 0; --j) {
        double col[N > 0 ? N : 1];
        for (int i = 0; i < N; ++i) {
            double z = 0.0;
            const int kmax = (j < N - 1) ? j : N - 1;
            for (int k = 0; k <= kmax; ++k) z += V[i * N + k] * H[k * N + j];
            col[i] = z;
        }
        for (int i = 0; i < N; ++i) V[i * N + j] = col[i];
    }
}

// ---------------------------------------------------------------------------
// Gauss-Jordan inversion with partial pivoting + an explicit rank guard.
// Returns SINGULAR_R when the pivot collapses relative to the matrix norm --
// the wet/dry defective case.  NO pseudo-inverse fallback: a pinv is not the
// inverse of any eigenbasis, so R|Lambda|L would silently stop being |A|.
// ---------------------------------------------------------------------------
template <int N>
AMREX_GPU_HOST_DEVICE
int invert(const double* R, double* L) noexcept
{
    double M[N * N];
    for (int i = 0; i < N * N; ++i) M[i] = R[i];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) L[i * N + j] = (i == j) ? 1.0 : 0.0;

    double anorm = 0.0;
    for (int i = 0; i < N; ++i) {
        double rs = 0.0;
        for (int j = 0; j < N; ++j) rs += dabs(M[i * N + j]);
        if (rs > anorm) anorm = rs;
    }
    if (anorm == 0.0) return SINGULAR_R;
    const double pivtol = 8.0 * N * EIG_EPS * anorm;

    for (int c = 0; c < N; ++c) {
        int piv = c;
        double best = dabs(M[c * N + c]);
        for (int i = c + 1; i < N; ++i) {
            const double v = dabs(M[i * N + c]);
            if (v > best) { best = v; piv = i; }
        }
        if (best <= pivtol) return SINGULAR_R;
        if (piv != c) {
            for (int j = 0; j < N; ++j) {
                double tmp = M[c * N + j]; M[c * N + j] = M[piv * N + j]; M[piv * N + j] = tmp;
                tmp = L[c * N + j]; L[c * N + j] = L[piv * N + j]; L[piv * N + j] = tmp;
            }
        }
        const double inv = 1.0 / M[c * N + c];
        for (int j = 0; j < N; ++j) { M[c * N + j] *= inv; L[c * N + j] *= inv; }
        for (int i = 0; i < N; ++i) {
            if (i == c) continue;
            const double f = M[i * N + c];
            if (f == 0.0) continue;
            for (int j = 0; j < N; ++j) {
                M[i * N + j] -= f * M[c * N + j];
                L[i * N + j] -= f * L[c * N + j];
            }
        }
    }
    return OK;
}

// ---------------------------------------------------------------------------
// Worker 1: eigenvalues only.  out[0..N-1] = spectrum (ascending).
// Stack: (N*N + 3N) doubles  (N=6 -> 432 B).
// ---------------------------------------------------------------------------
template <int N>
AMREX_GPU_HOST_DEVICE ZOOMY_EIG_NOINLINE
int values_core(const double* A_in, double* out) noexcept
{
    for (int i = 0; i < N; ++i) out[i] = ZOOMY_EIG_INF;
    for (int k = 0; k < N * N; ++k) if (!dfinite(A_in[k])) return NONFINITE;

    double H[N * N], d[N], e[N], scale[N];
    for (int k = 0; k < N * N; ++k) H[k] = A_in[k];
    balance<N>(H, scale);
    hessenberg<N, false>(H, nullptr);
    const int st = schur<N, false>(H, nullptr, d, e);
    if (st != OK) return st;

    // COMPLEX PAIR: unlike `system_core`, do NOT refuse.  The only consumer of
    // this kernel is max_i |lambda_i| -> the CFL bound, and the spectral radius
    // rho(A) = max |lambda| is perfectly well defined on a complex spectrum.
    // Refusing with +inf here would freeze dt on every state that is merely
    // weakly hyperbolic (measured: 1.7% of VAM level-2 states), which is a
    // solver stall, not a safety property.  We return the SIGNED modulus
    // sign(Re) * |lambda| so the sign information that a downstream
    // left/right-going split might use survives; |.| of it is exact rho.
    // `eigensystem` still refuses -- R|Lambda|L has no real-arithmetic meaning
    // there (see the divergence note in UserFunctions.H).
    for (int i = 0; i < N; ++i) {
        if (e[i] != 0.0) {
            const double m = dhypot(d[i], e[i]);
            d[i] = (d[i] < 0.0) ? -m : m;
        }
    }
    for (int i = 0; i < N; ++i) if (!dfinite(d[i])) return NOCONV;

    for (int i = 0; i < N; ++i) {           // selection sort, ascending
        int m = i;
        for (int j = i + 1; j < N; ++j) if (d[j] < d[m]) m = j;
        const double t = d[i]; d[i] = d[m]; d[m] = t;
    }
    for (int i = 0; i < N; ++i) out[i] = d[i];
    return OK;
}

// ---------------------------------------------------------------------------
// Worker 2: full eigensystem.  out = [ lambda(N) | R(N*N) | L(N*N) ], row major.
// Stack: (3*N*N + 3N + ...) doubles  (N=6 -> ~1.4 kB).
// ---------------------------------------------------------------------------
template <int N>
AMREX_GPU_HOST_DEVICE ZOOMY_EIG_NOINLINE
int system_core(const double* A_in, double* out) noexcept
{
    for (int i = 0; i < N; ++i) out[i] = ZOOMY_EIG_INF;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            out[N + i * N + j]         = (i == j) ? 1.0 : 0.0;
            out[N + N * N + i * N + j] = (i == j) ? 1.0 : 0.0;
        }
    for (int k = 0; k < N * N; ++k) if (!dfinite(A_in[k])) return NONFINITE;

    double H[N * N], V[N * N], d[N], e[N], scale[N];
    for (int k = 0; k < N * N; ++k) H[k] = A_in[k];
    balance<N>(H, scale);
    hessenberg<N, true>(H, V);
    const int st = schur<N, true>(H, V, d, e);
    if (st != OK) return st;

    for (int i = 0; i < N; ++i) if (e[i] != 0.0) return COMPLEX_EIG;
    for (int i = 0; i < N; ++i) if (!dfinite(d[i])) return NOCONV;

    eigvec_backsub<N>(H, V, d);

    // undo the (exact, power-of-two) balancing:  x = D x_bal
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) V[i * N + j] *= scale[i];

    // normalise each column to unit 2-norm, sign-fix on the largest component
    for (int j = 0; j < N; ++j) {
        double nrm = 0.0; int imax = 0; double vmax = -1.0;
        for (int i = 0; i < N; ++i) {
            nrm += V[i * N + j] * V[i * N + j];
            const double a = dabs(V[i * N + j]);
            if (a > vmax) { vmax = a; imax = i; }
        }
        nrm = dsqrt(nrm);
        if (nrm == 0.0) return SINGULAR_R;
        double sgn = (V[imax * N + j] < 0.0) ? -1.0 : 1.0;
        const double f = sgn / nrm;
        for (int i = 0; i < N; ++i) V[i * N + j] *= f;
    }

    // sort ascending by eigenvalue, permuting the columns of R with it
    int perm[N];
    for (int i = 0; i < N; ++i) perm[i] = i;
    for (int i = 0; i < N; ++i) {
        int m = i;
        for (int j = i + 1; j < N; ++j) if (d[perm[j]] < d[perm[m]]) m = j;
        const int t = perm[i]; perm[i] = perm[m]; perm[m] = t;
    }

    double Rm[N * N], Lm[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) Rm[i * N + j] = V[i * N + perm[j]];

    const int si = invert<N>(Rm, Lm);
    if (si != OK) return si;

    // Verify the contract in-kernel:  || L R - I ||_inf  must be at round-off.
    // This is the cheap (2 N^3 flops) guard that catches a near-singular basis
    // the pivot test alone lets through.
    double ierr = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k) s += Lm[i * N + k] * Rm[k * N + j];
            const double e2 = dabs(s - (i == j ? 1.0 : 0.0));
            if (e2 > ierr) ierr = e2;
        }
    if (!(ierr < 1.0e-8)) return SINGULAR_R;

    for (int i = 0; i < N; ++i) out[i] = d[perm[i]];
    for (int k = 0; k < N * N; ++k) out[N + k] = Rm[k];
    for (int k = 0; k < N * N; ++k) out[N + N * N + k] = Lm[k];
    return OK;
}

// compile-time integer square root, for inferring N from the variadic arity
constexpr int isqrt_ct(int m, int i = 0) { return (i * i >= m) ? i : isqrt_ct(m, i + 1); }

}  // namespace zoomy_eig
}  // namespace amrex_user
