// Contract test for Source/UserFunctions.H  (amrex_user::eigenvalues /
// eigensystem / solve).  Mirrors zoomy_dmplex/test_userfunctions.cpp: it pins
// the INVARIANTS and nothing else.
//
// Eigenvalue ORDER and eigenvector SIGN/SCALE are explicitly free -- |A| =
// R|Lambda|L is invariant under R -> R*D -- so nothing here byte-compares a
// lambda, an R or an L against another backend.  What is pinned:
//
//   * || R diag(lambda) L - A ||_inf ~ 0     on accepted cases
//   * || L R - I ||_inf              ~ 0     on accepted cases
//   * max|lambda| equals the analytic u +/- sqrt(gh) for SWE
//   * non-finite input               => +inf, R = L = I
//   * complex spectrum               => eigenvalues gives rho(A); eigensystem's
//                                       R / L blocks refuse with I
//   * defective basis (dry h = 0)    => eigensystem's LAMBDA block is still
//                                       exact (the CFL path reads only that and
//                                       would otherwise sit at dt = 0 forever);
//                                       only the R / L blocks refuse
//   * zero matrix and dry h = 0      => TERMINATE (the '<' vs '<=' deflation bug:
//                                       with an all-zero Jacobian s == 0 AND
//                                       norm == 0, so '<' never deflates and the
//                                       QR iteration burns its whole budget)
//
// BUILD (NDEBUG is REQUIRED -- several cases exercise the refusal path on
// purpose, and a debug build deliberately trips assert() there):
//
//   g++ -std=c++17 -O2 -DNDEBUG -I<amrex>/Src/Base -I../Source
//       test_userfunctions.cpp -o test_userfunctions && ./test_userfunctions

#include "UserFunctions.H"

#include <cmath>
#include <cstdio>

static int g_fail = 0;

static void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL  %s\n", what); ++g_fail; }
    else     { std::printf("ok    %s\n", what); }
}

static bool close(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

// ---------------------------------------------------------------------------
// n = 2 : SWE 1-D quasilinear Jacobian.  A = [[0, 1], [gh - u^2, 2u]],
// analytic spectrum u -/+ sqrt(gh).
// ---------------------------------------------------------------------------
static void test_swe_spectrum()
{
    const double g = 9.81, h = 2.0, u = 1.5;
    const double c = std::sqrt(g * h);
    const double a10 = g * h - u * u, a11 = 2.0 * u;

    const double l0 = amrex_user::eigenvalues(0, 0.0, 1.0, a10, a11);
    const double l1 = amrex_user::eigenvalues(1, 0.0, 1.0, a10, a11);

    // returned ascending, but only the SET is contractual
    const double lo = std::fmin(l0, l1), hi = std::fmax(l0, l1);
    check(close(lo, u - c, 1e-12), "swe: lambda_- == u - sqrt(gh)");
    check(close(hi, u + c, 1e-12), "swe: lambda_+ == u + sqrt(gh)");
    check(close(std::fmax(std::fabs(l0), std::fabs(l1)), u + c, 1e-12),
          "swe: max|lambda| == u + sqrt(gh)  (the CFL consumer)");
}

// ---------------------------------------------------------------------------
// Generic invariant harness for a full eigensystem read.
// ---------------------------------------------------------------------------
template <int N, typename F>
static void check_invariants(const double* A, F read, const char* tag, double tol)
{
    double lam[N], R[N * N], L[N * N];
    for (int i = 0; i < N; ++i)            lam[i] = read(i);
    for (int k = 0; k < N * N; ++k)        R[k]   = read(N + k);
    for (int k = 0; k < N * N; ++k)        L[k]   = read(N + N * N + k);

    double anorm = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += std::fabs(A[i * N + j]);
        if (s > anorm) anorm = s;
    }
    if (anorm == 0.0) anorm = 1.0;

    double rA = 0.0, rI = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0.0, t = 0.0;
            for (int k = 0; k < N; ++k) {
                s += R[i * N + k] * lam[k] * L[k * N + j];
                t += L[i * N + k] * R[k * N + j];
            }
            const double ea = std::fabs(s - A[i * N + j]) / anorm;
            const double ei = std::fabs(t - (i == j ? 1.0 : 0.0));
            // `!(x <= acc)` and NOT `x > acc`: a refused decomposition gives
            // lambda = +inf with R = L = I, so R diag(lam) L contains 0 * inf =
            // NaN.  With `x > acc` every comparison against NaN is false, the
            // accumulator stays 0, and the test reports a PERFECT residual on a
            // decomposition that was never computed.  That is exactly what
            // happened here: the original n4 fixture turned out to have a
            // complex pair, the kernel correctly refused, and this harness
            // printed 0.000e+00 and passed.  Fail closed on NaN.
            if (!(ea <= rA)) rA = ea;
            if (!(ei <= rI)) rI = ei;
        }

    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: ||R diag(lam) L - A||_inf / ||A|| = %.3e", tag, rA);
    check(!(rA >= tol), buf);
    std::snprintf(buf, sizeof(buf), "%s: ||L R - I||_inf = %.3e", tag, rI);
    check(!(rI >= tol), buf);
}

// ---------------------------------------------------------------------------
// n = 4 : a general NON-SYMMETRIC matrix with a real, well-separated spectrum.
// Constructed as V diag(1,2,3,5) V^-1 with a deliberately non-orthogonal V
// (cond(V) = 4.8, ||A - A^T||_inf = 2.54) so this exercises the general
// real-Schur + explicit-inverse path and not the symmetric special case.
// A symmetric fixture would pass with L = R^T and prove nothing.
// ---------------------------------------------------------------------------
static void test_eigensystem_n4()
{
    const double A[16] = {
         0.575758,  0.878788, -0.833333,  1.606061,
        -0.301299,  2.064935,  0.428571, -0.181818,
        -0.135065, -0.046753,  2.571429,  1.490909,
        -0.637229,  0.528139, -1.047619,  5.787879
    };
    auto read = [&](int idx) {
        return amrex_user::eigensystem(idx,
            A[0],  A[1],  A[2],  A[3],
            A[4],  A[5],  A[6],  A[7],
            A[8],  A[9],  A[10], A[11],
            A[12], A[13], A[14], A[15]);
    };
    check_invariants<4>(A, read, "n4", 1e-10);
}

// ---------------------------------------------------------------------------
// n = 6 : the VAM level-1 arity.  Row 0 is all zeros (the bathymetry row), so
// the matrix is singular -- but the spectrum is still real and simple here.
// ---------------------------------------------------------------------------
static void test_eigensystem_n6()
{
    const double A[36] = {
        0,   0,   0,   0,   0,   0,
        0,   0,   1,   0,   0,   0,
        1,   2,   3,   0.4, 0,   0,
        0,   0,   5,   6,   0.7, 0,
        0,   0,   0.8, 9,   1,   2,
        3,   0,   4,   5,   6,   7
    };
    auto read = [&](int idx) {
        return amrex_user::eigensystem(idx,
            A[0],  A[1],  A[2],  A[3],  A[4],  A[5],
            A[6],  A[7],  A[8],  A[9],  A[10], A[11],
            A[12], A[13], A[14], A[15], A[16], A[17],
            A[18], A[19], A[20], A[21], A[22], A[23],
            A[24], A[25], A[26], A[27], A[28], A[29],
            A[30], A[31], A[32], A[33], A[34], A[35]);
    };
    check_invariants<6>(A, read, "n6", 1e-9);
}

// ---------------------------------------------------------------------------
// Non-finite input => lambda = +inf, R = L = I  (MOOD feeds these BY DESIGN,
// REQ-168 addendum 1: candidate states are evaluated before being rejected).
// ---------------------------------------------------------------------------
static void test_nonfinite()
{
    const double nan = std::nan("");
    auto e = [&](int idx) {
        return amrex_user::eigensystem(idx, 1.0, 0.0, nan, 1.0);
    };
    bool lam_inf = true, is_I = true;
    for (int i = 0; i < 2; ++i) if (!(std::isinf(e(i)) && e(i) > 0.0)) lam_inf = false;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            if (e(2 + i * 2 + j) != want)     is_I = false;   // R
            if (e(2 + 4 + i * 2 + j) != want) is_I = false;   // L
        }
    check(lam_inf, "nonfinite: eigensystem lambda == +inf");
    check(is_I,    "nonfinite: eigensystem R == L == I  (NOT 0: 0*inf would be NaN)");

    const double v = amrex_user::eigenvalues(0, 1.0, 0.0, nan, 1.0);
    check(std::isinf(v) && v > 0.0, "nonfinite: eigenvalues == +inf");
}

// ---------------------------------------------------------------------------
// Complex spectrum.  A = [[0,1,0],[-2,0,1],[0,0,0.5]] : lambda = +/- i sqrt(2),
// 0.5, so rho(A) = sqrt(2).
//   eigenvalues  -> signed modulus, so max|.| == rho  (GRAFT: dt must not freeze)
//   eigensystem  -> refuses; R|Lambda|L has no real-arithmetic meaning
// ---------------------------------------------------------------------------
static void test_complex()
{
    const double a[9] = {0.0, 1.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.5};
    double rho = 0.0;
    bool finite = true;
    for (int i = 0; i < 3; ++i) {
        const double v = amrex_user::eigenvalues(i, a[0], a[1], a[2],
                                                    a[3], a[4], a[5],
                                                    a[6], a[7], a[8]);
        if (!std::isfinite(v)) finite = false;
        if (std::fabs(v) > rho) rho = std::fabs(v);
    }
    check(finite, "complex: eigenvalues stays FINITE (does not freeze dt)");
    check(close(rho, std::sqrt(2.0), 1e-12), "complex: max|lambda| == rho(A) == sqrt(2)");

    // eigensystem's R block (idx >= n) refuses; its lambda block follows the
    // lazy split and agrees with eigenvalues.
    const double r0 = amrex_user::eigensystem(3, a[0], a[1], a[2],
                                                 a[3], a[4], a[5],
                                                 a[6], a[7], a[8]);
    check(r0 == 1.0, "complex: eigensystem R block REFUSES with I");
}

// ---------------------------------------------------------------------------
// Termination guards.  Both of these are the '<' vs '<=' deflation bug: with an
// all-zero active block s == 0 AND norm == 0, so a strict '<' test never
// deflates and the QR loop runs out its whole budget instead of finishing.
// If either of these hangs, the criterion has regressed.
// ---------------------------------------------------------------------------
static void test_degenerate_terminates()
{
    double z[36] = {0.0};
    auto readz = [&](int idx) {
        return amrex_user::eigenvalues(idx,
            z[0],  z[1],  z[2],  z[3],  z[4],  z[5],
            z[6],  z[7],  z[8],  z[9],  z[10], z[11],
            z[12], z[13], z[14], z[15], z[16], z[17],
            z[18], z[19], z[20], z[21], z[22], z[23],
            z[24], z[25], z[26], z[27], z[28], z[29],
            z[30], z[31], z[32], z[33], z[34], z[35]);
    };
    bool allzero = true;
    for (int i = 0; i < 6; ++i) if (readz(i) != 0.0) allzero = false;
    check(allzero, "zero 6x6: terminates with lambda == 0 (not +inf/NOCONV)");

    // dry SWE cell, h = 0, u = 0:  A = [[0,1],[0,0]] -- defective Jordan block.
    const double d0 = amrex_user::eigenvalues(0, 0.0, 1.0, 0.0, 0.0);
    const double d1 = amrex_user::eigenvalues(1, 0.0, 1.0, 0.0, 0.0);
    check(d0 == 0.0 && d1 == 0.0, "dry h=0: terminates with lambda == {0, 0}");

    // ... and its eigensystem is genuinely defective.  LAZY SPLIT: the lambda
    // block (idx < n) must still be exact -- the CFL path reads only that, and
    // gating it on R's invertibility pinned dt at 0 for the whole VAM run --
    // while the R / L blocks (idx >= n) fall back to the identity, which is the
    // refusal (and is why Roe must not be wired to VAM).
    const double s0 = amrex_user::eigensystem(0, 0.0, 1.0, 0.0, 0.0);
    const double s1 = amrex_user::eigensystem(1, 0.0, 1.0, 0.0, 0.0);
    check(s0 == 0.0 && s1 == 0.0, "dry h=0: eigensystem lambda block is EXACT (not +inf)");
    bool RisI = true;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            if (amrex_user::eigensystem(2 + i * 2 + j, 0.0, 1.0, 0.0, 0.0) != want) RisI = false;
            if (amrex_user::eigensystem(6 + i * 2 + j, 0.0, 1.0, 0.0, 0.0) != want) RisI = false;
        }
    check(RisI, "dry h=0: eigensystem R/L blocks REFUSE with I (defective basis)");
}

// ---------------------------------------------------------------------------
// solve(idx, *A_flat, *b) : A^-1 b, arity 1 + n^2 + n.
// ---------------------------------------------------------------------------
static void test_solve()
{
    // [[2,1],[1,3]] x = [3,5]  =>  x = [0.8, 1.4]
    const double x0 = amrex_user::solve(0, 2.0, 1.0, 1.0, 3.0, 3.0, 5.0);
    const double x1 = amrex_user::solve(1, 2.0, 1.0, 1.0, 3.0, 3.0, 5.0);
    check(close(x0, 0.8, 1e-14) && close(x1, 1.4, 1e-14), "solve: 2x2 exact");

    // singular A must refuse, not return a plausible number
    const double s = amrex_user::solve(0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0);
    check(std::isinf(s) && s > 0.0, "solve: singular A refuses with +inf");
}

int main()
{
    test_swe_spectrum();
    test_eigensystem_n4();
    test_eigensystem_n6();
    test_nonfinite();
    test_complex();
    test_degenerate_terminates();
    test_solve();
    std::printf(g_fail ? "\n%d FAILURE(S)\n" : "\nall pass\n", g_fail);
    return g_fail ? 1 : 0;
}
