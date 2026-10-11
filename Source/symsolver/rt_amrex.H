#pragma once
/*---------------------------------------------------------------------------*\
    rt_amrex.H -- the AMReX runtime of the symbolic solvers: what every name
    in the printed C++ (zoomy_core.symsolver.CxxPrinter, the text foam runs)
    means on a single-level AMReX grid (2-D, Cartesian).

    Values: a Field is rows x (cells | faces) on the grid's BoxArray -- cells:
    one MultiFab with NG ghost cells; faces: one face-centred MultiFab per
    direction (the faces of each box's valid cells).  Fields are immutable
    after the block that makes them (shared storage, value semantics).
    ``ngv``: how many ghost layers hold valid data.  Pointwise blocks compute
    on the whole fab (ngv = min of the inputs), stencil blocks shrink it,
    ``halo`` (FillBoundary) restores it; a stencil block whose input is too
    thin exchanges it itself.

    Faces follow the unstructured convention of the JAX/foam runtimes: an
    interior face is owned by its lower cell, normal +e_d; a boundary face by
    its inner cell, normal outward.  A boundary value (Qb, Wb) lives in the
    out-of-domain ghost cell across its face.

    Every loop is an amrex::ParallelFor (CPU or GPU); reductions are
    ParReduce + ParallelDescriptor; the linear solve is amrex::GMRES.
\*---------------------------------------------------------------------------*/
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_GMRES.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_SmallMatrix.H>

#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(AMREX_SPACEDIM == 2, "rt_amrex.H: 2-D builds only");

namespace zs
{
using amrex::Real;
using amrex::IntVect;
template<int N> using Col = amrex::SmallMatrix<Real, (N > 0 ? N : 1), 1>;
constexpr Real NaN = std::numeric_limits<Real>::quiet_NaN();
constexpr int MAXR = 16;      // most rows a static row list carries

// ── settings (the driver's configure()) ─────────────────────────────────────
struct Settings { int lsq_degree = 1; };
inline Settings& settings() { static Settings s; return s; }
struct Implicit
{
    double tol = 1e-8;        // Newton: ||R||_2 <= tol
    int maxiter = 6;
    double ls_tol = 1e-7;     // GMRES: ||A x - b|| <= ls_tol ||b||
    int ls_restart = 20, ls_maxiter = 30;
    double ls_accept = -1;    // > 0: abort above this relative residual
};
inline Implicit& implicit() { static Implicit s; return s; }

// ── mesh ────────────────────────────────────────────────────────────────────
// Geo: what a device loop needs (trivially copyable).  LSQ: per cell class
// (clsx[i] * ncy + clsy[j]) a list of samples (kind 0: cell at (i+di, j+dj),
// kind 1: the boundary value in the ghost cell (i+di, j+dj)) and their
// weights per monomial -- the zoomy mesh's lsq_gradQ * lsq_scale_factors.
struct Geo
{
    int ilo, ihi, jlo, jhi;
    Real dx, dy, plox, ploy;
    int branch[4];                     // BC branch: x-lo, x-hi, y-lo, y-hi
    int ncy, nmon, gx, gy;             // gx, gy: monomial of d/dx, d/dy
    const int* clsx; const int* clsy; const int* off; const int* smp;
    const Real* w;
};

struct Mesh
{
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    int NG = 2, R = 1;
    std::vector<std::array<int, 3>> mon;
    amrex::Gpu::DeviceVector<int> clsx, clsy, off, smp;
    amrex::Gpu::DeviceVector<Real> w;
    Geo g{};

    // the LSQ table written by the driver (zoomy_amrex.symsolver._lsq_table)
    void load(const std::string& path, const int branch[4])
    {
        std::ifstream f(path);
        int nx, ny, ncx, ncy, nmon;
        f >> R >> nx >> ny >> ncx >> ncy >> nmon;
        NG = R + 1;
        std::vector<int> cx(nx), cy(ny), of{0}, sm;
        std::vector<Real> ww;
        for (auto& v : cx) f >> v;
        for (auto& v : cy) f >> v;
        mon.resize(nmon);
        for (auto& m : mon) { f >> m[0] >> m[1]; m[2] = 0; }
        for (int c = 0; c < ncx*ncy; ++c)
        {
            int ns; f >> ns;
            for (int s = 0; s < ns; ++s)
            {
                int kind, di, dj; f >> kind >> di >> dj;
                sm.insert(sm.end(), {kind, di, dj});
                for (int k = 0; k < nmon; ++k) { Real x; f >> x; ww.push_back(x); }
            }
            of.push_back(of.back() + ns);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.good(), "rt_amrex: bad LSQ table");
        auto up = [](auto& dv, const auto& hv)
        {
            dv.resize(hv.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, hv.begin(), hv.end(), dv.begin());
        };
        up(clsx, cx); up(clsy, cy); up(off, of); up(smp, sm); up(w, ww);
        const amrex::Box& d = geom.Domain();
        g.ilo = d.smallEnd(0); g.ihi = d.bigEnd(0);
        g.jlo = d.smallEnd(1); g.jhi = d.bigEnd(1);
        g.dx = geom.CellSize(0); g.dy = geom.CellSize(1);
        g.plox = geom.ProbLo(0); g.ploy = geom.ProbLo(1);
        for (int s = 0; s < 4; ++s) g.branch[s] = branch[s];
        g.ncy = ncy; g.nmon = nmon;
        g.gx = monomial(1, 0); g.gy = monomial(0, 1);
        g.clsx = clsx.data(); g.clsy = clsy.data(); g.off = off.data();
        g.smp = smp.data(); g.w = w.data();
    }
    int monomial(int a, int b) const
    {
        for (int k = 0; k < int(mon.size()); ++k)
            if (mon[k][0] == a && mon[k][1] == b) return k;
        return -1;
    }
};

// ── device geometry helpers ─────────────────────────────────────────────────
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool inside(const Geo& g, int i, int j)
{
    return i >= g.ilo && i <= g.ihi && j >= g.jlo && j <= g.jhi;
}

// face (i, j) of direction d (between cells (i,j) - e_d and (i,j)):
// owner, the other cell (out of the domain on a boundary face), normal
struct FaceInfo { int oi, oj, xi, xj; bool bnd; Real n0, n1; };
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FaceInfo face_info(const Geo& g, int d, int i, int j)
{
    const int li = i - (d == 0), lj = j - (d == 1);
    const bool blo = d == 0 ? i == g.ilo : j == g.jlo;
    const bool bhi = d == 0 ? i == g.ihi + 1 : j == g.jhi + 1;
    FaceInfo f;
    f.bnd = blo || bhi;
    if (blo) { f.oi = i; f.oj = j; f.xi = li; f.xj = lj; }
    else     { f.oi = li; f.oj = lj; f.xi = i; f.xj = j; }
    const Real s = blo ? -1.0 : 1.0;
    f.n0 = d == 0 ? s : 0.0;
    f.n1 = d == 1 ? s : 0.0;
    return f;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<3> cell_centre(const Geo& g, int i, int j)
{
    Col<3> X{};
    X(0, 0) = g.plox + (i + 0.5)*g.dx;
    X(1, 0) = g.ploy + (j + 0.5)*g.dy;
    return X;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<3> face_centre(const Geo& g, int d, int i, int j)
{
    Col<3> X{};
    X(0, 0) = g.plox + (i + (d == 0 ? 0.0 : 0.5))*g.dx;
    X(1, 0) = g.ploy + (j + (d == 1 ? 0.0 : 0.5))*g.dy;
    return X;
}

template<int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<N> normal(Real n0, Real n1)
{
    Col<N> n{};
    n(0, 0) = n0;
    if constexpr (N > 1) n(1, 0) = n1;
    return n;
}

template<int N, class A>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<N> col(const A& a, int i, int j, int nq)
{
    Col<N> v{};
    for (int q = 0; q < N && q < nq; ++q) v(q, 0) = a(i, j, 0, q);
    return v;
}

// ── fields ──────────────────────────────────────────────────────────────────
struct Field
{
    int nq = 0;
    bool face = false;
    int ngv = 0;
    std::shared_ptr<std::vector<amrex::MultiFab>> mf;
    std::shared_ptr<void> keep;        // the parent of an alias
    amrex::MultiFab& operator[](int d) const { return (*mf)[d]; }
    int ncomp() const { return face ? AMREX_SPACEDIM : 1; }
    int ng() const { return face ? 0 : (*mf)[0].nGrow(); }
};

inline Field make_like(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                       int nq, bool face, int ng)
{
    Field F;
    F.nq = nq; F.face = face;
    F.mf = std::make_shared<std::vector<amrex::MultiFab>>();
    for (int d = 0; d < (face ? AMREX_SPACEDIM : 1); ++d)
    {
        const amrex::BoxArray b = face ? amrex::convert(ba, IntVect::TheDimensionVector(d)) : ba;
        F.mf->emplace_back(b, dm, std::max(nq, 1), face ? 0 : ng);
        F.mf->back().setVal(0.0);
    }
    return F;
}
inline Field cells_of(const Mesh& M, int nq) { return make_like(M.ba, M.dm, nq, false, M.NG); }
inline Field faces_of(const Mesh& M, int nq) { return make_like(M.ba, M.dm, nq, true, 0); }
inline Field like(const Field& a, int nq)
{
    const amrex::MultiFab& m = a[0];
    amrex::BoxArray ba = m.boxArray();
    if (a.face) ba.enclosedCells();
    Field F = make_like(ba, m.DistributionMap(), nq, a.face, a.face ? 0 : m.nGrow());
    return F;
}

// rows ``first .. first + n`` of F, no copy
inline Field rows_alias(const Field& F, int first, int n)
{
    Field A;
    A.nq = n; A.face = F.face; A.ngv = F.ngv; A.keep = F.mf;
    A.mf = std::make_shared<std::vector<amrex::MultiFab>>();
    for (int d = 0; d < F.ncomp(); ++d) A.mf->emplace_back(F[d], amrex::make_alias, first, n);
    return A;
}

// ── elementwise arithmetic (one row broadcasts against many) ────────────────
struct Add { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const { return x + y; } };
struct Sub { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const { return x - y; } };
struct Mul { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const { return x*y; } };
struct Div { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const { return x/y; } };
struct Pow { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const { return std::pow(x, y); } };
struct Min
{
    AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const
    { return (x != x || y != y) ? NaN : (x < y ? x : y); }
};
struct Max
{
    AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real y) const
    { return (x != x || y != y) ? NaN : (x > y ? x : y); }
};

template<class F>
Field zip(const Field& a, const Field& b, F f)
{
    Field r = like(a, std::max(a.nq, b.nq));
    r.ngv = std::min(a.ngv, b.ngv);
    const int na = a.nq, nb = b.nq;
    for (int d = 0; d < r.ncomp(); ++d)
    {
        auto ra = r[d].arrays();
        auto aa = a[d].const_arrays();
        auto ba = b[d].const_arrays();
        amrex::ParallelFor(r[d], IntVect(r.ng()), r.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ra[bx](i, j, k, n) = f(aa[bx](i, j, k, na == 1 ? 0 : n),
                                   ba[bx](i, j, k, nb == 1 ? 0 : n));
        });
    }
    return r;
}

// f(x, s) (or f(s, x) with ``left``) for a number s
template<class F>
Field zip(const Field& a, Real s, F f, bool left)
{
    Field r = like(a, a.nq);
    r.ngv = a.ngv;
    for (int d = 0; d < r.ncomp(); ++d)
    {
        auto ra = r[d].arrays();
        auto aa = a[d].const_arrays();
        amrex::ParallelFor(r[d], IntVect(r.ng()), r.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            const Real x = aa[bx](i, j, k, n);
            ra[bx](i, j, k, n) = left ? f(s, x) : f(x, s);
        });
    }
    return r;
}

template<class T> using if_num = std::enable_if_t<std::is_arithmetic_v<T>, int>;
inline Field operator+(const Field& a, const Field& b) { return zip(a, b, Add{}); }
inline Field operator-(const Field& a, const Field& b) { return zip(a, b, Sub{}); }
inline Field operator*(const Field& a, const Field& b) { return zip(a, b, Mul{}); }
inline Field operator/(const Field& a, const Field& b) { return zip(a, b, Div{}); }
template<class T, if_num<T> = 0> Field operator+(const Field& a, T s) { return zip(a, Real(s), Add{}, false); }
template<class T, if_num<T> = 0> Field operator+(T s, const Field& a) { return zip(a, Real(s), Add{}, true); }
template<class T, if_num<T> = 0> Field operator-(const Field& a, T s) { return zip(a, Real(s), Sub{}, false); }
template<class T, if_num<T> = 0> Field operator-(T s, const Field& a) { return zip(a, Real(s), Sub{}, true); }
template<class T, if_num<T> = 0> Field operator*(const Field& a, T s) { return zip(a, Real(s), Mul{}, false); }
template<class T, if_num<T> = 0> Field operator*(T s, const Field& a) { return zip(a, Real(s), Mul{}, true); }
template<class T, if_num<T> = 0> Field operator/(const Field& a, T s) { return zip(a, Real(s), Div{}, false); }
template<class T, if_num<T> = 0> Field operator/(T s, const Field& a) { return zip(a, Real(s), Div{}, true); }
inline Field operator-(const Field& a) { return zip(a, -1.0, Mul{}, false); }
template<class T, if_num<T> = 0> Field pow(const Field& a, T e) { return zip(a, Real(e), Pow{}, false); }
struct Abs { AMREX_GPU_HOST_DEVICE Real operator()(Real x, Real) const { return std::abs(x); } };
inline Field abs(const Field& a) { return zip(a, 0.0, Abs{}, false); }
using std::abs;
using std::floor;
using std::fmod;
using std::pow;

// NaN propagates (std::min / std::max would swallow it)
inline double minimum(double a, double b) { return (a != a || b != b) ? NaN : std::min(a, b); }
inline double maximum(double a, double b) { return (a != a || b != b) ? NaN : std::max(a, b); }
inline Field minimum(const Field& a, const Field& b) { return zip(a, b, Min{}); }
template<class T, if_num<T> = 0> Field minimum(const Field& a, T s) { return zip(a, Real(s), Min{}, false); }
template<class T, if_num<T> = 0> Field minimum(T s, const Field& a) { return zip(a, Real(s), Min{}, true); }

// ── halo ────────────────────────────────────────────────────────────────────
// the copies of the neighbouring boxes' cells (any rank); a number passes
inline Field halo(const Field& X, const Mesh& M)
{
    if (X.face || X.ngv >= X.ng()) return X;
    X[0].FillBoundary(M.geom.periodicity());   // ghosts only: no value changes
    Field Y = X;
    Y.ngv = X.ng();
    return Y;
}
template<class T, if_num<T> = 0> T halo(T x, const Mesh&) { return x; }
inline Field need(const Field& X, int n, const Mesh& M) { return X.ngv >= n ? X : halo(X, M); }

// ── reductions (valid region, all ranks; NaN propagates) ────────────────────
inline double reduce_ext(const Field& F, bool want_min)
{
    double best = want_min ? std::numeric_limits<double>::infinity()
                           : -std::numeric_limits<double>::infinity();
    bool nan = false;
    const int nq = F.nq;
    for (int d = 0; d < F.ncomp(); ++d)
    {
        auto a = F[d].const_arrays();
        auto r = amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpMin, amrex::ReduceOpMax, amrex::ReduceOpLogicalOr>{},
            amrex::TypeList<Real, Real, int>{}, F[d], IntVect(0),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
                -> amrex::GpuTuple<Real, Real, int>
        {
            Real lo = std::numeric_limits<Real>::infinity(), hi = -lo;
            int bad = 0;
            for (int n = 0; n < nq; ++n)
            {
                const Real x = a[bx](i, j, k, n);
                if (x != x) bad = 1;
                else { lo = x < lo ? x : lo; hi = x > hi ? x : hi; }
            }
            return {lo, hi, bad};
        });
        best = want_min ? std::min(best, double(amrex::get<0>(r)))
                        : std::max(best, double(amrex::get<1>(r)));
        nan = nan || amrex::get<2>(r);
    }
    if (want_min) amrex::ParallelDescriptor::ReduceRealMin(best);
    else amrex::ParallelDescriptor::ReduceRealMax(best);
    amrex::ParallelDescriptor::ReduceBoolOr(nan);
    return nan ? NaN : best;
}
inline double reduce_min(const Field& F) { return reduce_ext(F, true); }
inline double reduce_max(const Field& F) { return reduce_ext(F, false); }
inline bool reduce_any(const Field& F)
{
    const int nq = F.nq;
    bool any = false;
    for (int d = 0; d < F.ncomp(); ++d)
    {
        auto a = F[d].const_arrays();
        any = any || amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalOr>{},
                                      amrex::TypeList<int>{}, F[d], IntVect(0),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
                -> amrex::GpuTuple<int>
        {
            int r = 0;
            for (int n = 0; n < nq; ++n) r = r || a[bx](i, j, k, n) != 0.0;
            return {r};
        });
    }
    amrex::ParallelDescriptor::ReduceBoolOr(any);
    return any;
}
template<class T, if_num<T> = 0> double reduce_min(T x) { return x; }
template<class T, if_num<T> = 0> double reduce_max(T x) { return x; }

// ── static row lists ────────────────────────────────────────────────────────
struct Rows { int n = 0; int r[MAXR]; };
template<class... I> Rows rows_of(const std::tuple<I...>& t)
{
    Rows R;
    std::apply([&](auto... i) { ((R.r[R.n++] = int(i)), ...); }, t);
    return R;
}

template<class... I, class V>
Field put(const Field& Q, const std::tuple<I...>& rows, const V& values)
{
    const Rows R = rows_of(rows);
    Field out = like(Q, Q.nq);
    out.ngv = Q.ngv;
    if constexpr (!std::is_arithmetic_v<V>) out.ngv = std::min(Q.ngv, values.ngv);
    for (int d = 0; d < Q.ncomp(); ++d)
    {
        amrex::MultiFab::Copy(out[d], Q[d], 0, 0, Q.nq, out.ng());
        for (int k = 0; k < R.n; ++k)
        {
            if constexpr (std::is_arithmetic_v<V>) out[d].setVal(Real(values), R.r[k], 1, out.ng());
            else amrex::MultiFab::Copy(out[d], values[d], values.nq == 1 ? 0 : k, R.r[k], 1, out.ng());
        }
    }
    return out;
}

template<class... I>
Field take(const Field& Q, const std::tuple<I...>& rows)
{
    const Rows R = rows_of(rows);
    Field out = like(Q, R.n);
    out.ngv = Q.ngv;
    for (int d = 0; d < Q.ncomp(); ++d)
        for (int k = 0; k < R.n; ++k)
            amrex::MultiFab::Copy(out[d], Q[d], R.r[k], k, 1, out.ng());
    return out;
}

inline Field zeros(int n, const Mesh& M)
{
    Field Z = cells_of(M, n);
    Z.ngv = M.NG;
    return Z;
}

// entries of ``rows`` in [-tol, 0) -> 0: round-off below zero
template<class... I>
Field clean_round_off(const Field& Q, const std::tuple<I...>& rows, double tol)
{
    const Rows R = rows_of(rows);
    Field out = like(Q, Q.nq);
    out.ngv = Q.ngv;
    for (int d = 0; d < Q.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        auto q = Q[d].const_arrays();
        amrex::ParallelFor(out[d], IntVect(out.ng()), Q.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            Real x = q[bx](i, j, k, n);
            for (int m = 0; m < R.n; ++m)
                if (R.r[m] == n && x < 0.0 && x >= -tol) x = 0.0;
            o[bx](i, j, k, n) = x;
        });
    }
    return out;
}

// select(mask, A, B): A where mask else B (fields broadcast, numbers too)
struct Src
{
    amrex::MultiArray4<Real const> a;
    int nq = 1;
    Real s = 0.0;
    bool num = true;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, int i, int j, int n) const
    { return num ? s : a[bx](i, j, 0, nq == 1 ? 0 : n); }
};
inline Src src(const Field& F, int d) { return Src{F[d].const_arrays(), F.nq, 0.0, false}; }
template<class T, if_num<T> = 0> Src src(T x, int) { Src s; s.s = Real(x); return s; }
inline int rows_n(const Field& F) { return F.nq; }
template<class T, if_num<T> = 0> int rows_n(T) { return 1; }
inline int ngv_of(const Field& F) { return F.ngv; }
template<class T, if_num<T> = 0> int ngv_of(T) { return 1 << 20; }

template<class A, class B>
Field select(const Field& m, const A& a, const B& b)
{
    Field out = like(m, std::max({m.nq, rows_n(a), rows_n(b)}));
    out.ngv = std::min({m.ngv, ngv_of(a), ngv_of(b)});
    for (int d = 0; d < m.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        const Src M = src(m, d), Sa = src(a, d), Sb = src(b, d);
        amrex::ParallelFor(out[d], IntVect(out.ng()), out.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
        {
            o[bx](i, j, 0, n) = M(bx, i, j, n) != 0.0 ? Sa(bx, i, j, n) : Sb(bx, i, j, n);
        });
    }
    return out;
}

// ── kernels on cells and faces ──────────────────────────────────────────────
// a cell-family kernel in every cell (ghosts included)
template<class K, class P, class T1, class T2>
Field cells(K k, const Field& Q, const Field& A, const P& p, T1 time_, T2 dt_,
            const Mesh& M)
{
    Field out = cells_of(M, K::n_out);
    out.ngv = std::min(Q.ngv, A.ngv);
    const Real time = time_, dt = dt_;
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IntVect(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        const auto v = k(col<K::n_in>(q[bx], i, j, nq), col<K::n_aux>(a[bx], i, j, na),
                         p, time, dt, cell_centre(g, i, j));
        for (int r = 0; r < K::n_out; ++r) o[bx](i, j, 0, r) = v(r, 0);
    });
    return out;
}

// a recon-family kernel takes (W, Qaux, p), a cell-family one gets time = dt
// = 0 and no position
template<class K, class X, class A, class P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto call_pointwise(const K& k, const X& x, const A& a, const P& p)
{
    if constexpr (K::family == 3) return k(x, a, p);
    else return k(x, a, p, 0.0, 0.0, Col<3>{});
}

// a kernel in every column (cells or faces); no time, step or position
template<class K, class P>
Field pointwise(K k, const Field& X, const Field& A, const P& p)
{
    Field out = like(X, K::n_out);
    out.ngv = std::min(X.ngv, A.ngv);
    const int nq = X.nq, na = A.nq;
    for (int d = 0; d < X.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        auto x = X[d].const_arrays();
        auto a = A[d].const_arrays();
        amrex::ParallelFor(out[d], IntVect(out.ng()),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
        {
            const auto xc = col<K::n_in>(x[bx], i, j, nq);
            const auto ac = col<K::n_aux>(a[bx], i, j, na);
            const auto v = call_pointwise(k, xc, ac, p);
            for (int r = 0; r < K::n_out; ++r) o[bx](i, j, 0, r) = v(r, 0);
        });
    }
    return out;
}

template<int N>
std::array<Field, N> split(const Field& F)
{
    std::array<Field, N> out;
    for (int g = 0; g < N; ++g) out[g] = rows_alias(F, g*(F.nq/N), F.nq/N);
    return out;
}

// the face kernel on every face, owner -> other with the owner's normal
template<class K, class P>
Field face_kernel(K k, const Field& QL, const Field& QR, const Field& AL, const Field& AR,
                  const P& p, const Mesh& M)
{
    Field out = faces_of(M, K::n_out);
    const Geo g = M.g;
    const int nq = QL.nq, na = AL.nq;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto o = out[d].arrays();
        auto l = QL[d].const_arrays();
        auto r = QR[d].const_arrays();
        auto al = AL[d].const_arrays();
        auto ar = AR[d].const_arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            const auto v = k(col<K::n_in>(l[bx], i, j, nq), col<K::n_in>(r[bx], i, j, nq),
                             col<K::n_aux>(al[bx], i, j, na), col<K::n_aux>(ar[bx], i, j, na),
                             p, normal<K::n_nrm>(f.n0, f.n1));
            for (int m = 0; m < K::n_out; ++m) o[bx](i, j, 0, m) = v(m, 0);
        });
    }
    return out;
}
template<int N = 1, class K, class P>
auto faces(K k, const Field& QL, const Field& QR, const Field& AL, const Field& AR,
           const P& p, const Mesh& M)
{
    const Field out = face_kernel(k, QL, QR, AL, AR, p, M);
    if constexpr (N == 1) return out;
    else return split<N>(out);
}

// owner and other cell aux per face (the inner cell twice at a boundary face)
inline std::tuple<Field, Field> face_aux(const Field& Qaux_, const Mesh& M)
{
    const Field Qaux = need(Qaux_, 1, M);
    Field L = faces_of(M, Qaux.nq), R = faces_of(M, Qaux.nq);
    const Geo g = M.g;
    const int na = Qaux.nq;
    auto a = Qaux[0].const_arrays();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IntVect(0), na,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            l[bx](i, j, 0, n) = a[bx](f.oi, f.oj, 0, n);
            r[bx](i, j, 0, n) = f.bnd ? a[bx](f.oi, f.oj, 0, n) : a[bx](f.xi, f.xj, 0, n);
        });
    }
    return {L, R};
}

// max |eigenvalue| per face over its two cells (the inner cell twice at a
// boundary face)
template<class K, class P>
Field face_wavespeed(K k, const Field& Q_, const Field& A_, const P& p, const Mesh& M)
{
    const Field Q = need(Q_, 1, M), A = need(A_, 1, M);
    Field out = faces_of(M, 1);
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto o = out[d].arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            const auto n = normal<K::n_nrm>(f.n0, f.n1);
            auto peak = [&](int ci, int cj)
            {
                const auto v = k(col<K::n_in>(q[bx], ci, cj, nq),
                                 col<K::n_aux>(a[bx], ci, cj, na), p, n);
                Real s = 0.0;
                for (int m = 0; m < K::n_out; ++m)
                {
                    const Real x = std::abs(v(m, 0));
                    s = (x != x || s != s) ? NaN : (x > s ? x : s);
                }
                return s;
            };
            const Real s0 = peak(f.oi, f.oj);
            const Real s1 = f.bnd ? s0 : peak(f.xi, f.xj);
            o[bx](i, j, 0, 0) = (s0 != s0 || s1 != s1) ? NaN : (s0 > s1 ? s0 : s1);
        });
    }
    return out;
}

// per-face cell size: d |V| / |dV| of the (uniform) cells
inline Field face_inradius(const Mesh& M)
{
    Field out = faces_of(M, 1);
    const Real r = 2.0*M.g.dx*M.g.dy/(2.0*M.g.dx + 2.0*M.g.dy);
    for (int d = 0; d < AMREX_SPACEDIM; ++d) out[d].setVal(r);
    return out;
}

// ── boundary ────────────────────────────────────────────────────────────────
// the BC kernel's state in every out-of-domain ghost cell across a face, from
// ``states`` (cells: the inner cell) and the inner cell's aux
template<class K, class P, class T>
Field bc_cells(K bc, const Field& Q, const Field& A, const P& p, T time_, const Mesh& M)
{
    Field out = cells_of(M, Q.nq);
    out.ngv = std::min(Q.ngv, A.ngv);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IntVect(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        const int ox = i < g.ilo ? -1 : (i > g.ihi ? 1 : 0);
        const int oy = j < g.jlo ? -1 : (j > g.jhi ? 1 : 0);
        if ((ox != 0) == (oy != 0)) return;                 // inside, or a corner
        const int ii = i - ox, jj = j - oy;                 // the inner cell
        if (!inside(g, ii, jj)) return;                     // two cells out
        const int side = ox < 0 ? 0 : ox > 0 ? 1 : oy < 0 ? 2 : 3;
        const int d = ox != 0 ? 0 : 1;
        const Col<3> X = face_centre(g, d, ox > 0 ? i : ii, oy > 0 ? j : jj);
        const Real dist = 0.5*(d == 0 ? g.dx : g.dy);
        const auto v = bc(g.branch[side], time, X, dist, col<K::n_in>(q[bx], ii, jj, nq),
                          col<K::n_aux>(a[bx], ii, jj, na), p,
                          normal<K::n_nrm>(Real(ox), Real(oy)));
        for (int m = 0; m < K::n_out && m < nq; ++m) o[bx](i, j, 0, m) = v(m, 0);
    });
    return out;
}

// the inner cell's values in every out-of-domain ghost cell across a face
inline Field inner_cells(const Field& A, const Mesh& M)
{
    Field out = cells_of(M, A.nq);
    out.ngv = A.ngv;
    const Geo g = M.g;
    auto o = out[0].arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IntVect(out.ng()), A.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
    {
        const int ox = i < g.ilo ? -1 : (i > g.ihi ? 1 : 0);
        const int oy = j < g.jlo ? -1 : (j > g.jhi ? 1 : 0);
        if ((ox != 0) == (oy != 0) || !inside(g, i - ox, j - oy)) return;
        o[bx](i, j, 0, n) = a[bx](i - ox, j - oy, 0, n);
    });
    return out;
}

template<class K, class P, class T>
Field boundary_ghosts(K bc, const Field& Q, const Field& A, const P& p, T time,
                      const Mesh& M)
{
    return bc_cells(bc, Q, A, p, time, M);
}

template<int N = 2, class K, class P, class T>
std::tuple<Field, Field> boundary_states(K bc, const Field& Q, const Field& A,
                                         const P& p, T time, const Mesh& M)
{
    return {bc_cells(bc, Q, A, p, time, M), inner_cells(A, M)};
}

// QR with BC(QL) on the boundary faces
template<class K, class P, class T>
Field boundary_traces(K bc, const Field&, const Field& QL, const Field& QR,
                      const Field& A, const P& p, T time_, const Mesh& M)
{
    Field out = like(QR, QR.nq);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = QL.nq, na = A.nq;
    auto a = A[0].const_arrays();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto o = out[d].arrays();
        auto l = QL[d].const_arrays();
        auto r = QR[d].const_arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            if (!f.bnd)
            {
                for (int m = 0; m < nq; ++m) o[bx](i, j, 0, m) = r[bx](i, j, 0, m);
                return;
            }
            const int side = d == 0 ? (f.n0 < 0 ? 0 : 1) : (f.n1 < 0 ? 2 : 3);
            const auto v = bc(g.branch[side], time, face_centre(g, d, i, j),
                              0.5*(d == 0 ? g.dx : g.dy), col<K::n_in>(l[bx], i, j, nq),
                              col<K::n_aux>(a[bx], f.oi, f.oj, na), p,
                              normal<K::n_nrm>(f.n0, f.n1));
            for (int m = 0; m < nq; ++m) o[bx](i, j, 0, m) = v(m, 0);
        });
    }
    return out;
}

// ── reconstruction and assembly ─────────────────────────────────────────────
// owner and other cell value per face; the boundary value on a boundary face
inline std::tuple<Field, Field> cell_traces(const Field& Q_, const Field& Qb, const Mesh& M)
{
    const Field Q = need(Q_, 1, M);
    Field L = faces_of(M, Q.nq), R = faces_of(M, Q.nq);
    const Geo g = M.g;
    auto q = Q[0].const_arrays();
    auto b = Qb[0].const_arrays();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IntVect(0), Q.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            l[bx](i, j, 0, n) = q[bx](f.oi, f.oj, 0, n);
            r[bx](i, j, 0, n) = f.bnd ? b[bx](f.xi, f.xj, 0, n) : q[bx](f.xi, f.xj, 0, n);
        });
    }
    return {L, R};
}

// LSQ fit of row ``n`` at cell (i, j), monomial ``m``: sum_s w_s,m delta_s.
// A boundary sample's delta: ghost(n) - u_c with ghost = Wb (``mode`` 0), the
// face value 2 Qb - Q_inner (1), or u_c itself (2: no boundary value)
template<class A, class B>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real lsq(const Geo& g, const A& u, const B& ub, int mode, int i, int j, int n, int m)
{
    const int c = g.clsx[i - g.ilo]*g.ncy + g.clsy[j - g.jlo];
    const Real uc = u(i, j, 0, n);
    Real s = 0.0;
    for (int t = g.off[c]; t < g.off[c + 1]; ++t)
    {
        const int kind = g.smp[3*t], ii = i + g.smp[3*t + 1], jj = j + g.smp[3*t + 2];
        Real delta;
        if (kind == 0) delta = u(ii, jj, 0, n) - uc;
        else if (mode == 0) delta = ub(ii, jj, 0, n) - uc;
        else if (mode == 1)
        {
            const int ci = ii < g.ilo ? g.ilo : (ii > g.ihi ? g.ihi : ii);
            const int cj = jj < g.jlo ? g.jlo : (jj > g.jhi ? g.jhi : jj);
            delta = 2.0*ub(ii, jj, 0, n) - u(ci, cj, 0, n) - uc;
        }
        else delta = 0.0;
        s += g.w[t*g.nmon + m]*delta;
    }
    return s;
}

// LSQ gradient of every row (row 2 q + a: d/dx_a of row q)
inline Field gradient(const Field& W_, const Field& Wb, const Mesh& M)
{
    const Field W = need(W_, M.R, M);
    Field G = cells_of(M, 2*W.nq);
    const int gr = std::max(0, std::min(W.ngv, Wb.ngv) - M.R);
    G.ngv = gr;
    const Geo g = M.g;
    auto o = G[0].arrays();
    auto w = W[0].const_arrays();
    auto wb = Wb[0].const_arrays();
    amrex::ParallelFor(G[0], IntVect(gr), W.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
    {
        if (!inside(g, i, j)) return;
        o[bx](i, j, 0, 2*n) = lsq(g, w[bx], wb[bx], 0, i, j, n, g.gx);
        o[bx](i, j, 0, 2*n + 1) = lsq(g, w[bx], wb[bx], 0, i, j, n, g.gy);
    });
    return G;
}

// Limiter factor of one cell from one face increment ``delta`` (cell bounds
// dmin/dmax); ``ev2`` = (cell size)^2 for Venkatakrishnan
enum class Limiter { minmod, venkatakrishnan, barth_jespersen };
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real limit(Limiter L, Real delta, Real dmin, Real dmax, Real ev2)
{
    const Real eps = 1e-30;
    if (!(delta > eps) && !(delta < -eps)) return 1.0;
    const Real dm = delta > eps ? dmax : dmin;
    Real r;
    if (L == Limiter::venkatakrishnan)
    {
        const Real num = dm*dm + ev2 + 2*delta*dm;
        const Real den = dm*dm + 2*delta*delta + delta*dm + ev2;
        r = num/(den > eps ? den : eps);
    }
    else r = delta > eps ? dm/(delta > eps ? delta : eps) : dm/(delta < -eps ? delta : -eps);
    return r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
}

// Limiter factors of ``rows``, 1 elsewhere; bounds from the face neighbours
// and the boundary ghosts 2 Wb - W
inline Field limiter_factors(Limiter L, const Rows R, const Field& W_, const Field& Wb,
                             const Field& G, const Mesh& M)
{
    const Field W = need(W_, 1, M);
    Field phi = cells_of(M, W.nq);
    const int gr = std::max(0, std::min({W.ngv - 1, G.ngv, Wb.ngv}));
    phi.ngv = gr;
    const Geo g = M.g;
    const Real ev2 = g.dx*g.dy;                         // (V^(1/2))^2
    auto o = phi[0].arrays();
    auto w = W[0].const_arrays();
    auto wb = Wb[0].const_arrays();
    auto gg = G[0].const_arrays();
    amrex::ParallelFor(phi[0], IntVect(gr), W.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
    {
        bool lim = false;
        for (int m = 0; m < R.n; ++m) lim = lim || R.r[m] == n;
        if (!lim || !inside(g, i, j)) { o[bx](i, j, 0, n) = 1.0; return; }
        const Real u = w[bx](i, j, 0, n);
        const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, -1, 1};
        Real lo = u, hi = u;
        for (int f = 0; f < 4; ++f)
        {
            const int ii = i + di[f], jj = j + dj[f];
            const Real v = inside(g, ii, jj) ? w[bx](ii, jj, 0, n)
                                             : 2.0*wb[bx](ii, jj, 0, n) - u;
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        Real p = 1.0;
        for (int f = 0; f < 4; ++f)
        {
            const Real delta = gg[bx](i, j, 0, 2*n)*0.5*di[f]*g.dx
                             + gg[bx](i, j, 0, 2*n + 1)*0.5*dj[f]*g.dy;
            const Real c = limit(L, delta, lo - u, hi - u, ev2);
            p = c < p ? c : p;
        }
        o[bx](i, j, 0, n) = p;
    });
    return phi;
}
template<class... I>
Field minmod(const std::tuple<I...>& rows, const Field& W, const Field& Wb,
             const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::minmod, rows_of(rows), W, Wb, G, M); }
template<class... I>
Field venkatakrishnan(const std::tuple<I...>& rows, const Field& W, const Field& Wb,
                      const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::venkatakrishnan, rows_of(rows), W, Wb, G, M); }
template<class... I>
Field barth_jespersen(const std::tuple<I...>& rows, const Field& W, const Field& Wb,
                      const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::barth_jespersen, rows_of(rows), W, Wb, G, M); }

// each cell's W moved to the face centre by phi G; Wb on the boundary faces
template<class P>
std::tuple<Field, Field> extrapolate(const Field& W_, const Field& Wb, const Field& G_,
                                     const P& phi_, const Mesh& M)
{
    const Field W = need(W_, 1, M), G = need(G_, 1, M);
    Field L = faces_of(M, W.nq), R = faces_of(M, W.nq);
    const Geo g = M.g;
    auto w = W[0].const_arrays();
    auto wb = Wb[0].const_arrays();
    auto gg = G[0].const_arrays();
    Src phi;
    if constexpr (std::is_arithmetic_v<P>) phi = src(phi_, 0);
    else phi = src(need(phi_, 1, M), 0);
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IntVect(0), W.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
        {
            const FaceInfo f = face_info(g, d, i, j);
            // the face centre from a cell centre: +- h/2 along d
            auto val = [&](int ci, int cj)
            {
                const Real s = (d == 0 ? (ci < i ? 0.5 : -0.5)*g.dx : (cj < j ? 0.5 : -0.5)*g.dy);
                return w[bx](ci, cj, 0, n) + phi(bx, ci, cj, n)*gg[bx](ci, cj, 0, 2*n + d)*s;
            };
            l[bx](i, j, 0, n) = val(f.oi, f.oj);
            r[bx](i, j, 0, n) = f.bnd ? wb[bx](f.xi, f.xj, 0, n) : val(f.xi, f.xj);
        });
    }
    return {L, R};
}

// per cell sum_f |f| W / |V| over its faces: Wm where it owns the face, Wp
// where it is the other cell
inline Field divergence(const Field& Wm, const Field& Wp, const Mesh& M)
{
    Field out = cells_of(M, Wm.nq);
    const Geo g = M.g;
    auto o = out[0].arrays();
    auto mx = Wm[0].const_arrays(); auto my = Wm[1].const_arrays();
    auto px = Wp[0].const_arrays(); auto py = Wp[1].const_arrays();
    amrex::ParallelFor(out[0], IntVect(0), Wm.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int, int n) noexcept
    {
        // lower face: this cell owns it only on the domain's lower boundary
        const Real xl = i == g.ilo ? mx[bx](i, j, 0, n) : px[bx](i, j, 0, n);
        const Real yl = j == g.jlo ? my[bx](i, j, 0, n) : py[bx](i, j, 0, n);
        o[bx](i, j, 0, n) = (xl + mx[bx](i + 1, j, 0, n))/g.dx
                          + (yl + my[bx](i, j + 1, 0, n))/g.dy;
    });
    return out;
}

// per cell the interior nonconservative integral over each of its faces
template<class K, class P>
Field edge_source(K k, const Field& Q, const Field& QL, const Field& QR,
                  const Field& A, const P& p, const Mesh& M)
{
    Field out = cells_of(M, Q.nq);
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    auto lx = QL[0].const_arrays(); auto ly = QL[1].const_arrays();
    auto rx = QR[0].const_arrays(); auto ry = QR[1].const_arrays();
    amrex::ParallelFor(out[0], IntVect(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        const auto qc = col<K::n_in>(q[bx], i, j, nq);
        const auto ac = col<K::n_aux>(a[bx], i, j, na);
        Col<K::n_out> s{};
        for (int d = 0; d < 2; ++d)
            for (int hi = 0; hi < 2; ++hi)
            {
                const int fi = i + (d == 0 && hi), fj = j + (d == 1 && hi);
                const FaceInfo f = face_info(g, d, fi, fj);
                const bool own = f.oi == i && f.oj == j;
                const auto L = d == 0 ? lx[bx] : ly[bx];
                const auto R = d == 0 ? rx[bx] : ry[bx];
                const Real sg = own ? 1.0 : -1.0;
                const auto v = k(qc, own ? col<K::n_in>(L, fi, fj, nq) : col<K::n_in>(R, fi, fj, nq),
                                 ac, ac, p, normal<K::n_nrm>(sg*f.n0, sg*f.n1));
                const Real h = d == 0 ? g.dx : g.dy;
                for (int m = 0; m < K::n_out; ++m) s(m, 0) += v(m, 0)/h;
            }
        for (int m = 0; m < K::n_out && m < nq; ++m) o[bx](i, j, 0, m) = s(m, 0);
    });
    return out;
}

// ── whole-domain operations ─────────────────────────────────────────────────
// Qaux with the derivative rows of ``spec`` from the LSQ fit; boundary
// samples carry no change (no Qb) or the face value 2 Qb - Q (with Qb, only
// the derivatives of state ``rows``)
template<class S>
Field derivative_aux_impl(const S& spec, const Field& Q_, const Field& Qaux,
                          const Mesh& M, const Field* Qb, const Rows rows)
{
    const Field Q = need(Q_, M.R, M);
    const int gr = std::max(0, std::min(Q.ngv, Qb ? Qb->ngv : Q.ngv) - M.R);
    Field out = like(Qaux, Qaux.nq);
    out.ngv = std::min(Qaux.ngv, gr);
    amrex::MultiFab::Copy(out[0], Qaux[0], 0, 0, Qaux.nq, out.ng());
    const Geo g = M.g;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto qb = (Qb ? *Qb : Q)[0].const_arrays();
    const int mode = Qb ? 1 : 2;
    for (const auto& e : spec)
    {
        if (Qb)
        {
            bool in = false;
            for (int m = 0; m < rows.n; ++m) in = in || rows.r[m] == e[1];
            if (!in) continue;
        }
        const int mi = M.monomial(e[2], e[3]);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mi >= 0 && e[4] == 0,
            "derivative_aux: the derivative exceeds the LSQ degree");
        const int row = e[0], s = e[1];
        amrex::ParallelFor(out[0], IntVect(gr),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
        {
            if (inside(g, i, j)) o[bx](i, j, 0, row) = lsq(g, q[bx], qb[bx], mode, i, j, s, mi);
        });
    }
    return out;
}
template<class S>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M)
{ return derivative_aux_impl(spec, Q, Qaux, M, nullptr, Rows{}); }
template<class S, class... I>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M,
                     const Field& Qb, const std::tuple<I...>& rows)
{ return derivative_aux_impl(spec, Q, Qaux, M, &Qb, rows_of(rows)); }

// cell mask: an admissibility row < 0 or a non-finite value
template<class K, class P>
Field inadmissible(K k, const Field& Q, const Field& A, const P& p)
{
    Field out = like(Q, 1);
    out.ngv = std::min(Q.ngv, A.ngv);
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IntVect(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        const auto x = col<K::n_in>(q[bx], i, j, nq);
        const auto v = k(x, col<K::n_aux>(a[bx], i, j, na), p, 0.0, 0.0, Col<3>{});
        bool bad = false;
        for (int m = 0; m < K::n_out; ++m) bad = bad || v(m, 0) < 0.0;
        for (int m = 0; m < nq; ++m) bad = bad || !std::isfinite(q[bx](i, j, 0, m));
        o[bx](i, j, 0, 0) = bad ? 1.0 : 0.0;
    });
    return out;
}

// ── implicit solves ─────────────────────────────────────────────────────────
inline double norm2(const Field& a)
{
    return std::sqrt(amrex::MultiFab::Dot(a[0], 0, a[0], 0, a.nq, 0));
}

// x + h e_k (h = sqrt(eps) max(1, |x_k|)) and h
std::pair<Field, Field> perturb(const Field& x, int k);
// the per-cell Newton update from the residual r and the perturbed columns
Field newton_update(const Field& x, const Field& r, const std::vector<Field>& cols,
                    const std::vector<Field>& hs);

inline std::pair<Field, Field> perturb(const Field& x, int kk)
{
    Field xk = like(x, x.nq), h = like(x, 1);
    xk.ngv = h.ngv = x.ngv;
    auto o = xk[0].arrays(); auto oh = h[0].arrays();
    auto a = x[0].const_arrays();
    const int nq = x.nq;
    amrex::ParallelFor(xk[0], IntVect(xk.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        for (int n = 0; n < nq; ++n) o[bx](i, j, 0, n) = a[bx](i, j, 0, n);
        const Real v = a[bx](i, j, 0, kk);
        const Real s = 1.4901161193847656e-08*(std::abs(v) > 1.0 ? std::abs(v) : 1.0);
        oh[bx](i, j, 0, 0) = s;
        o[bx](i, j, 0, kk) = v + s;
    });
    return {xk, h};
}

inline Field newton_update(const Field& x, const Field& r, const std::vector<Field>& cols,
                           const std::vector<Field>& hs)
{
    const int n = x.nq;
    AMREX_ALWAYS_ASSERT(n <= MAXR);
    amrex::MultiFab J(x[0].boxArray(), x[0].DistributionMap(), n*n, 0), H(
        x[0].boxArray(), x[0].DistributionMap(), n, 0);
    for (int k = 0; k < n; ++k)
    {
        for (int i = 0; i < n; ++i) amrex::MultiFab::Copy(J, cols[k][0], i, i*n + k, 1, 0);
        amrex::MultiFab::Copy(H, hs[k][0], 0, k, 1, 0);
    }
    Field out = like(x, n);
    amrex::MultiFab::Copy(out[0], x[0], 0, 0, n, 0);
    auto o = out[0].arrays();
    auto jj = J.const_arrays(); auto hh = H.const_arrays();
    auto rr = r[0].const_arrays();
    amrex::ParallelFor(out[0], IntVect(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        Real A[MAXR*MAXR], b[MAXR];
        for (int p = 0; p < n; ++p)
        {
            b[p] = -rr[bx](i, j, 0, p);
            for (int k = 0; k < n; ++k)
                A[p*n + k] = (jj[bx](i, j, 0, p*n + k) - rr[bx](i, j, 0, p))/hh[bx](i, j, 0, k);
        }
        for (int c = 0; c < n; ++c)                 // partial pivoting
        {
            int piv = c;
            for (int p = c + 1; p < n; ++p)
                if (std::abs(A[p*n + c]) > std::abs(A[piv*n + c])) piv = p;
            for (int k = 0; k < n; ++k) { Real t = A[c*n + k]; A[c*n + k] = A[piv*n + k]; A[piv*n + k] = t; }
            { Real t = b[c]; b[c] = b[piv]; b[piv] = t; }
            for (int p = c + 1; p < n; ++p)
            {
                const Real m = A[p*n + c]/A[c*n + c];
                for (int k = c; k < n; ++k) A[p*n + k] -= m*A[c*n + k];
                b[p] -= m*b[c];
            }
        }
        for (int p = n - 1; p >= 0; --p)
        {
            for (int k = p + 1; k < n; ++k) b[p] -= A[p*n + k]*b[k];
            b[p] /= A[p*n + p];
            o[bx](i, j, 0, p) += b[p];
        }
    });
    return out;
}

// Newton on a cell-local residual f (x -> R, same shape): each cell's n x n
// Jacobian by one-sided differences, one variable at a time in all cells.
template<class F>
Field root(F f, Field x)
{
    const auto& o = implicit();
    Field r = f(x);
    int it = 0;
    for (; it < o.maxiter && norm2(r) > o.tol; ++it)
    {
        std::vector<Field> cols(x.nq), hs(x.nq);
        for (int k = 0; k < x.nq; ++k)
        {
            auto ph = perturb(x, k);
            cols[k] = f(ph.first);
            hs[k] = ph.second;
        }
        x = newton_update(x, r, cols, hs);
        r = f(x);
    }
    static bool warned = false;
    if (norm2(r) > o.tol && !warned)
    {
        warned = true;
        amrex::Print() << "root: not converged after " << it << " Newton updates, ||R|| = "
                       << norm2(r) << " (warned once)\n";
    }
    return x;
}

// amrex::GMRES on the printed linear map, right-preconditioned by the exact
// cell blocks of the operator (probed by colouring: cells (R+1) apart do not
// share a stencil), inverted per cell.
struct LinOp
{
    using RT = Real;
    std::function<Field(Field)> op;
    const Mesh* M = nullptr;
    int nq = 0;
    amrex::MultiFab Binv;

    amrex::MultiFab makeVecRHS() const { return amrex::MultiFab(M->ba, M->dm, nq, 0); }
    amrex::MultiFab makeVecLHS() const { return amrex::MultiFab(M->ba, M->dm, nq, M->NG); }
    void assign(amrex::MultiFab& l, const amrex::MultiFab& r) { amrex::MultiFab::Copy(l, r, 0, 0, nq, 0); }
    RT dotProduct(const amrex::MultiFab& a, const amrex::MultiFab& b) { return amrex::MultiFab::Dot(a, 0, b, 0, nq, 0); }
    void increment(amrex::MultiFab& l, const amrex::MultiFab& r, RT a) { amrex::MultiFab::Saxpy(l, a, r, 0, 0, nq, 0); }
    void linComb(amrex::MultiFab& l, RT a, const amrex::MultiFab& x, RT b, const amrex::MultiFab& y)
    { amrex::MultiFab::LinComb(l, a, x, 0, b, y, 0, 0, nq, 0); }
    RT norm2(const amrex::MultiFab& v) { return std::sqrt(amrex::MultiFab::Dot(v, 0, v, 0, nq, 0)); }
    void scale(amrex::MultiFab& v, RT f) { v.mult(f, 0, nq, 0); }
    void setToZero(amrex::MultiFab& v) { v.setVal(0.0); }

    Field wrap(const amrex::MultiFab& x) const
    {
        Field X;
        X.nq = nq;
        X.mf = std::make_shared<std::vector<amrex::MultiFab>>();
        X.mf->emplace_back(x, amrex::make_alias, 0, nq);
        return X;
    }
    void apply(amrex::MultiFab& Ax, const amrex::MultiFab& x)
    {
        const Field y = op(wrap(x));
        amrex::MultiFab::Copy(Ax, y[0], 0, 0, nq, 0);
    }
    void precond(amrex::MultiFab& l, const amrex::MultiFab& r);
    void blocks();
};

inline void LinOp::blocks()
{
    const int n = nq, S = M->R + 1;
    AMREX_ALWAYS_ASSERT(n <= MAXR);
    amrex::MultiFab A(M->ba, M->dm, n*n, 0);
    Binv.define(M->ba, M->dm, n*n, 0);
    const Geo g = M->g;
    for (int cx = 0; cx < S; ++cx)
        for (int cy = 0; cy < S; ++cy)
            for (int m = 0; m < n; ++m)
            {
                amrex::MultiFab e(M->ba, M->dm, n, M->NG);
                e.setVal(0.0);
                auto ea = e.arrays();
                amrex::ParallelFor(e, IntVect(0),
                    [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
                {
                    if ((i - g.ilo) % S == cx && (j - g.jlo) % S == cy) ea[bx](i, j, 0, m) = 1.0;
                });
                const Field y = op(wrap(e));
                auto ya = y[0].const_arrays();
                auto aa = A.arrays();
                amrex::ParallelFor(A, IntVect(0),
                    [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
                {
                    if ((i - g.ilo) % S != cx || (j - g.jlo) % S != cy) return;
                    for (int p = 0; p < n; ++p) aa[bx](i, j, 0, p*n + m) = ya[bx](i, j, 0, p);
                });
            }
    auto aa = A.const_arrays();
    auto bi = Binv.arrays();
    amrex::ParallelFor(Binv, IntVect(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        Real a[MAXR*MAXR], v[MAXR*MAXR];
        for (int e = 0; e < n*n; ++e) { a[e] = aa[bx](i, j, 0, e); v[e] = 0.0; }
        for (int d = 0; d < n; ++d) v[d*n + d] = 1.0;
        for (int c = 0; c < n; ++c)                 // Gauss-Jordan, partial pivoting
        {
            int piv = c;
            for (int p = c + 1; p < n; ++p) if (std::abs(a[p*n + c]) > std::abs(a[piv*n + c])) piv = p;
            for (int k = 0; k < n; ++k)
            {
                Real t = a[c*n + k]; a[c*n + k] = a[piv*n + k]; a[piv*n + k] = t;
                t = v[c*n + k]; v[c*n + k] = v[piv*n + k]; v[piv*n + k] = t;
            }
            const Real ip = a[c*n + c] != 0.0 ? 1.0/a[c*n + c] : 0.0;
            for (int k = 0; k < n; ++k) { a[c*n + k] *= ip; v[c*n + k] *= ip; }
            for (int p = 0; p < n; ++p)
            {
                if (p == c) continue;
                const Real f = a[p*n + c];
                for (int k = 0; k < n; ++k) { a[p*n + k] -= f*a[c*n + k]; v[p*n + k] -= f*v[c*n + k]; }
            }
        }
        for (int e = 0; e < n*n; ++e) bi[bx](i, j, 0, e) = v[e];
    });
}

inline void LinOp::precond(amrex::MultiFab& l, const amrex::MultiFab& r)
{
    const int n = nq;
    auto la = l.arrays();
    auto ra = r.const_arrays();
    auto bi = Binv.const_arrays();
    amrex::ParallelFor(l, IntVect(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int) noexcept
    {
        Real x[MAXR];
        for (int p = 0; p < n; ++p)
        {
            x[p] = 0.0;
            for (int k = 0; k < n; ++k) x[p] += bi[bx](i, j, 0, p*n + k)*ra[bx](i, j, 0, k);
        }
        for (int p = 0; p < n; ++p) la[bx](i, j, 0, p) = x[p];
    });
}

Field linsolve_impl(std::function<Field(Field)> op, const Field& b, const Mesh& M);

inline Field linsolve_impl(std::function<Field(Field)> op, const Field& b, const Mesh& M)
{
    const auto& o = implicit();
    LinOp L;
    L.op = std::move(op); L.M = &M; L.nq = b.nq;
    L.blocks();
    amrex::GMRES<amrex::MultiFab, LinOp> gm;
    gm.define(L);
    gm.setRestartLength(o.ls_restart);
    gm.setMaxIters(o.ls_restart*o.ls_maxiter);
    Field x = cells_of(M, b.nq);
    amrex::MultiFab rhs(M.ba, M.dm, b.nq, 0);
    amrex::MultiFab::Copy(rhs, b[0], 0, 0, b.nq, 0);
    const double nb = L.norm2(rhs);
    if (nb > 0.0) gm.solve(x[0], rhs, o.ls_tol, 0.0);
    if (o.ls_accept > 0 && nb > 0.0)
    {
        amrex::MultiFab Ax = L.makeVecRHS();
        L.apply(Ax, x[0]);
        amrex::MultiFab::Subtract(Ax, rhs, 0, 0, b.nq, 0);
        const double rel = L.norm2(Ax)/nb;
        if (rel > o.ls_accept)
            amrex::Abort("linsolve: relative residual " + std::to_string(rel)
                         + " > accept gate " + std::to_string(o.ls_accept));
    }
    if (std::getenv("ZS_KRYLOV_LOG"))
        amrex::Print() << "linsolve: " << gm.getNumIters() << " iterations, residual "
                       << gm.getResidualNorm()/(nb > 0 ? nb : 1.0) << "\n";
    return x;
}

// the printed linsolve(op, b) carries no mesh: the driver's (one per run)
inline const Mesh*& the_mesh() { static const Mesh* m = nullptr; return m; }

template<class F>
Field linsolve(F op, const Field& b)
{
    return linsolve_impl(std::function<Field(Field)>(op), b, *the_mesh());
}

// ── control ─────────────────────────────────────────────────────────────────
template<class C, class B, class S>
S while_loop(C c, B b, S s)
{
    while (std::apply(c, s)) s = std::apply(b, s);
    return s;
}

template<class A, class B>
auto cond(bool p, A a, B b) { return p ? a() : b(); }

// ── effects ─────────────────────────────────────────────────────────────────
inline std::function<void(double, const Field&, const Field&)>& writer()
{
    static std::function<void(double, const Field&, const Field&)> w;
    return w;
}

inline std::tuple<long> io_write(long tok, double time, const Field& Q, const Field& Qaux)
{
    if (writer()) writer()(time, Q, Qaux);
    return {tok + 1};
}

inline std::tuple<long> check_cfl(long tok, double dt_cfl, double dt)
{
    static bool warned = false;
    if (dt > dt_cfl*(1 + 1e-12) && !warned)
    {
        warned = true;
        amrex::Print() << "warning: time step " << dt << " exceeds the CFL step "
                       << dt_cfl << " (warned once)\n";
    }
    return {tok + 1};
}

} // namespace zs
