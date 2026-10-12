#pragma once
/// \file
/// The AMReX runtime of the symbolic solvers: what every name in the printed C++ (zoomy_core.symsolver.CxxPrinter) means on a single-level Cartesian AMReX grid of dimension AMREX_SPACEDIM.
/// Loops are amrex::ParallelFor, reductions ParReduce, the linear solve amrex::GMRES; what all C++ runtimes share is zoomy_core's rt_common.H.
/// Ref: Zhang et al. 2019, https://doi.org/10.21105/joss.01370
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
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define ZS_HD AMREX_GPU_HOST_DEVICE
#include "rt_common.H"

namespace zs
{
using amrex::Real;
using amrex::IntVect;
using IV = amrex::IntVect;
/// Spatial dimension of the build (the model's dimension).
constexpr int D = AMREX_SPACEDIM;
/// An N-column of Real (a static matrix, usable in device code).
template<int N> using Col = amrex::SmallMatrix<Real, (N > 0 ? N : 1), 1>;
constexpr int MAXR = 16;      // most rows a static row list carries

/// Declare IV c, the cell of the current ParallelFor iteration.
#define ZS_CELL IV c(AMREX_D_DECL(i, j, k))

/// Unit IntVect e_d.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE IV unit(int d)
{
    IV u(0);
    u[d] = 1;
    return u;
}

/// Print a warning (amrex::Print: I/O rank only).
inline void warn(const std::string& message) { amrex::Print() << "warning: " << message << "\n"; }

// ── mesh ────────────────────────────────────────────────────────────────────
/// Geo: what a device loop needs (trivially copyable): domain box, periodicity, spacing, BC branch per side, and the LSQ tables (per-class sample lists, weights per monomial, closure classes).
struct Geo
{
    int lo[3], hi[3], per[3];
    Real dx[3], plo[3];
    int branch[6];                     // BC branch: x-lo, x-hi, y-lo, y-hi, z-lo, z-hi
    int ncls[3], nmon, gmon[3];        // gmon: the monomial of d/dx_a (or -1)
    int ncl, nrcl;
    const int* cls[3]; const int* off; const int* smp; const int* rcl;
    const Real* w;
};

struct Field;
/// Mesh: the grid (Geometry, BoxArray, DistributionMapping, NG ghost cells, LSQ radius R), the AMR hooks (halo fill, refluxing log) and the LSQ tables behind Geo.
struct Mesh
{
    amrex::Geometry geom;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    int NG = 2, R = 1;
    // an AMR level: ``fill`` = its halo (ghosts from its own boxes and from the
    // coarser level), ``log`` = the face terms of each divergence (refluxing)
    std::function<Field(const Field&)> fill;
    std::shared_ptr<std::vector<std::array<Field, 2>>> log;
    std::vector<std::array<int, 3>> mon;
    amrex::Gpu::DeviceVector<int> cls[3], off, smp, rcl;
    amrex::Gpu::DeviceVector<Real> w;
    Geo g{};

    /// Read the LSQ table written by the driver (zoomy_amrex.symsolver.lsq_table) into Geo.
    void load(const std::string& path, const int branch[6])
    {
        std::ifstream f(path);
        int dim, n[3] = {1, 1, 1}, nc[3] = {1, 1, 1}, nmon, ncl;
        f >> R >> dim;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dim == D, "rt_amrex: LSQ table of another dimension");
        for (int a = 0; a < D; ++a) f >> n[a];
        for (int a = 0; a < D; ++a) f >> nc[a];
        f >> nmon >> ncl;
        NG = R + 1;
        std::vector<int> cv[3], of, sm, rc;
        std::vector<Real> ww;
        for (int a = 0; a < D; ++a) { cv[a].resize(n[a]); for (auto& v : cv[a]) f >> v; }
        mon.resize(nmon);
        for (auto& m : mon) { m = {0, 0, 0}; for (int a = 0; a < D; ++a) f >> m[a]; }
        int nrc; f >> nrc;
        rc.resize(nrc);
        for (auto& v : rc) f >> v;
        int nsamp = 0;
        for (int c = 0; c < nc[0]*nc[1]*nc[2]; ++c)
            for (int l = 0; l <= ncl; ++l)
            {
                int ns; f >> ns;
                if (ns < 0)             // an inactive closure class: the base list
                {
                    const int b = of.size() - 2*l;
                    of.push_back(of[b]); of.push_back(of[b + 1]);
                    continue;
                }
                of.push_back(nsamp);
                for (int s = 0; s < ns; ++s)
                {
                    int kind, o[3] = {0, 0, 0}; f >> kind;
                    for (int a = 0; a < D; ++a) f >> o[a];
                    sm.insert(sm.end(), {kind, o[0], o[1], o[2]});
                    for (int k = 0; k < nmon; ++k) { Real x; f >> x; ww.push_back(x); }
                }
                nsamp += ns;
                of.push_back(nsamp);
            }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f.good(), "rt_amrex: bad LSQ table");
        auto up = [](auto& dv, const auto& hv)
        {
            dv.resize(hv.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice, hv.begin(), hv.end(), dv.begin());
        };
        for (int a = 0; a < D; ++a) up(cls[a], cv[a]);
        up(off, of); up(smp, sm); up(w, ww); up(rcl, rc);
        const amrex::Box& d = geom.Domain();
        for (int a = 0; a < 3; ++a)
        {
            g.lo[a] = a < D ? d.smallEnd(a) : 0;
            g.hi[a] = a < D ? d.bigEnd(a) : 0;
            g.per[a] = a < D ? int(geom.isPeriodic(a)) : 0;
            g.dx[a] = a < D ? geom.CellSize(a) : 1.0;
            g.plo[a] = a < D ? geom.ProbLo(a) : 0.0;
            g.ncls[a] = nc[a];
            g.cls[a] = a < D ? cls[a].data() : nullptr;
            g.gmon[a] = a < D ? monomial(a == 0, a == 1, a == 2) : -1;
        }
        for (int s = 0; s < 6; ++s) g.branch[s] = branch[s];
        g.nmon = nmon; g.ncl = ncl; g.nrcl = nrc;
        g.off = off.data(); g.smp = smp.data(); g.w = w.data(); g.rcl = rcl.data();
    }
    /// Index of the monomial x^a y^b z^c in the LSQ table, or -1.
    int monomial(int a, int b, int c) const
    {
        for (int k = 0; k < int(mon.size()); ++k)
            if (mon[k][0] == a && mon[k][1] == b && mon[k][2] == c) return k;
        return -1;
    }
};

// ── device geometry helpers ─────────────────────────────────────────────────
/// True if cell c lies in the domain or across a periodic side (its partner cell).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool inside(const Geo& g, const IV& c)
{
    for (int d = 0; d < D; ++d)
        if (!g.per[d] && (c[d] < g.lo[d] || c[d] > g.hi[d])) return false;
    return true;
}

/// Face c of direction d (between cells c - e_d and c): its owner (the lower cell; the inner cell on a boundary face), the other cell, a boundary flag and the owner's outward normal.
struct FaceInfo { IV o, x; bool bnd; Real n[3]; };
/// The FaceInfo of face c of direction d (unstructured convention of the JAX/foam runtimes).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FaceInfo face_info(const Geo& g, int d, const IV& c)
{
    const IV l = c - unit(d);
    const bool blo = !g.per[d] && c[d] == g.lo[d];
    const bool bhi = !g.per[d] && c[d] == g.hi[d] + 1;
    FaceInfo f;
    f.bnd = blo || bhi;
    f.o = blo ? c : l;
    f.x = blo ? l : c;
    for (int a = 0; a < 3; ++a) f.n[a] = a == d ? (blo ? -1.0 : 1.0) : 0.0;
    return f;
}

/// Centre of cell c.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<3> cell_centre(const Geo& g, const IV& c)
{
    Col<3> X{};
    for (int a = 0; a < D; ++a) X(a, 0) = g.plo[a] + (c[a] + 0.5)*g.dx[a];
    return X;
}

/// Centre of face c of direction d (the face between cells c - e_d and c).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<3> face_centre(const Geo& g, int d, const IV& c)
{
    Col<3> X{};
    for (int a = 0; a < D; ++a) X(a, 0) = g.plo[a] + (c[a] + (a == d ? 0.0 : 0.5))*g.dx[a];
    return X;
}

/// The first N components of s * v as a column.
template<int N>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<N> normal(const Real (&v)[3], Real s = 1.0)
{
    Col<N> n{};
    for (int a = 0; a < N && a < 3; ++a) n(a, 0) = s*v[a];
    return n;
}

/// The first min(N, nq) rows of a at cell c as a column.
template<int N, class A>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<N> col(const A& a, const IV& c, int nq)
{
    Col<N> v{};
    for (int q = 0; q < N && q < nq; ++q) v(q, 0) = a(c, q);
    return v;
}

// ── fields ──────────────────────────────────────────────────────────────────
/// Device view of a Field: entry (cell c, row n) in box bx; a one-row field broadcasts.
struct DLeaf
{
    amrex::MultiArray4<Real const> a;
    int nq;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    { return a[bx](c, nq == 1 ? 0 : n); }
};

/// A Field: nq rows over the cells (one MultiFab with ghosts) or the faces (one MultiFab per direction); ngv = ghost layers holding valid data. Immutable once built (shared storage).
struct Field
{
    int nq = 0;
    bool face = false;
    int ngv = 0;
    std::shared_ptr<std::vector<amrex::MultiFab>> mf;
    std::shared_ptr<void> keep;        // the parent of an alias
    /// The MultiFab of direction d (0 for cells).
    amrex::MultiFab& operator[](int d) const { return (*mf)[d]; }
    /// Number of MultiFabs: D for faces, 1 for cells.
    int ncomp() const { return face ? D : 1; }
    /// Ghost layers allocated.
    int ng() const { return face ? 0 : (*mf)[0].nGrow(); }
    /// Device view of the MultiFab d.
    DLeaf dev(int d) const { return {(*mf)[d].const_arrays(), nq}; }
    /// Itself, as the shape of an expression.
    const Field& shape() const { return *this; }
};

/// Storage is recycled: a dead Field's MultiFabs go to a pool and the next Field of the same shape takes them (no allocation, no blocking Array4-table upload); close_pool() empties it before amrex::Finalize.
using MFs = std::vector<amrex::MultiFab>;
/// A pooled MultiFab set with the BoxArray and mapping it was made for.
struct Pooled { MFs* mf; amrex::BoxArray ba; amrex::DistributionMapping dm; };
/// The free sets, keyed by (rows, face, ghosts).
inline std::map<std::tuple<int, bool, int>, std::vector<Pooled>>& pool()
{
    static auto* p = new std::map<std::tuple<int, bool, int>, std::vector<Pooled>>();
    return *p;
}
/// False once close_pool() has run (later frees delete instead of pooling).
inline bool& pool_open() { static bool o = true; return o; }
/// Empty the pool; call before amrex::Finalize.
inline void close_pool()
{
    pool_open() = false;
    for (auto& kv : pool())
        for (auto& e : kv.second) delete e.mf;
    pool().clear();
}

/// A Field of nq rows on (ba, dm), reusing a pooled MultiFab set when one of that shape is free.
inline Field make_like(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                       int nq, bool face, int ng)
{
    BL_PROFILE("zs::make_like");
    Field F;
    F.nq = nq; F.face = face;
    const auto key = std::make_tuple(std::max(nq, 1), face, face ? 0 : ng);
    MFs* m = nullptr;
    auto& free = pool()[key];
    for (size_t k = 0; k < free.size(); ++k)
        if (free[k].ba == ba && free[k].dm == dm)
        {
            m = free[k].mf;
            free.erase(free.begin() + long(k));
            break;
        }
    if (!m)
    {
        m = new MFs();
        for (int d = 0; d < (face ? D : 1); ++d)
        {
            const amrex::BoxArray b = face ? amrex::convert(ba, IV::TheDimensionVector(d)) : ba;
            m->emplace_back(b, dm, std::max(nq, 1), face ? 0 : ng);
        }
    }
#ifdef ZS_ZERO_INIT
    for (auto& x : *m) x.setVal(0.0);      // debugging: no block reads unset entries
#endif
    F.mf = std::shared_ptr<MFs>(m, [key, ba, dm](MFs* p)
    {
        if (pool_open()) pool()[key].push_back(Pooled{p, ba, dm});
        else delete p;
    });
    return F;
}
/// A cell Field of nq rows on the mesh's grid.
inline Field cells_of(const Mesh& M, int nq) { return make_like(M.ba, M.dm, nq, false, M.NG); }
/// A face Field of nq rows on the mesh's grid.
inline Field faces_of(const Mesh& M, int nq) { return make_like(M.ba, M.dm, nq, true, 0); }
/// A Field of nq rows with the layout of a.
inline Field like(const Field& a, int nq)
{
    const amrex::MultiFab& m = a[0];
    amrex::BoxArray ba = m.boxArray();
    if (a.face) ba.enclosedCells();
    Field F = make_like(ba, m.DistributionMap(), nq, a.face, a.face ? 0 : m.nGrow());
    return F;
}

/// Rows first .. first + n of F as an alias (no copy).
inline Field rows_alias(const Field& F, int first, int n)
{
    Field A;
    A.nq = n; A.face = F.face; A.ngv = F.ngv; A.keep = F.mf;
    A.mf = std::make_shared<std::vector<amrex::MultiFab>>();
    for (int d = 0; d < F.ncomp(); ++d) A.mf->emplace_back(F[d], amrex::make_alias, first, n);
    return A;
}

// ── elementwise arithmetic (rt_common.H's functors; one row broadcasts) ─────
/// Device leaf: the number s at every entry.
struct DNum
{
    Real s;
    AMREX_GPU_HOST_DEVICE Real operator()(int, const IV&, int) const { return s; }
};
/// Device node: op applied to the entries of a and b.
template<class Op, class A, class B> struct DBin
{
    Op op; A a; B b;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    { return op(a(bx, c, n), b(bx, c, n)); }
};
/// Host leaf: a number (no shape, nq = 0, ngv unbounded).
struct Num
{
    Real s;
    int nq = 0, ngv = 1 << 20;
    DNum dev(int) const { return {s}; }
};
/// The Field an expression takes its shape from (nullptr for a number).
inline const Field* shape_of(const Field& f) { return &f; }
/// Device view of a field read entry by entry, rows as stored (no one-row broadcast).
struct DRaw
{
    amrex::MultiArray4<Real const> a;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    { return a[bx](c, n); }
};
/// A Field as an expression leaf read without the one-row broadcast.
struct DLeafless
{
    Field f;
    int nq = f.nq, ngv = f.ngv;
    DRaw dev(int d) const { return {f[d].const_arrays()}; }
};
/// A number has no shape.
inline const Field* shape_of(const Num&) { return nullptr; }

template<class X> Field eval(const X& x);
/// Expression node op(a, b); arithmetic on Fields is lazy and evaluated by one ParallelFor where a block takes a Field (eval) or a reduction reads it.
template<class Op, class A, class B> struct XBin
{
    Op op; A a; B b;
    int nq, ngv;
    XBin(Op o, const A& a_, const B& b_)
        : op(o), a(a_), b(b_), nq(std::max(a_.nq, b_.nq)), ngv(std::min(a_.ngv, b_.ngv)) {}
    auto dev(int d) const
    { return DBin<Op, decltype(a.dev(d)), decltype(b.dev(d))>{op, a.dev(d), b.dev(d)}; }
    const Field& shape() const { const Field* f = shape_of(a); return f ? *f : *shape_of(b); }
    operator Field() const { return eval(*this); }
};
/// The shape of an expression node.
template<class Op, class A, class B> const Field* shape_of(const XBin<Op, A, B>& x) { return &x.shape(); }

/// Evaluate an expression into a new Field with one ParallelFor.
template<class X> Field eval(const X& x)
{
    BL_PROFILE("zs::eval");
    Field r = like(x.shape(), x.nq);
    r.ngv = std::min(x.ngv, r.ng());
    for (int d = 0; d < r.ncomp(); ++d)
    {
        const auto f = x.dev(d);
        auto ra = r[d].arrays();
        amrex::ParallelFor(r[d], IV(r.ng()), r.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        { ZS_CELL; ra[bx](c, n) = f(bx, c, n); });
    }
    return r;
}

/// Trait: T is a Field or an expression node.
template<class T> struct is_x : std::false_type {};
template<> struct is_x<Field> : std::true_type {};
template<class Op, class A, class B> struct is_x<XBin<Op, A, B>> : std::true_type {};
/// SFINAE guards: if_num<T> an arithmetic T, if_x<A, B> an operand pair with at least one expression.
template<class T> using if_num = std::enable_if_t<std::is_arithmetic_v<T>, int>;
template<class A, class B> using if_x = std::enable_if_t<
    (is_x<A>::value || is_x<B>::value) && (is_x<A>::value || std::is_arithmetic_v<A>)
    && (is_x<B>::value || std::is_arithmetic_v<B>), int>;
/// Wrap a number as a Num leaf; expressions pass through.
template<class T> auto wrap(const T& t)
{
    if constexpr (std::is_arithmetic_v<T>) return Num{Real(t)};
    else return t;
}
/// Build the expression node op(a, b).
template<class Op, class A, class B> auto xbin(Op op, const A& a, const B& b)
{ return XBin<Op, decltype(wrap(a)), decltype(wrap(b))>(op, wrap(a), wrap(b)); }
/// A block argument: a number as is, an expression evaluated into a Field.
template<class T> auto as_arg(const T& t)
{
    if constexpr (std::is_arithmetic_v<T>) return t;
    else return Field(t);
}

/// Lazy elementwise arithmetic and functions on expressions and numbers.
template<class A, class B, if_x<A, B> = 0> auto operator+(const A& a, const B& b) { return xbin(Add{}, a, b); }
template<class A, class B, if_x<A, B> = 0> auto operator-(const A& a, const B& b) { return xbin(Sub{}, a, b); }
template<class A, class B, if_x<A, B> = 0> auto operator*(const A& a, const B& b) { return xbin(Mul{}, a, b); }
template<class A, class B, if_x<A, B> = 0> auto operator/(const A& a, const B& b) { return xbin(Div{}, a, b); }
template<class A, std::enable_if_t<is_x<A>::value, int> = 0> auto operator-(const A& a) { return xbin(Mul{}, a, -1.0); }
template<class A, class T, if_x<A, T> = 0> auto pow(const A& a, T e) { return xbin(Pow{}, a, e); }
template<class A, std::enable_if_t<is_x<A>::value, int> = 0> auto abs(const A& a) { return xbin(Abs{}, a, 0.0); }
using std::abs;
using std::floor;
using std::fmod;
using std::pow;

/// Lazy elementwise minimum.
template<class A, class B, if_x<A, B> = 0> auto minimum(const A& a, const B& b) { return xbin(Min{}, a, b); }

// ── halo ────────────────────────────────────────────────────────────────────
/// Fill the ghost layers: copies of the neighbouring boxes' cells and the periodic partners (any rank); a number passes through.
inline Field halo(const Field& X, const Mesh& M)
{
    BL_PROFILE("zs::halo");
    if (X.face || X.ngv >= X.ng()) return X;
    if (M.fill) return M.fill(X);
    X[0].FillBoundary(M.geom.periodicity());   // ghosts only: no value changes
    Field Y = X;
    Y.ngv = X.ng();
    return Y;
}
/// A number needs no halo.
template<class T, if_num<T> = 0> T halo(T x, const Mesh&) { return x; }
/// X with at least n valid ghost layers (halo-exchanged when too thin).
inline Field need(const Field& X, int n, const Mesh& M) { return X.ngv >= n ? X : halo(X, M); }

// ── reductions (valid region, all ranks; NaN propagates) ────────────────────
/// Min or max over the valid region of all rows and ranks (ParReduce); a NaN anywhere gives NaN.
template<bool MIN, class X>
double reduce_ext(const X& x)
{
    BL_PROFILE("zs::reduce_ext");
    using Op = std::conditional_t<MIN, amrex::ReduceOpMin, amrex::ReduceOpMax>;
    constexpr Real init = MIN ? std::numeric_limits<Real>::infinity()
                              : -std::numeric_limits<Real>::infinity();
    double best = init;
    bool nan = false;
    const int nq = x.nq;
    const Field& F = x.shape();
    for (int d = 0; d < F.ncomp(); ++d)
    {
        const auto a = x.dev(d);
        auto r = amrex::ParReduce(amrex::TypeList<Op, amrex::ReduceOpLogicalOr>{},
                                  amrex::TypeList<Real, int>{}, F[d], IV(0),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
                -> amrex::GpuTuple<Real, int>
        {
            ZS_CELL;
            Real e = init;
            int bad = 0;
            for (int n = 0; n < nq; ++n)
            {
                const Real v = a(bx, c, n);
                if (v != v) bad = 1;
                else e = (MIN ? v < e : v > e) ? v : e;
            }
            return {e, bad};
        });
        best = MIN ? std::min(best, double(amrex::get<0>(r))) : std::max(best, double(amrex::get<0>(r)));
        nan = nan || amrex::get<1>(r);
    }
    if (MIN) amrex::ParallelDescriptor::ReduceRealMin(best);
    else amrex::ParallelDescriptor::ReduceRealMax(best);
    amrex::ParallelDescriptor::ReduceBoolOr(nan);
    return nan ? NaN : best;
}
/// Global minimum of an expression.
template<class X, std::enable_if_t<is_x<X>::value, int> = 0>
double reduce_min(const X& x) { return reduce_ext<true>(x); }
/// Global maximum of an expression.
template<class X, std::enable_if_t<is_x<X>::value, int> = 0>
double reduce_max(const X& x) { return reduce_ext<false>(x); }
/// True if any entry is nonzero on any rank.
inline bool reduce_any(const Field& F)
{
    BL_PROFILE("zs::reduce_any");
    const int nq = F.nq;
    bool any = false;
    for (int d = 0; d < F.ncomp(); ++d)
    {
        auto a = F[d].const_arrays();
        any = any || amrex::ParReduce(amrex::TypeList<amrex::ReduceOpLogicalOr>{},
                                      amrex::TypeList<int>{}, F[d], IV(0),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
                -> amrex::GpuTuple<int>
        {
            ZS_CELL;
            int r = 0;
            for (int n = 0; n < nq; ++n) r = r || a[bx](c, n) != 0.0;
            return {r};
        });
    }
    amrex::ParallelDescriptor::ReduceBoolOr(any);
    return any;
}
/// A number reduces to itself.
template<class T, if_num<T> = 0> double reduce_min(T x) { return x; }
template<class T, if_num<T> = 0> double reduce_max(T x) { return x; }

// ── static row lists ────────────────────────────────────────────────────────
/// A static list of up to MAXR row indices.
struct Rows { int n = 0; int r[MAXR]; };
/// The row list of an index tuple.
template<class... I> Rows rows_of(const std::tuple<I...>& t)
{
    Rows R;
    std::apply([&](auto... i) { ((R.r[R.n++] = int(i)), ...); }, t);
    return R;
}

/// Device view of a row subset (put / take are lazy too): row n is row R.r[n] of the field.
struct DRows
{
    DLeaf a;
    Rows R;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    { return a.a[bx](c, R.r[n]); }
};
/// Expression node for take: a row view of src, no copy.
struct XRows
{
    Field src;
    Rows R;
    int nq, ngv;
    DRows dev(int d) const { return {src.dev(d), R}; }
    const Field& shape() const { return src; }
    operator Field() const { return eval(*this); }
};
template<> struct is_x<XRows> : std::true_type {};
/// The shape of a take: its source.
inline const Field* shape_of(const XRows& x) { return &x.src; }

/// The rows `rows` of Q as a lazy row view.
template<class... I>
XRows take(const Field& Q, const std::tuple<I...>& rows)
{
    const Rows R = rows_of(rows);
    return XRows{Q, R, R.n, Q.ngv};
}

/// Device view of put: v for the rows in R, q elsewhere.
template<class V> struct DPut
{
    DLeaf q; V v; Rows R;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    {
        for (int m = 0; m < R.n; ++m) if (R.r[m] == n) return v(bx, c, m);
        return q.a[bx](c, n);
    }
};
/// Expression node for put: Q with rows R replaced by v, evaluated in the loop that copies the others.
template<class V> struct XPut
{
    Field Q; V v; Rows R;
    int nq, ngv;
    auto dev(int d) const { return DPut<decltype(v.dev(d))>{Q.dev(d), v.dev(d), R}; }
    const Field& shape() const { return Q; }
    operator Field() const { return eval(*this); }
};
template<class V> struct is_x<XPut<V>> : std::true_type {};
/// The shape of a put: Q.
template<class V> const Field* shape_of(const XPut<V>& x) { return &x.Q; }

/// Q with the rows `rows` replaced by `values`: lazy on the GPU (read by the consumer), one pass on the CPU.
template<class V, class... I>   // nvcc: the pack last (device lambda inside)
auto put(const Field& Q, const std::tuple<I...>& rows, const V& values)
{
    const auto v = wrap(values);
    const XPut<decltype(v)> x{Q, v, rows_of(rows), Q.nq, std::min(Q.ngv, v.ngv)};
#ifdef AMREX_USE_GPU
    return x;            // GPU: read by the consumer (memory-bound: no pass)
#else
    BL_PROFILE("zs::put");   // CPU: one pass; a re-read expression costs more there
    Field out = like(Q, Q.nq);
    out.ngv = x.ngv;
    const Rows R = x.R;
    for (int d = 0; d < Q.ncomp(); ++d)
    {
        const auto q = Q.dev(d);
        const auto f = v.dev(d);
        auto o = out[d].arrays();
        amrex::ParallelFor(out[d], IV(out.ng()), out.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            int s = -1;
            for (int m = 0; m < R.n; ++m) if (R.r[m] == n) s = m;
            o[bx](c, n) = s < 0 ? q.a[bx](c, n) : f(bx, c, s);
        });
    }
    return out;
#endif
}

/// A cell Field of n rows of zeros.
inline Field zeros(int n, const Mesh& M)
{
    Field Z = cells_of(M, n);
    Z[0].setVal(0.0);
    Z.ngv = M.NG;
    return Z;
}

/// Set the entries of `rows` in [-tol, 0) to 0 (round-off below zero).
template<class X, class... I>
Field clean_round_off(const X& x, const std::tuple<I...>& rows, double tol)
{
    BL_PROFILE("zs::clean_round_off");
    const Rows R = rows_of(rows);
    const Field& Q = x.shape();
    Field out = like(Q, x.nq);
    out.ngv = std::min(x.ngv, out.ng());
    for (int d = 0; d < Q.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        const auto f = x.dev(d);
        amrex::ParallelFor(out[d], IV(out.ng()), out.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            Real v = f(bx, c, n);
            for (int m = 0; m < R.n; ++m)
                if (R.r[m] == n && v < 0.0 && v >= -tol) v = 0.0;
            o[bx](c, n) = v;
        });
    }
    return out;
}

/// Device read of a Field row or a number, a one-row field broadcasting (operand of select).
struct Src
{
    amrex::MultiArray4<Real const> a;
    int nq = 1;
    Real s = 0.0;
    bool num = true;
    AMREX_GPU_HOST_DEVICE Real operator()(int bx, const IV& c, int n) const
    { return num ? s : a[bx](c, nq == 1 ? 0 : n); }
};
/// The Src of direction d of a Field, or of a number.
inline Src src(const Field& F, int d) { return Src{F[d].const_arrays(), F.nq, 0.0, false}; }
template<class T, if_num<T> = 0> Src src(T x, int) { Src s; s.s = Real(x); return s; }
/// Row count of a field (1 for a number).
inline int rows_n(const Field& F) { return F.nq; }
template<class T, if_num<T> = 0> int rows_n(T) { return 1; }
/// Valid ghost layers of a field (unbounded for a number).
inline int ngv_of(const Field& F) { return F.ngv; }
template<class T, if_num<T> = 0> int ngv_of(T) { return 1 << 20; }

/// select(mask, A, B): A where mask is nonzero, else B (fields broadcast, numbers too).
template<class A, class B>
Field select(const Field& m, const A& a_, const B& b_)
{
    BL_PROFILE("zs::select");
    const auto a = as_arg(a_);
    const auto b = as_arg(b_);
    Field out = like(m, std::max({m.nq, rows_n(a), rows_n(b)}));
    out.ngv = std::min({m.ngv, ngv_of(a), ngv_of(b)});
    for (int d = 0; d < m.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        const Src M = src(m, d), Sa = src(a, d), Sb = src(b, d);
        amrex::ParallelFor(out[d], IV(out.ng()), out.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            o[bx](c, n) = M(bx, c, n) != 0.0 ? Sa(bx, c, n) : Sb(bx, c, n);
        });
    }
    return out;
}

// ── kernels on cells and faces ──────────────────────────────────────────────
/// Apply a cell-family kernel (Q, Qaux, p, time, dt, position) in every cell, ghosts included.
template<class K, class P, class T1, class T2>
Field cells(K kn, const Field& Q, const Field& A, const P& p, T1 time_, T2 dt_,
            const Mesh& M)
{
    BL_PROFILE("zs::cells");
    Field out = cells_of(M, K::n_out);
    out.ngv = std::min(Q.ngv, A.ngv);
    const Real time = time_, dt = dt_;
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IV(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        const auto v = kn(col<K::n_in>(q[bx], c, nq), col<K::n_aux>(a[bx], c, na),
                          p, time, dt, cell_centre(g, c));
        for (int r = 0; r < K::n_out; ++r) o[bx](c, r) = v(r, 0);
    });
    return out;
}

/// Call a kernel pointwise: a recon-family kernel takes (W, Qaux, p), a cell-family one gets time = dt = 0 and no position.
template<class K, class X, class A, class P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
auto call_pointwise(const K& kn, const X& x, const A& a, const P& p)
{
    if constexpr (K::family == 3) return kn(x, a, p);
    else return kn(x, a, p, 0.0, 0.0, Col<3>{});
}

/// Column of the first min(N, nq) rows of an expression at cell c.
template<int N, class F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Col<N> colx(const F& f, int bx, const IV& c, int nq)
{
    Col<N> v{};
    for (int q = 0; q < N && q < nq; ++q) v(q, 0) = f(bx, c, q);
    return v;
}

/// Apply a kernel in every column (cells or faces), without time, step or position; X and A may be expressions (read in the same loop).
template<class K, class X, class A, class P>
Field pointwise(K kn, const X& X_, const A& A_, const P& p)
{
    BL_PROFILE("zs::pointwise");
    const auto x = wrap(X_);
    const auto a = wrap(A_);
    Field out = like(x.shape(), K::n_out);
    out.ngv = std::min(x.ngv, a.ngv);
    const int nq = x.nq, na = a.nq;
    for (int d = 0; d < out.ncomp(); ++d)
    {
        auto o = out[d].arrays();
        if constexpr (std::is_same_v<X, Field> && std::is_same_v<A, Field>)
        {   // fields: plain array reads (the CPU compiler vectorises them)
            auto xa = x[d].const_arrays();
            auto aa = a[d].const_arrays();
            amrex::ParallelFor(out[d], IV(out.ng()),
                [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
            {
                ZS_CELL;
                const auto v = call_pointwise(kn, col<K::n_in>(xa[bx], c, nq),
                                              col<K::n_aux>(aa[bx], c, na), p);
                for (int r = 0; r < K::n_out; ++r) o[bx](c, r) = v(r, 0);
            });
            continue;
        }
        const auto xf = x.dev(d);
        const auto af = a.dev(d);
        amrex::ParallelFor(out[d], IV(out.ng()),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            const auto v = call_pointwise(kn, colx<K::n_in>(xf, bx, c, nq),
                                          colx<K::n_aux>(af, bx, c, na), p);
            for (int r = 0; r < K::n_out; ++r) o[bx](c, r) = v(r, 0);
        });
    }
    return out;
}

/// Split F into N equal blocks of rows (aliases).
template<int N>
std::array<Field, N> split(const Field& F)
{
    std::array<Field, N> out;
    for (int g = 0; g < N; ++g) out[g] = rows_alias(F, g*(F.nq/N), F.nq/N);
    return out;
}

/// Apply the face kernel on every face, from the owner to the other cell, with the owner's normal.
template<class K, class P>
Field face_kernel(K kn, const Field& QL, const Field& QR, const Field& AL, const Field& AR,
                  const P& p, const Mesh& M)
{
    BL_PROFILE("zs::face_kernel");
    Field out = faces_of(M, K::n_out);
    const Geo g = M.g;
    const int nq = QL.nq, na = AL.nq;
    for (int d = 0; d < D; ++d)
    {
        auto o = out[d].arrays();
        auto l = QL[d].const_arrays();
        auto r = QR[d].const_arrays();
        auto al = AL[d].const_arrays();
        auto ar = AR[d].const_arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            const auto v = kn(col<K::n_in>(l[bx], c, nq), col<K::n_in>(r[bx], c, nq),
                              col<K::n_aux>(al[bx], c, na), col<K::n_aux>(ar[bx], c, na),
                              p, normal<K::n_nrm>(f.n));
            for (int m = 0; m < K::n_out; ++m) o[bx](c, m) = v(m, 0);
        });
    }
    return out;
}
/// face_kernel, split into N row blocks when N > 1.
template<int N = 1, class K, class P>
auto faces(K kn, const Field& QL, const Field& QR, const Field& AL, const Field& AR,
           const P& p, const Mesh& M)
{
    const Field out = face_kernel(kn, QL, QR, AL, AR, p, M);
    if constexpr (N == 1) return out;
    else return split<N>(out);
}

/// Owner and other cell aux per face (the inner cell twice at a boundary face).
inline std::tuple<Field, Field> face_aux(const Field& Qaux_, const Mesh& M)
{
    BL_PROFILE("zs::face_aux");
    const Field Qaux = need(Qaux_, 1, M);
    Field L = faces_of(M, Qaux.nq), R = faces_of(M, Qaux.nq);
    const Geo g = M.g;
    const int na = Qaux.nq;
    auto a = Qaux[0].const_arrays();
    for (int d = 0; d < D; ++d)
    {
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IV(0), na,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            l[bx](c, n) = a[bx](f.o, n);
            r[bx](c, n) = f.bnd ? a[bx](f.o, n) : a[bx](f.x, n);
        });
    }
    return {L, R};
}

/// Max |eigenvalue| per face over its two cells (the inner cell twice at a boundary face); NaN propagates.
template<class K, class P>
Field face_wavespeed(K kn, const Field& Q_, const Field& A_, const P& p, const Mesh& M)
{
    BL_PROFILE("zs::face_wavespeed");
    const Field Q = need(Q_, 1, M), A = need(A_, 1, M);
    Field out = faces_of(M, 1);
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    for (int d = 0; d < D; ++d)
    {
        auto o = out[d].arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            const auto n = normal<K::n_nrm>(f.n);
            auto peak = [&](const IV& cc)
            {
                const auto v = kn(col<K::n_in>(q[bx], cc, nq), col<K::n_aux>(a[bx], cc, na), p, n);
                Real s = 0.0;
                for (int m = 0; m < K::n_out; ++m)
                {
                    const Real x = std::abs(v(m, 0));
                    s = (x != x || s != s) ? NaN : (x > s ? x : s);
                }
                return s;
            };
            const Real s0 = peak(f.o);
            const Real s1 = f.bnd ? s0 : peak(f.x);
            o[bx](c, 0) = (s0 != s0 || s1 != s1) ? NaN : (s0 > s1 ? s0 : s1);
        });
    }
    return out;
}

/// Cell size per face: the inradius as zoomy_core's mesh defines it (shortest normal distance from the cell centre to a face plane), min dx / 2 of the uniform cells, one number.
inline Real face_inradius(const Mesh& M)
{
    Real h = M.g.dx[0];
    for (int d = 1; d < D; ++d) h = std::min(h, M.g.dx[d]);
    return 0.5*h;
}

// ── boundary ────────────────────────────────────────────────────────────────
/// True if c is an out-of-domain ghost cell across exactly one boundary face; gives its direction d, side (2 d + upper) and inner cell.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool boundary_ghost(const Geo& g, const IV& c, int& d, int& side, IV& in)
{
    int n = 0;
    in = c;
    for (int a = 0; a < D; ++a)
    {
        const int o = c[a] < g.lo[a] ? -1 : (c[a] > g.hi[a] ? 1 : 0);
        if (o == 0 || g.per[a]) continue;               // periodic: a partner cell
        ++n; d = a; side = 2*a + (o > 0);
        in[a] -= o;
    }
    return n == 1 && inside(g, in);
}

/// The ghost kernel's value in every out-of-domain ghost cell across a face, from the inner cell and its aux (Cartesian: the foot of the perpendicular is the face centre, normal distance h/2).
template<class K, class P, class T>
Field bc_cells(K ghost, const Field& Q, const Field& A, const P& p, T time_, const Mesh& M)
{
    BL_PROFILE("zs::bc_cells");
    Field out = cells_of(M, Q.nq);
    out.ngv = std::min(Q.ngv, A.ngv);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IV(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        int d = 0, side = 0;
        IV in = c;
        if (!boundary_ghost(g, c, d, side, in)) return;
        const IV fc = (side & 1) ? c : in;                // the face index
        Real n[3] = {0.0, 0.0, 0.0};
        n[d] = (side & 1) ? 1.0 : -1.0;
        const auto v = ghost(g.branch[side], time, face_centre(g, d, fc), 0.5*g.dx[d],
                             col<K::n_in>(q[bx], in, nq), col<K::n_aux>(a[bx], in, na), p,
                             normal<K::n_nrm>(n));
        for (int m = 0; m < K::n_out && m < nq; ++m) o[bx](c, m) = v(m, 0);
    });
    return out;
}

/// The inner cell's values in every out-of-domain ghost cell across a face.
inline Field inner_cells(const Field& A, const Mesh& M)
{
    BL_PROFILE("zs::inner_cells");
    Field out = cells_of(M, A.nq);
    out.ngv = A.ngv;
    const Geo g = M.g;
    auto o = out[0].arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IV(out.ng()), A.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
    {
        ZS_CELL;
        int d = 0, side = 0;
        IV in = c;
        if (boundary_ghost(g, c, d, side, in)) o[bx](c, n) = a[bx](in, n);
    });
    return out;
}

/// The boundary ghosts Wg of the BC ghost kernel (bc_cells).
template<class K, class P, class T>
Field boundary_ghosts(K ghost, const Field& Q, const Field& A, const P& p, T time,
                      const Mesh& M)
{
    return bc_cells(ghost, Q, A, p, time, M);
}

/// The boundary ghosts (bc_cells) and the inner cells' aux in the ghost cells.
template<int N = 2, class K, class P, class T>
std::tuple<Field, Field> boundary_states(K ghost, const Field& Q, const Field& A,
                                         const P& p, T time, const Mesh& M)
{
    return {bc_cells(ghost, Q, A, p, time, M), inner_cells(A, M)};
}

/// QR with the BC's outer face state bc(QL) on the boundary faces (the face kernel, at the face centre).
template<class K, class P, class T>
Field boundary_traces(K bc, const Field&, const Field& QL, const Field& QR,
                      const Field& A, const P& p, T time_, const Mesh& M)
{
    BL_PROFILE("zs::boundary_traces");
    Field out = like(QR, QR.nq);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = QL.nq, na = A.nq;
    auto a = A[0].const_arrays();
    for (int d = 0; d < D; ++d)
    {
        auto o = out[d].arrays();
        auto l = QL[d].const_arrays();
        auto r = QR[d].const_arrays();
        amrex::ParallelFor(out[d],
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            if (!f.bnd)
            {
                for (int m = 0; m < nq; ++m) o[bx](c, m) = r[bx](c, m);
                return;
            }
            const int side = 2*d + (f.n[d] > 0);
            const auto v = bc(g.branch[side], time, face_centre(g, d, c), 0.5*g.dx[d],
                              col<K::n_in>(l[bx], c, nq), col<K::n_aux>(a[bx], f.o, na), p,
                              normal<K::n_nrm>(f.n));
            for (int m = 0; m < nq; ++m) o[bx](c, m) = v(m, 0);
        });
    }
    return out;
}

// ── reconstruction and assembly ─────────────────────────────────────────────
/// Owner and other cell value per face; a boundary face's outer slot holds the inner value until boundary_traces sets the BC's face state.
template<class X>
std::tuple<Field, Field> cell_traces_of(const X& Q, const Mesh& M)
{
    BL_PROFILE("zs::cell_traces");
    Field L = faces_of(M, Q.nq), R = faces_of(M, Q.nq);
    const Geo g = M.g;
    for (int d = 0; d < D; ++d)
    {
        const auto q = Q.dev(0);
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IV(0), Q.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            l[bx](c, n) = q(bx, f.o, n);
            r[bx](c, n) = q(bx, f.bnd ? f.o : f.x, n);
        });
    }
    return {L, R};
}
/// Cell traces of an expression (read lazily when its halo is valid).
template<class X>
std::tuple<Field, Field> cell_traces(const X& Q, const Mesh& M)
{
    if (Q.ngv >= 1) return cell_traces_of(Q, M);   // an expression: read lazily
    return cell_traces_of(need(Field(Q), 1, M), M);
}
/// Cell traces of a Field (halo-exchanged when too thin).
inline std::tuple<Field, Field> cell_traces(const Field& Q, const Mesh& M)
{
    return cell_traces_of(DLeafless{need(Q, 1, M)}, M);
}

/// LSQ fit of row n at cell c, monomial m: sum_s w_s,m delta_s; a boundary sample carries ug(n) - u_c (ghost ug) or no change, k = the row's boundary-closure class (-1: none).
/// Ref: Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
template<class A, class B>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real lsq(const Geo& g, const A& u, const B& ug, bool ghost, const IV& c, int n, int m, int k = -1)
{
    if (m < 0) return 0.0;
    IV cw = c;                          // c in the domain (a periodic image wrapped)
    int cl = 0;
    for (int a = 0; a < D; ++a)
    {
        const int na = g.hi[a] - g.lo[a] + 1;
        cw[a] = g.lo[a] + ((c[a] - g.lo[a]) % na + na) % na;
        cl = cl*g.ncls[a] + g.cls[a][cw[a] - g.lo[a]];
    }
    const Real uc = u(c, n);
    Real s = 0.0;
    const int li = 2*(cl*(g.ncl + 1) + (k >= 0 && k < g.ncl ? 1 + k : 0));
    for (int t = g.off[li]; t < g.off[li + 1]; ++t)
    {
        const int kind = g.smp[4*t];
        IV cc = c;
        for (int a = 0; a < D; ++a) cc[a] += g.smp[4*t + 1 + a];
        const Real delta = kind == 0 ? u(cc, n) - uc : (ghost ? ug(cc, n) - uc : 0.0);
        s += g.w[t*g.nmon + m]*delta;
    }
    return s;
}

/// LSQ gradient of every row of the cells (row D q + a: d/dx_a of row q); the boundary samples are the ghosts Wg.
/// Ref: Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
inline Field gradient(const Field& W_, const Field& Wg, const Mesh& M)
{
    BL_PROFILE("zs::gradient");
    const Field W = need(W_, M.R, M);
    Field G = cells_of(M, D*W.nq);
    const int gr = std::max(0, std::min(W.ngv, Wg.ngv) - M.R);
    G.ngv = gr;
    const Geo g = M.g;
    auto o = G[0].arrays();
    auto w = W[0].const_arrays();
    auto wg = Wg[0].const_arrays();
    amrex::ParallelFor(G[0], IV(gr), W.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
    {
        ZS_CELL;
        if (!inside(g, c)) return;
        const int kc = n < g.nrcl ? g.rcl[n] : -1;      // the closure class of W row n
        for (int a = 0; a < D; ++a) o[bx](c, D*n + a) = lsq(g, w[bx], wg[bx], true, c, n, g.gmon[a], kc);
    });
    return G;
}

/// The slope limiters of a reconstruction.
enum class Limiter { minmod, venkatakrishnan, barth_jespersen };
/// Limiter factor of one cell from one face increment delta (cell bounds dmin, dmax); ev2 = (cell size)^2 for Venkatakrishnan.
/// Ref: Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
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

/// Limiter factors of `rows` (1 elsewhere), the bounds from the face neighbours and the boundary ghosts Wg (at the ghost position, as in gradient).
/// Ref: Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
inline Field limiter_factors(Limiter L, const Rows R, const Field& W_, const Field& Wg,
                             const Field& G, const Mesh& M)
{
    BL_PROFILE("zs::limiter_factors");
    const Field W = need(W_, 1, M);
    Field phi = cells_of(M, W.nq);
    const int gr = std::max(0, std::min({W.ngv - 1, G.ngv, Wg.ngv}));
    phi.ngv = gr;
    const Geo g = M.g;
    Real vol = 1.0;
    for (int d = 0; d < D; ++d) vol *= g.dx[d];
    const Real ev2 = std::pow(vol, 2.0/D);              // (V^(1/d))^2
    auto o = phi[0].arrays();
    auto w = W[0].const_arrays();
    auto wg = Wg[0].const_arrays();
    auto gg = G[0].const_arrays();
    amrex::ParallelFor(phi[0], IV(gr), W.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
    {
        ZS_CELL;
        bool lim = false;
        for (int m = 0; m < R.n; ++m) lim = lim || R.r[m] == n;
        if (!lim || !inside(g, c)) { o[bx](c, n) = 1.0; return; }
        const Real u = w[bx](c, n);
        Real lo = u, hi = u;
        for (int f = 0; f < 2*D; ++f)
        {
            IV cc = c;
            cc[f/2] += (f & 1) ? 1 : -1;
            // a boundary ghost holds Wg; a periodic one the partner cell
            const Real v = inside(g, cc) ? w[bx](cc, n) : wg[bx](cc, n);
            lo = v < lo ? v : lo;
            hi = v > hi ? v : hi;
        }
        Real ph = 1.0;
        for (int f = 0; f < 2*D; ++f)
        {
            const int d = f/2;
            const Real delta = gg[bx](c, D*n + d)*((f & 1) ? 0.5 : -0.5)*g.dx[d];
            const Real cf = limit(L, delta, lo - u, hi - u, ev2);
            ph = cf < ph ? cf : ph;
        }
        o[bx](c, n) = ph;
    });
    return phi;
}
/// Limiter factors with the minmod limiter.
template<class... I>
Field minmod(const std::tuple<I...>& rows, const Field& W, const Field& Wg,
             const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::minmod, rows_of(rows), W, Wg, G, M); }
/// Limiter factors with the Venkatakrishnan limiter.
template<class... I>
Field venkatakrishnan(const std::tuple<I...>& rows, const Field& W, const Field& Wg,
                      const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::venkatakrishnan, rows_of(rows), W, Wg, G, M); }
/// Limiter factors with the Barth-Jespersen limiter.
template<class... I>
Field barth_jespersen(const std::tuple<I...>& rows, const Field& W, const Field& Wg,
                      const Field& G, const Mesh& M)
{ return limiter_factors(Limiter::barth_jespersen, rows_of(rows), W, Wg, G, M); }

/// Each cell's W moved to the face centre by phi G; a boundary face's outer slot holds the inner trace until boundary_traces sets the BC's face state.
/// Ref: van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
template<class P>
std::tuple<Field, Field> extrapolate(const Field& W_, const Field& G_, const P& phi_,
                                     const Mesh& M)
{
    BL_PROFILE("zs::extrapolate");
    const Field W = need(W_, 1, M), G = need(G_, 1, M);
    Field L = faces_of(M, W.nq), R = faces_of(M, W.nq);
    const Geo g = M.g;
    auto w = W[0].const_arrays();
    auto gg = G[0].const_arrays();
    Src phi;
    Field keep;                         // phi's storage lives through the launches
    if constexpr (std::is_arithmetic_v<P>) phi = src(phi_, 0);
    else { keep = need(Field(phi_), 1, M); phi = src(keep, 0); }
    for (int d = 0; d < D; ++d)
    {
        auto l = L[d].arrays();
        auto r = R[d].arrays();
        amrex::ParallelFor(L[d], IV(0), W.nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            const FaceInfo f = face_info(g, d, c);
            // the face centre from a cell centre: +- h/2 along d
            auto val = [&](const IV& cc)
            {
                const Real s = (cc[d] < c[d] ? 0.5 : -0.5)*g.dx[d];
                return w[bx](cc, n) + phi(bx, cc, n)*gg[bx](cc, D*n + d)*s;
            };
            l[bx](c, n) = val(f.o);
            r[bx](c, n) = val(f.bnd ? f.o : f.x);
        });
    }
    return {L, R};
}

/// Per cell sum_f |f| W / |V| over its faces: Wm where it owns the face, Wp where it is the other cell; the face terms are logged for refluxing on an AMR level.
inline Field divergence(const Field& Wm, const Field& Wp, const Mesh& M)
{
    BL_PROFILE("zs::divergence");
    if (M.log) M.log->push_back({Wm, Wp});
    Field out = cells_of(M, Wm.nq);
    const Geo g = M.g;
    auto o = out[0].arrays();
    amrex::GpuArray<amrex::MultiArray4<Real const>, D> m, pp;
    for (int d = 0; d < D; ++d) { m[d] = Wm[d].const_arrays(); pp[d] = Wp[d].const_arrays(); }
    amrex::ParallelFor(out[0], IV(0), Wm.nq,
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
    {
        ZS_CELL;
        Real s = 0.0;
        for (int d = 0; d < D; ++d)
        {
            // lower face: this cell owns it only on the domain's lower boundary
            const bool own = !g.per[d] && c[d] == g.lo[d];
            const Real lower = own ? m[d][bx](c, n) : pp[d][bx](c, n);
            s += (lower + m[d][bx](c + unit(d), n))/g.dx[d];
        }
        o[bx](c, n) = s;
    });
    return out;
}

/// Per cell the interior nonconservative integral over each of its faces.
/// Ref: Pares 2006, https://doi.org/10.1137/050628052
template<class K, class P>
Field edge_source(K kn, const Field& Q, const Field& QL, const Field& QR,
                  const Field& A, const P& p, const Mesh& M)
{
    BL_PROFILE("zs::edge_source");
    Field out = cells_of(M, Q.nq);
    const Geo g = M.g;
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::GpuArray<amrex::MultiArray4<Real const>, D> ql, qr;
    for (int d = 0; d < D; ++d) { ql[d] = QL[d].const_arrays(); qr[d] = QR[d].const_arrays(); }
    amrex::ParallelFor(out[0], IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        const auto qc = col<K::n_in>(q[bx], c, nq);
        const auto ac = col<K::n_aux>(a[bx], c, na);
        Col<K::n_out> s{};
        for (int d = 0; d < D; ++d)
            for (int up = 0; up < 2; ++up)
            {
                const IV fc = up ? c + unit(d) : c;
                const FaceInfo f = face_info(g, d, fc);
                const bool own = f.o == c;
                const auto v = kn(qc, own ? col<K::n_in>(ql[d][bx], fc, nq)
                                          : col<K::n_in>(qr[d][bx], fc, nq),
                                  ac, ac, p, normal<K::n_nrm>(f.n, own ? 1.0 : -1.0));
                for (int mm = 0; mm < K::n_out; ++mm) s(mm, 0) += v(mm, 0)/g.dx[d];
            }
        for (int mm = 0; mm < K::n_out && mm < nq; ++mm) o[bx](c, mm) = s(mm, 0);
    });
    return out;
}

// ── whole-domain operations ─────────────────────────────────────────────────
/// Qaux with the derivative rows of spec from the LSQ fit; boundary samples carry no change (no Qg) or the ghost Qg (with Qg, only the derivatives of the states in rows).
/// Ref: Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
template<class S>
Field derivative_aux_impl(const S& spec, const Field& Q_, const Field& Qaux,
                          const Mesh& M, const Field* Qg, const Rows rows)
{
    BL_PROFILE("zs::derivative_aux_impl");
    const Field Q = need(Q_, M.R, M);
    const int gr = std::max(0, std::min(Q.ngv, Qg ? Qg->ngv : Q.ngv) - M.R);
    Field out = like(Qaux, Qaux.nq);
    out.ngv = std::min(Qaux.ngv, gr);
    amrex::MultiFab::Copy(out[0], Qaux[0], 0, 0, Qaux.nq, out.ng());
    const Geo g = M.g;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto qg = (Qg ? *Qg : Q)[0].const_arrays();
    const bool ghost = Qg != nullptr;
    for (const auto& e : spec)
    {
        if (Qg)
        {
            bool in = false;
            for (int m = 0; m < rows.n; ++m) in = in || rows.r[m] == e[1];
            if (!in) continue;
        }
        const int mi = M.monomial(e[2], e[3], e[4]);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mi >= 0,
            "derivative_aux: the derivative exceeds the LSQ degree");
        const int row = e[0], s = e[1], kc = ghost ? e[5] : -1;   // e[5]: closure class
        amrex::ParallelFor(out[0], IV(gr),
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            if (inside(g, c)) o[bx](c, row) = lsq(g, q[bx], qg[bx], ghost, c, s, mi, kc);
        });
    }
    return out;
}
/// derivative_aux_impl without ghosts.
template<class S>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M)
{ return derivative_aux_impl(spec, Q, Qaux, M, nullptr, Rows{}); }
/// derivative_aux_impl with the ghosts Qg for the states in `rows`.
template<class S, class... I>
Field derivative_aux(const S& spec, const Field& Q, const Field& Qaux, const Mesh& M,
                     const Field& Qg, const std::tuple<I...>& rows)
{ return derivative_aux_impl(spec, Q, Qaux, M, &Qg, rows_of(rows)); }

/// Cell mask: 1 where an admissibility row of kn is < 0 or a state is non-finite.
template<class K, class P>
Field inadmissible(K kn, const Field& Q, const Field& A, const P& p)
{
    BL_PROFILE("zs::inadmissible");
    Field out = like(Q, 1);
    out.ngv = std::min(Q.ngv, A.ngv);
    const int nq = Q.nq, na = A.nq;
    auto o = out[0].arrays();
    auto q = Q[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IV(out.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        const auto x = col<K::n_in>(q[bx], c, nq);
        const auto v = kn(x, col<K::n_aux>(a[bx], c, na), p, 0.0, 0.0, Col<3>{});
        bool bad = false;
        for (int m = 0; m < K::n_out; ++m) bad = bad || v(m, 0) < 0.0;
        for (int m = 0; m < nq; ++m) bad = bad || !std::isfinite(q[bx](c, m));
        o[bx](c, 0) = bad ? 1.0 : 0.0;
    });
    return out;
}

// ── implicit solves ─────────────────────────────────────────────────────────
/// L2 norm of a Field's rows over the valid cells (all ranks).
inline double norm(const Field& a)
{
    return std::sqrt(amrex::MultiFab::Dot(a[0], 0, a[0], 0, a.nq, 0));
}

/// x + h e_kk (h = fd_step(x_kk)) and h: the finite-difference column kk.
inline std::pair<Field, Field> perturb(const Field& x, int kk)
{
    BL_PROFILE("zs::perturb");
    Field xk = like(x, x.nq), h = like(x, 1);
    xk.ngv = h.ngv = x.ngv;
    auto o = xk[0].arrays(); auto oh = h[0].arrays();
    auto a = x[0].const_arrays();
    const int nq = x.nq;
    amrex::ParallelFor(xk[0], IV(xk.ng()),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        for (int n = 0; n < nq; ++n) o[bx](c, n) = a[bx](c, n);
        const Real v = a[bx](c, kk);
        const Real s = fd_step(v);
        oh[bx](c, 0) = s;
        o[bx](c, kk) = v + s;
    });
    return {xk, h};
}

/// The per-cell Newton update from the residual r and the perturbed columns (finite-difference Jacobian, solved per cell).
inline Field newton_update(const Field& x, const Field& r, const std::vector<Field>& cols,
                           const std::vector<Field>& hs)
{
    BL_PROFILE("zs::newton_update");
    const int n = x.nq;
    AMREX_ALWAYS_ASSERT(n <= MAXR);
    amrex::MultiFab J(x[0].boxArray(), x[0].DistributionMap(), n*n, 0), H(
        x[0].boxArray(), x[0].DistributionMap(), n, 0);
    for (int q = 0; q < n; ++q)
    {
        for (int p = 0; p < n; ++p) amrex::MultiFab::Copy(J, cols[q][0], p, p*n + q, 1, 0);
        amrex::MultiFab::Copy(H, hs[q][0], 0, q, 1, 0);
    }
    Field out = like(x, n);
    amrex::MultiFab::Copy(out[0], x[0], 0, 0, n, 0);
    auto o = out[0].arrays();
    auto jj = J.const_arrays(); auto hh = H.const_arrays();
    auto rr = r[0].const_arrays();
    amrex::ParallelFor(out[0], IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        Real A[MAXR*MAXR], b[MAXR];
        for (int p = 0; p < n; ++p)
        {
            b[p] = -rr[bx](c, p);
            for (int q = 0; q < n; ++q)
                A[p*n + q] = (jj[bx](c, p*n + q) - rr[bx](c, p))/hh[bx](c, q);
        }
        solve_small(A, b, n);
        for (int p = 0; p < n; ++p) o[bx](c, p) += b[p];
    });
    return out;
}

/// Newton on f(x) = 0 with a per-cell (block-diagonal) Jacobian (newton_cellwise of rt_common.H).
template<class F> Field root(F f, Field x) { return newton_cellwise(f, std::move(x)); }

/// amrex::GMRES operator on the printed linear map, right-preconditioned by the exact cell blocks of the operator (probed by colouring: cells R + 1 apart share no stencil) inverted per cell.
/// Ref: Saad, Schultz 1986, https://doi.org/10.1137/0907058
struct LinOp
{
    using RT = Real;
    std::function<Field(Field)> op;
    const Mesh* M = nullptr;
    int nq = 0;
    amrex::MultiFab Binv;

    /// The vector-space interface amrex::GMRES requires of its operator (allocation, copy, dot, axpy, norm, scale).
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

    /// Alias of a MultiFab as a Field.
    Field wrap(const amrex::MultiFab& x) const
    {
        Field X;
        X.nq = nq;
        X.mf = std::make_shared<std::vector<amrex::MultiFab>>();
        X.mf->emplace_back(x, amrex::make_alias, 0, nq);
        return X;
    }
    /// Ax = op(x).
    void apply(amrex::MultiFab& Ax, const amrex::MultiFab& x)
    {
        const Field y = op(wrap(x));
        amrex::MultiFab::Copy(Ax, y[0], 0, 0, nq, 0);
    }
    /// l = Binv r, the cell-block preconditioner (defined below).
    void precond(amrex::MultiFab& l, const amrex::MultiFab& r);
    /// Probe op and invert its cell blocks into Binv (defined below).
    void blocks();
};

/// Colour of cell c: (c - lo) mod S per axis, combined (cells S apart share no stencil).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int colour(const Geo& g, const IV& c, int S)
{
    int cl = 0;
    for (int d = D - 1; d >= 0; --d) cl = cl*S + (c[d] - g.lo[d]) % S;
    return cl;
}

/// Probe op with one unit vector per colour and row, and invert the n x n cell blocks by Gauss-Jordan with partial pivoting into Binv.
inline void LinOp::blocks()
{
    BL_PROFILE("zs::LinOp::blocks");
    const int n = nq, S = M->R + 1;
    int ncol = 1;
    for (int d = 0; d < D; ++d) ncol *= S;
    AMREX_ALWAYS_ASSERT(n <= MAXR);
    amrex::MultiFab A(M->ba, M->dm, n*n, 0);
    Binv.define(M->ba, M->dm, n*n, 0);
    const Geo g = M->g;
    for (int cl = 0; cl < ncol; ++cl)
        for (int m = 0; m < n; ++m)
        {
            amrex::MultiFab e(M->ba, M->dm, n, M->NG);
            e.setVal(0.0);
            auto ea = e.arrays();
            amrex::ParallelFor(e, IV(0),
                [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
            {
                ZS_CELL;
                if (colour(g, c, S) == cl) ea[bx](c, m) = 1.0;
            });
            const Field y = op(wrap(e));
            auto ya = y[0].const_arrays();
            auto aa = A.arrays();
            amrex::ParallelFor(A, IV(0),
                [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
            {
                ZS_CELL;
                if (colour(g, c, S) != cl) return;
                for (int p = 0; p < n; ++p) aa[bx](c, p*n + m) = ya[bx](c, p);
            });
        }
    auto aa = A.const_arrays();
    auto bi = Binv.arrays();
    amrex::ParallelFor(Binv, IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        Real a[MAXR*MAXR], v[MAXR*MAXR];
        for (int e = 0; e < n*n; ++e) { a[e] = aa[bx](c, e); v[e] = 0.0; }
        for (int d = 0; d < n; ++d) v[d*n + d] = 1.0;
        for (int cc = 0; cc < n; ++cc)              // Gauss-Jordan, partial pivoting
        {
            int piv = cc;
            for (int p = cc + 1; p < n; ++p) if (std::abs(a[p*n + cc]) > std::abs(a[piv*n + cc])) piv = p;
            for (int q = 0; q < n; ++q)
            {
                Real t = a[cc*n + q]; a[cc*n + q] = a[piv*n + q]; a[piv*n + q] = t;
                t = v[cc*n + q]; v[cc*n + q] = v[piv*n + q]; v[piv*n + q] = t;
            }
            const Real ip = a[cc*n + cc] != 0.0 ? 1.0/a[cc*n + cc] : 0.0;
            for (int q = 0; q < n; ++q) { a[cc*n + q] *= ip; v[cc*n + q] *= ip; }
            for (int p = 0; p < n; ++p)
            {
                if (p == cc) continue;
                const Real f = a[p*n + cc];
                for (int q = 0; q < n; ++q) { a[p*n + q] -= f*a[cc*n + q]; v[p*n + q] -= f*v[cc*n + q]; }
            }
        }
        for (int e = 0; e < n*n; ++e) bi[bx](c, e) = v[e];
    });
}

/// l = Binv r: the exact cell blocks as a right preconditioner.
inline void LinOp::precond(amrex::MultiFab& l, const amrex::MultiFab& r)
{
    BL_PROFILE("zs::LinOp::precond");
    const int n = nq;
    auto la = l.arrays();
    auto ra = r.const_arrays();
    auto bi = Binv.const_arrays();
    amrex::ParallelFor(l, IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        Real x[MAXR];
        for (int p = 0; p < n; ++p)
        {
            x[p] = 0.0;
            for (int q = 0; q < n; ++q) x[p] += bi[bx](c, p*n + q)*ra[bx](c, q);
        }
        for (int p = 0; p < n; ++p) la[bx](c, p) = x[p];
    });
}

/// Solve op(x) = b with amrex::GMRES (restart, iterations and tolerance from implicit()); aborts above the accept gate.
/// Ref: Saad, Schultz 1986, https://doi.org/10.1137/0907058
inline Field linsolve_impl(std::function<Field(Field)> op, const Field& b, const Mesh& M)
{
    BL_PROFILE("zs::linsolve_impl");
    const auto& o = implicit();
    LinOp L;
    L.op = std::move(op); L.M = &M; L.nq = b.nq;
    L.blocks();
    amrex::GMRES<amrex::MultiFab, LinOp> gm;
    gm.define(L);
    gm.setRestartLength(o.ls_restart);
    gm.setMaxIters(o.ls_restart*o.ls_maxiter);
    Field x = cells_of(M, b.nq);
    x[0].setVal(0.0);
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

/// The driver's mesh (one per run): the printed linsolve(op, b) carries none.
inline const Mesh*& the_mesh() { static const Mesh* m = nullptr; return m; }

/// The printed linear solve: linsolve_impl on the driver's mesh.
template<class F>
Field linsolve(F op, const Field& b)
{
    return linsolve_impl(std::function<Field(Field)>(op), b, *the_mesh());
}

// ── diffusion ───────────────────────────────────────────────────────────────
/// The two-point part of div(A : grad X) per cell (zoomy_core's diffusion block): per face n.A.n (X_out - X_in) / d |f| / |V|, A the mean of the two cells' tensors, X_out the ghost across a boundary face and the partner cell across a periodic side; not refluxed (no AMR).
/// Ref: Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
template<class K, class G, class P, class T>
Field diffusion_tpfa(K kn, G ghost, const Field& X_, const Field& A_, const P& p, T time_,
                     const Mesh& M)
{
    BL_PROFILE("zs::diffusion_tpfa");
    constexpr int N = K::n_in;
    static_assert(K::n_out == N*N*D*D, "diffusion: a tensor (n, n, D, D)");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!M.log, "diffusion: AMR does not reflux diffusive fluxes");
    const Field X = need(X_, 1, M), A = need(A_, 1, M);
    const Field Xg = bc_cells(ghost, X, A, p, time_, M);
    Field out = cells_of(M, X.nq);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = X.nq, na = A.nq;
    auto o = out[0].arrays();
    auto x = X[0].const_arrays();
    auto xg = Xg[0].const_arrays();
    auto a = A[0].const_arrays();
    amrex::ParallelFor(out[0], IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        auto tensor = [&](const IV& cc)
        {
            return kn(col<N>(x[bx], cc, nq), col<K::n_aux>(a[bx], cc, na), p, time, 0.0,
                      cell_centre(g, cc));
        };
        const auto Ac = tensor(c);
        Real s[N] = {};
        for (int d = 0; d < D; ++d)
            for (int up = 0; up < 2; ++up)
            {
                const IV cc = up ? c + unit(d) : c - unit(d);
                const bool in = inside(g, cc);
                const auto An = in ? tensor(cc) : Ac;
                const Real w = 1.0/(g.dx[d]*g.dx[d]);
                for (int r = 0; r < N; ++r)
                {
                    Real f = 0.0;
                    for (int q = 0; q < N; ++q)
                    {
                        const int e = ((r*N + q)*D + d)*D + d;
                        const Real t = in ? 0.5*(Ac(e, 0) + An(e, 0)) : Ac(e, 0);
                        f += t*((in ? x[bx](cc, q) : xg[bx](cc, q)) - x[bx](c, q));
                    }
                    s[r] += f*w;
                }
            }
        for (int r = 0; r < N && r < nq; ++r) o[bx](c, r) = s[r];
    });
    return out;
}

/// The rest of the face flux n.A.grad_f beyond diffusion_tpfa (2-D, 3-D): grad_f = the mean of the two cells' LSQ gradients (the derivative-aux fit, ghosts sampled), whose normal part the two-point difference replaces, so on a Cartesian face only its tangential part enters.
/// Ref: Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
template<class K, class G, class P, class T>
Field diffusion_correction(K kn, G ghost, const Field& X_, const Field& A_, const P& p, T time_,
                           const Mesh& M)
{
    BL_PROFILE("zs::diffusion_correction");
    constexpr int N = K::n_in;
    const int r1 = std::max(1, M.R);
    const Field X = need(X_, r1, M), A = need(A_, r1, M);
    const Field Xg = bc_cells(ghost, X, A, p, time_, M);
    const Geo g = M.g;
    const Real time = time_;
    const int nq = X.nq, na = A.nq;
    auto x = X[0].const_arrays();
    auto a = A[0].const_arrays();
    Field Gc = cells_of(M, D*nq);              // row D q + a: d/dx_a of row q
    {
        auto o = Gc[0].arrays();
        auto xg = Xg[0].const_arrays();
        amrex::ParallelFor(Gc[0], IV(0), nq,
            [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
        {
            ZS_CELL;
            if (!inside(g, c)) return;
            for (int e = 0; e < D; ++e) o[bx](c, D*n + e) = lsq(g, x[bx], xg[bx], true, c, n, g.gmon[e], -1);
        });
    }
    const Field Gr = need(Gc, 1, M);
    Field out = cells_of(M, nq);
    auto o = out[0].arrays();
    auto gr = Gr[0].const_arrays();
    amrex::ParallelFor(out[0], IV(0),
        [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        auto tensor = [&](const IV& cc)
        {
            return kn(col<N>(x[bx], cc, nq), col<K::n_aux>(a[bx], cc, na), p, time, 0.0,
                      cell_centre(g, cc));
        };
        const auto Ac = tensor(c);
        Real s[N] = {};
        for (int d = 0; d < D; ++d)
            for (int up = 0; up < 2; ++up)
            {
                const IV cc = up ? c + unit(d) : c - unit(d);
                const bool in = inside(g, cc);
                const auto An = in ? tensor(cc) : Ac;
                const Real w = (up ? 1.0 : -1.0)/g.dx[d];
                for (int r = 0; r < N; ++r)
                {
                    Real f = 0.0;
                    for (int q = 0; q < N; ++q)
                        for (int e = 0; e < D; ++e)
                        {
                            if (e == d) continue;
                            const int m = ((r*N + q)*D + d)*D + e;
                            const Real t = in ? 0.5*(Ac(m, 0) + An(m, 0)) : Ac(m, 0);
                            const Real ge = in ? 0.5*(gr[bx](c, D*q + e) + gr[bx](cc, D*q + e))
                                               : gr[bx](c, D*q + e);
                            f += t*ge;
                        }
                    s[r] += f*w;
                }
            }
        for (int r = 0; r < N && r < nq; ++r) o[bx](c, r) = s[r];
    });
    return out;
}

/// div(A : grad X) per cell (zoomy_core's diffusion block): diffusion_tpfa plus diffusion_correction.
template<class K, class G, class P, class T>
Field diffusion(K kn, G ghost, const Field& X, const Field& A, const P& p, T time,
                const Mesh& M)
{
    if constexpr (D == 1) return diffusion_tpfa(kn, ghost, X, A, p, time, M);
    else return Field(diffusion_tpfa(kn, ghost, X, A, p, time, M)
                      + diffusion_correction(kn, ghost, X, A, p, time, M));
}

/// One Crank-Nicolson step of diffusion_tpfa from time to time + dt plus the correction explicit at Q (deferred): R(X) = X - Q - dt/2 (D(X) + D(Q) + 2 C(Q)) = 0, A at X (Qaux frozen), by Newton-Krylov.
/// Ref: Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
template<class K, class G, class P, class T, class S>
Field implicit_diffusion(K kn, G ghost, const Field& Q, const Field& A, const P& p,
                         T time_, S dt_, const Mesh& M)
{
    BL_PROFILE("zs::implicit_diffusion");
    const Real time = time_, dt = dt_;
    const Field D0 = diffusion_tpfa(kn, ghost, Q, A, p, time, M);
    const Field b = [&]() -> Field
    {
        if constexpr (D == 1) return Q + 0.5*dt*D0;
        else return Q + 0.5*dt*(D0 + 2.0*diffusion_correction(kn, ghost, Q, A, p, time, M));
    }();
    auto R = [&](const Field& X) -> Field
    { return X - b - 0.5*dt*diffusion_tpfa(kn, ghost, X, A, p, time + dt, M); };
    return halo(newton_krylov(R, Q), M);
}

// ── control ─────────────────────────────────────────────────────────────────
/// Iterations of every run-time loop (the steps of a hyperbolic march's time loop).
inline long& loop_iterations() { static long n = 0; return n; }

/// A carried loop value; expressions are evaluated (the carry has one type).
template<class T> auto carried(const T& v)
{
    if constexpr (is_x<T>::value) return Field(v);
    else return v;
}

/// The printed while loop: the body b applied to the carried tuple while c holds.
template<class C, class B, class S0>
auto while_loop(C c, B b, S0 s0)
{
    auto s = std::apply([](const auto&... v) { return std::make_tuple(carried(v)...); }, s0);
    while (std::apply(c, s)) { s = std::apply(b, s); ++loop_iterations(); }
    return s;
}

/// The printed conditional on a host bool; an expression and a Field branch are unified as Fields.
template<class A, class B>
auto cond(bool p, A a, B b)
{
    using RA = decltype(a());
    if constexpr (std::is_same_v<RA, decltype(b())>) return p ? a() : b();
    else return p ? Field(a()) : Field(b());      // an expression and a Field
}

#ifdef ZS_PRECICE
// ── preCICE ─────────────────────────────────────────────────────────────────
/// The coupled side (2 d + upper) and, per face, its preCICE vertex (-1 if not coupled); filled by couple_setup.
/// Ref: Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
struct Interface
{
    int side = -1;                                  // 2 d + upper
    Field index;                                    // vertex of a coupled face, -1 else
};
/// The coupling interface of this rank.
inline Interface& interface() { static Interface i; return i; }

/// Set up preCICE: the participant and one vertex per coupled boundary face of this rank's boxes on `side`, numbered box by box (x fastest); fills coupling() (rt_common.H) and interface().
inline void couple_setup(const Mesh& M, const std::string& name, const std::string& config,
                         const std::string& mesh, const std::vector<std::string>& write,
                         const std::vector<std::string>& read, int side)
{
    auto& cpl = coupling();
    auto& itf = interface();
    cpl.mesh = mesh; cpl.write = write; cpl.read = read; itf.side = side;
    const int d = side/2, plane = (side & 1) ? M.g.hi[d] + 1 : M.g.lo[d];
    itf.index = faces_of(M, 1);
    std::vector<int> first;                         // per local box: its first vertex
    std::vector<double> xyz;
    const int dims = std::max(D, 2);                // the preCICE mesh: 2-D at least
    for (amrex::MFIter mfi(itf.index[d]); mfi.isValid(); ++mfi)
    {
        first.push_back(int(xyz.size())/dims);
        amrex::Box b = mfi.validbox();
        if (b.smallEnd(d) > plane || b.bigEnd(d) < plane) continue;
        b.setSmall(d, plane); b.setBig(d, plane);
        amrex::LoopOnCpu(b, [&](int i, int j, int k)
        {
            const Col<3> X = face_centre(M.g, d, IV(AMREX_D_DECL(i, j, k)));
            for (int a = 0; a < dims; ++a) xyz.push_back(a < D ? X(a, 0) : 0.0);
        });
    }
    amrex::Gpu::DeviceVector<int> f0(first.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, first.begin(), first.end(), f0.begin());
    const int* fp = f0.data();
    for (int a = 0; a < D; ++a)
    {
        auto o = itf.index[a].arrays();
        amrex::ParallelFor(itf.index[a], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            if (a != d || c[d] != plane) { o[bx](c, 0) = -1.0; return; }
            const auto lo = amrex::lbound(o[bx]), hi = amrex::ubound(o[bx]);
            const int l[3] = {lo.x, lo.y, lo.z}, n[3] = {hi.x - lo.x + 1, hi.y - lo.y + 1, hi.z - lo.z + 1};
            int idx = 0;
            for (int e = D - 1; e >= 0; --e)
                if (e != d) idx = idx*n[e] + (c[e] - l[e]);
            o[bx](c, 0) = fp[bx] + idx;
        });
    }
    amrex::Gpu::streamSynchronize();
    cpl.ids.resize(xyz.size()/dims);
    cpl.participant = std::make_unique<precice::Participant>(
        name, config, amrex::ParallelDescriptor::MyProc(), amrex::ParallelDescriptor::NProcs());
    cpl.participant->setMeshVertices(cpl.mesh, xyz, cpl.ids);
}

/// Face mask: 1 on the coupled faces.
inline Field coupled_faces(const Mesh&)
{
    const Field& I = interface().index;
    Field out = like(I, 1);
    for (int d = 0; d < D; ++d)
    {
        auto o = out[d].arrays();
        auto x = I[d].const_arrays();
        amrex::ParallelFor(out[d], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        { ZS_CELL; o[bx](c, 0) = x[bx](c, 0) >= 0.0 ? 1.0 : 0.0; });
    }
    return out;
}

/// The inner cell's state at every coupled face (vertex order), on the host.
inline Vertices interface_states(const Field& Q, const Mesh& M)
{
    const auto& itf = interface();
    const int d = itf.side/2, up = itf.side & 1, nq = Q.nq, nv = int(coupling().ids.size());
    amrex::Gpu::DeviceVector<double> v(size_t(nq)*std::max(nv, 1));
    double* vp = v.data();
    auto x = itf.index[d].const_arrays();
    auto q = Q[0].const_arrays();
    amrex::ParallelFor(itf.index[d], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        const int jv = int(x[bx](c, 0));
        if (jv < 0) return;
        const IV in = up ? c - unit(d) : c;
        for (int r = 0; r < nq; ++r) vp[r*nv + jv] = q[bx](in, r);
    });
    Vertices out{nq, nv, std::vector<double>(size_t(nq)*nv)};
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, v.begin(), v.begin() + long(nq)*nv, out.v.begin());
    amrex::Gpu::streamSynchronize();
    (void)M;
    return out;
}

/// One Riemann problem per coupled face, posed low to high (kn = numerical_face_sides): row 0 for the low side, row 1 for the high side; each state's aux from the pointwise kernel on a zero template (false: zero).
template<class K, class KA, class P>
Field interface_flux(K kn, KA kaux, int side, const Field& Q, const Field& Qaux,
                     const Vertices& peer, const P& p, const Mesh& M)
{
    const auto& itf = interface();
    const int d = itf.side/2, up = itf.side & 1, nq = Q.nq, nv = peer.n;
    Field out = faces_of(M, nq);
    for (int a = 0; a < D; ++a) out[a].setVal(0.0);
    amrex::Gpu::DeviceVector<double> pv(std::max<size_t>(peer.v.size(), 1));
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, peer.v.begin(), peer.v.end(), pv.begin());
    const double* pp = pv.data();
    const Geo g = M.g;
    auto x = itf.index[d].const_arrays();
    auto q = Q[0].const_arrays();
    auto o = out[d].arrays();
    constexpr int NQ = K::n_in, NA = K::n_aux;
    (void)Qaux;
    amrex::ParallelFor(out[d], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
    {
        ZS_CELL;
        const int jv = int(x[bx](c, 0));
        if (jv < 0) return;
        const IV in = up ? c - unit(d) : c;
        Col<NQ> own{}, other{};
        for (int r = 0; r < NQ && r < nq; ++r) { own(r, 0) = q[bx](in, r); other(r, 0) = pp[r*nv + jv]; }
        const Col<NQ>& lo = side == 0 ? own : other;
        const Col<NQ>& hi = side == 0 ? other : own;
        auto aux = [&](const Col<NQ>& s)
        {
            if constexpr (std::is_same_v<KA, bool>) return Col<NA>{};
            else
            {
                const auto v = kaux(s, Col<KA::n_aux>{}, p, 0.0, 0.0, face_centre(g, d, c));
                Col<NA> r{};
                for (int m = 0; m < NA && m < KA::n_out; ++m) r(m, 0) = v(m, 0);
                return r;
            }
        };
        Real nn[3] = {0.0, 0.0, 0.0};
        nn[d] = 1.0;                                // low -> high
        const auto W = kn(lo, hi, aux(lo), aux(hi), p, normal<K::n_nrm>(nn));
        for (int r = 0; r < nq; ++r) o[bx](c, r) = W(side*nq + r, 0);
    });
    amrex::Gpu::streamSynchronize();                // pv dies here
    return out;
}

#endif

} // namespace zs
