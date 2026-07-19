#pragma once
// Jacobian-Free Newton-Krylov (JFNK) backward-Euler implicit-source solve.
//
// Solves  Q = Qexp + dt*S(Q)  for the whole field, i.e. F(Q) = Q - Qexp - dt*S(Q) = 0,
// by an outer Newton iteration whose linear system J*delta = -F is solved with
// amrex::GMRES.  J is applied MATRIX-FREE via a finite-difference JVP:
//     J*x = x - (dt/eps) * ( S(Qbase + eps*x) - S(Qbase) ).
// Because each source evaluation refreshes ghost cells + auxiliaries, the source
// may be NONLOCAL (gradient/derivative-coupled) — the FD-JVP captures the
// neighbour coupling automatically; no per-cell locality is assumed.
//
// This file is #included exactly once (into ZoomyAmr.cpp) and defines the
// ZoomyAmr methods plus the matrix-free GMRES operator.

#include <AMReX_MultiFab.H>
#include <AMReX_GMRES.H>
#include "Model.H"
#include "constants.H"

// ---- S(Q) over the valid region; ghosts + aux refreshed (nonlocal-ready) ----
inline void ZoomyAmr::ComputeSourceField(amrex::MultiFab& Qin, amrex::MultiFab& Sout,
                                         int lev, amrex::Real time)
{
    using namespace amrex;
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();          // REQ-185
    auto plo = geom.ProbLoArray();
    auto const& p = p_mat;

    // Refresh ghosts: interior exchange + physical BC dispatch.  This is what
    // makes a nonlocal source well-defined at domain boundaries.
    Qin.FillBoundary(geom.periodicity());
    MultiFab auxs(Qin.boxArray(), Qin.DistributionMap(),
                  std::max(Model::n_dof_qaux, 1), Qin.nGrow());
    auxs.setVal(0.0);
    FillPhysicalBC_mf(Qin, auxs, lev, time);

    for (MFIter mfi(Sout); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto Qa = Qin.const_array(mfi);
        auto Sa = Sout.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            SmallMatrix<Real, Model::n_dof_q, 1> q;
            SmallMatrix<Real, Model::n_dof_qaux, 1> a;
            for (int n = 0; n < Model::n_dof_q; ++n) q(n, 0) = Qa(i, j, 0, n);
            for (int n = 0; n < Model::n_dof_qaux; ++n) a(n, 0) = 0.0;
            SmallMatrix<Real, 3, 1> X{};   // REQ-185
            X(0, 0) = plo[0] + (i + 0.5) * dx[0];
            X(1, 0) = (Model::dimension == 2) ? plo[1] + (j + 0.5) * dx[1] : Real(0.0);
            X(2, 0) = Real(0.0);
            a = Model::update_aux_variables(q, a, p, time, X);
            auto S = Model::source(q, a, p, time, X);
            for (int n = 0; n < Model::n_dof_q; ++n) Sa(i, j, 0, n) = S(n, 0);
        });
    }
}

namespace {

// Matrix-free linear operator for amrex::GMRES: V = MultiFab, applies J = ∂F/∂Q
// at the frozen Newton iterate (Qbase, Sbase) via finite-difference JVP.
struct ImpSrcOp {
    using RT = amrex::Real;

    ZoomyAmr* solver = nullptr;
    int lev = 0, ncomp = 0, ng = 0;
    amrex::Real time = 0.0, dt = 0.0, nrmQ = 0.0;
    amrex::BoxArray ba;
    amrex::DistributionMapping dm;
    const amrex::MultiFab* Qbase = nullptr;  // frozen during one GMRES solve
    const amrex::MultiFab* Sbase = nullptr;  // S(Qbase)

    amrex::MultiFab makeVecRHS() const { return amrex::MultiFab(ba, dm, ncomp, 0); }
    amrex::MultiFab makeVecLHS() const { return amrex::MultiFab(ba, dm, ncomp, ng); }
    void assign(amrex::MultiFab& l, amrex::MultiFab const& r) { amrex::MultiFab::Copy(l, r, 0, 0, ncomp, 0); }
    RT dotProduct(amrex::MultiFab const& a, amrex::MultiFab const& b) { return amrex::MultiFab::Dot(a, 0, b, 0, ncomp, 0); }
    void increment(amrex::MultiFab& l, amrex::MultiFab const& r, RT a) { amrex::MultiFab::Saxpy(l, a, r, 0, 0, ncomp, 0); }
    void linComb(amrex::MultiFab& l, RT a, amrex::MultiFab const& ra, RT b, amrex::MultiFab const& rb) { amrex::MultiFab::LinComb(l, a, ra, 0, b, rb, 0, 0, ncomp, 0); }
    RT norm2(amrex::MultiFab const& v) { return std::sqrt(amrex::MultiFab::Dot(v, 0, v, 0, ncomp, 0)); }
    void scale(amrex::MultiFab& v, RT f) { v.mult(f, 0, ncomp, 0); }
    void setToZero(amrex::MultiFab& v) { v.setVal(0.0); }
    // Right-preconditioner: identity (the friction/relaxation Jacobian is
    // diagonally dominant, so unpreconditioned GMRES converges quickly).
    void precond(amrex::MultiFab& l, amrex::MultiFab const& r) { amrex::MultiFab::Copy(l, r, 0, 0, ncomp, 0); }

    void apply(amrex::MultiFab& Ax, amrex::MultiFab const& x) {
        RT nx = norm2(x);
        if (nx == RT(0)) { setToZero(Ax); return; }
        const RT eps = RT(1e-7) * (RT(1.0) + nrmQ) / nx;   // JFNK step
        amrex::MultiFab Qpert(ba, dm, ncomp, ng);
        amrex::MultiFab::LinComb(Qpert, RT(1.0), *Qbase, 0, eps, x, 0, 0, ncomp, 0);
        amrex::MultiFab Spert(ba, dm, ncomp, 0);
        solver->ComputeSourceField(Qpert, Spert, lev, time);
        // Ax = x - (dt/eps)*(Spert - Sbase)
        amrex::MultiFab::LinComb(Ax, RT(1.0), x, 0, -dt / eps, Spert, 0, 0, ncomp, 0);
        amrex::MultiFab::Saxpy(Ax, dt / eps, *Sbase, 0, 0, ncomp, 0);
    }
};

} // namespace

inline void ZoomyAmr::SolveImplicitSourceGlobal(int lev, amrex::Real dt, amrex::Real time)
{
    using namespace amrex;
    const BoxArray& ba = Q[lev].boxArray();
    const DistributionMapping& dm = Q[lev].DistributionMap();
    const int ncomp = Model::n_dof_q;
    const int ng = nghost();

    MultiFab Qexp(ba, dm, ncomp, 0);
    MultiFab::Copy(Qexp, Q[lev], 0, 0, ncomp, 0);   // post-flux state (frozen)
    MultiFab Sbase(ba, dm, ncomp, 0);
    MultiFab F(ba, dm, ncomp, 0);
    MultiFab rhs(ba, dm, ncomp, 0);
    MultiFab delta(ba, dm, ncomp, ng);

    ImpSrcOp op;
    op.solver = this; op.lev = lev; op.time = time; op.dt = dt;
    op.ba = ba; op.dm = dm; op.ncomp = ncomp; op.ng = ng;

    GMRES<MultiFab, ImpSrcOp> gmres;
    gmres.define(op);
    gmres.setVerbose(0);

    const int newton_max = 12;
    Real resid0 = -1.0;
    for (int it = 0; it < newton_max; ++it) {
        ComputeSourceField(Q[lev], Sbase, lev, time);          // S at current iterate
        MultiFab::LinComb(F, 1.0, Q[lev], 0, -1.0, Qexp, 0, 0, ncomp, 0);
        MultiFab::Saxpy(F, -dt, Sbase, 0, 0, ncomp, 0);        // F = Q - Qexp - dt*S
        const Real resid = std::sqrt(MultiFab::Dot(F, 0, F, 0, ncomp, 0));
        if (it == 0) resid0 = resid;
        if (resid < 1e-12 || (resid0 > 0 && resid < 1e-9 * resid0)) break;

        MultiFab::Copy(rhs, F, 0, 0, ncomp, 0);
        rhs.mult(-1.0, 0, ncomp, 0);                            // rhs = -F

        op.Qbase = &Q[lev];
        op.Sbase = &Sbase;
        op.nrmQ  = std::sqrt(MultiFab::Dot(Q[lev], 0, Q[lev], 0, ncomp, 0));
        delta.setVal(0.0);
        gmres.solve(delta, rhs, 1e-4, 0.0);                    // J*delta = -F

        MultiFab::Saxpy(Q[lev], 1.0, delta, 0, 0, ncomp, 0);   // Q += delta
        // REQ-200(c): the private depth floor that used to sit here
        //     if (Qa(i,j,0,idx_h) < 1e-8) Qa(i,j,0,idx_h) = 1e-8;
        // is REMOVED. It was not a "floor" in the safe sense -- it RAISED h, i.e.
        // it fabricated water on every dry cell of every Newton iterate, so the
        // implicit-source path could not conserve mass by construction. Standing
        // mandate: h is NEVER clamped; wet_dry_eps is only ever a threshold for
        // forming the RECIPROCAL of h.
        // Nothing here needs replacing: dry-safety already lives in the operator,
        // which is where core (REQ-200) argues it belongs -- the model's
        // update_aux_variables forms hinv = 1/max(h, wet_dry_eps), so the source
        // is finite for any depth, and guard_eigenvalue_powers (run_case.py:70)
        // keeps Max(0, .) under the sqrt so a roundoff-negative h cannot NaN the
        // spectrum. A call-site clamp on top of that is exactly how the fused and
        // legacy paths came to disagree.
    }
}
