#pragma once
// Shared Chorin helpers (state layout, FD aux, params, BCs, matrix-free
// pressure operator + preconditioners) used by chorin_main.cpp (single level)
// and chorin_amr.cpp (AmrCore multi-level). Split out of chorin_main.cpp.

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GMRES.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_SmallMatrix.H>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <memory>

#include "ModelPred.H"
#include "NumericsPred.H"
#include "ModelPress.H"
#include "ModelCorr.H"

using namespace amrex;

// forward decl: raster loader (init_solution.cpp) — bed/IC ingest, shared w/ SWE
void readRasterIntoComponent(const std::string& filename, const Geometry& geom,
                             MultiFab& mf, int comp);

static constexpr int NS = ModelPred::n_dof_q;   // shared state slots (= n_state)

// ── central-FD derivative of component `s` to order (xo,yo) at cell (i,j) ────
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real fd_deriv(Array4<const Real> const& Q, int i, int j, int s,
              int xo, int yo, Real dx, Real dy)
{
    // apply the x-stencil, then the y-stencil (separable central differences).
    auto dxop = [&](int ii, int jj) -> Real {
        if (xo == 0) return Q(ii, jj, 0, s);
        if (xo == 1) return (Q(ii+1, jj, 0, s) - Q(ii-1, jj, 0, s)) / (2.0*dx);
        return (Q(ii+1, jj, 0, s) - 2.0*Q(ii, jj, 0, s) + Q(ii-1, jj, 0, s)) / (dx*dx);
    };
    if (yo == 0) return dxop(i, j);
    if (yo == 1) return (dxop(i, j+1) - dxop(i, j-1)) / (2.0*dy);
    return (dxop(i, j+1) - 2.0*dxop(i, j) + dxop(i, j-1)) / (dy*dy);
}

// ── fill the derivative-aux rows of `Qaux` from a spec table (generic) ──────
template <int NSPEC>
void fill_deriv_aux(MultiFab& Q, MultiFab& Qaux, const int (&spec)[NSPEC][4],
                    int nspec, Real dx, Real dy)
{
    for (MFIter mfi(Qaux); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(Qaux.nGrow());
        auto Qa = Q.const_array(mfi);
        auto Aa = Qaux.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            for (int e = 0; e < nspec; ++e)
                Aa(i, j, 0, spec[e][0]) =
                    fd_deriv(Qa, i, j, spec[e][1], spec[e][2], spec[e][3], dx, dy);
        });
    }
}

// ── pack std::vector params into the per-model SmallMatrix (sizes differ) ────
template <int N>
static amrex::SmallMatrix<Real,N,1> packp(const std::vector<Real>& v)
{
    amrex::SmallMatrix<Real,N,1> m;
    for (int i = 0; i < N; ++i) m(i,0) = v[i];
    return m;
}

// ── physical BCs: zeroGradient (extrap) everywhere, optional left inflow ────
// Generic + case-driven: the case sets `inflow.*` in the inputs to prescribe
// state components on the x-lo face (a Lambda-style inflow); all other faces and
// components are zero-gradient.  No model structure here.
struct BCInfo {
    int n_in = 0;
    std::vector<int> comp;       // state slots to prescribe at x-lo (inflow)
    std::vector<Real> val;
    int n_pin = 0;               // state slots Dirichlet-pinned (e.g. P=0)
    std::vector<int> pin_comp;   // -> the Chorin pressure reference
    std::vector<Real> pin_val;
    int pin_all = 0;             // 0: pin at x-hi only; 1: pin on ALL boundaries
    // ── model-driven BCs (preferred): dispatch ModelPred::boundary_conditions ──
    bool model_bc = false;       // if true, ignore the inflow/pin fields above and
    int side_tag[4] = {-1,-1,-1,-1};  // use the MODEL's per-tag BC. side_tag =
    std::vector<Real> params;    //   {x-lo,x-hi,y-lo,y-hi} tag index (-1 -> zeroGrad).
    Real time = 0.0;
};
// Physical-boundary ghost fill.  Preferred path (model_bc): dispatch the model's
// own ``boundary_conditions(bc_idx, ...)`` Piecewise (inflow / wall / extrap / the
// pressure pin all come from the MODEL, exactly like the SWE ZoomyAmr driver).
// Fallback (no model tags): zero-gradient + the inputs-driven inflow/pin.
static void fillBC(MultiFab& Q, const Geometry& geom, const BCInfo& bc)
{
    Q.FillBoundary(geom.periodicity());
    const Box& dom = geom.Domain();
    const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
    const int jl = dom.smallEnd(1), jh = dom.bigEnd(1);
    if (bc.model_bc) {                                   // ── MODEL boundary_conditions
        auto P = packp<ModelPred::n_parameters>(bc.params);
        const int s0=bc.side_tag[0], s1=bc.side_tag[1], s2=bc.side_tag[2], s3=bc.side_tag[3];
        const Real t = bc.time;
        auto plo = geom.ProbLoArray(); auto dxv = geom.CellSizeArray();
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& gbx = mfi.growntilebox(Q.nGrow());
            auto a = Q.array(mfi);
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                int ii = amrex::min(amrex::max(i, ilo), ihi);
                int jj = amrex::min(amrex::max(j, jl), jh);
                if (i == ii && j == jj) return;          // interior
                amrex::ignore_unused(s2, s3);
                int bc_idx = -1; SmallMatrix<Real,ModelPred::dimension,1> nh{};
                if (i < ilo)      { bc_idx = s0; nh(0,0) = -1.0; }
                else if (i > ihi) { bc_idx = s1; nh(0,0) =  1.0; }
                else if constexpr (ModelPred::dimension >= 2) {
                    if (j < jl)      { bc_idx = s2; nh(1,0) = -1.0; }
                    else if (j > jh) { bc_idx = s3; nh(1,0) =  1.0; }
                }
                if (bc_idx < 0) {                        // untagged side -> zeroGrad
                    for (int n=0;n<NS;++n) a(i,j,0,n) = a(ii,jj,0,n); return; }
                SmallMatrix<Real,NS,1> qint;
                for (int n=0;n<NS;++n) qint(n,0) = a(ii,jj,0,n);
                SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1> aint{};
                SmallMatrix<Real,3,1> X;
                X(0,0)=plo[0]+(i+0.5)*dxv[0];
                X(1,0)=(ModelPred::dimension>=2)?plo[1]+(j+0.5)*dxv[1]:0.0; X(2,0)=0.0;
                auto qg = ModelPred::boundary_conditions(bc_idx, qint, aint, nh, X, t, dxv[0]);
                for (int n=0;n<NS;++n) a(i,j,0,n) = qg(n,0);
            });
        }
        return;
    }
    const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1);
    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox(Q.nGrow());
        auto a = Q.array(mfi);
        const int ncomp = NS;
        const int nin = bc.n_in, npin = bc.n_pin, pinall = bc.pin_all;
        int  ic[16]; Real iv[16]; int pc[16]; Real pv[16];
        for (int m = 0; m < nin  && m < 16; ++m) { ic[m] = bc.comp[m];     iv[m] = bc.val[m]; }
        for (int m = 0; m < npin && m < 16; ++m) { pc[m] = bc.pin_comp[m]; pv[m] = bc.pin_val[m]; }
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int ii = amrex::min(amrex::max(i, ilo), ihi);   // clamp into domain
            int jj = amrex::min(amrex::max(j, jlo), jhi);
            if (i == ii && j == jj) return;                 // interior: skip
            for (int n = 0; n < ncomp; ++n) a(i, j, 0, n) = a(ii, jj, 0, n);  // zeroGrad
            if (i < ilo)                                    // x-lo inflow override
                for (int m = 0; m < nin; ++m) a(i, j, 0, ic[m]) = iv[m];
            const bool do_pin = pinall ? true : (i > ihi);  // all faces, or x-hi only
            if (do_pin)                                     // Dirichlet pin ghost = 2*val-edge
                for (int m = 0; m < npin; ++m)
                    a(i, j, 0, pc[m]) = 2.0*pv[m] - a(ii, jj, 0, pc[m]);
        });
    }
}

// ── Chorin pressure solve: matrix-free GMRES on the coupled pressure block ──
// The pressure system is LINEAR: source(P) = A·P + R0, so the operator is applied
// matrix-free as  A·x = source(x) - source(0)  (no FD-JVP epsilon needed).  This
// is dimension- and mode-AGNOSTIC — the vector carries n_dof_eq pressure modes per
// cell and A·x is a full ModelPress::source sweep (central-FD ∂P/∂²P over the
// structured grid, any dimension) — so the same solver serves 2-D/3-D moment
// systems and ML-VAM with no per-model code (option 2 of the Escalante fast-solve
// discussion: MG/Krylov in space, block over the modes).  Preconditioner is
// identity; the x-hi pressure pin makes A non-singular.
static constexpr int NP = ModelPress::n_dof_eq;

// scatter a mode-vector X (ncomp=NP) into the shared state's pressure slots e2s
static void scatterP(MultiFab& Q, const MultiFab& X)
{
    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto Qa = Q.array(mfi); auto Xa = X.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            for (int m=0;m<NP;++m) Qa(i,j,0,ModelPress::e2s[m]) = Xa(i,j,0,m);
        });
    }
}

// evaluate ModelPress::source into `out` (ncomp=NP) for the current P in Q; refills
// the LIVE derivative aux (P-dependent) — the FROZEN input_aux stay as pre-filled.
static void pressSource(MultiFab& Q, MultiFab& Aqs, const Geometry& geom,
                        const BCInfo& bc, const std::vector<Real>& pv,
                        Real dx, Real dy, MultiFab& out)
{
    fillBC(Q, geom, bc);
    fill_deriv_aux(Q, Aqs, ModelPress::deriv_aux, ModelPress::n_deriv_aux, dx, dy);
    Aqs.FillBoundary(geom.periodicity());
    auto P = packp<ModelPress::n_parameters>(pv);
    for (MFIter mfi(out); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto Qa = Q.const_array(mfi); auto Aa = Aqs.const_array(mfi); auto Oa = out.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            SmallMatrix<Real,NS,1> q;
            SmallMatrix<Real,std::max(ModelPress::n_dof_qaux,1),1> a;
            for (int n=0;n<NS;++n) q(n,0)=Qa(i,j,0,n);
            for (int n=0;n<ModelPress::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
            auto r = ModelPress::source(q,a,P);
            for (int m=0;m<NP;++m) Oa(i,j,0,m)=r(m,0);
        });
    }
}

// amrex::GMRES linear operator: A·x = source(x) - R0  (R0 = source at P=0)
struct ChorinPressOp {
    using RT = amrex::Real;
    MultiFab* Q=nullptr; MultiFab* Aqs=nullptr; const Geometry* geom=nullptr;
    const BCInfo* bc=nullptr; const std::vector<Real>* pv=nullptr;
    const MultiFab* R0=nullptr; Real dx=0, dy=0;
    BoxArray ba; DistributionMapping dm;
    int ptype = 0;              // 0 identity, 1 Jacobi, 2 multigrid, 3 block-Jacobi
    MultiFab Dinv;             // 1/diag(A) for Jacobi (rebuilt per solve)
    MultiFab Binv;             // per-cell NP×NP inverse mode-block (coupling-aware)
    Real mgscale[8] = {0};     // per-mode  1/c  so  c·∇² ≈ A  (from the diagonal)
    Real mgsign = 1.0;         // MLPoisson sign convention (−∇² vs ∇²): flip if needed
    int mg_vcycles = 4;
    std::unique_ptr<amrex::MLPoisson> mlp;   // scalar-Poisson MG preconditioner
    Box domain;

    void setupMG() {
        amrex::LPInfo info; info.setAgglomeration(true).setConsolidation(true);
        mlp = std::make_unique<amrex::MLPoisson>(
            amrex::Vector<amrex::Geometry>{*geom}, amrex::Vector<amrex::BoxArray>{ba},
            amrex::Vector<amrex::DistributionMapping>{dm}, info);
        mlp->setMaxOrder(2);
        mlp->setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,   // P pinned = 0
                                       amrex::LinOpBCType::Dirichlet,
                                       amrex::LinOpBCType::Dirichlet)},
                         {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                                       amrex::LinOpBCType::Dirichlet,
                                       amrex::LinOpBCType::Dirichlet)});
    }

    MultiFab makeVecRHS() const { return MultiFab(ba, dm, NP, 0); }
    MultiFab makeVecLHS() const { return MultiFab(ba, dm, NP, 0); }
    void assign(MultiFab& l, MultiFab const& r) { MultiFab::Copy(l,r,0,0,NP,0); }
    RT dotProduct(MultiFab const& a, MultiFab const& b) { return MultiFab::Dot(a,0,b,0,NP,0); }
    void increment(MultiFab& l, MultiFab const& r, RT a) { MultiFab::Saxpy(l,a,r,0,0,NP,0); }
    void linComb(MultiFab& l, RT a, MultiFab const& x, RT b, MultiFab const& y) { MultiFab::LinComb(l,a,x,0,b,y,0,0,NP,0); }
    RT norm2(MultiFab const& v) { return std::sqrt(MultiFab::Dot(v,0,v,0,NP,0)); }
    void scale(MultiFab& v, RT f) { v.mult(f,0,NP,0); }
    void setToZero(MultiFab& v) { v.setVal(0.0); }
    void apply(MultiFab& Ax, MultiFab const& x) {
        scatterP(*Q, x);
        pressSource(*Q, *Aqs, *geom, *bc, *pv, dx, dy, Ax);   // Ax = source(x)
        MultiFab::Subtract(Ax, *R0, 0, 0, NP, 0);             // A·x = source(x) - R0
    }
    // M⁻¹ r : identity (0), Jacobi diagonal (1), or per-mode Poisson MG V-cycles (2)
    void precond(MultiFab& l, MultiFab const& r) {
        if (ptype == 1) {
            MultiFab::Copy(l,r,0,0,NP,0); MultiFab::Multiply(l,Dinv,0,0,NP,0);
        } else if (ptype == 2) {
            if (!mlp) setupMG();
            MultiFab rmf(ba,dm,1,0), phi(ba,dm,1,1);
            for (int m=0;m<NP;++m) {
                MultiFab::Copy(rmf, r, m, 0, 1, 0);
                phi.setVal(0.0); mlp->setLevelBC(0, &phi);   // homogeneous Dirichlet
                amrex::MLMG mlmg(*mlp);
                mlmg.setVerbose(0); mlmg.setBottomVerbose(0);
                mlmg.setFixedIter(mg_vcycles);               // K V-cycles as preconditioner
                mlmg.solve({&phi}, {&rmf}, 1.e-10, 0.0);     // ∇²φ = r_m
                MultiFab::Copy(l, phi, 0, m, 1, 0);
                l.mult(mgscale[m], m, 1, 0);                 // A⁻¹ ≈ (1/c)(∇²)⁻¹
            }
        } else if (ptype == 3) {                             // point-block-Jacobi
            for (MFIter mfi(l); mfi.isValid(); ++mfi) {
                const Box& bx=mfi.validbox();
                auto la=l.array(mfi); auto ra=r.const_array(mfi); auto bi=Binv.const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    for (int mo=0;mo<NP;++mo){ Real s=0.0;
                        for (int mc=0;mc<NP;++mc) s += bi(i,j,0,mo*NP+mc)*ra(i,j,0,mc);
                        la(i,j,0,mo)=s; }
                });
            }
        } else if (ptype == 4) {                             // block-diagonal MG
            MultiFab rmf(ba,dm,1,0), phi(ba,dm,1,1);
            for (int m=0;m<NP;++m) {
                MultiFab::Copy(rmf, r, m, 0, 1, 0);
                phi.setVal(0.0); Dk[m]->setLevelBC(0, &phi);
                amrex::MLMG mlmg(*Dk[m]);
                mlmg.setVerbose(0); mlmg.setBottomVerbose(0);
                mlmg.setFixedIter(mg_vcycles);
                mlmg.solve({&phi}, {&rmf}, 1.e-10, 0.0);     // D_m φ = r_m
                MultiFab::Copy(l, phi, 0, m, 1, 0);
            }
        } else {
            MultiFab::Copy(l,r,0,0,NP,0);
        }
    }
    // block-diagonal MG (ptype 4): one MLABecLaplacian per mode,
    // D_m = a·A0[m][m] − ∇·(β∇),  β_x=−Axx[m][m], β_y=−Ayy[m][m] (analytic REQ-94
    // coefficients). Reaction+anisotropic diffusion = the true diagonal block; the
    // off-diagonal mode coupling is left to the outer GMRES.
    std::vector<std::unique_ptr<amrex::MLABecLaplacian>> Dk;
    void setupBlockMG() {
        amrex::LPInfo info; info.setAgglomeration(true).setConsolidation(true);
        Dk.clear();
        for (int m=0;m<NP;++m) {
            auto op = std::make_unique<amrex::MLABecLaplacian>(
                amrex::Vector<amrex::Geometry>{*geom}, amrex::Vector<amrex::BoxArray>{ba},
                amrex::Vector<amrex::DistributionMapping>{dm}, info);
            op->setMaxOrder(2);
            op->setDomainBC({AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                                          amrex::LinOpBCType::Dirichlet,
                                          amrex::LinOpBCType::Dirichlet)},
                            {AMREX_D_DECL(amrex::LinOpBCType::Dirichlet,
                                          amrex::LinOpBCType::Dirichlet,
                                          amrex::LinOpBCType::Dirichlet)});
            op->setScalars(1.0, 1.0);
            Dk.push_back(std::move(op));
        }
    }
    void assembleBlockMG() {
        if (Dk.empty()) setupBlockMG();
        auto P = packp<ModelPress::n_parameters>(*pv);
        MultiFab acc(ba,dm,NP,0), cxx(ba,dm,NP,1), cyy(ba,dm,NP,1);
        for (MFIter mfi(acc); mfi.isValid(); ++mfi) {
            const Box& bx=mfi.validbox();
            auto qa=Q->const_array(mfi); auto aa=Aqs->const_array(mfi);
            auto ac=acc.array(mfi); auto bx_=cxx.array(mfi); auto by_=cyy.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                SmallMatrix<Real,NS,1> q;
                SmallMatrix<Real,std::max(ModelPress::n_dof_qaux,1),1> a;
                for (int n=0;n<NS;++n) q(n,0)=qa(i,j,0,n);
                for (int n=0;n<ModelPress::n_dof_qaux;++n) a(n,0)=aa(i,j,0,n);
                auto A0=ModelPress::pressure_A0(q,a,P);
                auto Axx=ModelPress::pressure_Axx(q,a,P);
                auto Ayy=ModelPress::pressure_Ayy(q,a,P);
                for (int m=0;m<NP;++m){                 // clamp ≥0 for an SPD
                    ac(i,j,0,m)=amrex::max(A0(m*NP+m,0), 0.0);   // preconditioner operator
                    bx_(i,j,0,m)=amrex::max(-Axx(m*NP+m,0), 1e-30);
                    by_(i,j,0,m)=amrex::max(-Ayy(m*NP+m,0), 0.0); // 0 for 1-horizontal
                }
            });
        }
        cxx.FillBoundary(geom->periodicity()); cyy.FillBoundary(geom->periodicity());
        for (int m=0;m<NP;++m) {
            MultiFab am(ba,dm,1,0); MultiFab::Copy(am,acc,m,0,1,0);
            Dk[m]->setACoeffs(0, am);
            Array<MultiFab,AMREX_SPACEDIM> face;
            for (int d=0;d<AMREX_SPACEDIM;++d)
                face[d].define(amrex::convert(ba, IntVect::TheDimensionVector(d)), dm, 1, 0);
            for (MFIter mfi(am); mfi.isValid(); ++mfi) {
                auto fx=face[0].array(mfi); auto cx=cxx.const_array(mfi);
                ParallelFor(mfi.nodaltilebox(0), [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    fx(i,j,0)=0.5*(cx(i-1,j,0,m)+cx(i,j,0,m)); });
#if (AMREX_SPACEDIM >= 2)
                auto fy=face[1].array(mfi); auto cy=cyy.const_array(mfi);
                ParallelFor(mfi.nodaltilebox(1), [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    fy(i,j,0)=0.5*(cy(i,j-1,0,m)+cy(i,j,0,m)); });
#endif
            }
            Dk[m]->setBCoeffs(0, amrex::GetArrOfConstPtrs(face));
        }
    }
    // Extract diag(A) matrix-free by a 9-colour (i%3,j%3) × NP-mode probe: exact
    // for the 5-point (radius-1) pressure stencil.  Rebuilt each solve (depth-
    // dependent).  ~9·NP matvecs; amortised by the GMRES-iteration cut.
    void buildDiagonal() {
        if (ptype < 1) return;
        if (!Dinv.ok()) Dinv.define(ba, dm, NP, 0);
        if (ptype==3 && !Binv.ok()) Binv.define(ba, dm, NP*NP, 0);
        MultiFab diag(ba,dm,NP,0), probe(ba,dm,NP,0), Ap(ba,dm,NP,0);
        MultiFab Ablk; if (ptype==3) Ablk.define(ba,dm,NP*NP,0);
        diag.setVal(0.0);
        const int ilo=domain.smallEnd(0), jlo=domain.smallEnd(1);
        const int pt=ptype;
        for (int mc=0;mc<NP;++mc) for (int cx=0;cx<3;++cx) for (int cy=0;cy<3;++cy) {
            probe.setVal(0.0);
            for (MFIter mfi(probe); mfi.isValid(); ++mfi) {
                const Box& bx=mfi.validbox(); auto pa=probe.array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    if (((i-ilo)%3)==cx && ((j-jlo)%3)==cy) pa(i,j,0,mc)=1.0; });
            }
            apply(Ap, probe);                                // A·e_colour (column mc)
            auto* blk = (pt==3) ? &Ablk : nullptr;
            for (MFIter mfi(diag); mfi.isValid(); ++mfi) {
                const Box& bx=mfi.validbox(); auto da=diag.array(mfi); auto aa=Ap.const_array(mfi);
                auto ba_= (pt==3)?blk->array(mfi):Array4<Real>{};
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    if (((i-ilo)%3)!=cx || ((j-jlo)%3)!=cy) return;
                    da(i,j,0,mc)=aa(i,j,0,mc);               // diagonal (mc,mc)
                    if (pt==3) for (int mo=0;mo<NP;++mo) ba_(i,j,0,mo*NP+mc)=aa(i,j,0,mo);
                });
            }
        }
        for (MFIter mfi(Dinv); mfi.isValid(); ++mfi) {       // Dinv = 1/diag (guarded)
            const Box& bx=mfi.validbox(); auto di=Dinv.array(mfi); auto da=diag.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                for (int m=0;m<NP;++m){ Real d=da(i,j,0,m); di(i,j,0,m)=(std::abs(d)>1e-30)?1.0/d:1.0; } });
        }
        if (ptype==3) {                                      // invert the NP×NP block per cell
            for (MFIter mfi(Binv); mfi.isValid(); ++mfi) {
                const Box& bx=mfi.validbox(); auto bi=Binv.array(mfi); auto ab=Ablk.const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
                    Real A[NP*NP], I[NP*NP];
                    for (int e=0;e<NP*NP;++e){ A[e]=ab(i,j,0,e); I[e]=0.0; }
                    for (int d=0;d<NP;++d) I[d*NP+d]=1.0;
                    for (int col=0;col<NP;++col){         // Gauss-Jordan (NP small)
                        Real piv=A[col*NP+col]; if (std::abs(piv)<1e-30) piv=1e-30;
                        Real ip=1.0/piv;
                        for (int c=0;c<NP;++c){ A[col*NP+c]*=ip; I[col*NP+c]*=ip; }
                        for (int r=0;r<NP;++r){ if(r==col) continue; Real f=A[r*NP+col];
                            for (int c=0;c<NP;++c){ A[r*NP+c]-=f*A[col*NP+c]; I[r*NP+c]-=f*I[col*NP+c]; } }
                    }
                    for (int e=0;e<NP*NP;++e) bi(i,j,0,e)=I[e];
                });
            }
        }
        if (ptype == 2) {                                    // per-mode Laplacian scale
            const Real ldiag = -2.0/(dx*dx) - 2.0/(dy*dy);   // ∇² discrete diagonal
            const Real ncell = Real(domain.numPts());
            for (int m=0;m<NP;++m) {
                Real meandiag = diag.sum(m) / ncell;         // mean diag(A) for mode m
                mgscale[m] = mgsign * ((std::abs(meandiag)>1e-30) ? ldiag/meandiag : 1.0);
            }
        }
    }
};
