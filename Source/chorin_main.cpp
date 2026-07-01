/*---------------------------------------------------------------------------*\
  chorin_main — generic Chorin pressure-projection driver for a zoomy
  NON-hydrostatic split model (task 0029).  It is MODEL-AGNOSTIC: every
  model/Riemann structure comes from the printed headers

      ModelPred.H  / NumericsPred.H   (explicit predictor sub-model + flux)
      ModelPress.H                    (elliptic pressure block)
      ModelCorr.H                     (velocity corrector)
      UserFunctions.H

  and the driver only does generic wiring over their compile-time meta
  (n_dof_q, n_dof_eq, e2s[], deriv_aux[][4], input_aux[][4], n_dof_qaux).
  Anything zoomy_core can chorin-split + print runs here unchanged.

  Per step (shared state Q[0..n_state-1]):
    predictor : explicit Rusanov (NumericsPred) on SM_pred's e2s rows; P frozen.
    pressure  : solve ModelPress::source(P, dP, ddP)=0 for the pressure modes
                (linear → residual-by-probe + dense solve; the residual fills the
                derivative aux by central FD on the structured grid).
    corrector : ModelCorr::update_variables(Q,Qaux,p) → the velocity-mode rows.

  The derivative auxiliaries (foam fills them in a mesh-aware update_aux) are
  filled here by central finite differences from each header's deriv_aux/
  input_aux specs {aux_row, state_index, x_order, y_order}.

  Single level, structured 2-D (a 1-D case is a 1-cell-thick slice).  The dense
  pressure solve is single-rank (fine for verification cases); a matrix-free
  amrex::GMRES is the drop-in scale upgrade (same residual functor).
\*---------------------------------------------------------------------------*/
#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_GMRES.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_MLMG.H>
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
};                               //    (symmetric far-field ref for radial cases)
static void fillBC(MultiFab& Q, const Geometry& geom, const BCInfo& bc)
{
    Q.FillBoundary(geom.periodicity());
    const Box& dom = geom.Domain();
    const int ilo = dom.smallEnd(0), ihi = dom.bigEnd(0);
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
        } else {
            MultiFab::Copy(l,r,0,0,NP,0);
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

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
    // ── inputs / geometry ───────────────────────────────────────────────────
    ParmParse ppa("amr");   Vector<int> ncell;  ppa.getarr("n_cell", ncell);
    ParmParse ppg("geometry");
    Vector<Real> plo, phi;  ppg.getarr("prob_lo", plo); ppg.getarr("prob_hi", phi);
    Vector<int> isper(AMREX_SPACEDIM, 0);
    if (ppg.contains("is_periodic")) ppg.getarr("is_periodic", isper);
    int max_grid = 1024; ppa.query("max_grid_size", max_grid);

    Box domain(IntVect(0,0), IntVect(ncell[0]-1, ncell[1]-1));
    RealBox rb({plo[0], plo[1]}, {phi[0], phi[1]});
    Geometry geom(domain, rb, 0, {isper[0], isper[1]});
    BoxArray ba(domain); ba.maxSize(max_grid);
    DistributionMapping dm(ba);
    const Real dx = geom.CellSize(0), dy = geom.CellSize(1);

    ParmParse pps("solver");
    Real t_end = 1.0, cfl = 0.3, dt_fixed = -1.0, plot_dt = 1e30;
    pps.query("time_end", t_end); pps.query("cfl", cfl);
    pps.query("dt", dt_fixed);
    ParmParse ppo("output"); ppo.query("plot_dt_interval", plot_dt);

    // ── state + per-sub-model aux ───────────────────────────────────────────
    const int ng = 2;
    MultiFab Q(ba, dm, NS, ng);  Q.setVal(0.0);
    MultiFab Aqp(ba, dm, std::max(ModelPred::n_dof_qaux,1), 1);  Aqp.setVal(0.0);
    MultiFab Aqs(ba, dm, std::max(ModelPress::n_dof_qaux,1), 1); Aqs.setVal(0.0);
    MultiFab Aqc(ba, dm, std::max(ModelCorr::n_dof_qaux,1), 1);  Aqc.setVal(0.0);

    // ── IC: raster bed (comp 0) + depth (comp 1); other modes start at 0 ─────
    std::string dem, rel;
    { ParmParse pp("init"); pp.query("dem_file", dem); pp.query("release_file", rel); }
    if (!rel.empty()) readRasterIntoComponent(rel, geom, Q, 1);
    if (!dem.empty()) readRasterIntoComponent(dem, geom, Q, 0);

    // ── parameters (model defaults + inputs overrides); dt is the LAST slot ──
    auto mkparams = [&](auto names, auto defs) {
        std::vector<Real> p(defs.begin(), defs.end());
        ParmParse pp("params");
        for (size_t i = 0; i < names.size(); ++i) { Real v; if (pp.query(names[i].c_str(), v)) p[i] = v; }
        return p;
    };
    std::vector<Real> pPred  = mkparams(ModelPred::parameter_names(),  ModelPred::default_parameters());
    std::vector<Real> pPress = mkparams(ModelPress::parameter_names(), ModelPress::default_parameters());
    std::vector<Real> pCorr  = mkparams(ModelCorr::parameter_names(),  ModelCorr::default_parameters());
    auto setdt = [&](std::vector<Real>& p, Real dt) { p.back() = dt; };   // dt in p(last)

    // ── inflow BC spec (case-level): inflow.comp = {slots}, inflow.val = {vals} ─
    BCInfo bc;
    { ParmParse pp("inflow"); Vector<int> c; Vector<Real> v;
      if (pp.queryarr("comp", c)) { pp.getarr("val", v); bc.n_in = c.size();
          bc.comp.assign(c.begin(), c.end()); bc.val.assign(v.begin(), v.end()); } }
    { ParmParse pp("pin"); Vector<int> c; Vector<Real> v;       // Dirichlet pressure pin
      if (pp.queryarr("comp", c)) { pp.getarr("val", v); bc.n_pin = c.size();
          bc.pin_comp.assign(c.begin(), c.end()); bc.pin_val.assign(v.begin(), v.end()); }
      pp.query("all", bc.pin_all); }


    // ── predictor explicit RHS (Rusanov flux + NCP + source) on e2s rows ────
    auto predictor = [&](Real dt) {
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqp, ModelPred::deriv_aux, ModelPred::n_deriv_aux, dx, dy);
        Aqp.FillBoundary(geom.periodicity());
        auto P = packp<ModelPred::n_parameters>(pPred);
        MultiFab RHS(ba, dm, ModelPred::n_dof_eq, 0); RHS.setVal(0.0);
        for (MFIter mfi(RHS); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto Qa = Q.const_array(mfi); auto Aa = Aqp.const_array(mfi);
            auto Ra = RHS.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                constexpr int NE = ModelPred::n_dof_eq;       // numerics state = e2s rows
                SmallMatrix<Real,NE,1> qC, qL, qR, qD, qU;
                SmallMatrix<Real,NS,1> q8;                    // full state for the source
                SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1> aC,aL,aR,aD,aU;
                auto load = [&](int ii,int jj, SmallMatrix<Real,NE,1>& q,
                                SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1>& a){
                    for (int n=0;n<NE;++n) q(n,0)=Qa(ii,jj,0,ModelPred::e2s[n]);
                    for (int n=0;n<ModelPred::n_dof_qaux;++n) a(n,0)=Aa(ii,jj,0,n); };
                load(i,j,qC,aC); load(i-1,j,qL,aL); load(i+1,j,qR,aR);
                load(i,j-1,qD,aD); load(i,j+1,qU,aU);
                for (int n=0;n<NS;++n) q8(n,0)=Qa(i,j,0,n);
                SmallMatrix<Real,ModelPred::dimension,1> nx{}; nx(0,0)=1.0;
                auto fXR = NumericsPred::numerical_flux(qC,qR,aC,aR,P,nx);
                auto fXL = NumericsPred::numerical_flux(qL,qC,aL,aC,P,nx);
                auto gXR = NumericsPred::numerical_fluctuations(qC,qR,aC,aR,P,nx);
                auto gXL = NumericsPred::numerical_fluctuations(qL,qC,aL,aC,P,nx);
                auto src = ModelPred::source(q8,aC,P);
                for (int n=0;n<NE;++n)
                    Ra(i,j,0,n) = (-(fXR(n,0)-fXL(n,0)) - gXR(NE+n,0) - gXL(n,0))/dx + src(n,0);
                if constexpr (ModelPred::dimension >= 2) {   // y faces (2-horizontal models)
                    SmallMatrix<Real,ModelPred::dimension,1> ny{}; ny(1,0)=1.0;
                    auto fYR = NumericsPred::numerical_flux(qC,qU,aC,aU,P,ny);
                    auto fYL = NumericsPred::numerical_flux(qD,qC,aD,aC,P,ny);
                    auto gYR = NumericsPred::numerical_fluctuations(qC,qU,aC,aU,P,ny);
                    auto gYL = NumericsPred::numerical_fluctuations(qD,qC,aD,aC,P,ny);
                    for (int n=0;n<NE;++n)
                        Ra(i,j,0,n) += (-(fYR(n,0)-fYL(n,0)) - gYR(NE+n,0) - gYL(n,0))/dy;
                }
            });
        }
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto Qa = Q.array(mfi); auto Ra = RHS.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                for (int n=0;n<ModelPred::n_dof_eq;++n)
                    Qa(i,j,0,ModelPred::e2s[n]) += dt * Ra(i,j,0,n);
            });
        }
    };

    // ── pressure solve: matrix-free GMRES on the coupled pressure block ─────
    // A·P = -R0 with A·x = source(x)-source(0); R0 = source(0) is the frozen RHS.
    // Dimension/mode-agnostic (option 2): scales to 2-D/3-D moment systems &
    // ML-VAM with no per-model code — MG/Krylov in space, block over the modes.
    MultiFab Pmf(ba, dm, NP, 0), R0(ba, dm, NP, 0), rhs(ba, dm, NP, 0);
    int precond_type = 0, mg_vcycles = 4;
    Real mg_sign = 1.0;
    { ParmParse pp("precond"); pp.query("type", precond_type); pp.query("vcycles", mg_vcycles);
      pp.query("mgsign", mg_sign); }
    ChorinPressOp op;
    op.Q=&Q; op.Aqs=&Aqs; op.geom=&geom; op.bc=&bc; op.pv=&pPress;
    op.R0=&R0; op.dx=dx; op.dy=dy; op.ba=ba; op.dm=dm;
    op.ptype=precond_type; op.domain=domain; op.mg_vcycles=mg_vcycles; op.mgsign=mg_sign;
    int gmres_restart = 30;
    { ParmParse pp("gmres"); pp.query("restart", gmres_restart); }
    GMRES<MultiFab, ChorinPressOp> gmres;
    gmres.define(op); gmres.setVerbose(0); gmres.setRestartLength(gmres_restart);
    long presIters = 0, presSolves = 0;                      // iteration accounting
    auto solvePressure = [&](Real dt) {
        setdt(pPress, dt);
        // frozen input_aux: derivatives of the (predictor-updated) state — once
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqs, ModelPress::input_aux, ModelPress::n_input_aux, dx, dy);
        Aqs.FillBoundary(geom.periodicity());
        Pmf.setVal(0.0);                                     // R0 = source(P=0)
        scatterP(Q, Pmf);
        pressSource(Q, Aqs, geom, bc, pPress, dx, dy, R0);
        op.buildDiagonal();                                  // Jacobi setup (ptype 1)
        MultiFab::Copy(rhs, R0, 0, 0, NP, 0); rhs.mult(-1.0, 0, NP, 0);   // rhs = -R0
        Pmf.setVal(0.0);
        gmres.solve(Pmf, rhs, 1e-10, 0.0);                   // A·P = -R0
        presIters += gmres.getNumIters(); ++presSolves;
        scatterP(Q, Pmf);
        fillBC(Q, geom, bc);
    };

    // ── corrector: ModelCorr::update_variables → e2s_corr rows ──────────────
    auto corrector = [&](Real dt) {
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqc, ModelCorr::deriv_aux, ModelCorr::n_deriv_aux, dx, dy);
        Aqc.FillBoundary(geom.periodicity());
        setdt(pCorr, dt); auto P = packp<ModelCorr::n_parameters>(pCorr);
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto Qa = Q.array(mfi); auto Aa = Aqc.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                SmallMatrix<Real,NS,1> q;
                SmallMatrix<Real,std::max(ModelCorr::n_dof_qaux,1),1> a;
                for (int n=0;n<NS;++n) q(n,0)=Qa(i,j,0,n);
                for (int n=0;n<ModelCorr::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
                auto u = ModelCorr::update_variables(q,a,P);
                for (int m=0;m<ModelCorr::n_dof_eq;++m) Qa(i,j,0,ModelCorr::e2s[m]) = u(m,0);
            });
        }
    };

    // ── dt from the predictor max wavespeed ─────────────────────────────────
    auto compute_dt = [&]() -> Real {
        if (dt_fixed > 0) return dt_fixed;
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqp, ModelPred::deriv_aux, ModelPred::n_deriv_aux, dx, dy);
        auto P = packp<ModelPred::n_parameters>(pPred);
        Real maxev = 0.0;
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto Qa = Q.const_array(mfi); auto Aa = Aqp.const_array(mfi);
            LoopOnCpu(bx, [&](int i,int j,int k){
                SmallMatrix<Real,ModelPred::n_dof_eq,1> q;
                SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1> a;
                for (int n=0;n<ModelPred::n_dof_eq;++n) q(n,0)=Qa(i,j,0,ModelPred::e2s[n]);
                for (int n=0;n<ModelPred::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
                SmallMatrix<Real,ModelPred::dimension,1> nx{}; nx(0,0)=1.0;
                Real ex = NumericsPred::local_max_abs_eigenvalue(q,a,P,nx)(0,0);
                Real inv = ex/dx;
                if constexpr (ModelPred::dimension >= 2) {      // y-direction wavespeed
                    SmallMatrix<Real,ModelPred::dimension,1> ny{}; ny(1,0)=1.0;
                    Real ey = NumericsPred::local_max_abs_eigenvalue(q,a,P,ny)(0,0);
                    inv = amrex::max(inv, ey/dy);
                }
                if (inv>maxev) maxev=inv;                       // maxev := max |λ|/Δ
            });
        }
        ParallelDescriptor::ReduceRealMax(maxev);
        return (maxev>1e-14) ? cfl/maxev : 1e-3;
    };

    // ── time loop ───────────────────────────────────────────────────────────
    Print() << "chorin: n_state="<<NS<<" n_eq_pred="<<ModelPred::n_dof_eq
            << " nP="<<ModelPress::n_dof_eq<<" cells="<<domain.numPts()<<"\n";
    Real time = 0.0; int step = 0, pstep = 0; Real next_plot = 0.0;
    auto writeplt = [&](){
        Vector<std::string> nm(NS); for (int n=0;n<NS;++n) nm[n]="var"+std::to_string(n);
        WriteSingleLevelPlotfile(Concatenate("chk_",pstep,5), Q, nm, geom, time, step);
        ++pstep;
    };
    writeplt();
    while (time < t_end - 1e-12) {
        Real dt = compute_dt(); dt = amrex::min(dt, t_end - time);
        predictor(dt);
        solvePressure(dt);
        corrector(dt);
        time += dt; ++step;
        if (step % 200 == 0 || time >= t_end - 1e-12)
            Print() << "step "<<step<<" t="<<time<<" dt="<<dt<<"\n";
        if (time >= next_plot) { writeplt(); next_plot += plot_dt; }
    }
    writeplt();
    Print() << "chorin done: steps="<<step<<" t="<<time
            << "  pressure GMRES: "<<presSolves<<" solves, avg "
            << (presSolves? double(presIters)/presSolves : 0.0) <<" iters/solve\n";

    // convenience: ASCII dump (x [y] + all state slots) for verification plots
    // without an AMReX plotfile reader.  1-D grid -> the j=jmid ray (profile.dat);
    // 2-D grid -> the full field, all cells (field.dat, for radial-symmetry checks).
    {
        fillBC(Q, geom, bc);
        const int jmid = (domain.smallEnd(1)+domain.bigEnd(1))/2;
        const bool twoD = (domain.length(1) > 1);
        FILE* fp = (ParallelDescriptor::IOProcessor())
                       ? std::fopen(twoD ? "field.dat" : "profile.dat", "w") : nullptr;
        if (fp) { std::fprintf(fp, "# x%s", twoD ? " y" : "");
                  for (int n=0;n<NS;++n) std::fprintf(fp," var%d",n); std::fprintf(fp,"\n"); }
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox(); auto a = Q.const_array(mfi);
            LoopOnCpu(bx, [&](int i,int j,int k){
                if (!fp || (!twoD && j!=jmid)) return;
                Real x = geom.ProbLo(0) + (i+0.5)*dx;
                std::fprintf(fp, "%.10g", x);
                if (twoD) std::fprintf(fp, " %.10g", geom.ProbLo(1) + (j+0.5)*dy);
                for (int n=0;n<NS;++n) std::fprintf(fp, " %.10g", a(i,j,0,n));
                std::fprintf(fp, "\n");
            });
        }
        if (fp) { std::fclose(fp); Print() << "wrote " << (twoD?"field.dat":"profile.dat") << "\n"; }
    }
    }
    amrex::Finalize();
    return 0;
}
