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
#include <AMReX_Geometry.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_SmallMatrix.H>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>

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
    int n_pin = 0;               // state slots Dirichlet-pinned at x-hi (e.g. P=0)
    std::vector<int> pin_comp;   // -> the Chorin pressure reference
    std::vector<Real> pin_val;
};
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
        const int nin = bc.n_in, npin = bc.n_pin;
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
            if (i > ihi)                                    // x-hi Dirichlet pin
                for (int m = 0; m < npin; ++m)              //   ghost = 2*val - edge
                    a(i, j, 0, pc[m]) = 2.0*pv[m] - a(ii, jj, 0, pc[m]);
        });
    }
}

// ── dense Gaussian elimination (small systems; single-rank pressure solve) ──
static bool solveDense(std::vector<double>& A, std::vector<double>& b, int n)
{
    for (int col = 0; col < n; ++col) {
        int piv = col; double best = std::abs(A[col*n+col]);
        for (int r = col+1; r < n; ++r) {
            double v = std::abs(A[r*n+col]); if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-30) return false;
        if (piv != col) {
            for (int c = 0; c < n; ++c) std::swap(A[col*n+c], A[piv*n+c]);
            std::swap(b[col], b[piv]);
        }
        double d = A[col*n+col];
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = A[r*n+col] / d;
            if (f == 0.0) continue;
            for (int c = col; c < n; ++c) A[r*n+c] -= f * A[col*n+c];
            b[r] -= f * b[col];
        }
    }
    for (int i = 0; i < n; ++i) b[i] /= A[i*n+i];
    return true;
}

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
    { ParmParse pp("pin"); Vector<int> c; Vector<Real> v;       // x-hi Dirichlet pin
      if (pp.queryarr("comp", c)) { pp.getarr("val", v); bc.n_pin = c.size();
          bc.pin_comp.assign(c.begin(), c.end()); bc.pin_val.assign(v.begin(), v.end()); } }


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

    // ── pressure residual functor (fills live deriv aux, evaluates source) ──
    // input_aux (frozen: derivatives of the predictor state) are filled ONCE.
    auto pressureResidual = [&](MultiFab& Rout, Real dt) {
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqs, ModelPress::deriv_aux, ModelPress::n_deriv_aux, dx, dy);
        Aqs.FillBoundary(geom.periodicity());
        setdt(pPress, dt); auto P = packp<ModelPress::n_parameters>(pPress);
        for (MFIter mfi(Rout); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto Qa = Q.const_array(mfi); auto Aa = Aqs.const_array(mfi);
            auto Ra = Rout.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                SmallMatrix<Real,NS,1> q;
                SmallMatrix<Real,std::max(ModelPress::n_dof_qaux,1),1> a;
                for (int n=0;n<NS;++n) q(n,0)=Qa(i,j,0,n);
                for (int n=0;n<ModelPress::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
                auto r = ModelPress::source(q,a,P);
                for (int m=0;m<ModelPress::n_dof_eq;++m) Ra(i,j,0,m)=r(m,0);
            });
        }
    };
    auto solvePressure = [&](Real dt) {
        const int nP = ModelPress::n_dof_eq;
        const long nc = domain.numPts();
        const long N = nP * nc;
        // frozen input_aux: derivatives of the (now predictor-updated) state
        fillBC(Q, geom, bc);
        fill_deriv_aux(Q, Aqs, ModelPress::input_aux, ModelPress::n_input_aux, dx, dy);
        Aqs.FillBoundary(geom.periodicity());
        auto scatterP = [&](const std::vector<double>& Pv) {
            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox(); auto Qa = Q.array(mfi);
                LoopOnCpu(bx, [&](int i,int j,int k){
                    long c = (long)(i-domain.smallEnd(0)) + (long)ncell[0]*(j-domain.smallEnd(1));
                    for (int m=0;m<nP;++m) Qa(i,j,0,ModelPress::e2s[m]) = Pv[m*nc + c];
                });
            }
        };
        auto gatherR = [&](MultiFab& R, std::vector<double>& out) {
            for (MFIter mfi(R); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox(); auto Ra = R.const_array(mfi);
                LoopOnCpu(bx, [&](int i,int j,int k){
                    long c = (long)(i-domain.smallEnd(0)) + (long)ncell[0]*(j-domain.smallEnd(1));
                    for (int m=0;m<nP;++m) out[m*nc + c] = Ra(i,j,0,m);
                });
            }
        };
        MultiFab R(ba, dm, nP, 0);
        std::vector<double> zero(N,0.0), R0(N), rj(N);
        scatterP(zero); pressureResidual(R, dt); gatherR(R, R0);
        std::vector<double> A((long)N*N, 0.0);
        std::vector<double> ej(N, 0.0);
        for (long jcol=0;jcol<N;++jcol) {
            ej[jcol]=1.0; scatterP(ej); pressureResidual(R,dt); gatherR(R,rj); ej[jcol]=0.0;
            for (long i=0;i<N;++i) A[i*N+jcol] = rj[i]-R0[i];     // matvec(e_j)
        }
        std::vector<double> b(N); for (long i=0;i<N;++i) b[i]=-R0[i];
        if (!solveDense(A,b,(int)N)) { amrex::Print()<<"  pressure: singular\n"; scatterP(zero); return; }
        scatterP(b);
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
                auto ev = NumericsPred::local_max_abs_eigenvalue(q,a,P,nx);
                if (ev(0,0)>maxev) maxev=ev(0,0);
            });
        }
        ParallelDescriptor::ReduceRealMax(maxev);
        return (maxev>1e-14) ? cfl*amrex::min(dx,dy)/maxev : 1e-3;
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
    Print() << "chorin done: steps="<<step<<" t="<<time<<"\n";

    // convenience: 1-D ASCII profile (x + all state slots) for verification
    // plots without an AMReX plotfile reader (single-rank gather along j=jmid).
    {
        fillBC(Q, geom, bc);
        const int jmid = (domain.smallEnd(1)+domain.bigEnd(1))/2;
        FILE* fp = (ParallelDescriptor::IOProcessor()) ? std::fopen("profile.dat","w") : nullptr;
        if (fp) { std::fprintf(fp, "# x"); for (int n=0;n<NS;++n) std::fprintf(fp," var%d",n);
                  std::fprintf(fp,"\n"); }
        for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox(); auto a = Q.const_array(mfi);
            LoopOnCpu(bx, [&](int i,int j,int k){
                if (j!=jmid || !fp) return;
                Real x = geom.ProbLo(0) + (i+0.5)*dx;
                std::fprintf(fp, "%.10g", x);
                for (int n=0;n<NS;++n) std::fprintf(fp, " %.10g", a(i,j,0,n));
                std::fprintf(fp, "\n");
            });
        }
        if (fp) { std::fclose(fp); Print() << "wrote profile.dat\n"; }
    }
    }
    amrex::Finalize();
    return 0;
}
