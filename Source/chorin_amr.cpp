/*---------------------------------------------------------------------------*\
  chorin_amr — AMR (AmrCore) Chorin pressure-projection driver.

  Multi-level extension of chorin_main.cpp: block-structured adaptive refinement
  (Berger–Colella / AmrCore) around the moving front + bump, single-cycle (no
  subcycling — all levels advance with the same dt), reusing the generic Chorin
  helpers in chorin_common.H (predictor Rusanov, matrix-free GMRES pressure solve
  with the point-block-Jacobi preconditioner, ModelCorr corrector).

  Coarse–fine coupling: the PREDICTOR (hyperbolic) ghosts are interpolated from the
  coarse level via FillPatch (the load-bearing transport coupling). The per-level
  PRESSURE solve is done with the domain pin + zero-gradient at the coarse–fine
  interface (a level-by-level projection; the coarse P is not imposed as a C-F
  Dirichlet) — a documented approximation; levels are synced by average_down. Good
  enough for an honest AMR run + performance assessment; a full composite-elliptic
  C-F solve is the follow-up.
\*---------------------------------------------------------------------------*/
#include <AMReX_AmrCore.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_PlotFileUtil.H>
#include <iomanip>
#include "chorin_common.H"

void readRasterIntoComponent(const std::string&, const Geometry&, MultiFab&, int);

static constexpr int IDX_B = 0, IDX_H = 1;   // state layout: [b, h, ...]

struct NullFill {
    void operator()(MultiFab&, int, int, IntVect const&, Real, int) const {}
};

class ChorinAmr : public AmrCore {
public:
    ChorinAmr();
    void InitData();
    void Evolve();

protected:
    void MakeNewLevelFromScratch(int lev, Real t, const BoxArray& ba,
                                 const DistributionMapping& dm) override;
    void MakeNewLevelFromCoarse(int lev, Real t, const BoxArray& ba,
                                const DistributionMapping& dm) override;
    void RemakeLevel(int lev, Real t, const BoxArray& ba,
                     const DistributionMapping& dm) override;
    void ClearLevel(int lev) override;
    void ErrorEst(int lev, TagBoxArray& tags, Real t, int ng) override;

private:
    void FillPatch(int lev, MultiFab& mf, int icomp, int ncomp);
    void FillCoarsePatch(int lev, MultiFab& mf, int icomp, int ncomp);
    void physBC(int lev);
    void predictor(int lev, Real dt);
    void solvePressure(int lev, Real dt);
    void corrector(int lev, Real dt);
    Real computeDt();
    Real totalMass();
    void writePlot(int step, Real time);

    Vector<MultiFab> Q;                        // state per level
    std::vector<Real> pPred, pPress, pCorr;
    BCInfo bc;
    Vector<BCRec> bcs;
    int precond_type = 3, gmres_restart = 30, mg_vcycles = 4;
    Real time_end = 1.0, cfl = 0.3, plot_dt = 1e30, tag_threshold = 0.02;
    std::string dem_file, release_file;
    int ng = 2;
    long presIters = 0, presSolves = 0;
};

ChorinAmr::ChorinAmr()
{
    { ParmParse pp("solver"); pp.query("time_end", time_end); pp.query("cfl", cfl); }
    { ParmParse pp("output"); pp.query("plot_dt_interval", plot_dt); }
    { ParmParse pp("tagging"); pp.query("threshold", tag_threshold); }
    { ParmParse pp("init"); pp.query("dem_file", dem_file); pp.query("release_file", release_file); }
    { ParmParse pp("precond"); pp.query("type", precond_type); pp.query("vcycles", mg_vcycles); }
    { ParmParse pp("gmres"); pp.query("restart", gmres_restart); }
    // params (model defaults + inputs overrides); dt appended for press/corr
    auto mk = [](auto names, auto defs){ std::vector<Real> p(defs.begin(),defs.end());
        ParmParse pp("params"); for (size_t i=0;i<names.size();++i){ Real v; if (pp.query(names[i].c_str(),v)) p[i]=v; } return p; };
    pPred  = mk(ModelPred::parameter_names(),  ModelPred::default_parameters());
    pPress = mk(ModelPress::parameter_names(), ModelPress::default_parameters());
    pCorr  = mk(ModelCorr::parameter_names(),  ModelCorr::default_parameters());
    { ParmParse pp("inflow"); Vector<int> c; Vector<Real> v;
      if (pp.queryarr("comp",c)){ pp.getarr("val",v); bc.n_in=c.size();
          bc.comp.assign(c.begin(),c.end()); bc.val.assign(v.begin(),v.end()); } }
    { ParmParse pp("pin"); Vector<int> c; Vector<Real> v;
      if (pp.queryarr("comp",c)){ pp.getarr("val",v); bc.n_pin=c.size();
          bc.pin_comp.assign(c.begin(),c.end()); bc.pin_val.assign(v.begin(),v.end()); }
      pp.query("all", bc.pin_all); }
    bcs.resize(NS);
    for (int n=0;n<NS;++n) for (int d=0;d<AMREX_SPACEDIM;++d){
        bcs[n].setLo(d, BCType::foextrap); bcs[n].setHi(d, BCType::foextrap); }
    Q.resize(max_level+1);
}

void ChorinAmr::MakeNewLevelFromScratch(int lev, Real, const BoxArray& ba,
                                        const DistributionMapping& dm)
{
    Q[lev].define(ba, dm, NS, ng); Q[lev].setVal(0.0);
}
void ChorinAmr::MakeNewLevelFromCoarse(int lev, Real, const BoxArray& ba,
                                       const DistributionMapping& dm)
{
    Q[lev].define(ba, dm, NS, ng); FillCoarsePatch(lev, Q[lev], 0, NS);
}
void ChorinAmr::RemakeLevel(int lev, Real, const BoxArray& ba,
                            const DistributionMapping& dm)
{
    MultiFab nQ(ba, dm, NS, ng); FillPatch(lev, nQ, 0, NS); std::swap(Q[lev], nQ);
}
void ChorinAmr::ClearLevel(int lev) { Q[lev].clear(); }

void ChorinAmr::ErrorEst(int lev, TagBoxArray& tags, Real, int)
{
    const Real thr = tag_threshold;
    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto q = Q[lev].const_array(mfi); auto tag = tags.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            Real gx = std::abs(q(i+1,j,0,IDX_H)-q(i-1,j,0,IDX_H));
            Real g = gx*gx;
            if constexpr (ModelPred::dimension>=2){
                Real gy = std::abs(q(i,j+1,0,IDX_H)-q(i,j-1,0,IDX_H)); g += gy*gy; }
            if (std::sqrt(g) > thr) tag(i,j,k) = TagBox::SET;   // refine the front
        });
    }
}

void ChorinAmr::FillPatch(int lev, MultiFab& mf, int icomp, int ncomp)
{
    NullFill nf;
    if (lev==0) {
        amrex::FillPatchSingleLevel(mf, 0.0, {&Q[lev]}, {0.0}, 0, icomp, ncomp,
                                    Geom(lev), nf, 0);
    } else {
        amrex::FillPatchTwoLevels(mf, 0.0, {&Q[lev-1]}, {0.0}, {&Q[lev]}, {0.0},
                                  0, icomp, ncomp, Geom(lev-1), Geom(lev),
                                  nf, 0, nf, 0, refRatio(lev-1), &cell_cons_interp, bcs, 0);
    }
    physBC(lev);
}
void ChorinAmr::FillCoarsePatch(int lev, MultiFab& mf, int icomp, int ncomp)
{
    NullFill nf;
    amrex::InterpFromCoarseLevel(mf, 0.0, Q[lev-1], 0, icomp, ncomp,
                                 Geom(lev-1), Geom(lev), nf, 0, nf, 0,
                                 refRatio(lev-1), &cell_cons_interp, bcs, 0);
    physBC(lev);
}
void ChorinAmr::physBC(int lev) { fillBC(Q[lev], Geom(lev), bc); }  // zeroGrad+inflow+pin

void ChorinAmr::InitData()
{
    InitFromScratch(0.0);
    if (!release_file.empty()) readRasterIntoComponent(release_file, Geom(0), Q[0], 1);
    if (!dem_file.empty())     readRasterIntoComponent(dem_file,     Geom(0), Q[0], 0);
    if (max_level > 0) {
        for (int i=0;i<4;++i){
            for (int lev=1; lev<=finest_level; ++lev) FillCoarsePatch(lev, Q[lev], 0, NS);
            int of = finest_level; regrid(0, 0.0);
            if (of==finest_level && i>0) break;
        }
        for (int lev=1; lev<=finest_level; ++lev) FillCoarsePatch(lev, Q[lev], 0, NS);
    }
    for (int lev=0; lev<=finest_level; ++lev) physBC(lev);
}

// ── per-level Chorin stages (reuse chorin_common.H helpers on Q[lev]) ────────
void ChorinAmr::predictor(int lev, Real dt)
{
    const auto& geom = Geom(lev); const Real dx=geom.CellSize(0), dy=geom.CellSize(1);
    const BoxArray& ba = Q[lev].boxArray(); const DistributionMapping& dm = Q[lev].DistributionMap();
    FillPatch(lev, Q[lev], 0, NS);                               // coarse-fine + phys ghosts
    MultiFab Aqp(ba, dm, std::max(ModelPred::n_dof_qaux,1), 1);
    fill_deriv_aux(Q[lev], Aqp, ModelPred::deriv_aux, ModelPred::n_deriv_aux, dx, dy);
    Aqp.FillBoundary(geom.periodicity());
    auto P = packp<ModelPred::n_parameters>(pPred);
    MultiFab RHS(ba, dm, ModelPred::n_dof_eq, 0); RHS.setVal(0.0);
    for (MFIter mfi(RHS); mfi.isValid(); ++mfi) {
        const Box& bx=mfi.validbox(); auto Qa=Q[lev].const_array(mfi); auto Aa=Aqp.const_array(mfi);
        auto Ra=RHS.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            constexpr int NE=ModelPred::n_dof_eq;
            SmallMatrix<Real,NE,1> qC,qL,qR,qD,qU; SmallMatrix<Real,NS,1> q8;
            SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1> aC,aL,aR,aD,aU;
            auto ld=[&](int ii,int jj,SmallMatrix<Real,NE,1>&q,
                        SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1>&a){
                for(int n=0;n<NE;++n) q(n,0)=Qa(ii,jj,0,ModelPred::e2s[n]);
                for(int n=0;n<ModelPred::n_dof_qaux;++n) a(n,0)=Aa(ii,jj,0,n); };
            ld(i,j,qC,aC); ld(i-1,j,qL,aL); ld(i+1,j,qR,aR); ld(i,j-1,qD,aD); ld(i,j+1,qU,aU);
            for(int n=0;n<NS;++n) q8(n,0)=Qa(i,j,0,n);
            SmallMatrix<Real,ModelPred::dimension,1> nx{}; nx(0,0)=1.0;
            auto fXR=NumericsPred::numerical_flux(qC,qR,aC,aR,P,nx);
            auto fXL=NumericsPred::numerical_flux(qL,qC,aL,aC,P,nx);
            auto gXR=NumericsPred::numerical_fluctuations(qC,qR,aC,aR,P,nx);
            auto gXL=NumericsPred::numerical_fluctuations(qL,qC,aL,aC,P,nx);
            auto src=ModelPred::source(q8,aC,P);
            for(int n=0;n<NE;++n)
                Ra(i,j,0,n)=(-(fXR(n,0)-fXL(n,0))-gXR(NE+n,0)-gXL(n,0))/dx + src(n,0);
            if constexpr (ModelPred::dimension>=2){
                SmallMatrix<Real,ModelPred::dimension,1> ny{}; ny(1,0)=1.0;
                auto fYR=NumericsPred::numerical_flux(qC,qU,aC,aU,P,ny);
                auto fYL=NumericsPred::numerical_flux(qD,qC,aD,aC,P,ny);
                auto gYR=NumericsPred::numerical_fluctuations(qC,qU,aC,aU,P,ny);
                auto gYL=NumericsPred::numerical_fluctuations(qD,qC,aD,aC,P,ny);
                for(int n=0;n<NE;++n)
                    Ra(i,j,0,n)+=(-(fYR(n,0)-fYL(n,0))-gYR(NE+n,0)-gYL(n,0))/dy;
            }
        });
    }
    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx=mfi.validbox(); auto Qa=Q[lev].array(mfi); auto Ra=RHS.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            for(int n=0;n<ModelPred::n_dof_eq;++n) Qa(i,j,0,ModelPred::e2s[n])+=dt*Ra(i,j,0,n); });
    }
}

void ChorinAmr::solvePressure(int lev, Real dt)
{
    const auto& geom = Geom(lev); const Real dx=geom.CellSize(0), dy=geom.CellSize(1);
    const BoxArray& ba = Q[lev].boxArray(); const DistributionMapping& dm = Q[lev].DistributionMap();
    std::vector<Real> pv = pPress; pv.back() = dt;
    MultiFab Aqs(ba, dm, std::max(ModelPress::n_dof_qaux,1), 1);
    physBC(lev);
    fill_deriv_aux(Q[lev], Aqs, ModelPress::input_aux, ModelPress::n_input_aux, dx, dy);
    Aqs.FillBoundary(geom.periodicity());
    MultiFab Pmf(ba,dm,NP,0), R0(ba,dm,NP,0), rhs(ba,dm,NP,0);
    ChorinPressOp op; op.Q=&Q[lev]; op.Aqs=&Aqs; op.geom=&geom; op.bc=&bc; op.pv=&pv;
    op.R0=&R0; op.dx=dx; op.dy=dy; op.ba=ba; op.dm=dm; op.ptype=precond_type;
    op.domain=geom.Domain(); op.mg_vcycles=mg_vcycles;
    GMRES<MultiFab, ChorinPressOp> gmres; gmres.define(op);
    gmres.setVerbose(0); gmres.setRestartLength(gmres_restart);
    Pmf.setVal(0.0); scatterP(Q[lev], Pmf);
    pressSource(Q[lev], Aqs, geom, bc, pv, dx, dy, R0);
    op.buildDiagonal();
    MultiFab::Copy(rhs, R0, 0,0,NP,0); rhs.mult(-1.0,0,NP,0);
    Pmf.setVal(0.0);
    gmres.solve(Pmf, rhs, 1e-8, 0.0);
    presIters += gmres.getNumIters(); ++presSolves;
    scatterP(Q[lev], Pmf); physBC(lev);
}

void ChorinAmr::corrector(int lev, Real dt)
{
    const auto& geom = Geom(lev); const Real dx=geom.CellSize(0), dy=geom.CellSize(1);
    const BoxArray& ba = Q[lev].boxArray(); const DistributionMapping& dm = Q[lev].DistributionMap();
    std::vector<Real> pv = pCorr; pv.back() = dt;
    MultiFab Aqc(ba, dm, std::max(ModelCorr::n_dof_qaux,1), 1);
    physBC(lev);
    fill_deriv_aux(Q[lev], Aqc, ModelCorr::deriv_aux, ModelCorr::n_deriv_aux, dx, dy);
    Aqc.FillBoundary(geom.periodicity());
    auto P = packp<ModelCorr::n_parameters>(pv);
    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx=mfi.validbox(); auto Qa=Q[lev].array(mfi); auto Aa=Aqc.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k){
            SmallMatrix<Real,NS,1> q; SmallMatrix<Real,std::max(ModelCorr::n_dof_qaux,1),1> a;
            for(int n=0;n<NS;++n) q(n,0)=Qa(i,j,0,n);
            for(int n=0;n<ModelCorr::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
            auto u=ModelCorr::update_variables(q,a,P);
            for(int m=0;m<ModelCorr::n_dof_eq;++m) Qa(i,j,0,ModelCorr::e2s[m])=u(m,0);
        });
    }
}

Real ChorinAmr::computeDt()
{
    Real maxinv = 0.0;
    for (int lev=0; lev<=finest_level; ++lev) {
        const auto& geom=Geom(lev); const Real dx=geom.CellSize(0), dy=geom.CellSize(1);
        const BoxArray& ba=Q[lev].boxArray(); const DistributionMapping& dm=Q[lev].DistributionMap();
        MultiFab Aqp(ba,dm,std::max(ModelPred::n_dof_qaux,1),1);
        physBC(lev);
        fill_deriv_aux(Q[lev], Aqp, ModelPred::deriv_aux, ModelPred::n_deriv_aux, dx, dy);
        auto P=packp<ModelPred::n_parameters>(pPred);
        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi){
            const Box& bx=mfi.validbox(); auto Qa=Q[lev].const_array(mfi); auto Aa=Aqp.const_array(mfi);
            LoopOnCpu(bx, [&](int i,int j,int k){
                SmallMatrix<Real,ModelPred::n_dof_eq,1> q;
                SmallMatrix<Real,std::max(ModelPred::n_dof_qaux,1),1> a;
                for(int n=0;n<ModelPred::n_dof_eq;++n) q(n,0)=Qa(i,j,0,ModelPred::e2s[n]);
                for(int n=0;n<ModelPred::n_dof_qaux;++n) a(n,0)=Aa(i,j,0,n);
                SmallMatrix<Real,ModelPred::dimension,1> nx{}; nx(0,0)=1.0;
                Real inv=NumericsPred::local_max_abs_eigenvalue(q,a,P,nx)(0,0)/dx;
                if constexpr (ModelPred::dimension>=2){
                    SmallMatrix<Real,ModelPred::dimension,1> ny{}; ny(1,0)=1.0;
                    inv=amrex::max(inv, NumericsPred::local_max_abs_eigenvalue(q,a,P,ny)(0,0)/dy); }
                if (inv>maxinv) maxinv=inv;
            });
        }
    }
    ParallelDescriptor::ReduceRealMax(maxinv);
    return (maxinv>1e-14)? cfl/maxinv : 1e-3;
}

Real ChorinAmr::totalMass()
{
    Real tot=0.0;
    for (int lev=0; lev<=finest_level; ++lev){
        const Real cv=Geom(lev).CellSize(0)*Geom(lev).CellSize(1);
        MultiFab h(Q[lev].boxArray(), Q[lev].DistributionMap(), 1, 0);
        MultiFab::Copy(h, Q[lev], IDX_H, 0, 1, 0);
        if (lev<finest_level){
            MultiFab mask=makeFineMask(Q[lev].boxArray(), Q[lev].DistributionMap(),
                                       boxArray(lev+1), refRatio(lev), Real(1.0), Real(0.0));
            MultiFab::Multiply(h, mask, 0,0,1,0);
        }
        tot += h.sum(0)*cv;
    }
    return tot;
}

void ChorinAmr::Evolve()
{
    Real time=0.0; int step=0; Real next_plot=0.0;
    Real t0=ParallelDescriptor::second();
    Real mass0=totalMass();
    Print()<<"chorin-amr: cells(l0)="<<boxArray(0).numPts()<<" mass0="<<std::setprecision(10)<<mass0<<"\n";
    while (time < time_end - 1e-12) {
        if (max_level>0 && step>0) regrid(0, time);
        if (time >= next_plot) { writePlot(step, time); next_plot += plot_dt; }
        Real dt=computeDt(); dt=amrex::min(dt, time_end-time);
        for (int lev=0; lev<=finest_level; ++lev) predictor(lev, dt);
        for (int lev=0; lev<=finest_level; ++lev) solvePressure(lev, dt);
        for (int lev=0; lev<=finest_level; ++lev) corrector(lev, dt);
        for (int lev=finest_level-1; lev>=0; --lev)
            amrex::average_down(Q[lev+1], Q[lev], Geom(lev+1), Geom(lev), 0, NS, refRatio(lev));
        time+=dt; ++step;
        if (step%50==0 || time>=time_end-1e-12){
            long ncells=0; for(int l=0;l<=finest_level;++l) ncells+=boxArray(l).numPts();
            Print()<<"step "<<step<<" t="<<time<<" dt="<<dt<<" levels="<<finest_level+1
                   <<" cells="<<ncells<<" mass="<<std::setprecision(12)<<totalMass()<<"\n";
        }
    }
    writePlot(step, time);
    Real et=ParallelDescriptor::second()-t0;
    long ncells=0; for(int l=0;l<=finest_level;++l) ncells+=boxArray(l).numPts();
    Print()<<"chorin-amr done: steps="<<step<<" t="<<time<<" final_cells="<<ncells
           <<" wall="<<et<<"s  pressure GMRES avg "
           <<(presSolves?double(presIters)/presSolves:0.0)<<" iters/solve\n";
}

void ChorinAmr::writePlot(int step, Real time)
{
    const std::string pf = Concatenate("chk_", step, 5);
    Vector<std::string> nm(NS); for(int n=0;n<NS;++n) nm[n]="var"+std::to_string(n);
    Vector<const MultiFab*> mfs(finest_level+1); Vector<Geometry> gs(finest_level+1);
    Vector<int> ls(finest_level+1, step); Vector<IntVect> rr(finest_level);
    for(int l=0;l<=finest_level;++l){ mfs[l]=&Q[l]; gs[l]=Geom(l); }
    for(int l=0;l<finest_level;++l) rr[l]=refRatio(l);
    WriteMultiLevelPlotfile(pf, finest_level+1, mfs, nm, gs, time, ls, rr);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    { ChorinAmr amr; amr.InitData(); amr.Evolve(); }
    amrex::Finalize();
    return 0;
}
