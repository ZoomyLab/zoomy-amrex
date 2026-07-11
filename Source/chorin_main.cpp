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
#include "chorin_common.H"

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
    // model-driven BCs — the structured-mesh face convention (no parameter):
    //   x-lo=West  x-hi=East  y-lo=South  y-hi=North  (z-lo=Bottom z-hi=Top for a
    //   true 3-D mesh).  Each face maps to the model's same-named boundary tag; the
    //   driver dispatches the model's own boundary_conditions there (inflow / wall /
    //   extrap / the non-hydrostatic pressure pin all live on the MODEL).
    { auto tags = ModelPred::get_boundary_tags();
      auto tidx = [&](const char* nm){ for (int t=0;t<(int)tags.size();++t) if (tags[t]==std::string(nm)) return t; return -1; };
      bc.side_tag[0]=tidx("West"); bc.side_tag[1]=tidx("East");
      bc.side_tag[2]=tidx("South"); bc.side_tag[3]=tidx("North");
      bc.params = pPred;
      bc.model_bc = (ModelPred::n_boundary_tags > 0); }


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
        op.buildDiagonal();                                  // Jacobi/block-Jacobi setup
        if (precond_type == 4) op.assembleBlockMG();         // block-diagonal MG coeffs
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
