#include "ZoomyAmr.H"
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_GMRES.H>
#include "imex_solver.H"
#include <iomanip>

using namespace amrex;

// Forward declarations from init_solution.cpp
void init_solution(const Geometry& geom, MultiFab& solution);
void readRasterIntoComponent(const std::string& filename, const Geometry& geom, MultiFab& mf, int comp);

// No-op BC functor (we apply our own BCs after FillPatch)
struct NullFill {
    void operator()(MultiFab& /*mf*/, int /*dcomp*/, int /*ncomp*/,
                    IntVect const& /*nghost*/, Real /*time*/, int /*bccomp*/) const {}
};

// ==========================================================================
//  Constructor
// ==========================================================================
ZoomyAmr::ZoomyAmr()
{
    { ParmParse pp("init");
      pp.query("dem_file", dem_file);
      pp.query("release_file", release_file);
      pp.query("friction_file", friction_file);
      Vector<std::string> sr; if (pp.queryarr("state_rasters", sr))  // REQ-123: full-state IC
          state_rasters.assign(sr.begin(), sr.end());
    }
    { ParmParse pp("output");
      pp.query("identifier", identifier);
      pp.query("plot_dt_interval", plot_dt_interval);
    }
    { ParmParse pp("solver");
      pp.query("time_end", time_end);
      pp.query("max_step", max_step);
      pp.query("cfl", cfl);
      pp.query("dtmin", dtmin);
      pp.query("dtmax", dtmax);
      pp.query("spatial_order", spatial_order);
      pp.query("implicit_source", implicit_source);
      pp.query("implicit_global", implicit_global);
      pp.query("well_balanced", well_balanced);
      pp.query("clamp_positivity", clamp_positivity);
      pp.query("positivity", positivity_method);
    }
    { ParmParse pp("tagging");
      pp.query("threshold", tag_threshold);
      pp.query("b_max", tag_b_max);
    }
    { ParmParse pp("bc");
      pp.query("x_lo", bc_x_lo);
      pp.query("x_hi", bc_x_hi);
      pp.query("y_lo", bc_y_lo);
      pp.query("y_hi", bc_y_hi);
    }

    auto p_std = Model::default_parameters();
    for (int i = 0; i < Model::n_parameters; ++i)
        p_mat(i, 0) = p_std[i];

    // Optional per-parameter overrides from inputs: `params.<name> = value`
    // (e.g. params.n_m = 0.05 for Manning friction).  Lets a case tune any
    // model parameter without regenerating Model.H.
    {
        ParmParse pp("params");
        auto names = Model::parameter_names();
        for (int i = 0; i < Model::n_parameters; ++i) {
            amrex::Real v;
            if (pp.query(names[i].c_str(), v)) {
                p_mat(i, 0) = v;
                amrex::Print() << "param override: " << names[i] << " = " << v << "\n";
            }
        }
    }
    // The driver never regularizes or floors h.  Wet/dry robustness lives entirely
    // in the MODEL: its wet_dry_eps parameter desingularizes 1/h (hinv) and gates
    // the eigenvalues via max(h, wet_dry_eps).  The driver reads no wet/dry constant.

    bcs.resize(Model::n_dof_q);
    for (int n = 0; n < Model::n_dof_q; ++n) {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            bcs[n].setLo(idim, BCType::foextrap);
            bcs[n].setHi(idim, BCType::foextrap);
        }
    }

    int nlevs_max = max_level + 1;
    Q.resize(nlevs_max);
    Qtmp.resize(nlevs_max);
    Qold.resize(nlevs_max);
    Qaux.resize(nlevs_max);
    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.0);
    dt_level.resize(nlevs_max, 1.0e100);
    nsubsteps.resize(nlevs_max, 1);
    for (int lev = 1; lev <= max_level; ++lev)
        nsubsteps[lev] = MaxRefRatio(lev - 1);
}

// ==========================================================================
//  Initialization
// ==========================================================================
void ZoomyAmr::InitData()
{
    InitFromScratch(0.0);

    // Two-step IC.  STEP 1 (always): the model's analytic initial condition on
    // ALL state rows (b, h, momentum, passive tracers) — supplied here as one
    // raster per row (init.state_rasters), evaluated from model.initial_conditions.
    // STEP 2 (overwrite): specific fields from measured rasters — the DEM bed
    // (comp 0) and a release depth (comp 1) — layered on top of step 1.
    for (int c = 0; c < (int)state_rasters.size() && c < Model::n_dof_q; ++c)
        if (FileSystem::Exists(state_rasters[c]))
            readRasterIntoComponent(state_rasters[c], Geom(0), Q[0], c);
    if (!dem_file.empty() && FileSystem::Exists(dem_file))
        readRasterIntoComponent(dem_file, Geom(0), Q[0], 0);          // DEM -> b
    if (!release_file.empty() && FileSystem::Exists(release_file))
        readRasterIntoComponent(release_file, Geom(0), Q[0], 1);      // release -> h
    if (!friction_file.empty() && FileSystem::Exists(friction_file))
        readRasterIntoComponent(friction_file, Geom(0), Qaux[0], 1);

    if (max_level > 0) {
        // The raster IC was loaded ONLY into level 0. Any finer levels created
        // by InitFromScratch carry the flat init_solution default, and regrid's
        // RemakeLevel (FillPatch) would preserve that. So after each regrid we
        // re-initialise every fine level from the (correct) coarse level via
        // FillCoarsePatch, propagating the bathymetry + IC down the hierarchy.
        for (int i = 0; i < 4; ++i) {
            for (int lev = 1; lev <= finest_level; ++lev)
                FillCoarsePatch(lev, 0.0, Q[lev], 0, Model::n_dof_q);
            int old_finest = finest_level;
            regrid(0, 0.0);
            if (old_finest == finest_level && i > 0) break;
        }
        for (int lev = 1; lev <= finest_level; ++lev)
            FillCoarsePatch(lev, 0.0, Q[lev], 0, Model::n_dof_q);
    }

    for (int lev = 0; lev <= finest_level; ++lev)
        dt_level[lev] = ComputeDt(lev);
}

// ==========================================================================
//  AmrCore overrides
// ==========================================================================
void ZoomyAmr::MakeNewLevelFromScratch(int lev, Real time,
                                        const BoxArray& ba,
                                        const DistributionMapping& dm)
{
    int ng = nghost();
    Q[lev].define(ba, dm, Model::n_dof_q, ng);
    Qtmp[lev].define(ba, dm, Model::n_dof_q, ng);
    Qold[lev].define(ba, dm, Model::n_dof_q, 0);
    Qaux[lev].define(ba, dm, std::max(Model::n_dof_qaux, 1), ng);

    t_new[lev] = time;
    t_old[lev] = time - 1.0;

    init_solution(Geom(lev), Q[lev]);
    init_solution(Geom(lev), Qaux[lev]);
}

void ZoomyAmr::MakeNewLevelFromCoarse(int lev, Real time,
                                       const BoxArray& ba,
                                       const DistributionMapping& dm)
{
    int ng = nghost();
    Q[lev].define(ba, dm, Model::n_dof_q, ng);
    Qtmp[lev].define(ba, dm, Model::n_dof_q, ng);
    Qold[lev].define(ba, dm, Model::n_dof_q, 0);
    Qaux[lev].define(ba, dm, std::max(Model::n_dof_qaux, 1), ng);

    t_new[lev] = time;
    t_old[lev] = time - 1.0;

    FillCoarsePatch(lev, time, Q[lev], 0, Model::n_dof_q);
}

void ZoomyAmr::RemakeLevel(int lev, Real time,
                            const BoxArray& ba,
                            const DistributionMapping& dm)
{
    int ng = nghost();
    MultiFab new_Q(ba, dm, Model::n_dof_q, ng);
    MultiFab new_Qaux(ba, dm, std::max(Model::n_dof_qaux, 1), ng);

    FillPatch(lev, time, new_Q, 0, Model::n_dof_q);

    std::swap(Q[lev], new_Q);
    std::swap(Qaux[lev], new_Qaux);

    Qtmp[lev].define(ba, dm, Model::n_dof_q, ng);
    Qold[lev].define(ba, dm, Model::n_dof_q, 0);

    t_new[lev] = time;
    t_old[lev] = time - 1.0;
}

void ZoomyAmr::ClearLevel(int lev)
{
    Q[lev].clear();
    Qtmp[lev].clear();
    Qold[lev].clear();
    Qaux[lev].clear();
}

// ==========================================================================
//  Error estimation for AMR tagging
// ==========================================================================
void ZoomyAmr::ErrorEst(int lev, TagBoxArray& tags, Real /*time*/, int /*ngrow*/)
{
    const auto& mf = Q[lev];
    const Real threshold = tag_threshold;
    const Real bmax = tag_b_max;   // only refine cells with bed < bmax

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& Q_arr = mf.const_array(mfi);
        auto const& tag = tags.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Bed-based domain gate: never refine cells essentially outside the
            // domain (the high exterior wall bed) -> they stay at the coarsest
            // level. Inside (b < bmax), refine the flood front (|grad h|) and
            // any wet cell so the moving water is well resolved.
            if (Q_arr(i, j, 0, idx_b) >= bmax) return;
            Real grad = 0.0;
            Real dhdx = std::abs(Q_arr(i + 1, j, 0, idx_h) - Q_arr(i - 1, j, 0, idx_h));
            grad += dhdx * dhdx;
            if constexpr (Model::dimension >= 2) {
                Real dhdy = std::abs(Q_arr(i, j + 1, 0, idx_h) - Q_arr(i, j - 1, 0, idx_h));
                grad += dhdy * dhdy;
            }
            if (std::sqrt(grad) > threshold || Q_arr(i, j, 0, idx_h) > 1.0e-2)
                tag(i, j, k) = TagBox::SET;
        });
    }
}

// ==========================================================================
//  Fill patch (ghost cells from coarse level or own level)
// ==========================================================================
void ZoomyAmr::FillPatch(int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    BL_PROFILE("ZoomyAmr::FillPatch");
    NullFill null_fill;
    if (lev == 0) {
        amrex::FillPatchSingleLevel(mf, time, {&Q[lev]}, {t_new[lev]},
                                     0, icomp, ncomp, Geom(lev), null_fill, 0);
    } else {
        amrex::FillPatchTwoLevels(mf, time,
                                   {&Q[lev - 1]}, {t_new[lev - 1]},
                                   {&Q[lev]}, {t_new[lev]},
                                   0, icomp, ncomp,
                                   Geom(lev - 1), Geom(lev),
                                   null_fill, 0, null_fill, 0,
                                   refRatio(lev - 1),
                                   &cell_cons_interp, bcs, 0);
    }
    // Apply our own physical BCs after fill patch
    FillPhysicalBC(lev, time);
}

void ZoomyAmr::FillCoarsePatch(int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
    NullFill null_fill;
    amrex::InterpFromCoarseLevel(mf, time,
                                  Q[lev - 1], 0, icomp, ncomp,
                                  Geom(lev - 1), Geom(lev),
                                  null_fill, 0, null_fill, 0,
                                  refRatio(lev - 1),
                                  &cell_cons_interp, bcs, 0);
}

// ==========================================================================
//  State update and boundary conditions (same logic as original)
// ==========================================================================
void ZoomyAmr::UpdateState(int lev, Real time)
{
    BL_PROFILE("ZoomyAmr::UpdateState");
    auto const& p = p_mat;
    // REQ-185: cell position for a time/space-dependent update_aux_variables.
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();
    auto plo = geom.ProbLoArray();
    // MOOD (REQ-175) is the positivity mechanism when active; the non-conservative
    // clamp must stay OFF so the order-1 redo is the ONLY thing touching h<0 cells.
    const bool clamp = clamp_positivity && (positivity_method != "mood");
    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        auto Q_arr = Q[lev].array(mfi);
        auto Qaux_arr = Qaux[lev].array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            SmallMatrix<Real, Model::n_dof_q, 1> q;
            SmallMatrix<Real, Model::n_dof_qaux, 1> a;
            for (int n = 0; n < Model::n_dof_q; ++n) q(n, 0) = Q_arr(i, j, 0, n);
            for (int n = 0; n < Model::n_dof_qaux; ++n) a(n, 0) = Qaux_arr(i, j, 0, n);

            SmallMatrix<Real, 3, 1> X{};
            X(0, 0) = plo[0] + (i + 0.5) * dx[0];
            X(1, 0) = (Model::dimension == 2) ? plo[1] + (j + 0.5) * dx[1] : Real(0.0);
            X(2, 0) = Real(0.0);

            // All wet/dry hygiene is INHERITED from the (Numerical)SystemModel:
            // update_variables caps |momentum|/h and zeros dry-cell momentum;
            // update_aux_variables computes the KP-desingularised hinv. The
            // driver adds nothing. The optional positivity clamp (no magic
            // constant, just h>=0) is off by default for an exactly mass-
            // conserving run.
            q = Model::update_variables(q, a, p);
            if (clamp && q(idx_h, 0) < 0.0) q(idx_h, 0) = 0.0;
            a = Model::update_aux_variables(q, a, p, time, X);

            for (int n = 0; n < Model::n_dof_q; ++n) Q_arr(i, j, 0, n) = q(n, 0);
            for (int n = 0; n < Model::n_dof_qaux; ++n) Qaux_arr(i, j, 0, n) = a(n, 0);
        });
    }
}

void ZoomyAmr::FillPhysicalBC(int lev, Real time)
{
    // Delegate to the MF-generic version on the level's own state arrays.
    FillPhysicalBC_mf(Q[lev], Qaux[lev], lev, time);
}

void ZoomyAmr::FillPhysicalBC_mf(MultiFab& Qmf, MultiFab& Qauxmf, int lev, Real time)
{
    BL_PROFILE("ZoomyAmr::FillPhysicalBC");
    const auto& geom = Geom(lev);
    const auto& domain = geom.Domain();
    const int dom_lo_x = domain.smallEnd(0), dom_hi_x = domain.bigEnd(0);
    const int dom_lo_y = domain.smallEnd(1), dom_hi_y = domain.bigEnd(1);
    auto const& p = p_mat;

    // Resolve each domain side's configured BC tag name to its index in the
    // model's sorted tag list (Model::boundary_conditions is a Piecewise over
    // that index).  Empty/unknown name => tag index 0.  Done on the host once.
    // An EMPTY name means "no BC configured for this side" and legitimately
    // falls back to tag 0.  A NON-EMPTY name that matches nothing is a WIRING
    // ERROR and now aborts: it used to return 0 silently, so a model whose tags
    // are not the West/East/South/North that run_case.py hardcodes had ALL FOUR
    // sides collapse onto tag 0 — wrong boundary conditions, no diagnostic, and
    // a green run.  Exactly the silent-false-green class the backend suite
    // exists to kill, so it must not live in the driver that suite runs on.
    auto tag_index = [](const std::string& name) -> int {
        if (name.empty()) return 0;
        auto tags = Model::get_boundary_tags();
        for (int t = 0; t < (int)tags.size(); ++t)
            if (tags[t] == name) return t;
        std::string known;
        for (auto const& t : tags) known += (known.empty() ? "" : ", ") + t;
        amrex::Abort("boundary tag '" + name + "' is not declared by this model. "
                     "Known tags: [" + known + "]. Check the bc.* inputs keys "
                     "against the model's boundary_conditions tags.");
        return 0;
    };
    const int bcix_xlo = tag_index(bc_x_lo);
    const int bcix_xhi = tag_index(bc_x_hi);
    const int bcix_ylo = tag_index(bc_y_lo);
    const int bcix_yhi = tag_index(bc_y_hi);

    for (MFIter mfi(Qmf); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox();
        auto Q_arr = Qmf.array(mfi);
        auto Qaux_arr = Qauxmf.array(mfi);
        auto prob_lo = geom.ProbLoArray();
        auto dx = geom.CellSizeArray();

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            bool out_x = (i < dom_lo_x || i > dom_hi_x);
            bool out_y = (Model::dimension == 2) && (j < dom_lo_y || j > dom_hi_y);
            if (!out_x && !out_y) return;

            // REFLECT the ghost index about the domain face (not clamp to the
            // nearest interior cell): ghost layer k must mirror interior cell
            // (face-1-k).  A clamp fills every ghost layer from the SAME nearest
            // cell, so the 2nd ghost layer is a wrong mirror -> the 2nd-order
            // reconstruction at a reflective wall no longer gives exactly zero
            // mass flux (order-1 is unaffected; it only reads the 1st layer).
            // Reflection makes both layers true mirrors -> wall conserves mass to
            // machine precision at 2nd order.  Only the outside coordinate is
            // reflected; an interior coordinate is left as-is.
            int i_int = i;
            if (i < dom_lo_x)      i_int = 2 * dom_lo_x - 1 - i;
            else if (i > dom_hi_x) i_int = 2 * dom_hi_x + 1 - i;
            int j_int = j;
            if (Model::dimension == 2) {
                if (j < dom_lo_y)      j_int = 2 * dom_lo_y - 1 - j;
                else if (j > dom_hi_y) j_int = 2 * dom_hi_y + 1 - j;
            }

            int bc_idx = -1;
            // Force-capture the per-side BC indices in the lambda body BEFORE the
            // constexpr-if: nvcc cannot first-capture a variable inside a
            // constexpr-if branch of an extended __device__ lambda.
            amrex::ignore_unused(bcix_xlo, bcix_xhi, bcix_ylo, bcix_yhi);
            SmallMatrix<Real, Model::dimension, 1> n_hat{};
            if constexpr (Model::dimension == 1) {
                if (i < dom_lo_x) { bc_idx = bcix_xlo; n_hat(0, 0) = -1.0; }
                else if (i > dom_hi_x) { bc_idx = bcix_xhi; n_hat(0, 0) = 1.0; }
            } else if constexpr (Model::dimension == 2) {
                if (i < dom_lo_x)      { bc_idx = bcix_xlo; n_hat(0, 0) = -1.0; }
                else if (i > dom_hi_x) { bc_idx = bcix_xhi; n_hat(0, 0) =  1.0; }
                else if (j < dom_lo_y) { bc_idx = bcix_ylo; n_hat(1, 0) = -1.0; }
                else if (j > dom_hi_y) { bc_idx = bcix_yhi; n_hat(1, 0) =  1.0; }
            }

            if (bc_idx != -1) {
                SmallMatrix<Real, Model::n_dof_q, 1> q_int;
                SmallMatrix<Real, Model::n_dof_qaux, 1> a_int;
                for (int n = 0; n < Model::n_dof_q; ++n) q_int(n, 0) = Q_arr(i_int, j_int, 0, n);
                for (int n = 0; n < Model::n_dof_qaux; ++n) a_int(n, 0) = Qaux_arr(i_int, j_int, 0, n);

                SmallMatrix<Real, 3, 1> X;
                X(0, 0) = prob_lo[0] + (i + 0.5) * dx[0];
                X(1, 0) = (Model::dimension == 2) ? prob_lo[1] + (j + 0.5) * dx[1] : 0.0;
                X(2, 0) = 0.0;

                auto q_ghost = Model::boundary_conditions(bc_idx, q_int, a_int, n_hat, X, time, dx[0]);
                auto a_ghost = Model::aux_boundary_conditions(bc_idx, q_int, a_int, n_hat, X, time, dx[0]);

                for (int n = 0; n < Model::n_dof_q; ++n) Q_arr(i, j, 0, n) = q_ghost(n, 0);
                for (int n = 0; n < Model::n_dof_qaux; ++n) Qaux_arr(i, j, 0, n) = a_ghost(n, 0);
            }
        });
    }
}

// ==========================================================================
//  Time stepping
// ==========================================================================
Real ZoomyAmr::ComputeDt(int lev)
{
    BL_PROFILE("ZoomyAmr::ComputeDt");
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();
    auto const& p = p_mat;

    // REQ-186: LOCAL per-cell, per-direction CFL.  Reduce max over (cell, dir) of
    // |lambda_dir| / dx_dir, then dt = cfl / that — i.e. dt = min_{i,dir}(cfl *
    // dx_dir / |lambda_dir(i)|).  This couples each cell's OWN size and wave speed
    // instead of global_min(dx) / global_max(|lambda|); a single small or slow
    // cell no longer shrinks dt over the whole level, and anisotropic dx / per-
    // direction wave speeds get their tight (correct) bound.  On a uniform square
    // grid this is identical to the old cfl*min_dx/global_max.
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);

    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto Q_arr = Q[lev].const_array(mfi);
        auto Qaux_arr = Qaux[lev].const_array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) -> GpuTuple<Real> {
                SmallMatrix<Real, Model::n_dof_q, 1> q;
                SmallMatrix<Real, Model::n_dof_qaux, 1> a;
                for (int n = 0; n < Model::n_dof_q; ++n) q(n, 0) = Q_arr(i, j, 0, n);
                for (int n = 0; n < Model::n_dof_qaux; ++n) a(n, 0) = Qaux_arr(i, j, 0, n);

                // The driver never floors h; the model's eigenvalue is responsible
                // for gating dry cells (it uses max(h, wet_dry_eps) internally).
                Real max_inv = 0.0;   // max over dir of |lambda_dir| / dx_dir
                for (int dir = 0; dir < Model::dimension; ++dir) {
                    SmallMatrix<Real, Model::dimension, 1> n_hat{};
                    n_hat(dir, 0) = 1.0;
                    auto ev = Numerics::local_max_abs_eigenvalue(q, a, p, n_hat);
                    Real inv = ev(0, 0) / dx[dir];
                    if (inv > max_inv) max_inv = inv;
                }
                return max_inv;
            });
    }

    Real global_max_inv = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(global_max_inv);

    // REQ-188: a WAVE-FREE domain (all cells dry / sub-wet_dry_eps -> every gated
    // eigenvalue is 0) imposes NO CFL constraint, so step at dtmax (the driver then
    // clamps to the output cadence / time_end) instead of a magic 1e-3 floor. dtmax
    // is the solver parameter. Stopgap adoption by river; core to generalize dtmax
    // onto the NSM so every backend inherits the same dry-domain guard (REQ-188).
    if (global_max_inv < 1e-14) return dtmax;
    Real dt = cfl / global_max_inv;
    return amrex::max(amrex::min(dt, dtmax), dtmin);
}

Real ZoomyAmr::ComputeTotalMass()
{
    BL_PROFILE("ZoomyAmr::ComputeTotalMass");
    Real total = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev) {
        const Real cell_vol = Geom(lev).CellSize(0) * Geom(lev).CellSize(1);
        MultiFab h(boxArray(lev), DistributionMap(lev), 1, 0);
        MultiFab::Copy(h, Q[lev], idx_h, 0, 1, 0);
        if (lev < finest_level) {
            // zero the cells covered by the finer level so they are not
            // double-counted (mask: uncovered=1, covered=0).
            MultiFab mask = makeFineMask(boxArray(lev), DistributionMap(lev),
                                         boxArray(lev + 1), refRatio(lev),
                                         Real(1.0), Real(0.0));
            MultiFab::Multiply(h, mask, 0, 0, 1, 0);
        }
        total += h.sum(0) * cell_vol;
    }
    return total;
}

void ZoomyAmr::Advance(int lev, Real time, Real dt)
{
    BL_PROFILE("ZoomyAmr::Advance");
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();
    auto plo = geom.ProbLoArray();   // REQ-185: cell position for source/aux
    auto const& p = p_mat;
    int order = spatial_order;
    bool impl_src = implicit_source;
    bool wb = well_balanced;

    auto do_stage = [&](int ord) {
        UpdateState(lev, time);
        // Coarse-fine-aware ghost fill. A refined level's ghost cells at the
        // coarse-fine interface must be interpolated from the coarse level
        // (FillPatch), NOT just FillBoundary (which only copies same-level
        // ghosts). Without it, fine-boundary cells read uninitialised ghosts ->
        // NaN -> average_down poisons the coarse level -> the refinement
        // collapses after one regrid. FillPatch also applies the physical BC.
        FillPatch(lev, time, Q[lev], 0, Model::n_dof_q);
        Q[lev].FillBoundary(geom.periodicity());
        Qaux[lev].FillBoundary(geom.periodicity());

        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
            auto Q_arr = Q[lev].const_array(mfi);
            auto Qaux_arr = Qaux[lev].const_array(mfi);
            auto RHS_arr = Qtmp[lev].array(mfi);
            ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    compute_cell_rhs(i, j, Q_arr, Qaux_arr, RHS_arr,
                                     dx[0], dx[1], ord, impl_src, p,
                                     time, plo[0], plo[1], wb);   // REQ-185
                });
        }

        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
            auto Q_arr = Q[lev].array(mfi);
            auto RHS_arr = Qtmp[lev].const_array(mfi);
            ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int n = 0; n < Model::n_dof_q; ++n)
                        Q_arr(i, j, 0, n) += dt * RHS_arr(i, j, 0, n);
                });
        }

        if (impl_src) {
            if (implicit_global) {
                // General matrix-free Newton-Krylov backward-Euler source solve
                // (nonlocal-capable). Q[lev] currently holds the post-flux Qexp.
                SolveImplicitSourceGlobal(lev, dt, time);
            } else {
                for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
                    auto Q_arr = Q[lev].array(mfi);
                    auto Qaux_arr = Qaux[lev].const_array(mfi);
                    ParallelFor(mfi.validbox(),
                        [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            SmallMatrix<Real, 3, 1> X{};   // REQ-185
                            X(0, 0) = plo[0] + (i + 0.5) * dx[0];
                            X(1, 0) = (Model::dimension == 2) ? plo[1] + (j + 0.5) * dx[1] : Real(0.0);
                            X(2, 0) = Real(0.0);
                            apply_implicit_source(i, j, Q_arr, Qaux_arr, dt, p, time, X);
                        });
                }
            }
        }
    };

    // Heun (SSP-RK2) combine: Q <- 0.5*Q^n + 0.5*Q**   (Q^n held in Qold).
    auto heun = [&]() {
        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
            auto Q_arr = Q[lev].array(mfi);
            auto Qold_arr = Qold[lev].const_array(mfi);
            ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int n = 0; n < Model::n_dof_q; ++n)
                        Q_arr(i, j, 0, n) = 0.5 * Qold_arr(i, j, 0, n) + 0.5 * Q_arr(i, j, 0, n);
                });
        }
    };

    if (spatial_order == 1) {
        do_stage(1);
    } else {
        MultiFab::Copy(Qold[lev], Q[lev], 0, 0, Model::n_dof_q, 0);   // Qold = Q^n
        do_stage(order);
        do_stage(order);
        heun();                                                       // order-2 candidate

        // ---- a-posteriori MOOD positivity (REQ-175) ----
        // If the order-2 candidate produced h<0 (or non-finite) anywhere, redo the
        // WHOLE step at order 1 from Q^n (order-1 SSP-RK2, PCM).
        //
        // Why whole-step and not per-cell: amrex's order-2 is a TWO-stage SSP-RK2, so
        // overriding only troubled cells from Q^n (dmplex's local recipe) mixes a
        // single-stage order-1 flux into cells whose neighbours kept the two-stage
        // order-2 flux -> the shared face fluxes no longer cancel and mass leaks
        // (measured 14% at an over-CFL front).  A whole-step order-1 redo keeps every
        // face flux shared between its two cells, so it is EXACTLY mass-conserving AND
        // positivity-preserving (order-1 SSP is monotone under CFL<=1).  The cost is
        // that a troubled step runs order-1 across the whole level, not just the
        // front -- but troubled steps are rare (only when order-2 would go negative),
        // and exact mass conservation is non-negotiable.  Supersedes the clamp.
        if (positivity_method == "mood") {
            MultiFab mask(boxArray(lev), DistributionMap(lev), 1, 0);
            for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
                auto qc = Q[lev].const_array(mfi);
                auto m  = mask.array(mfi);
                ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        amrex::Real h = qc(i, j, 0, idx_h);
                        // troubled: h<0, NaN (fails h>=0), or +/-inf (fails |h|<1e300)
                        m(i, j, 0) = (!(h >= 0.0) || !(h < 1.0e300)) ? 1.0 : 0.0;
                    });
            }
            const amrex::Real n_troubled = mask.sum(0);
            if (n_troubled > 0.0) {
                amrex::Print() << "  [MOOD] lev " << lev << " troubled cells: "
                               << (long)n_troubled << " -> whole-step order-1 redo\n";
                MultiFab::Copy(Q[lev], Qold[lev], 0, 0, Model::n_dof_q, 0);  // Q <- Q^n
                do_stage(1);
                do_stage(1);
                heun();                                                     // order-1 SSP-RK2
            }
        }
    }

    // Clean the post-update state with the MODEL's update_variables (caps |u|,
    // zeros dry-cell momentum). The conservative update can leave a thin cell
    // with large momentum; without this the NEXT ComputeDt (which runs before
    // the next stage's UpdateState) would read that un-capped state and see a
    // spurious huge wavespeed -> dt collapses to dtmin. Capping here uses the
    // model only (no driver wet/dry constants) and does not touch the depth, so
    // mass stays exactly conserved.
    UpdateState(lev, time);
}

void ZoomyAmr::TimeStep(int lev, Real time, int iteration)
{
    if (lev < max_level && iteration > 0) {
        regrid(lev, time);
    }

    dt_level[lev] = ComputeDt(lev);

    // Land exactly on time_end instead of stepping past it. Without this the
    // last coarse step overshoots by up to a full dt (a ~20% overshoot on a
    // short run), so the final frame sits at a different time than every other
    // backend and the run does extra work. Only level 0 needs the cap: Evolve
    // drives it, and sub-cycled fine levels are bounded by the coarse step.
    if (lev == 0) {
        const amrex::Real remaining = time_end - time;
        if (remaining > 0.0 && dt_level[lev] > remaining) dt_level[lev] = remaining;
    }

    Advance(lev, time, dt_level[lev]);
    t_old[lev] = t_new[lev];
    t_new[lev] = time + dt_level[lev];

    if (lev < finest_level) {
        for (int i = 1; i <= nsubsteps[lev + 1]; ++i) {
            TimeStep(lev + 1, time + (i - 1) * dt_level[lev + 1], i);
        }
        amrex::average_down(Q[lev + 1], Q[lev], Geom(lev + 1), Geom(lev),
                            0, Model::n_dof_q, refRatio(lev));
    }
}

// ==========================================================================
//  Main evolution loop
// ==========================================================================
void ZoomyAmr::Evolve()
{
    Real time = 0.0;
    int step = 0;
    next_plot_time = 0.0;

    Real evo_start = ParallelDescriptor::second();

    const Real mass0 = ComputeTotalMass();
    Print() << "Initial total mass = " << std::setprecision(12) << mass0 << "\n";

    // max_step < 0 disables the step cap, so the default is the historical
    // time-only loop, bit-for-bit. The final WritePlotFile below runs either
    // way, so a step-capped run still emits its last state.
    while (time < time_end && (max_step < 0 || step < max_step)) {
        if (time >= next_plot_time) {
            WritePlotFile(plot_step, time);
            next_plot_time += plot_dt_interval;
            plot_step++;
        }

        TimeStep(0, time, step);

        Real dt0 = dt_level[0];
        time += dt0;
        step++;

        Real mass = ComputeTotalMass();
        Real drift = (mass0 != 0.0) ? (mass - mass0) / mass0 : (mass - mass0);
        // REQ-207(B): std::setprecision is STICKY on the stream, so a trailing
        // setprecision(4) for `drift` silently degraded `dt`/`time` on every
        // subsequent line. Any conservation check that reconstructs an expected
        // mass from the parsed log then sees a fabricated ~1e-4 "drift" that is
        // pure formatting (this cost river a false-positive mass-leak hunt).
        // Every field now carries its OWN precision, so the line is independent
        // of prior stream state and cannot leak into the next step.
        Print() << "Step " << step
                << " dt: "   << std::setprecision(12) << dt0
                << " time: " << std::setprecision(12) << time << "s"
                << " levels: " << finest_level + 1
                << " mass: "  << std::setprecision(14) << mass
                << " drift: " << std::setprecision(4)  << drift << "\n";
    }

    WritePlotFile(plot_step, time);

    Real evo_time = ParallelDescriptor::second() - evo_start;
    Print() << "Total evolution time = " << evo_time << " seconds\n";
}

// ==========================================================================
//  Output (multi-level AMReX plotfile)
// ==========================================================================
void ZoomyAmr::WritePlotFile(int step, Real time)
{
    BL_PROFILE("ZoomyAmr::WritePlotFile");
    const std::string pltfile = Concatenate(Concatenate("plt_", identifier), step, 5);

    // Q AND Qaux.  Writing Q alone was a REGRESSION from the AmrCore refactor
    // (8aaa626): the single-level driver it replaced called
    // write_plotfiles(identifier, step, Q, Qaux, geom, time) (main.cpp:199,316
    // at 8aaa626^) and emitted both.  write_plotfiles.cpp still carries that
    // stitching logic but has had no caller since, so aux silently stopped
    // being observable.
    //
    // CORRECTED 2026-07-20.  An earlier version of this comment claimed aux was
    // a cross-step RECURRENCE and therefore unrecoverable.  That was WRONG, and
    // it briefly propagated into the v6 design before the user rejected it.
    // Qaux is BY CONTRACT a per-cell local formula in (Q, parameters):
    // system_model.py:635-640, and the emitted kernel proves it — the generated
    // update_aux_variables body reads only Q(1,0) and p(5,0) (the KP
    // desingularised hinv) and never puts Qaux on a right-hand side.  Qaux sits
    // in the signature only so the operator can return a full-length vector.
    // The misreading came from the STALE checked-in Source/Model.H, whose
    // placeholder body is a 4-row identity — the generated header is authority,
    // never the repo copy.
    //
    // Writing aux is still right, on two honest grounds:
    //   * it lets a test check the solver ACTUALLY HELD to that contract.
    //     Recomputing aux in post would test the recomputation, and a solver
    //     whose aux had drifted from f(Q,p) would still pass.
    //   * the non-local LSQ derivative aux rows would need the mesh gradient
    //     machinery to rebuild in post — cheap to write, expensive to redo.
    constexpr int ncomp_plot = Model::n_dof_q + Model::n_dof_qaux;

    Vector<std::string> var_names;
    for (int n = 0; n < Model::n_dof_q; ++n)
        var_names.push_back(Concatenate("var", n, 1));
    for (int n = 0; n < Model::n_dof_qaux; ++n)
        var_names.push_back(Concatenate("aux", n, 1));

    Vector<MultiFab> plot_mf(finest_level + 1);
    Vector<const MultiFab*> mf_ptrs(finest_level + 1);
    Vector<Geometry> geoms(finest_level + 1);
    Vector<int> level_steps(finest_level + 1, step);
    Vector<IntVect> ref_ratios(finest_level);

    for (int lev = 0; lev <= finest_level; ++lev) {
        // One fab per level, since WriteMultiLevelPlotfile takes a single fab
        // per level.  No ghost cells: a plotfile carries valid data only.
        plot_mf[lev].define(Q[lev].boxArray(), Q[lev].DistributionMap(),
                            ncomp_plot, 0);
        MultiFab::Copy(plot_mf[lev], Q[lev], 0, 0, Model::n_dof_q, 0);
        if constexpr (Model::n_dof_qaux > 0)
            MultiFab::Copy(plot_mf[lev], Qaux[lev], 0, Model::n_dof_q,
                           Model::n_dof_qaux, 0);
        mf_ptrs[lev] = &plot_mf[lev];
        geoms[lev] = Geom(lev);
    }
    for (int lev = 0; lev < finest_level; ++lev)
        ref_ratios[lev] = refRatio(lev);

    WriteMultiLevelPlotfile(pltfile, finest_level + 1, mf_ptrs,
                            var_names, geoms, time, level_steps, ref_ratios);
}
