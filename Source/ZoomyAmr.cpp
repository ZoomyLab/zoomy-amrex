#include "ZoomyAmr.H"
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>

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
    }
    { ParmParse pp("output");
      pp.query("identifier", identifier);
      pp.query("plot_dt_interval", plot_dt_interval);
    }
    { ParmParse pp("solver");
      pp.query("time_end", time_end);
      pp.query("cfl", cfl);
      pp.query("dtmin", dtmin);
      pp.query("dtmax", dtmax);
      pp.query("spatial_order", spatial_order);
      pp.query("implicit_source", implicit_source);
    }
    { ParmParse pp("tagging");
      pp.query("threshold", tag_threshold);
    }

    auto p_std = Model::default_parameters();
    for (int i = 0; i < Model::n_parameters; ++i)
        p_mat(i, 0) = p_std[i];

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

    if (!release_file.empty() && FileSystem::Exists(release_file))
        readRasterIntoComponent(release_file, Geom(0), Q[0], 1);
    if (!dem_file.empty() && FileSystem::Exists(dem_file))
        readRasterIntoComponent(dem_file, Geom(0), Q[0], 0);
    if (!friction_file.empty() && FileSystem::Exists(friction_file))
        readRasterIntoComponent(friction_file, Geom(0), Qaux[0], 1);

    if (max_level > 0) {
        for (int i = 0; i < 3; ++i) {
            int old_finest = finest_level;
            regrid(0, 0.0);
            if (old_finest == finest_level) break;
        }
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

    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const& Q_arr = mf.const_array(mfi);
        auto const& tag = tags.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Real grad = 0.0;
            Real dhdx = std::abs(Q_arr(i + 1, j, 0, idx_h) - Q_arr(i - 1, j, 0, idx_h));
            grad += dhdx * dhdx;
            if constexpr (Model::dimension >= 2) {
                Real dhdy = std::abs(Q_arr(i, j + 1, 0, idx_h) - Q_arr(i, j - 1, 0, idx_h));
                grad += dhdy * dhdy;
            }
            if (std::sqrt(grad) > threshold)
                tag(i, j, k) = TagBox::SET;
        });
    }
}

// ==========================================================================
//  Fill patch (ghost cells from coarse level or own level)
// ==========================================================================
void ZoomyAmr::FillPatch(int lev, Real time, MultiFab& mf, int icomp, int ncomp)
{
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
void ZoomyAmr::UpdateState(int lev)
{
    auto const& p = p_mat;
    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        auto Q_arr = Q[lev].array(mfi);
        auto Qaux_arr = Qaux[lev].array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            SmallMatrix<Real, Model::n_dof_q, 1> q;
            SmallMatrix<Real, Model::n_dof_qaux, 1> a;
            for (int n = 0; n < Model::n_dof_q; ++n) q(n, 0) = Q_arr(i, j, 0, n);
            for (int n = 0; n < Model::n_dof_qaux; ++n) a(n, 0) = Qaux_arr(i, j, 0, n);

            q = Model::update_variables(q, a, p);
            a = Model::update_aux_variables(q, a, p);

            for (int n = 0; n < Model::n_dof_q; ++n) Q_arr(i, j, 0, n) = q(n, 0);
            for (int n = 0; n < Model::n_dof_qaux; ++n) Qaux_arr(i, j, 0, n) = a(n, 0);
        });
    }
}

void ZoomyAmr::FillPhysicalBC(int lev, Real time)
{
    const auto& geom = Geom(lev);
    const auto& domain = geom.Domain();
    const int dom_lo_x = domain.smallEnd(0), dom_hi_x = domain.bigEnd(0);
    const int dom_lo_y = domain.smallEnd(1), dom_hi_y = domain.bigEnd(1);
    auto const& p = p_mat;

    for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox();
        auto Q_arr = Q[lev].array(mfi);
        auto Qaux_arr = Qaux[lev].array(mfi);
        auto prob_lo = geom.ProbLoArray();
        auto dx = geom.CellSizeArray();

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            bool out_x = (i < dom_lo_x || i > dom_hi_x);
            bool out_y = (Model::dimension == 2) && (j < dom_lo_y || j > dom_hi_y);
            if (!out_x && !out_y) return;

            int i_int = amrex::max(dom_lo_x, amrex::min(dom_hi_x, i));
            int j_int = amrex::max(dom_lo_y, amrex::min(dom_hi_y, j));

            int bc_idx = -1;
            SmallMatrix<Real, Model::dimension, 1> n_hat{};
            if constexpr (Model::dimension == 1) {
                if (i < dom_lo_x) { bc_idx = 3; n_hat(0, 0) = -1.0; }
                else if (i > dom_hi_x) { bc_idx = 0; n_hat(0, 0) = 1.0; }
            } else if constexpr (Model::dimension == 2) {
                if (i < dom_lo_x)      { bc_idx = 3; n_hat(0, 0) = -1.0; }
                else if (i > dom_hi_x) { bc_idx = 0; n_hat(0, 0) =  1.0; }
                else if (j < dom_lo_y) { bc_idx = 2; n_hat(1, 0) = -1.0; }
                else if (j > dom_hi_y) { bc_idx = 1; n_hat(1, 0) =  1.0; }
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
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();
    Real min_dx = amrex::min(dx[0], dx[1]);
    auto const& p = p_mat;

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

                Real max_ev = 0.0;
                for (int dir = 0; dir < Model::dimension; ++dir) {
                    SmallMatrix<Real, Model::dimension, 1> n_hat{};
                    n_hat(dir, 0) = 1.0;
                    auto ev = Numerics::local_max_abs_eigenvalue(q, a, p, n_hat);
                    if (ev(0, 0) > max_ev) max_ev = ev(0, 0);
                }
                return max_ev;
            });
    }

    Real global_max = amrex::get<0>(reduce_data.value(reduce_op));
    ParallelDescriptor::ReduceRealMax(global_max);

    if (global_max < 1e-14) return 1e-3;
    Real dt = cfl * min_dx / global_max;
    return amrex::max(amrex::min(dt, dtmax), dtmin);
}

void ZoomyAmr::Advance(int lev, Real time, Real dt)
{
    const auto& geom = Geom(lev);
    auto dx = geom.CellSizeArray();
    auto const& p = p_mat;
    int order = spatial_order;
    bool impl_src = implicit_source;

    auto do_stage = [&]() {
        UpdateState(lev);
        FillPhysicalBC(lev, time);
        Q[lev].FillBoundary(geom.periodicity());
        Qaux[lev].FillBoundary(geom.periodicity());

        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
            auto Q_arr = Q[lev].const_array(mfi);
            auto Qaux_arr = Qaux[lev].const_array(mfi);
            auto RHS_arr = Qtmp[lev].array(mfi);
            ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    compute_cell_rhs(i, j, Q_arr, Qaux_arr, RHS_arr,
                                     dx[0], dx[1], order, impl_src, p);
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
            for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
                auto Q_arr = Q[lev].array(mfi);
                auto Qaux_arr = Qaux[lev].const_array(mfi);
                ParallelFor(mfi.validbox(),
                    [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        apply_implicit_source(i, j, Q_arr, Qaux_arr, dt, p);
                    });
            }
        }
    };

    if (spatial_order == 1) {
        do_stage();
    } else {
        MultiFab::Copy(Qold[lev], Q[lev], 0, 0, Model::n_dof_q, 0);
        do_stage();
        do_stage();
        for (MFIter mfi(Q[lev]); mfi.isValid(); ++mfi) {
            auto Q_arr = Q[lev].array(mfi);
            auto Qold_arr = Qold[lev].const_array(mfi);
            ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int n = 0; n < Model::n_dof_q; ++n)
                        Q_arr(i, j, 0, n) = 0.5 * Qold_arr(i, j, 0, n) + 0.5 * Q_arr(i, j, 0, n);
                });
        }
    }
}

void ZoomyAmr::TimeStep(int lev, Real time, int iteration)
{
    if (lev < max_level && iteration > 0) {
        regrid(lev, time);
    }

    dt_level[lev] = ComputeDt(lev);

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

    while (time < time_end) {
        if (time >= next_plot_time) {
            WritePlotFile(plot_step, time);
            next_plot_time += plot_dt_interval;
            plot_step++;
        }

        TimeStep(0, time, step);

        Real dt0 = dt_level[0];
        time += dt0;
        step++;

        Print() << "Step " << step << " dt: " << dt0
                << " time: " << time << "s"
                << " levels: " << finest_level + 1 << "\n";
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
    const std::string pltfile = Concatenate(Concatenate("plt_", identifier), step, 5);

    Vector<std::string> var_names;
    for (int n = 0; n < Model::n_dof_q; ++n)
        var_names.push_back(Concatenate("var", n, 1));

    Vector<const MultiFab*> mf_ptrs(finest_level + 1);
    Vector<Geometry> geoms(finest_level + 1);
    Vector<int> level_steps(finest_level + 1, step);
    Vector<IntVect> ref_ratios(finest_level);

    for (int lev = 0; lev <= finest_level; ++lev) {
        mf_ptrs[lev] = &Q[lev];
        geoms[lev] = Geom(lev);
    }
    for (int lev = 0; lev < finest_level; ++lev)
        ref_ratios[lev] = refRatio(lev);

    WriteMultiLevelPlotfile(pltfile, finest_level + 1, mf_ptrs,
                            var_names, geoms, time, level_steps, ref_ratios);
}
