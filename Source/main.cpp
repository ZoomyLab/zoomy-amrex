#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFabUtil.H>
#include <iostream> 

#include "make_rhs.H"
#include "Model.H"

using namespace amrex;

void init_solution(const Geometry& geom, MultiFab& solution);
void readRasterIntoComponent(const std::string& filename, const Geometry& geom, MultiFab& solution, int comp);
void write_plotfiles(const int identifier, const int step, MultiFab & solution, MultiFab & solution_aux, Geometry const& geom, const Real time);

double compute_dt(const MultiFab& Q, const MultiFab& Qaux, Real cfl, Real min_dx, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat) {
    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);

    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto Q_arr = Q.const_array(mfi);
        auto Qaux_arr = Qaux.const_array(mfi);

        reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> GpuTuple<Real> {
            amrex::SmallMatrix<Real, Model::n_dof_q, 1> q;
            amrex::SmallMatrix<Real, Model::n_dof_qaux, 1> a;
            for(int n=0; n<Model::n_dof_q; ++n) q(n,0) = Q_arr(i,j,0,n);
            for(int n=0; n<Model::n_dof_qaux; ++n) a(n,0) = Qaux_arr(i,j,0,n);
            
            Real max_ev = 0.0;
            for (int dir=0; dir<Model::dimension; ++dir) {
                amrex::SmallMatrix<Real, Model::dimension, 1> n_hat{}; n_hat(dir, 0) = 1.0;
                auto ev = Numerics::local_max_abs_eigenvalue(q, a, p_mat, n_hat);
                if (ev(0,0) > max_ev) max_ev = ev(0,0);
            }
            return max_ev;
        });
    }
    Real global_max_ev = amrex::get<0>(reduce_data.value(reduce_op));
    amrex::ParallelDescriptor::ReduceRealMax(global_max_ev);
    
    if (global_max_ev < 1e-14) return 1e-3;
    return cfl * min_dx / global_max_ev;
}

void update_state(MultiFab& Q, MultiFab& Qaux, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat) {
    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox();
        auto Q_arr = Q.array(mfi);
        auto Qaux_arr = Qaux.array(mfi);
        
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> q;
            amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a;
            for(int n=0; n<Model::n_dof_q; ++n) q(n,0) = Q_arr(i,j,0,n);
            for(int n=0; n<Model::n_dof_qaux; ++n) a(n,0) = Qaux_arr(i,j,0,n);
            
            q = Model::update_variables(q, a, p_mat);
            a = Model::update_aux_variables(q, a, p_mat);
            
            for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i,j,0,n) = q(n,0);
            for(int n=0; n<Model::n_dof_qaux; ++n) Qaux_arr(i,j,0,n) = a(n,0);
        });
    }
}

void FillPhysicalBoundary(MultiFab& Q, MultiFab& Qaux, const Geometry& geom, Real time, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat) {
    const auto& domain = geom.Domain();
    const int dom_lo_x = domain.smallEnd(0), dom_hi_x = domain.bigEnd(0);
    const int dom_lo_y = domain.smallEnd(1), dom_hi_y = domain.bigEnd(1);

    for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox();
        auto Q_arr = Q.array(mfi);
        auto Qaux_arr = Qaux.array(mfi);
        auto prob_lo = geom.ProbLoArray();
        auto dx = geom.CellSizeArray();

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            bool out_x = (i < dom_lo_x || i > dom_hi_x);
            bool out_y = (Model::dimension == 2) && (j < dom_lo_y || j > dom_hi_y);
            if (!out_x && !out_y) return; 

            int i_int = amrex::max(dom_lo_x, amrex::min(dom_hi_x, i));
            int j_int = amrex::max(dom_lo_y, amrex::min(dom_hi_y, j));

            int bc_idx = -1;
            amrex::SmallMatrix<amrex::Real, Model::dimension, 1> n_hat{};
            if constexpr (Model::dimension == 1) {
                if (i < dom_lo_x) { bc_idx = 3; n_hat(0,0) = -1.0; }
                else if (i > dom_hi_x) { bc_idx = 0; n_hat(0,0) = 1.0; }
            } else if constexpr (Model::dimension == 2) {
                if (i < dom_lo_x) { bc_idx = 3; n_hat(0,0) = -1.0; n_hat(1,0) = 0.0; }
                else if (i > dom_hi_x) { bc_idx = 0; n_hat(0,0) = 1.0; n_hat(1,0) = 0.0; }
                else if (j < dom_lo_y) { bc_idx = 2; n_hat(0,0) = 0.0; n_hat(1,0) = -1.0; }
                else if (j > dom_hi_y) { bc_idx = 1; n_hat(0,0) = 0.0; n_hat(1,0) = 1.0; }
            }

            if (bc_idx != -1) {
                amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> q_int;
                amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a_int;
                for(int n=0; n<Model::n_dof_q; ++n) q_int(n,0) = Q_arr(i_int, j_int, 0, n);
                for(int n=0; n<Model::n_dof_qaux; ++n) a_int(n,0) = Qaux_arr(i_int, j_int, 0, n);

                amrex::SmallMatrix<amrex::Real, 3, 1> X;
                X(0,0) = prob_lo[0] + (i + 0.5)*dx[0];
                if constexpr (Model::dimension == 2) X(1,0) = prob_lo[1] + (j + 0.5)*dx[1];
                else X(1,0) = 0.0;
                X(2,0) = 0.0;

                auto q_ghost = Model::boundary_conditions(bc_idx, q_int, a_int, n_hat, X, time, dx[0]);
                auto a_ghost = Model::aux_boundary_conditions(bc_idx, q_int, a_int, n_hat, X, time, dx[0]);

                for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i, j, 0, n) = q_ghost(n,0);
                for(int n=0; n<Model::n_dof_qaux; ++n) Qaux_arr(i, j, 0, n) = a_ghost(n,0);
            }
        });
    }
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    int   n_cell_x = 1, n_cell_y = 1, n_cell_z = 1;
    int   max_grid_size_x = 32, max_grid_size_y = 32, max_grid_size_z = 1;
    Real  phy_bb_x0 = 0., phy_bb_y0 = 0., phy_bb_x1 = 1., phy_bb_y1 = 1.;
    Real  plot_dt_interval = 0.1, time_end = 1.0, CFL = 0.5, dtmin=1.0e-7, dtmax=1.e-2;
    int   identifier = 0, spatial_order = 1;
    bool  implicit_source = true;
    std::string dem_file, release_file, friction_file;

    { ParmParse pp("init"); pp.query("dem_file", dem_file); pp.query("release_file", release_file);
        pp.query("friction_file", friction_file);
    }
    { ParmParse pp("output"); pp.query("identifier", identifier); pp.query("plot_dt_interval", plot_dt_interval); }
    { ParmParse pp("solver"); 
        pp.query("time_end", time_end); pp.query("cfl", CFL); pp.query("dtmin", dtmin); pp.query("dtmax", dtmax); 
        pp.query("spatial_order", spatial_order); pp.query("implicit_source", implicit_source);
    }
    { ParmParse pp("geometry"); 
        pp.query("n_cell_x", n_cell_x); pp.query("n_cell_y", n_cell_y);
        pp.query("phy_bb_x0", phy_bb_x0); pp.query("phy_bb_y0", phy_bb_y0);
        pp.query("phy_bb_x1", phy_bb_x1); pp.query("phy_bb_y1", phy_bb_y1);
    }
    { ParmParse pp("grid"); pp.query("max_grid_size_x", max_grid_size_x); pp.query("max_grid_size_y", max_grid_size_y); }

    int Nghost = (spatial_order == 2) ? 2 : 1; 

    BoxArray ba;
    Geometry geom;
    IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect dom_hi(AMREX_D_DECL(n_cell_x-1, n_cell_y-1, n_cell_z-1));
    Box domain(dom_lo, dom_hi);
    ba.define(domain);
    ba.maxSize(IntVect(AMREX_D_DECL(max_grid_size_x,max_grid_size_y,max_grid_size_z)));
    RealBox real_box({AMREX_D_DECL( phy_bb_x0, phy_bb_y0, 0.)}, {AMREX_D_DECL( phy_bb_x1, phy_bb_y1, 1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    Real cell_size = amrex::min(dx[0], dx[1]);

    DistributionMapping dm(ba);
    MultiFab Q(ba, dm, Model::n_dof_q, Nghost);
    MultiFab Qtmp(ba, dm, Model::n_dof_q, Nghost);
    MultiFab Qold(ba, dm, Model::n_dof_q, 0); 
    MultiFab Qaux(ba, dm, Model::n_dof_qaux, Nghost);

    auto p_std = Model::default_parameters();
    amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1> p_mat;
    for(int i=0; i<Model::n_parameters; ++i) p_mat(i,0) = p_std[i];

    init_solution(geom, Q);
    init_solution(geom, Qtmp);
    init_solution(geom, Qaux);
    readRasterIntoComponent(release_file, geom, Q, 1);
    readRasterIntoComponent(dem_file, geom, Q, 0);
    readRasterIntoComponent(friction_file, geom, Qaux, 1);

    Real time = 0.0, next_write = 0.0;
    int step = 0, iteration = 0;

    Real evolution_start_time = ParallelDescriptor::second();

    while (time < time_end) {
        Real dt = compute_dt(Q, Qaux, CFL, cell_size, p_mat);
        dt = amrex::max(amrex::min(dt, dtmax), dtmin);
        if (time + dt > time_end) dt = time_end - time;

        if (time >= next_write) {
            write_plotfiles(identifier, step, Q, Qaux, geom, time);
            next_write += plot_dt_interval;
            step += 1;
        }

        Real step_start_time = ParallelDescriptor::second();

        if (spatial_order == 1) { // 1st Order Explicit Euler
            update_state(Q, Qaux, p_mat);
            FillPhysicalBoundary(Q, Qaux, geom, time, p_mat);
            Q.FillBoundary(geom.periodicity()); Qaux.FillBoundary(geom.periodicity());

            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto Qaux_arr = Qaux.const_array(mfi);
                auto RHS_arr = Qtmp.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    compute_cell_rhs(i, j, Q_arr, Qaux_arr, RHS_arr, dx[0], dx[1], 1, implicit_source, p_mat);
                });
            }
            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto RHS_arr = Qtmp.const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i,j,0,n) += dt * RHS_arr(i,j,0,n);
                });
            }
            if (implicit_source) {
                for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                    auto Q_arr = Q.array(mfi);
                    auto Qaux_arr = Qaux.const_array(mfi);
                    ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        apply_implicit_source(i, j, Q_arr, Qaux_arr, dt, p_mat);
                    });
                }
            }
        } 
        else if (spatial_order == 2) { // 2nd Order SSP-RK2
            MultiFab::Copy(Qold, Q, 0, 0, Model::n_dof_q, 0);

            // Stage 1
            update_state(Q, Qaux, p_mat);
            FillPhysicalBoundary(Q, Qaux, geom, time, p_mat);
            Q.FillBoundary(geom.periodicity()); Qaux.FillBoundary(geom.periodicity());

            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto Qaux_arr = Qaux.const_array(mfi);
                auto RHS_arr = Qtmp.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    compute_cell_rhs(i, j, Q_arr, Qaux_arr, RHS_arr, dx[0], dx[1], 2, implicit_source, p_mat);
                });
            }
            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto RHS_arr = Qtmp.const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i,j,0,n) += dt * RHS_arr(i,j,0,n);
                });
            }
            if (implicit_source) {
                for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                    auto Q_arr = Q.array(mfi);
                    auto Qaux_arr = Qaux.const_array(mfi);
                    ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        apply_implicit_source(i, j, Q_arr, Qaux_arr, dt, p_mat);
                    });
                }
            }

            // Stage 2
            update_state(Q, Qaux, p_mat);
            FillPhysicalBoundary(Q, Qaux, geom, time + dt, p_mat);
            Q.FillBoundary(geom.periodicity()); Qaux.FillBoundary(geom.periodicity());

            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto Qaux_arr = Qaux.const_array(mfi);
                auto RHS_arr = Qtmp.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    compute_cell_rhs(i, j, Q_arr, Qaux_arr, RHS_arr, dx[0], dx[1], 2, implicit_source, p_mat);
                });
            }
            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto RHS_arr = Qtmp.const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i,j,0,n) += dt * RHS_arr(i,j,0,n);
                });
            }
            if (implicit_source) {
                for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                    auto Q_arr = Q.array(mfi);
                    auto Qaux_arr = Qaux.const_array(mfi);
                    ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                        apply_implicit_source(i, j, Q_arr, Qaux_arr, dt, p_mat);
                    });
                }
            }
            
            // Averaging (RK2)
            for (MFIter mfi(Q); mfi.isValid(); ++mfi) {
                auto Q_arr = Q.array(mfi);
                auto Qold_arr = Qold.const_array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    for(int n=0; n<Model::n_dof_q; ++n) Q_arr(i,j,0,n) = 0.5 * Qold_arr(i,j,0,n) + 0.5 * Q_arr(i,j,0,n);
                });
            }
        }

        Real step_stop_time = ParallelDescriptor::second() - step_start_time;
        amrex::Print() << "Step " << iteration << " dt: " << dt << " time: " << time << "s in " << step_stop_time << "s\n";
        
        time += dt;
        iteration += 1;
    }

    if (time >= next_write) write_plotfiles(identifier, step, Q, Qaux, geom, time);
    
    Real evolution_stop_time = ParallelDescriptor::second() - evolution_start_time;
    amrex::Print() << "Total evolution time = " << evolution_stop_time << " seconds\n";

    amrex::Finalize();
    return 0;
}