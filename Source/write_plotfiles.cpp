#include "plotfile_utils.H"
#include "AMReX_MultiFabUtil.H"

#include "Model.H"

using namespace amrex;

void write_plotfiles_2d (const int identifier, const int step, MultiFab const& solution, MultiFab const& solution_aux, Geometry const& geom, const Real time)
{
    const std::string& pltfile = amrex::Concatenate(amrex::Concatenate("plt_2d_", identifier),step,5);

    int ncomp_q = solution.nComp();
    int ncomp_aux = solution_aux.nComp();
    int ncomp_total = ncomp_q + ncomp_aux;

    // 1. Create a combined MultiFab with enough components for both Q and Qaux
    MultiFab plot_mf(solution.boxArray(), solution.DistributionMap(), ncomp_total, 0);

    // 2. Stitch the data into the combined MultiFab
    amrex::MultiFab::Copy(plot_mf, solution,     0, 0,       ncomp_q,   0);
    amrex::MultiFab::Copy(plot_mf, solution_aux, 0, ncomp_q, ncomp_aux, 0);

    // 3. Setup variable names WITHOUT underscores so ParaView doesn't auto-group them!
    Vector<std::string> var_names;
    for (int n=0; n<ncomp_q; ++n) {
        var_names.push_back(amrex::Concatenate("var", n, 1));
    }
    for (int n=0; n<ncomp_aux; ++n) {
        var_names.push_back(amrex::Concatenate("aux", n, 1));
    }

    WriteSingleLevelPlotfile(pltfile, plot_mf, var_names, geom, time, step);
}

void write_plotfiles_3d (const int identifier, const int step, MultiFab const& solution, MultiFab const& solution_aux, Geometry const& geom, const Real time)
{
    const std::string& pltfile = amrex::Concatenate(amrex::Concatenate("plt_3d_", identifier),step,5);

    int ifac = 8;
    Real fac = static_cast<Real>(ifac);

    Vector<std::string> var_names;
    var_names.push_back("b");
    var_names.push_back("h");
    var_names.push_back("u");
    var_names.push_back("v");
    var_names.push_back("w");
    var_names.push_back("p");

    BoxArray ba = solution.boxArray();
    Geometry geom_3d;

    Box domain_3d = geom.Domain();
    domain_3d.setBig(2,ifac);

    RealBox rb = geom.ProbDomain();
    rb.setHi(2,fac*geom.ProbDomain().hi(2));

    geom_3d.define(domain_3d, rb, CoordSys::cartesian, geom.isPeriodic());

    int ncomp  = 6;
    int nghost = 0;

    int n_dof = Model::n_dof_q;
    int n_dof_aux = Model::n_dof_qaux;

    AMREX_ALWAYS_ASSERT (ncomp == var_names.size());

    BoxList bl3d = BoxList(ba);
    for (auto& b : bl3d) {
        b.setBig(2,ifac);
    }
    BoxArray ba_3d(std::move(bl3d));

    MultiFab solution_3d(ba_3d, solution.DistributionMap(), ncomp, nghost);

    // Dynamically extract parameters using Model::n_parameters to avoid hardcoded bounds
    auto p_std = Model::default_parameters();
    amrex::GpuArray<amrex::Real, Model::n_parameters> params_arr;
    for(int i=0; i<Model::n_parameters; ++i) params_arr[i] = p_std[i];

    // Loop over boxes
    for (MFIter mfi(solution_3d); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real const>& sol_2d_arr = solution.const_array(mfi);
        const Array4<Real const>& sol_2d_arr_aux = solution_aux.const_array(mfi);
        const Array4<Real      >& sol_3d_arr = solution_3d.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real dz = 1. / ifac;
            Real z = (k) * dz;
            
            amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> Q;
            amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> Qaux;
            amrex::SmallMatrix<amrex::Real, 6, 1> Q3d; 
            amrex::SmallMatrix<amrex::Real, 3, 1> X;
            amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1> p;

            for (int n=0;n<n_dof; ++n) Q(n, 0) = sol_2d_arr(i, j, 0, n);
            for (int n=0;n<n_dof_aux; ++n) Qaux(n, 0) = sol_2d_arr_aux(i, j, 0, n);
            for (int n=0;n<Model::n_parameters; ++n) p(n, 0) = params_arr[n];
            
            X(0, 0) = 0.;
            X(1, 0) = 0.;
            X(2, 0) = z; 
            
            Q3d = Model::project_2d_to_3d(X, Q, Qaux, p);

            for (int n=0;n<6; ++n) {
                sol_3d_arr(i, j, k, n) = Q3d(n, 0);
            }
        });
    } 

    WriteSingleLevelPlotfile(pltfile, solution_3d, var_names, geom_3d, time, step);
}

void write_plotfiles (const int identifier, const int step, MultiFab& solution, MultiFab& solution_aux, Geometry const& geom, const Real time)
{
    solution.FillBoundary(geom.periodicity());
    solution_aux.FillBoundary(geom.periodicity());
    
    // Explicitly pass solution_aux down to both writing functions
    write_plotfiles_2d (identifier ,step, solution, solution_aux, geom, time);
    write_plotfiles_3d (identifier, step, solution, solution_aux, geom, time);
}