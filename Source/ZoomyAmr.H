#pragma once

#include <AMReX_AmrCore.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_Interpolater.H>
#include <AMReX_FillPatchUtil.H>

#include "Model.H"
#include "Numerics.H"
#include "make_rhs.H"
#include "constants.H"

class ZoomyAmr : public amrex::AmrCore
{
public:
    ZoomyAmr();
    ~ZoomyAmr() override = default;

    // Main driver
    void InitData();
    void Evolve();

    // Raster loading (level 0 only)
    void LoadRaster(const std::string& filename, int comp, bool is_aux = false);

private:
    // ---- AmrCore pure virtual overrides ----
    void MakeNewLevelFromScratch(int lev, amrex::Real time,
                                 const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override;

    void MakeNewLevelFromCoarse(int lev, amrex::Real time,
                                const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override;

    void RemakeLevel(int lev, amrex::Real time,
                     const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override;

    void ClearLevel(int lev) override;

    void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time,
                  int ngrow) override;

    // ---- Time stepping ----
    void TimeStep(int lev, amrex::Real time, int iteration);
    void Advance(int lev, amrex::Real time, amrex::Real dt);
    amrex::Real ComputeDt(int lev);

    // ---- Fill patch (ghost cells) ----
    void FillPatch(int lev, amrex::Real time,
                   amrex::MultiFab& mf, int icomp, int ncomp);
    void FillCoarsePatch(int lev, amrex::Real time,
                         amrex::MultiFab& mf, int icomp, int ncomp);

    // ---- Helpers ----
    void UpdateState(int lev);
    void FillPhysicalBC(int lev, amrex::Real time);
    void WritePlotFile(int step, amrex::Real time);

    // ---- Data per level ----
    amrex::Vector<amrex::MultiFab>  Q;       // Primary variables
    amrex::Vector<amrex::MultiFab>  Qtmp;    // RHS / scratch
    amrex::Vector<amrex::MultiFab>  Qold;    // For RK staging
    amrex::Vector<amrex::MultiFab>  Qaux;    // Auxiliary variables
    // TODO: add FluxRegister for conservation correction at coarse-fine boundaries

    amrex::Vector<amrex::Real>      t_new;   // Current time per level
    amrex::Vector<amrex::Real>      t_old;   // Previous time per level
    amrex::Vector<amrex::Real>      dt_level; // dt per level
    amrex::Vector<int>              nsubsteps; // subcycling steps per level

    // Parameters (read from inputs)
    amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1> p_mat;
    amrex::Real time_end  = 1.0;
    amrex::Real cfl       = 0.5;
    amrex::Real dtmin     = 1.0e-7;
    amrex::Real dtmax     = 1.0e-2;
    int spatial_order     = 1;
    bool implicit_source  = true;
    amrex::Real plot_dt_interval = 0.1;
    int identifier        = 0;
    amrex::Real tag_threshold = 0.01; // gradient threshold for refinement

    // Ghost cells
    int nghost() const { return (spatial_order == 2) ? 2 : 1; }

    // BC descriptors
    amrex::Vector<amrex::BCRec> bcs;

    // IO
    std::string dem_file, release_file, friction_file;
    int plot_step = 0;
    amrex::Real next_plot_time = 0.0;
};
