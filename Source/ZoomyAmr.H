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

    // ---- Implicit-source helpers (public so the JFNK GMRES operator can call
    //      them). General matrix-free Newton-Krylov backward-Euler source solve;
    //      the source may be nonlocal (the FD-JVP matvec recomputes ghosts/aux). ----
    // Fill ghost cells (FillBoundary + physical BC dispatch) on an arbitrary MF.
    void FillPhysicalBC_mf(amrex::MultiFab& Qmf, amrex::MultiFab& Qauxmf,
                           int lev, amrex::Real time);
    // S(Q) into Sout (valid region): fills Qin ghosts/aux, evaluates Model::source.
    void ComputeSourceField(amrex::MultiFab& Qin, amrex::MultiFab& Sout,
                            int lev, amrex::Real time);
    // Solve Q = Qexp + dt*S(Q) for Q[lev] in place (matrix-free Newton + GMRES).
    void SolveImplicitSourceGlobal(int lev, amrex::Real dt, amrex::Real time);

    // NOTE: public (not private). nvcc forbids an extended __device__ lambda
    // (amrex::ParallelFor) inside a private/protected member function, and
    // several of these methods launch device kernels.
public:
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

    // Conservative total water mass sum_lev sum(h)*dV over uncovered cells.
    amrex::Real ComputeTotalMass();

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
    bool implicit_global  = false;  // true: matrix-free JFNK (nonlocal-capable)
                                    // false: per-cell local Newton
    bool well_balanced    = false;  // true: Audusse hydrostatic reconstruction
                                    // (lake-at-rest preserving; SWE bed slope)
    bool clamp_positivity = true;   // clamp negative depth to 0 each stage.
                                    // Stabilising but NOT mass-conserving; turn
                                    // off (+ closed walls + CFL<=0.5) for an
                                    // exactly mass-conserving run.
    std::string positivity_method = "none";  // "none" | "mood" (REQ-175).  "mood"
                                    // = a-posteriori local MOOD: after the order-2
                                    // step, cells that went h<0 are redone by a
                                    // conservative order-1 update from the saved
                                    // stage state.  Supersedes the clamp (which is
                                    // forced off) — order-2 everywhere healthy,
                                    // order-1 only at troubled wet/dry cells.
    amrex::Real plot_dt_interval = 0.1;
    int identifier        = 0;
    amrex::Real tag_threshold = 0.01; // gradient threshold for refinement
    amrex::Real tag_b_max = 1.0e30;   // only refine cells with bed < tag_b_max
                                      // (keeps the exterior wall bed coarse)

    // Ghost cells
    int nghost() const { return (spatial_order == 2) ? 2 : 1; }

    // BC descriptors
    amrex::Vector<amrex::BCRec> bcs;

    // Per-side boundary-condition tag names (resolved to the sorted index in
    // Model::get_boundary_tags() at fill time).  Empty => tag index 0.
    std::string bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi;

    // IO
    std::string dem_file, release_file, friction_file;
    std::vector<std::string> state_rasters;  // REQ-123: one raster per state row (all comps)
    int plot_step = 0;
    amrex::Real next_plot_time = 0.0;
};
