#pragma once

#include <AMReX_Array4.H>
#include "Model.H"
#include "Numerics.H"
#include "constants.H"

// Simple MinMod Limiter for 2nd-order TVD stability
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod(amrex::Real a, amrex::Real b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

// REQ-185: cell-centre position (x,y,z) for the time/space-aware source and
// update_aux_variables operators.  z=0 (structured 2-D); mirrors the BC path.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::SmallMatrix<amrex::Real, 3, 1> cell_center(
        int i, int j, amrex::Real dx, amrex::Real dy,
        amrex::Real plox, amrex::Real ploy) {
    amrex::SmallMatrix<amrex::Real, 3, 1> X{};
    X(0, 0) = plox + (i + 0.5) * dx;
    X(1, 0) = (Model::dimension == 2) ? ploy + (j + 0.5) * dy : amrex::Real(0.0);
    X(2, 0) = amrex::Real(0.0);
    return X;
}

// ----------------------------------------------------------------------------
// Reconstructs states at a face and evaluates numerical fluxes
// ----------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void eval_face(int iL, int jL, int iR, int jR, int dir, int order,
               amrex::Array4<const amrex::Real> const& Q,
               amrex::Array4<const amrex::Real> const& Qaux,
               const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
               // Back to n_dof_q: core dropped the trailing face-wavespeed row
               // again in zoomy_core 7dd2ac9 (v6 design, reverts REQ-212) —
               // eigenvalues move to a pre-step dt_pass storing two scalars per
               // face. The n_dof_q+1 accommodation added in 03ce140 was always
               // flagged temporary, so this is its planned removal.
               amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1>& flux,
               amrex::SmallMatrix<amrex::Real, 2 * Model::n_dof_q, 1>& fluct, // <-- Flat 1D Vector
               amrex::Real time, amrex::Real dx, amrex::Real dy,             // REQ-185
               amrex::Real plox, amrex::Real ploy,                          // REQ-185
               bool well_balanced = false,
               amrex::Real* hLstar = nullptr, amrex::Real* hRstar = nullptr)
{
    amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> qL, qR;
    amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> aL, aR;
    amrex::SmallMatrix<amrex::Real, Model::dimension, 1> n_hat{};
    
    for(int d=0; d<Model::dimension; ++d) {
        n_hat(d, 0) = 0.0;
    }
    n_hat(dir, 0) = 1.0;

    for(int n=0; n<Model::n_dof_q; ++n) {
        if (order == 1) {
            qL(n,0) = Q(iL, jL, 0, n);
            qR(n,0) = Q(iR, jR, 0, n);
        } else {
            // TVD MinMod Reconstruction (prevents artificial oscillations and NaNs)
            if (dir == 0) { 
                amrex::Real gradL = minmod(Q(iL, jL, 0, n) - Q(iL-1, jL, 0, n), Q(iL+1, jL, 0, n) - Q(iL, jL, 0, n));
                amrex::Real gradR = minmod(Q(iR, jR, 0, n) - Q(iR-1, jR, 0, n), Q(iR+1, jR, 0, n) - Q(iR, jR, 0, n));
                qL(n,0) = Q(iL, jL, 0, n) + 0.5 * gradL;
                qR(n,0) = Q(iR, jR, 0, n) - 0.5 * gradR;
            } else { 
                amrex::Real gradL = minmod(Q(iL, jL, 0, n) - Q(iL, jL-1, 0, n), Q(iL, jL+1, 0, n) - Q(iL, jL, 0, n));
                amrex::Real gradR = minmod(Q(iR, jR, 0, n) - Q(iR, jR-1, 0, n), Q(iR, jR+1, 0, n) - Q(iR, jR, 0, n));
                qL(n,0) = Q(iL, jL, 0, n) + 0.5 * gradL;
                qR(n,0) = Q(iR, jR, 0, n) - 0.5 * gradR;
            }
        }
    }
    
    // Audusse hydrostatic reconstruction (well-balanced SWE bed slope).
    // Reconstruct the depth from the free surface eta=h+b about the higher of
    // the two beds, b_face=max(bL,bR): h* = max(0, eta - b_face). Momentum is
    // rescaled to preserve the velocity, and the face bed is set equal on both
    // sides so the generic path-conservative bed NCP term cancels (Delta b=0);
    // the balancing bed source is added per cell in compute_cell_rhs. This
    // makes lake-at-rest (eta=const, u=0) exact. No-op when well_balanced=false.
    amrex::Real hLs = qL(idx_h,0), hRs = qR(idx_h,0);
    if (well_balanced) {
        amrex::Real bL = qL(idx_b,0), bR = qR(idx_b,0);
        amrex::Real bf = amrex::max(bL, bR);
        hLs = amrex::max(amrex::Real(0.0), qL(idx_h,0) + bL - bf);
        hRs = amrex::max(amrex::Real(0.0), qR(idx_h,0) + bR - bf);
        amrex::Real sL = (qL(idx_h,0) > amrex::Real(0)) ? hLs / qL(idx_h,0) : amrex::Real(0.0);
        amrex::Real sR = (qR(idx_h,0) > amrex::Real(0)) ? hRs / qR(idx_h,0) : amrex::Real(0.0);
        for (int d = 0; d < Model::dimension; ++d) {
            qL(idx_h+1+d, 0) *= sL;
            qR(idx_h+1+d, 0) *= sR;
        }
        qL(idx_h,0) = hLs;  qR(idx_h,0) = hRs;
        qL(idx_b,0) = bf;   qR(idx_b,0) = bf;
    }
    if (hLstar) *hLstar = hLs;
    if (hRstar) *hRstar = hRs;

    // NB: the driver NEVER floors/clamps h.  Wet/dry robustness is the MODEL's job
    // (its update_aux_variables computes hinv = 1/max(h, wet_dry_eps) and its
    // eigenvalues gate on max(h, wet_dry_eps)); positivity is the WB reconstruction
    // above.  The reconstructed face depth is passed through unmodified.

    for(int n=0; n<Model::n_dof_qaux; ++n) {
        aL(n,0) = Qaux(iL, jL, 0, n);
        aR(n,0) = Qaux(iR, jR, 0, n);
    }

    auto X_L = cell_center(iL, jL, dx, dy, plox, ploy);   // REQ-185
    auto X_R = cell_center(iR, jR, dx, dy, plox, ploy);
    aL = Model::update_aux_variables(qL, aL, p_mat, time, X_L);
    aR = Model::update_aux_variables(qR, aR, p_mat, time, X_R);

    // REQ-212(a): back to the two legacy calls.  The fused numerical_face
    // (REQ-194/207) is being REMOVED from core per a user design decision: the
    // CSE-silo problem it addressed will instead be solved by a printer option
    // that resolves opaque functions so CSE sees through them, with max-|lambda|
    // becoming an extra OUTPUT of numerical_flux.
    // The measurement agreed with dropping it -- on GPU the fusion was a
    // PESSIMISATION, not a win: 23.18 -> 24.02 s, registers 112 -> 137, occupancy
    // halved. The -18.3% op count never reached the clock because the packed
    // kernel's live range blew the register budget.
    // Safe to revert: these names never stopped being registered and come from the
    // same cached definitions -- the A/B in 3997673 showed byte-identical dt either
    // way (Step 1 0.0158738780034 ... Step 7 0.00676464827122).
    flux = Numerics::numerical_flux(qL, qR, aL, aR, p_mat, n_hat);
    fluct = Numerics::numerical_fluctuations(qL, qR, aL, aR, p_mat, n_hat);
}

// ----------------------------------------------------------------------------
// Shared RHS arithmetic — ONE definition, used by BOTH the fused kernel
// (compute_cell_rhs) and the profile-split two-pass kernels below, so the two
// paths cannot drift apart.
// ----------------------------------------------------------------------------

// Conservative flux difference + path-conservative fluctuations along one axis.
// D+ lives at index [n], D- at [n_dof_q + n]: cell i receives D- from its right
// face and D+ from its left face.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real face_divergence(amrex::Real f_r, amrex::Real f_l,
                            amrex::Real dminus_r, amrex::Real dplus_l)
{
    return - (f_r - f_l) - dminus_r - dplus_l;
}

// Audusse well-balanced bed source: the hydrostatic-reconstruction counterpart
// of the cancelled bed NCP. For momentum component k in direction d it is
// (g/2/dx_d)[ (h*_right)^2 - (h*_left)^2 ], using THIS cell's depths
// reconstructed at its two faces. Together with the reconstructed pressure flux
// this is exact for lake-at-rest.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void add_wb_bed_source(int i, int j, amrex::Array4<amrex::Real> const& RHS,
                       amrex::Real hL_xr, amrex::Real hR_xl,
                       amrex::Real hL_yr, amrex::Real hR_yl,
                       amrex::Real g, amrex::Real dx, amrex::Real dy)
{
    RHS(i,j,0,idx_h+1) += 0.5 * g * (hL_xr*hL_xr - hR_xl*hR_xl) / dx;
    if constexpr (Model::dimension == 2)
        RHS(i,j,0,idx_h+2) += 0.5 * g * (hL_yr*hL_yr - hR_yl*hR_yl) / dy;
}

// ---- Diffusive flux (optional, compile-time gated) ----
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void add_diffusive_flux(int i, int j, amrex::Array4<const amrex::Real> const& Q,
                        amrex::Array4<const amrex::Real> const& Qaux,
                        amrex::Array4<amrex::Real> const& RHS,
                        amrex::Real dx, amrex::Real dy,
                        const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat)
{
    if constexpr (Model::has_diffusion) {
        constexpr int n_q = Model::n_dof_q;
        constexpr int dim = Model::dimension;
        constexpr int n_gradQ = Model::n_dof_gradQ; // n_q * dim

        // Compute the normal-projected diffusive flux at a single face.
        // dir  = face-normal direction (0=x, 1=y)
        // (iL,jL) = left cell,  (iR,jR) = right cell
        // dx_face = cell spacing in the face-normal direction
        // Returns the face-normal component of the diffusive flux for each variable.
        auto diff_flux_at_face = [&](int iL, int jL, int iR, int jR,
                                     int dir, amrex::Real dx_face)
            -> amrex::SmallMatrix<amrex::Real, n_q, 1>
        {
            // Face-averaged state
            amrex::SmallMatrix<amrex::Real, n_q, 1> q_face;
            amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a_face;
            for (int v = 0; v < n_q; ++v)
                q_face(v, 0) = 0.5 * (Q(iL, jL, 0, v) + Q(iR, jR, 0, v));
            for (int v = 0; v < Model::n_dof_qaux; ++v)
                a_face(v, 0) = 0.5 * (Qaux(iL, jL, 0, v) + Qaux(iR, jR, 0, v));

            // Face-normal gradient via central difference
            // gradQ layout: [dQ0_d0, dQ0_d1, ..., dQ1_d0, dQ1_d1, ...]
            amrex::SmallMatrix<amrex::Real, n_gradQ, 1> gradQ_face{};
            for (int v = 0; v < n_q; ++v) {
                gradQ_face(v * dim + dir, 0) =
                    (Q(iR, jR, 0, v) - Q(iL, jL, 0, v)) / dx_face;
            }

            // Evaluate model diffusive flux (returns n_q * dim flat)
            auto Fd = Model::diffusive_flux(q_face, a_face, gradQ_face, p_mat);

            // Project onto face-normal direction
            amrex::SmallMatrix<amrex::Real, n_q, 1> Fd_n;
            for (int v = 0; v < n_q; ++v)
                Fd_n(v, 0) = Fd(v * dim + dir, 0);
            return Fd_n;
        };

        // Divergence form: RHS += (Fd_right - Fd_left)/dx [+ (Fd_top - Fd_bot)/dy]
        // This mirrors the convective flux pattern (eval_face at xl,xr,yl,yr).
        auto Fd_xr = diff_flux_at_face(i,   j, i+1, j, 0, dx);
        auto Fd_xl = diff_flux_at_face(i-1, j, i,   j, 0, dx);

        for (int v = 0; v < n_q; ++v)
            RHS(i, j, 0, v) += (Fd_xr(v, 0) - Fd_xl(v, 0)) / dx;

        if constexpr (dim == 2) {
            auto Fd_yr = diff_flux_at_face(i, j,   i, j+1, 1, dy);
            auto Fd_yl = diff_flux_at_face(i, j-1, i, j,   1, dy);

            for (int v = 0; v < n_q; ++v)
                RHS(i, j, 0, v) += (Fd_yr(v, 0) - Fd_yl(v, 0)) / dy;
        }
    }
}

// ----------------------------------------------------------------------------
// Calculates the total explicit RHS for a single cell
// ----------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_cell_rhs(int i, int j, amrex::Array4<const amrex::Real> const& Q, amrex::Array4<const amrex::Real> const& Qaux,
                      amrex::Array4<amrex::Real> const& RHS, amrex::Real dx, amrex::Real dy,
                      int order, bool implicit_source, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
                      amrex::Real time, amrex::Real plox, amrex::Real ploy,   // REQ-185
                      bool well_balanced = false)
{
    amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> flux_xr, flux_xl, flux_yr, flux_yl;

    // Flat 1D Vectors
    amrex::SmallMatrix<amrex::Real, 2 * Model::n_dof_q, 1> fluct_xr, fluct_xl, fluct_yr, fluct_yl;

    // Reconstructed depths of THIS cell at each of its faces (Audusse source).
    amrex::Real hL_xr = 0.0, hR_xl = 0.0, hL_yr = 0.0, hR_yl = 0.0, dum;
    eval_face(i, j, i+1, j, 0, order, Q, Qaux, p_mat, flux_xr, fluct_xr, time, dx, dy, plox, ploy, well_balanced, &hL_xr, &dum);
    eval_face(i-1, j, i, j, 0, order, Q, Qaux, p_mat, flux_xl, fluct_xl, time, dx, dy, plox, ploy, well_balanced, &dum, &hR_xl);

    if constexpr (Model::dimension == 2) {
        eval_face(i, j, i, j+1, 1, order, Q, Qaux, p_mat, flux_yr, fluct_yr, time, dx, dy, plox, ploy, well_balanced, &hL_yr, &dum);
        eval_face(i, j-1, i, j, 1, order, Q, Qaux, p_mat, flux_yl, fluct_yl, time, dx, dy, plox, ploy, well_balanced, &dum, &hR_yl);
    }

    amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> q_curr, source_term;
    amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a_curr;

    for(int n=0; n<Model::n_dof_q; ++n) q_curr(n,0) = Q(i,j,0,n);
    for(int n=0; n<Model::n_dof_qaux; ++n) a_curr(n,0) = Qaux(i,j,0,n);

    for(int n=0; n<Model::n_dof_q; ++n) source_term(n,0) = 0.0;
    // No dry guard: the model's friction source uses pow(max(h_floor, h)) and
    // hinv, so it is finite for any depth.
    if (!implicit_source)
        source_term = Model::source(q_curr, a_curr, p_mat, time,
                                    cell_center(i, j, dx, dy, plox, ploy));  // REQ-185

    for(int n=0; n<Model::n_dof_q; ++n) {

        amrex::Real F_x = face_divergence(flux_xr(n,0), flux_xl(n,0),
                                          fluct_xr(Model::n_dof_q + n, 0),
                                          fluct_xl(n, 0));

        amrex::Real F_y = 0.0;
        if constexpr (Model::dimension == 2) {
            F_y = face_divergence(flux_yr(n,0), flux_yl(n,0),
                                  fluct_yr(Model::n_dof_q + n, 0),
                                  fluct_yl(n, 0));
        }

        RHS(i,j,0,n) = F_x / dx + F_y / dy;
        if (!implicit_source) RHS(i,j,0,n) += source_term(n,0);
    }

    if (well_balanced)
        add_wb_bed_source(i, j, RHS, hL_xr, hR_xl, hL_yr, hR_yl, p_mat(0,0), dx, dy);

    add_diffusive_flux(i, j, Q, Qaux, RHS, dx, dy, p_mat);
}

// ============================================================================
// PROFILE-SPLIT PATH (solver.profile_split = 1) — INSTRUMENTATION ONLY
// ============================================================================
// `compute_cell_rhs` above is a single fused device kernel, so its internal
// face/flux vs cell/assembly split cannot be timed: BL_PROFILE is host-side and
// does not exist inside device code.  To MEASURE that split the kernel is cut in
// two, each half launched from its own host-side BL_PROFILE region:
//
//   pass 1  compute_face_data     -> face fluxes/fluctuations/h* into a face MF
//   pass 2  assemble_cell_rhs     -> reads the face MF, assembles the cell RHS
//
// Both passes call the SAME eval_face / face_divergence / add_wb_bed_source /
// add_diffusive_flux as the fused kernel, so the arithmetic has one definition
// and the two paths cannot drift.  The split is NOT free — it adds a full
// round-trip of the face data through global memory that the fused kernel keeps
// in registers — so the split-path total is an UPPER bound on the fused cost and
// must be reported alongside the unsplit number.
//
// Face-MF component layout (per direction):
//   [0, n_q)            numerical flux
//   [n_q, 3*n_q)        numerical fluctuations, flat (D+ then D-)
//   3*n_q               h*_L   reconstructed depth of the LEFT  cell at this face
//   3*n_q + 1           h*_R   reconstructed depth of the RIGHT cell at this face
static constexpr int n_face_comp = 3 * Model::n_dof_q + 2;

// Pass 1: one face per thread.  (i,j) indexes the face nodal in `dir`, i.e. the
// face between cell (i-1,j) and (i,j) for dir=0.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_face_data(int i, int j, int dir, int order,
                       amrex::Array4<const amrex::Real> const& Q,
                       amrex::Array4<const amrex::Real> const& Qaux,
                       amrex::Array4<amrex::Real> const& F,
                       const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
                       amrex::Real time, amrex::Real dx, amrex::Real dy,
                       amrex::Real plox, amrex::Real ploy, bool well_balanced)
{
    constexpr int n_q = Model::n_dof_q;
    amrex::SmallMatrix<amrex::Real, n_q, 1> flux;
    amrex::SmallMatrix<amrex::Real, 2 * n_q, 1> fluct;
    amrex::Real hLs = 0.0, hRs = 0.0;

    const int iL = (dir == 0) ? i - 1 : i;
    const int jL = (dir == 1) ? j - 1 : j;
    eval_face(iL, jL, i, j, dir, order, Q, Qaux, p_mat, flux, fluct,
              time, dx, dy, plox, ploy, well_balanced, &hLs, &hRs);

    for (int n = 0; n < n_q; ++n)       F(i, j, 0, n)         = flux(n, 0);
    for (int n = 0; n < 2 * n_q; ++n)   F(i, j, 0, n_q + n)   = fluct(n, 0);
    F(i, j, 0, 3 * n_q)     = hLs;
    F(i, j, 0, 3 * n_q + 1) = hRs;
}

// Pass 2: one cell per thread.  Cell (i,j)'s left x-face is Fx(i,j), its right
// x-face is Fx(i+1,j) (AMReX face indexing), and likewise in y.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void assemble_cell_rhs(int i, int j,
                       amrex::Array4<const amrex::Real> const& Q,
                       amrex::Array4<const amrex::Real> const& Qaux,
                       amrex::Array4<const amrex::Real> const& Fx,
                       amrex::Array4<const amrex::Real> const& Fy,
                       amrex::Array4<amrex::Real> const& RHS,
                       amrex::Real dx, amrex::Real dy, bool implicit_source,
                       const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
                       amrex::Real time, amrex::Real plox, amrex::Real ploy,
                       bool well_balanced)
{
    constexpr int n_q = Model::n_dof_q;

    amrex::SmallMatrix<amrex::Real, n_q, 1> q_curr, source_term;
    amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a_curr;

    for(int n=0; n<n_q; ++n) q_curr(n,0) = Q(i,j,0,n);
    for(int n=0; n<Model::n_dof_qaux; ++n) a_curr(n,0) = Qaux(i,j,0,n);

    for(int n=0; n<n_q; ++n) source_term(n,0) = 0.0;
    if (!implicit_source)
        source_term = Model::source(q_curr, a_curr, p_mat, time,
                                    cell_center(i, j, dx, dy, plox, ploy));

    for(int n=0; n<n_q; ++n) {
        amrex::Real F_x = face_divergence(Fx(i+1, j, 0, n), Fx(i, j, 0, n),
                                          Fx(i+1, j, 0, n_q + n_q + n),   // D- right
                                          Fx(i,   j, 0, n_q + n));        // D+ left

        amrex::Real F_y = 0.0;
        if constexpr (Model::dimension == 2) {
            F_y = face_divergence(Fy(i, j+1, 0, n), Fy(i, j, 0, n),
                                  Fy(i, j+1, 0, n_q + n_q + n),
                                  Fy(i, j,   0, n_q + n));
        }

        RHS(i,j,0,n) = F_x / dx + F_y / dy;
        if (!implicit_source) RHS(i,j,0,n) += source_term(n,0);
    }

    if (well_balanced) {
        const amrex::Real hL_xr = Fx(i+1, j, 0, 3*n_q);      // this cell, right face
        const amrex::Real hR_xl = Fx(i,   j, 0, 3*n_q + 1);  // this cell, left  face
        amrex::Real hL_yr = 0.0, hR_yl = 0.0;
        if constexpr (Model::dimension == 2) {
            hL_yr = Fy(i, j+1, 0, 3*n_q);
            hR_yl = Fy(i, j,   0, 3*n_q + 1);
        }
        add_wb_bed_source(i, j, RHS, hL_xr, hR_xl, hL_yr, hR_yl, p_mat(0,0), dx, dy);
    }

    add_diffusive_flux(i, j, Q, Qaux, RHS, dx, dy, p_mat);
}

// ----------------------------------------------------------------------------
// Point-implicit (linearly-implicit) source update — ONE step, no iteration
// ----------------------------------------------------------------------------
// Solves the linearisation of the backward-Euler source update about the
// incoming state:
//
//     (I - dt*J) dq = dt*S(q_old),    J = dS/dQ + (dS/dA)(dA/dQ)
//     q_new = q_old + dq
//
// i.e. exactly ONE Newton iteration started from q_old — the standard
// point-implicit update. Unconditionally stable for the stiff LINEAR part of
// the source, which is what matters here: Manning friction is stiff on thin
// films, tt = dt*g*n^2*|u|/h^(4/3) reaching 6.2 at p99 on LowerTriangle.
// Explicit forward Euler multiplies the momentum by (1 - tt) — by -5.2 there —
// so it REVERSES and amplifies the flow instead of damping it, and the wet/dry
// momentum cap then pins those cells at Froude ~3.5 forever.
//
// A CONSTANT source is reproduced EXACTLY by this update, which is why rainfall
// needs no separate explicit path: rain does not depend on the state, so its
// Jacobian row is zero, the mass row of (I - dt*J) is the identity row, and the
// step returns dh = dt*r_o to machine precision.
//
// Replaces a 25-iteration Newton with damping and a convergence test. That loop
// lost 30.6% of the catchment's water (308805 against the exact 444888 m3): the
// damping heuristic halved the update on ALL rows including mass, so rainfall
// accumulation was under-relaxed and never caught up, and a non-converged cell
// had its entire source discarded. One fixed-cost solve has no damping, no
// convergence test and no discard path, so none of those failure modes exist.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_implicit_source(int i, int j, amrex::Array4<amrex::Real> const& Q, amrex::Array4<const amrex::Real> const& Qaux,
                           amrex::Real dt, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
                           amrex::Real time, const amrex::SmallMatrix<amrex::Real, 3, 1>& X)   // REQ-185
{
    constexpr int n_q = Model::n_dof_q;
    constexpr int n_a = Model::n_dof_qaux;

    amrex::SmallMatrix<amrex::Real, n_q, 1> q_old;
    amrex::SmallMatrix<amrex::Real, n_a, 1> a_in;
    for(int n=0; n<n_q; ++n) q_old(n,0) = Q(i,j,0,n);
    for(int n=0; n<n_a; ++n) a_in(n,0)  = Qaux(i,j,0,n);

    // NO dry-cell skip and NO depth floor. Both used to live here, and both
    // hardcoded model facts into the driver: that state row 1 is the depth, and
    // a 1e-6 wet/dry threshold the model already owns as wet_dry_eps. The skip
    // made a dry-start rainfall run impossible — every cell begins at h = 0, so
    // every cell returned, the rain source never fired, and mass stayed exactly
    // 0 for the whole run. The floor broke the standing never-floor-h rule and,
    // once the skip was gone, would have created mass in every dry cell.
    // Whether a source is evaluable as h -> 0 is the MODEL's business; it
    // desingularises its own friction denominator. The driver only integrates.
    auto S     = Model::source(q_old, a_in, p_mat, time, X);
    auto dS_dQ = Model::source_jacobian_wrt_variables(q_old, a_in, p_mat);
    auto dS_dA = Model::source_jacobian_wrt_aux_variables(q_old, a_in, p_mat);
    auto dA_dQ = Model::update_aux_variables_jacobian_wrt_variables(q_old, a_in, p_mat, time, X);

    amrex::SmallMatrix<amrex::Real, n_q, n_q> A;
    amrex::SmallMatrix<amrex::Real, n_q, 1> rhs;
    for(int r=0; r<n_q; ++r) {
        rhs(r,0) = dt * S(r,0);
        for(int c=0; c<n_q; ++c) {
            amrex::Real term = dS_dQ(r*n_q + c, 0);
            for(int k=0; k<n_a; ++k) term += dS_dA(r*n_a + k, 0) * dA_dQ(k*n_q + c, 0);
            A(r,c) = (r == c ? 1.0 : 0.0) - dt * term;
        }
    }

    // Gaussian elimination with partial pivoting on the cell-local n_q x n_q
    // system (n_q = 4 for SWE, so this is small and fully unrolled).
    bool ok = true;
    for(int k=0; k<n_q && ok; ++k) {
        int piv = k;
        amrex::Real best = std::abs(A(k,k));
        for(int r=k+1; r<n_q; ++r)
            if (std::abs(A(r,k)) > best) { best = std::abs(A(r,k)); piv = r; }
        if (!(best > 0.0) || !std::isfinite(best)) { ok = false; break; }
        if (piv != k) {
            for(int c=k; c<n_q; ++c) { amrex::Real t = A(k,c); A(k,c) = A(piv,c); A(piv,c) = t; }
            amrex::Real t = rhs(k,0); rhs(k,0) = rhs(piv,0); rhs(piv,0) = t;
        }
        for(int r=k+1; r<n_q; ++r) {
            amrex::Real f = A(r,k) / A(k,k);
            for(int c=k; c<n_q; ++c) A(r,c) -= f * A(k,c);
            rhs(r,0) -= f * rhs(k,0);
        }
    }

    if (ok) {
        amrex::SmallMatrix<amrex::Real, n_q, 1> dq;
        for(int r=n_q-1; r>=0; --r) {
            amrex::Real sum = 0.0;
            for(int c=r+1; c<n_q; ++c) sum += A(r,c) * dq(c,0);
            dq(r,0) = (rhs(r,0) - sum) / A(r,r);
            if (!std::isfinite(dq(r,0))) { ok = false; break; }
        }
        if (ok) {
            for(int n=0; n<n_q; ++n) Q(i,j,0,n) = q_old(n,0) + dq(n,0);
            return;
        }
    }

    // Singular or non-finite linearisation: fall back to the EXPLICIT source.
    // A numerical guard, not a physics decision — and it still applies the
    // source, so mass is never silently dropped the way the old
    // `if (converged)` gate dropped it.
    for(int n=0; n<n_q; ++n) {
        amrex::Real v = q_old(n,0) + dt * S(n,0);
        Q(i,j,0,n) = std::isfinite(v) ? v : q_old(n,0);
    }
}
