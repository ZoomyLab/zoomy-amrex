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

        // D+ is at index [n]. D- is offset by [n_dof_q + n]
        // Cell `i` receives D- from the right face, and D+ from the left face!
        amrex::Real F_x = - (flux_xr(n,0) - flux_xl(n,0))
                          - fluct_xr(Model::n_dof_q + n, 0) // D-
                          - fluct_xl(n, 0);                 // D+

        amrex::Real F_y = 0.0;
        if constexpr (Model::dimension == 2) {
            F_y = - (flux_yr(n,0) - flux_yl(n,0))
                  - fluct_yr(Model::n_dof_q + n, 0)         // D-
                  - fluct_yl(n, 0);                         // D+
        }

        RHS(i,j,0,n) = F_x / dx + F_y / dy;
        if (!implicit_source) RHS(i,j,0,n) += source_term(n,0);
    }

    // Audusse well-balanced bed source: the hydrostatic-reconstruction
    // counterpart of the cancelled bed NCP. For momentum component k in
    // direction d it is (g/2/dx_d)[ (h*_right)^2 - (h*_left)^2 ], using THIS
    // cell's depths reconstructed at its two faces. Together with the
    // reconstructed pressure flux this is exact for lake-at-rest.
    if (well_balanced) {
        amrex::Real g = p_mat(0,0);
        RHS(i,j,0,idx_h+1) += 0.5 * g * (hL_xr*hL_xr - hR_xl*hR_xl) / dx;
        if constexpr (Model::dimension == 2)
            RHS(i,j,0,idx_h+2) += 0.5 * g * (hL_yr*hL_yr - hR_yl*hR_yl) / dy;
    }

    // ---- Diffusive flux (optional, compile-time gated) ----
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
// Local Newton-Raphson Solver for the Implicit Source Term
// ----------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_implicit_source(int i, int j, amrex::Array4<amrex::Real> const& Q, amrex::Array4<const amrex::Real> const& Qaux,
                           amrex::Real dt, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat,
                           amrex::Real time, const amrex::SmallMatrix<amrex::Real, 3, 1>& X)   // REQ-185
{
    constexpr int n_q = Model::n_dof_q;
    constexpr int n_a = Model::n_dof_qaux;

    amrex::SmallMatrix<amrex::Real, n_q, 1> q_old, q_curr;
    amrex::SmallMatrix<amrex::Real, n_a, 1> a_curr;

    for(int n=0; n<n_q; ++n) {
        q_old(n,0) = Q(i,j,0,n);
        q_curr(n,0) = Q(i,j,0,n);
    }
    for(int n=0; n<n_a; ++n) a_curr(n,0) = Qaux(i,j,0,n);

    if (q_curr(1,0) < 1e-6) return; // Dry cell skip 

    bool converged = false;

    for(int iter=0; iter<25; ++iter) {
        a_curr = Model::update_aux_variables(q_curr, a_curr, p_mat, time, X);   // REQ-185
        auto S = Model::source(q_curr, a_curr, p_mat, time, X);

        amrex::Real res_norm = 0.0;
        amrex::SmallMatrix<amrex::Real, n_q, 1> residual;
        for(int n=0; n<n_q; ++n) {
            residual(n,0) = q_curr(n,0) - q_old(n,0) - dt * S(n,0);
            res_norm += residual(n,0) * residual(n,0);
        }
        
        if (std::sqrt(res_norm) < 1e-5) { 
            converged = true; 
            break; 
        }

        auto dS_dQ = Model::source_jacobian_wrt_variables(q_curr, a_curr, p_mat);
        auto dS_dA = Model::source_jacobian_wrt_aux_variables(q_curr, a_curr, p_mat);
        auto dA_dQ = Model::update_aux_variables_jacobian_wrt_variables(q_curr, a_curr, p_mat, time, X);

        amrex::SmallMatrix<amrex::Real, n_q, n_q> J;
        for(int r=0; r<n_q; ++r) {
            for(int c=0; c<n_q; ++c) {
                amrex::Real term = dS_dQ(r*n_q + c, 0);
                for(int k=0; k<n_a; ++k) term += dS_dA(r*n_a + k, 0) * dA_dQ(k*n_q + c, 0);
                J(r,c) = (r == c ? 1.0 : 0.0) - dt * term;
            }
        }

        amrex::SmallMatrix<amrex::Real, n_q, 1> delta;
        for(int r=0; r<n_q; ++r) residual(r,0) = -residual(r,0);

        bool singular = false;
        for(int k=0; k<n_q; ++k) {
            int max_row = k;
            amrex::Real max_val = std::abs(J(k,k));
            for(int r=k+1; r<n_q; ++r) {
                if (std::abs(J(r,k)) > max_val) { max_val = std::abs(J(r,k)); max_row = r; }
            }
            if (max_val < 1e-14 || std::isnan(max_val)) { singular = true; break; }
            
            if (max_row != k) {
                for(int c=k; c<n_q; ++c) { amrex::Real tmp = J(k,c); J(k,c) = J(max_row,c); J(max_row,c) = tmp; }
                amrex::Real tmp = residual(k,0); residual(k,0) = residual(max_row,0); residual(max_row,0) = tmp;
            }
            for(int r=k+1; r<n_q; ++r) {
                amrex::Real factor = J(r,k) / J(k,k);
                for(int c=k; c<n_q; ++c) J(r,c) -= factor * J(k,c);
                residual(r,0) -= factor * residual(k,0);
            }
        }
        if (singular) break;

        for(int r=n_q-1; r>=0; --r) {
            amrex::Real sum = 0.0;
            for(int c=r+1; c<n_q; ++c) sum += J(r,c) * delta(c,0);
            delta(r,0) = (residual(r,0) - sum) / J(r,r);
        }

        amrex::Real damping = 1.0;
        if (std::abs(delta(1,0)) > 0.5 * q_curr(1,0)) {
            damping = 0.5;
        }

        for(int n=0; n<n_q; ++n) q_curr(n,0) += damping * delta(n,0);

        if (q_curr(1,0) < 1e-6) q_curr(1,0) = 1e-6;
    }

    if (converged) {
        for(int n=0; n<n_q; ++n) Q(i,j,0,n) = q_curr(n,0);
    }
}