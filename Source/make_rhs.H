#pragma once

#include <AMReX_Array4.H>
#include "Model.H"
#include "Numerics.H"

// Simple MinMod Limiter for 2nd-order TVD stability
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real minmod(amrex::Real a, amrex::Real b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
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
               amrex::SmallMatrix<amrex::Real, 2 * Model::n_dof_q, 1>& fluct) // <-- Flat 1D Vector
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
    
    for(int n=0; n<Model::n_dof_qaux; ++n) {
        aL(n,0) = Qaux(iL, jL, 0, n);
        aR(n,0) = Qaux(iR, jR, 0, n);
    }

    aL = Model::update_aux_variables(qL, aL, p_mat);
    aR = Model::update_aux_variables(qR, aR, p_mat);

    flux = Numerics::numerical_flux(qL, qR, aL, aR, p_mat, n_hat);
    fluct = Numerics::numerical_fluctuations(qL, qR, aL, aR, p_mat, n_hat);
}

// ----------------------------------------------------------------------------
// Calculates the total explicit RHS for a single cell
// ----------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_cell_rhs(int i, int j, amrex::Array4<const amrex::Real> const& Q, amrex::Array4<const amrex::Real> const& Qaux,
                      amrex::Array4<amrex::Real> const& RHS, amrex::Real dx, amrex::Real dy, 
                      int order, bool implicit_source, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat)
{
    amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> flux_xr, flux_xl, flux_yr, flux_yl;
    
    // Flat 1D Vectors
    amrex::SmallMatrix<amrex::Real, 2 * Model::n_dof_q, 1> fluct_xr, fluct_xl, fluct_yr, fluct_yl; 

    eval_face(i, j, i+1, j, 0, order, Q, Qaux, p_mat, flux_xr, fluct_xr);
    eval_face(i-1, j, i, j, 0, order, Q, Qaux, p_mat, flux_xl, fluct_xl);

    if constexpr (Model::dimension == 2) {
        eval_face(i, j, i, j+1, 1, order, Q, Qaux, p_mat, flux_yr, fluct_yr);
        eval_face(i, j-1, i, j, 1, order, Q, Qaux, p_mat, flux_yl, fluct_yl);
    }

    amrex::SmallMatrix<amrex::Real, Model::n_dof_q, 1> q_curr, source_term;
    amrex::SmallMatrix<amrex::Real, Model::n_dof_qaux, 1> a_curr;
    
    for(int n=0; n<Model::n_dof_q; ++n) q_curr(n,0) = Q(i,j,0,n);
    for(int n=0; n<Model::n_dof_qaux; ++n) a_curr(n,0) = Qaux(i,j,0,n);
    
    if (!implicit_source) source_term = Model::source(q_curr, a_curr, p_mat);

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
}

// ----------------------------------------------------------------------------
// Local Newton-Raphson Solver for the Implicit Source Term
// ----------------------------------------------------------------------------
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_implicit_source(int i, int j, amrex::Array4<amrex::Real> const& Q, amrex::Array4<const amrex::Real> const& Qaux,
                           amrex::Real dt, const amrex::SmallMatrix<amrex::Real, Model::n_parameters, 1>& p_mat) 
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
        a_curr = Model::update_aux_variables(q_curr, a_curr, p_mat);
        auto S = Model::source(q_curr, a_curr, p_mat);

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
        auto dA_dQ = Model::update_aux_variables_jacobian_wrt_variables(q_curr, a_curr, p_mat);

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