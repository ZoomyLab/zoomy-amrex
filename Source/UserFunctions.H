#pragma once
// Hand-written AMReX implementations of the opaque "kernel" placeholder
// functions (zoomy_core.model.kernel_functions).  The code printer emits
// `amrex_user::<name>(...)` calls; these are their device-side definitions.
// SWE/SME with a Rusanov flux does not invoke eigensystem / compute_derivative,
// so these are stubs until a Roe scheme or non-local aux is wired (task 0026).
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>

namespace amrex_user {

// Placeholder: a Roe-style eigensystem helper.  Returns its first argument.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T eigensystem(T a) noexcept { return a; }

// Placeholder: non-local spatial derivative (LSQ gradient) aux.  The driver
// supplies real gradients via Qaux; this opaque fallback returns 0.
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T compute_derivative(T) noexcept { return T(0); }

}  // namespace amrex_user
