#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include "ZoomyAmr.H"

#ifdef AMREX_TINY_PROFILING
#include <chrono>
#include <cstdio>
#endif

int main(int argc, char* argv[])
{
    // amrex::Initialize() creates the CUDA context and reserves The_Arena, and
    // amrex::Finalize() tears them down.  BOTH sit OUTSIDE TinyProfiler's window
    // (the profiler is started by Initialize and stopped by Finalize), so a
    // "startup cost" read as exe-wall minus evolution time is a DIFFERENCE of two
    // other numbers, not a measurement.  These two host timers make it a READING.
    // Compiled in only for TINY_PROFILE builds, so a shipping build's stdout is
    // unchanged.
#ifdef AMREX_TINY_PROFILING
    const auto t_pre = std::chrono::steady_clock::now();
#endif
    amrex::Initialize(argc, argv);
#ifdef AMREX_TINY_PROFILING
    amrex::Print() << "amrex::Initialize time = "
                   << std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t_pre).count()
                   << " seconds\n";
#endif
    {
        ZoomyAmr solver;
        solver.InitData();
        solver.Evolve();
    }
#ifdef AMREX_TINY_PROFILING
    const auto t_pre_final = std::chrono::steady_clock::now();
    const bool io_proc = amrex::ParallelDescriptor::IOProcessor();
#endif
    amrex::Finalize();
#ifdef AMREX_TINY_PROFILING
    if (io_proc) {
        std::printf("amrex::Finalize time = %g seconds\n",
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_pre_final).count());
    }
#endif
    return 0;
}
