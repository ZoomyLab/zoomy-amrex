#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include "ZoomyAmr.H"

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ZoomyAmr solver;
        solver.InitData();
        solver.Evolve();
    }
    amrex::Finalize();
    return 0;
}
