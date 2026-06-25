
#include "AMReX_MultiFab.H"
#include "constants.H"

using namespace amrex;

void readRasterIntoComponent (const std::string& filename,
                                     const Geometry&    geom,
                                     MultiFab&          mf,
                                     int                comp)
{
    const Box& domain = geom.Domain();
    const int nx  = domain.length(0);
    const int ny  = domain.length(1);
    const std::size_t ncell  = static_cast<std::size_t>(nx)*ny;
    const std::size_t nbytes = ncell * sizeof(Real);

    // ----------------------------------------------------------------
    // 1. open file
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs.good()) {
        amrex::Abort("### Cannot open raw file: " + filename);
    }

    // 2. check size
    ifs.seekg(0, std::ios::end);
    const std::size_t actual = ifs.tellg();
    ifs.seekg(0);
    if (actual < nbytes) {
        amrex::Print() << "### Raw file too small: " << actual
                       << " < expected " << nbytes << " bytes ("
                       << nx << "×" << ny
#ifdef AMREX_USE_DOUBLE
                       << " ×8"
#else
                       << " ×4"
#endif
                       << ")\n";
        amrex::Abort("Raw file size mismatch");
    }

    // 3. read
    std::vector<Real> hostBuf(ncell);
    ifs.read(reinterpret_cast<char*>(hostBuf.data()), nbytes);
    if (!ifs.good()) amrex::Abort("### Error while reading " + filename);
    ifs.close();

    // 4. copy into the MultiFab
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        auto const& arr = mf.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i,int j,int k)
        {
            const std::size_t idx = i + nx*j;   // 2-D
            arr(i,j,k,comp) = hostBuf[idx];
        });
    }

    amrex::Gpu::streamSynchronize();

    if (ParallelDescriptor::IOProcessor())
        amrex::Print() << "✓ loaded '" << filename << "' into component "
                       << comp << " (" << nx << "×" << ny << ")\n";
}


void init_solution(const Geometry&     geom,
                   MultiFab&           solution)
{
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> problo = geom.ProbLoArray();
    GpuArray<Real,AMREX_SPACEDIM> probhi = geom.ProbHiArray();

    const int ncomp = solution.nComp();
    // Default initial condition: a flat-bed 2-D dam break in x.
    // Applied only to the conserved-state field (ncomp >= 2, i.e. has h);
    // auxiliary fields (ncomp == 1) are zero-filled.  When a release/DEM raster
    // is supplied, InitData() overlays it on top of this default afterwards.
    const Real x_mid = 0.5 * (problo[0] + probhi[0]);
    const Real h_left  = 2.0;
    const Real h_right = 1.0;
    const bool is_state = (ncomp >= 2);

    // Loop over boxes
    for (MFIter mfi(solution); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& sol_array = solution.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            for (int n = 0; n < ncomp; n++) {
                sol_array(i,j,k,n) = 0.0;
            }
            if (is_state) {
                const Real x = problo[0] + (i + 0.5) * dx[0];
                sol_array(i,j,k,idx_b) = 0.0;                       // flat bed
                sol_array(i,j,k,idx_h) = (x < x_mid) ? h_left : h_right;
            }
        });
    } // mfi
    //
}

