/*---------------------------------------------------------------------------*\
    symAmrex -- runs a printed symbolic solver (Solver.H) on a single-level
    AMReX grid (rt_amrex.H, CPU / GPU / MPI).

    inputs (ParmParse, prefix zs.):  n_cell, prob_lo, prob_hi, max_grid_size,
    time_end, branch (BC branch of x-lo x-hi y-lo y-hi), lsq (the LSQ table),
    q0 / qaux0 (raw doubles, rows x ny x nx, x fastest), out (directory).
    Writes every io.write snapshot as a plotfile <out>/plt<k> and a raw dump
    <out>/<k>.bin, the final state as <out>/final.bin: (time, rows, nx, ny,
    data) as doubles, gathered on rank 0.
\*---------------------------------------------------------------------------*/
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <chrono>
#include <cstdio>
#include <string>

#include "rt_amrex.H"
#include "Kernels.H"

namespace zs
{
#include "Solver.H"
}

namespace
{
void load(const std::string& path, zs::Field& F, const zs::Mesh& M)
{
    const int nx = M.g.ihi - M.g.ilo + 1, ny = M.g.jhi - M.g.jlo + 1, nq = F.nq;
    amrex::Vector<double> v(size_t(nq)*nx*ny, 0.0);
    if (!path.empty())
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(f, "symAmrex: cannot open " + path);
        AMREX_ALWAYS_ASSERT(std::fread(v.data(), sizeof(double), v.size(), f) == v.size());
        std::fclose(f);
    }
    for (amrex::MFIter mfi(F[0]); mfi.isValid(); ++mfi)
    {
        const amrex::Box& b = mfi.validbox();
        amrex::FArrayBox host(b, F[0].nComp(), amrex::The_Pinned_Arena());
        auto h = host.array();
        amrex::LoopOnCpu(b, nq, [&](int i, int j, int, int n)
        { h(i, j, 0, n) = v[(size_t(n)*ny + j)*nx + i]; });
        F[0][mfi].copy<amrex::RunOn::Device>(host, b, 0, b, 0, F[0].nComp());   // pinned: device-visible
        amrex::Gpu::streamSynchronize();               // before ``host`` is freed
    }
    F = zs::halo(F, M);
}

void dump(const std::string& path, double time, const zs::Field& F, const zs::Mesh& M)
{
    const amrex::Box dom = M.geom.Domain();
    amrex::MultiFab all(amrex::BoxArray(dom), amrex::DistributionMapping(amrex::Vector<int>{0}),
                        F.nq, 0);
    all.ParallelCopy(F[0], 0, 0, F.nq);
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    amrex::FArrayBox host(dom, F.nq, amrex::The_Pinned_Arena());
    host.copy<amrex::RunOn::Device>(all[0], dom, 0, dom, 0, F.nq);
    amrex::Gpu::streamSynchronize();
    const double head[4] = {time, double(F.nq), double(dom.length(0)), double(dom.length(1))};
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(head, sizeof(double), 4, f);
    std::fwrite(host.dataPtr(), sizeof(double), size_t(F.nq)*dom.numPts(), f);
    std::fclose(f);
}
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        amrex::ParmParse pp("zs");
        amrex::Vector<int> n_cell, branch(4, 0);
        amrex::Vector<amrex::Real> lo, hi;
        int max_grid_size = 64;
        double time_end = 0.0;
        std::string lsq, q0, qaux0, out = ".";
        pp.getarr("n_cell", n_cell);
        pp.getarr("prob_lo", lo);
        pp.getarr("prob_hi", hi);
        pp.query("max_grid_size", max_grid_size);
        pp.get("time_end", time_end);
        pp.queryarr("branch", branch);
        pp.get("lsq", lsq);
        pp.query("q0", q0);
        pp.query("qaux0", qaux0);
        pp.query("out", out);

        const amrex::Box dom(amrex::IntVect(0, 0), amrex::IntVect(n_cell[0] - 1, n_cell[1] - 1));
        zs::Mesh M;
        M.geom.define(dom, amrex::RealBox(lo.data(), hi.data()), amrex::CoordSys::cartesian,
                      amrex::Array<int, 2>{0, 0});
        M.ba = amrex::BoxArray(dom);
        M.ba.maxSize(max_grid_size);
        M.dm = amrex::DistributionMapping(M.ba);
        const int br[4] = {branch[0], branch[1], branch[2], branch[3]};
        M.load(lsq, br);
        zs::the_mesh() = &M;
        zs::configure();
        amrex::Print() << "symAmrex: " << n_cell[0] << " x " << n_cell[1] << " cells, "
                       << M.ba.size() << " boxes, " << amrex::ParallelDescriptor::NProcs()
                       << " ranks, LSQ reach " << M.R << "\n";

        zs::Field Q = zs::cells_of(M, zs::System::n_state);
        zs::Field Qaux = zs::cells_of(M, zs::System::n_aux);
        load(q0, Q, M);
        load(qaux0, Qaux, M);
        zs::Col<zs::NPAR> p{};
        const auto pv = zs::System::parameters();
        for (int i = 0; i < zs::NPAR; ++i) p(i, 0) = pv[i];

        amrex::UtilCreateDirectory(out, 0755);
        int k = 0;
        amrex::Vector<std::string> names;
        for (int i = 0; i < Q.nq; ++i) names.push_back("Q" + std::to_string(i));
        zs::writer() = [&](double t, const zs::Field& q, const zs::Field&)
        {
            amrex::MultiFab v(M.ba, M.dm, q.nq, 0);
            amrex::MultiFab::Copy(v, q[0], 0, 0, q.nq, 0);
            amrex::WriteSingleLevelPlotfile(out + "/plt" + std::to_string(k), v, names,
                                            M.geom, t, k);
            dump(out + "/" + std::to_string(k) + ".bin", t, q, M);
            ++k;
        };

        Qaux = std::get<0>(zs::initial_aux(Q, Qaux, p, std::cref(M), 0L));
        amrex::ParallelDescriptor::Barrier();
        const auto t0 = std::chrono::steady_clock::now();
        const auto res = zs::march(Q, Qaux, p, time_end, std::cref(M));
        amrex::Gpu::streamSynchronize();
        amrex::ParallelDescriptor::Barrier();
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        amrex::Print() << "symAmrex: march wall time " << wall << " s\n";
        dump(out + "/final.bin", std::get<0>(res), std::get<1>(res), M);
        amrex::Print() << "symAmrex: t = " << std::get<0>(res) << "\n";
    }
    amrex::Finalize();
    return 0;
}
