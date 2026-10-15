/// \file
/// symAmrex: runs a printed symbolic solver (Solver.H) on a single-level AMReX grid (rt_amrex.H; CPU / GPU / MPI), or with zs.amr = 1 on an AMR hierarchy (amr_amrex.H), optionally as one preCICE participant.
/// Inputs are the ParmParse zs.* keys (n_cell, prob_lo / prob_hi, branch, periodic, lsq, q0 / qaux0, out, time_start, restart, checkpoint); it writes each io.write snapshot as plotfile plt<k> and raw dump <k>.bin, the final state as final.bin and each io.save_state as the checkpoint directory chk<t>.
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "rt_amrex.H"
#include "Kernels.H"

namespace zs
{
#include "Solver.H"
}

namespace
{
/// Flat index of cell c, row n in the driver's raw files (x fastest).
size_t flat(const amrex::Box& dom, const amrex::IntVect& c, int n)
{
    size_t idx = size_t(n);
    for (int d = AMREX_SPACEDIM - 1; d >= 0; --d) idx = idx*dom.length(d) + c[d];
    return idx;
}

/// Read a raw-double file (rows x ny x nx, x fastest; an empty path gives zeros) into the cell Field F and fill its halo.
void load(const std::string& path, zs::Field& F, const zs::Mesh& M)
{
    const amrex::Box dom = M.geom.Domain();
    const int nq = F.nq;
    amrex::Vector<double> v(size_t(nq)*dom.numPts(), 0.0);
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
        amrex::LoopOnCpu(b, nq, [&](int i, int j, int k, int n)
        {
            const amrex::IntVect c(AMREX_D_DECL(i, j, k));
            h(c, n) = v[flat(dom, c, n)];
        });
        F[0][mfi].copy<amrex::RunOn::Device>(host, b, 0, b, 0, F[0].nComp());   // pinned: device-visible
        amrex::Gpu::streamSynchronize();               // before ``host`` is freed
    }
    F = zs::halo(F, M);
}

/// Write (time, rows, the D cell counts, then the rows with x fastest) as doubles gathered on rank 0 (NaN where a level lacks cells).
void dump(const std::string& path, double time, const zs::Field& F, const zs::Mesh& M)
{
    const amrex::Box dom = M.geom.Domain();
    amrex::MultiFab all(amrex::BoxArray(dom), amrex::DistributionMapping(amrex::Vector<int>{0}),
                        F.nq, 0);
    all.setVal(std::numeric_limits<double>::quiet_NaN());     // cells the level lacks
    all.ParallelCopy(F[0], 0, 0, F.nq);
    if (!amrex::ParallelDescriptor::IOProcessor()) return;
    amrex::FArrayBox host(dom, F.nq, amrex::The_Pinned_Arena());
    host.copy<amrex::RunOn::Device>(all[0], dom, 0, dom, 0, F.nq);
    amrex::Gpu::streamSynchronize();
    std::vector<double> head{time, double(F.nq)};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) head.push_back(double(dom.length(d)));
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(head.data(), sizeof(double), head.size(), f);
    std::fwrite(host.dataPtr(), sizeof(double), size_t(F.nq)*dom.numPts(), f);
    std::fclose(f);
}

/// The restart state as an AMReX checkpoint directory: Header (format, time, rows and valid ghost
/// layers of Q and Qaux, the BoxArray) and the MultiFabs Q, Qaux (VisMF: every rank its own boxes,
/// ghost cells included).
void save_state(const std::string& dir, double time, const zs::Field& Q, const zs::Field& Qaux,
                const zs::Mesh& M)
{
    amrex::UtilCreateCleanDirectory(dir, true);
    amrex::VisMF::Write(Q[0], dir + "/Q");
    amrex::VisMF::Write(Qaux[0], dir + "/Qaux");
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream h(dir + "/Header");
        h << "zoomy-restart-1\n" << std::setprecision(17) << time << "\n" << Q.nq << " " << Q.ngv
          << " " << Qaux.nq << " " << Qaux.ngv << "\n";
        M.ba.writeOn(h);
        h << "\n";
    }
    amrex::ParallelDescriptor::Barrier();
}

/// Read a save_state checkpoint into Q, Qaux; returns its time.  The same BoxArray: the boxes with
/// their ghost cells as saved; another one: the valid cells redistributed, the halo refilled.
double load_state(const std::string& dir, zs::Field& Q, zs::Field& Qaux, const zs::Mesh& M)
{
    std::ifstream h(dir + "/Header");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(h.good(), "symAmrex: no restart state " + dir + "/Header");
    std::string fmt;
    double time = 0.0;
    int nq = -1, ngq = 0, na = -1, nga = 0;
    amrex::BoxArray ba;
    h >> fmt >> time >> nq >> ngq >> na >> nga;
    ba.readFrom(h);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(fmt == "zoomy-restart-1" && nq == Q.nq && na == Qaux.nq,
                                     "symAmrex: " + dir + " is not a restart state of this model");
    const bool same = ba == M.ba;
    for (auto [F, name, ngv] : {std::make_tuple(&Q, "Q", ngq), std::make_tuple(&Qaux, "Qaux", nga)})
    {
        if (same)
        {
            amrex::VisMF::Read((*F)[0], dir + "/" + name);
            F->ngv = ngv;
            continue;
        }
        amrex::MultiFab tmp;
        amrex::VisMF::Read(tmp, dir + "/" + name);
        (*F)[0].ParallelCopy(tmp, 0, 0, (*F)[0].nComp());
        F->ngv = 0;
        *F = zs::halo(*F, M);
    }
    return time;
}

/// Directory name of a time (12 significant digits).
std::string time_name(double t)
{
    std::ostringstream s;
    s << std::setprecision(12) << t;
    return s.str();
}
}

#ifdef ZS_AMR_PROCEDURES                // the program has the AMR procedures
#include "amr_amrex.H"
#endif

/// Entry point: read the zs.* inputs, build the mesh, run the printed march (single level, or run_amr), write the snapshots and the final state.
int main(int argc, char* argv[])
{
    // one GPU stream: the march runs without MFIter's end-of-loop synchronisation (a no-sync
    // region), so AMReX's per-box loops (MultiFab setVal / Copy / ... on ranks with 2 to 8
    // boxes above 65^3 cells, which it does not fuse) must stay ordered with the fused kernels
    // that follow them (ANALYSIS 55: 3 boxes on 2 GPUs assembled garbage pressure rows)
    amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, []
    {
        amrex::ParmParse pp("amrex");
        if (!pp.contains("max_gpu_streams")) pp.add("max_gpu_streams", 1);
    });
    amrex::ExecOnFinalize(zs::close_pool);
    int amr = 0;
    amrex::ParmParse("zs").query("amr", amr);
#ifdef ZS_AMR_PROCEDURES
    if (amr) zs::run_amr();
    else
#else
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!amr, "symAmrex: a program without AMR procedures");
#endif
    {
        amrex::ParmParse pp("zs");
        amrex::Vector<int> n_cell, branch(6, -1), periodic(AMREX_SPACEDIM, 0);
        amrex::Vector<amrex::Real> lo, hi;
        int max_grid_size = 64;
        double time_end = 0.0, time_start = 0.0;
        std::string lsq, q0, qaux0, out = ".", restart, checkpoint;
        pp.getarr("n_cell", n_cell);
        pp.getarr("prob_lo", lo);
        pp.getarr("prob_hi", hi);
        pp.query("max_grid_size", max_grid_size);
        pp.get("time_end", time_end);
        pp.queryarr("branch", branch);
        pp.queryarr("periodic", periodic);
        branch.resize(6, -1);
        periodic.resize(AMREX_SPACEDIM, 0);
        pp.get("lsq", lsq);
        pp.query("q0", q0);
        pp.query("qaux0", qaux0);
        pp.query("out", out);
        pp.query("time_start", time_start);
        pp.query("restart", restart);           // a save_state checkpoint directory
        checkpoint = out;
        pp.query("checkpoint", checkpoint);     // where save_state writes chk<t>

        amrex::IntVect hi_cell;
        amrex::Array<int, AMREX_SPACEDIM> per;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { hi_cell[d] = n_cell[d] - 1; per[d] = periodic[d]; }
        const amrex::Box dom(amrex::IntVect(0), hi_cell);
        zs::Mesh M;
        M.geom.define(dom, amrex::RealBox(lo.data(), hi.data()), amrex::CoordSys::cartesian, per);
        M.ba = amrex::BoxArray(dom);
        M.ba.maxSize(max_grid_size);
        M.dm = amrex::DistributionMapping(M.ba);
        const int br[6] = {branch[0], branch[1], branch[2], branch[3], branch[4], branch[5]};
        M.load(lsq, br);
        zs::the_mesh() = &M;
        zs::configure();
        amrex::Print() << "symAmrex: " << AMREX_SPACEDIM << "-D, " << dom.numPts() << " cells, "
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
        int nsnap = 0;
        amrex::Vector<std::string> names;
        for (int i = 0; i < Q.nq; ++i) names.push_back("Q" + std::to_string(i));
        zs::writer() = [&](double t, const zs::Field& q, const zs::Field&)
        {
            amrex::MultiFab v(M.ba, M.dm, q.nq, 0);
            amrex::MultiFab::Copy(v, q[0], 0, 0, q.nq, 0);
            amrex::WriteSingleLevelPlotfile(out + "/plt" + std::to_string(nsnap), v, names,
                                            M.geom, t, nsnap);
            dump(out + "/" + std::to_string(nsnap) + ".bin", t, q, M);
            ++nsnap;
        };

        Qaux = std::get<0>(zs::initial_aux(Q, Qaux, p, std::cref(M), 0L));
        if (!restart.empty())
        {
            zs::Field Qr = zs::cells_of(M, Q.nq), Ar = zs::cells_of(M, Qaux.nq);
            const double tr = load_state(restart, Qr, Ar, M);
            zs::loader() = [Qr, Ar, tr](double& t, zs::Field& q, zs::Field& a)
            {
                t = tr;
                q = Qr;
                a = Ar;
            };
        }
        amrex::UtilCreateDirectory(checkpoint, 0755);
        zs::saver() = [&](double t, const zs::Field& q, const zs::Field& qa)
        {
            save_state(checkpoint + "/chk" + time_name(t), t, q, qa, M);
        };
#ifdef ZS_PRECICE
        {   // zs.precice = participant config mesh; zs.precice_write / _read: data
            // names; zs.precice_side: the coupled side (2 d + upper)
            amrex::Vector<std::string> pc, wr, rd;
            int side = -1;
            pp.getarr("precice", pc);
            pp.getarr("precice_write", wr);
            pp.getarr("precice_read", rd);
            pp.get("precice_side", side);
            int layers = 0;           // zs.precice_overlap: an overlap participant's layers
            pp.query("precice_overlap", layers);
            if (layers > 0)
                zs::couple_setup_overlap(M, pc[0], pc[1], pc[2], {wr.begin(), wr.end()},
                                         {rd.begin(), rd.end()}, side, layers);
            else
                zs::couple_setup(M, pc[0], pc[1], pc[2], {wr.begin(), wr.end()},
                                 {rd.begin(), rd.end()}, side);
        }
#endif
        amrex::ParallelDescriptor::Barrier();
        const auto t0 = std::chrono::steady_clock::now();
        // the blocks synchronise where they read results on the host
        // (reductions, halo exchange, output); no sync after every copy
        const auto res = [&]
        {
            amrex::Gpu::NoSyncRegion no_sync;
            return zs::march(Q, Qaux, p, time_start, time_end, std::cref(M));
        }();
        zs::loader() = nullptr;                  // its Fields before amrex::Finalize
        amrex::Gpu::streamSynchronize();
        amrex::ParallelDescriptor::Barrier();
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        amrex::Print() << "symAmrex: march wall time " << wall << " s\n"
                       << "symAmrex: loop iterations (steps) " << zs::loop_iterations() << "\n";
#ifdef ZS_PRECICE
        zs::coupling().participant->finalize();
#endif
        dump(out + "/final.bin", std::get<0>(res), std::get<1>(res), M);
        amrex::Print() << "symAmrex: t = " << std::get<0>(res) << "\n";
    }
    amrex::Finalize();
    return 0;
}
