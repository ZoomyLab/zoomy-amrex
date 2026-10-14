#pragma once
/// \file
/// The AMR level loop of the symbolic hyperbolic solver (amrex::AmrCore, one dt for every level, no subcycling) around the printed single-level procedures of Solver.H (amr_stage<i> or step, dt_of, initial_aux, amr_w / amr_q, rk_weights).
/// A step advances the levels stage by stage (ZS_AMR_STAGES: the composite RK stage, MOOD on the composite) or, for IMEX-SSP2, the whole step per level coarse to fine; then fine to coarse the face terms are refluxed and the levels averaged down; the grid is regridded by tags every regrid_int steps.
#include <AMReX_AmrCore.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_TagBox.H>

namespace zs
{
/// Fill functor for the out-of-domain ghosts: a no-op, since the template's boundary blocks own them (the interpolation sees first-order extrapolation, BCRec foextrap).
struct NoFill
{
    AMREX_GPU_DEVICE void operator()(const IV&, amrex::Array4<Real> const&, int, int,
                                     amrex::GeometryData const&, Real, const amrex::BCRec*,
                                     int, int) const {}
};
using PhysBC = amrex::PhysBCFunct<amrex::GpuBndryFuncFab<NoFill>>;

/// Scope for AMReX's own AMR routines (interpolation, flux registers, average_down, regrid), which rely on MFIter's end-of-loop synchronisation: streams synchronised on entry and exit, the march's no-sync region lifted meanwhile.
struct Synced
{
    bool ns;
    Synced() { amrex::Gpu::streamSynchronize(); ns = amrex::Gpu::setNoSyncRegion(false); }
    ~Synced() { amrex::Gpu::streamSynchronize(); amrex::Gpu::setNoSyncRegion(ns); }
};

/// Tagging rule: mode 0 row > thr, 1 largest undivided difference of row > thr, 2 the region [lo, hi].
struct Tagging { int row = 0, mode = 0; Real thr = 0.0, lo[3] = {}, hi[3] = {}; };

/// The AMR level loop on amrex::AmrCore: per-level Mesh, state Q, aux A and flux register, one dt for every level.
/// Ref: Berger, Oliger 1984, https://doi.org/10.1016/0021-9991(84)90073-1
struct Amr : public amrex::AmrCore
{
    std::vector<std::unique_ptr<Mesh>> mesh;
    std::vector<Field> Q, A;
    std::vector<std::unique_ptr<amrex::FluxRegister>> fr;
    std::vector<std::string> lsq, q0;
    std::string qaux0;
    int branch[6];
    Col<NPAR> p{};
    amrex::Vector<amrex::BCRec> bcs;
    Tagging tag;
    bool reflux = true;
    // the coarse state and aux a finer level's halo is interpolated from (default Q, A: the
    // levels' current state; a composite stage: the stage states), piecewise constant if pc
    const std::vector<Field>* src_q = nullptr;
    const std::vector<Field>* src_a = nullptr;
    bool pc = false;

    /// Per-level storage, the BC types (periodic or foextrap) and the run inputs (LSQ tables, initial-data files, BC branches, parameters).
    Amr(const std::vector<std::string>& lsq_, const std::vector<std::string>& q0_,
        const std::string& qaux0_, const int br[6], const Col<NPAR>& p_)
        : lsq(lsq_), q0(q0_), qaux0(qaux0_), p(p_)
    {
        const int n = max_level + 1;
        mesh.resize(n); Q.resize(n); A.resize(n); fr.resize(n);
        for (int s = 0; s < 6; ++s) branch[s] = br[s];
        bcs.resize(std::max({System::n_state, System::n_aux, 1}));
        for (auto& b : bcs)
            for (int d = 0; d < D; ++d)
            {
                const int t = Geom(0).isPeriodic(d) ? amrex::BCType::int_dir
                                                    : amrex::BCType::foextrap;
                b.setLo(d, t); b.setHi(d, t);
            }
    }

    /// Create or update level lev's Mesh (grid, LSQ table, refluxing log) and, above level 0, its halo fill and flux register.
    void setup_level(int lev, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm)
    {
        if (!mesh[lev])
        {
            mesh[lev] = std::make_unique<Mesh>();
            mesh[lev]->geom = Geom(lev);
            mesh[lev]->load(lsq[lev], branch);
        }
        Mesh& M = *mesh[lev];
        M.ba = ba; M.dm = dm;
        M.log = std::make_shared<std::vector<std::array<Field, 2>>>();
        M.dlog = std::make_shared<std::vector<std::pair<Real, Field>>>();
        if (lev == 0) return;
        M.fill = [this, lev](const Field& X) { return fill_halo(lev, X); };
        fr[lev] = std::make_unique<amrex::FluxRegister>(ba, dm, refRatio(lev - 1), lev,
                                                        System::n_state);
    }

    /// A field of level lev - 1 interpolated onto (ba, dm) of level lev, ghosts too (conservative linear; piecewise constant if pc).
    Field interp(int lev, const Field& C, const amrex::BoxArray& ba,
                 const amrex::DistributionMapping& dm, bool pc = false)
    {
        Field F = make_like(ba, dm, C.nq, false, mesh[lev]->NG);
        F.ngv = F.ng();
        if (C.nq == 0) return F;
        const Synced scope;
        PhysBC cbc(Geom(lev - 1), bcs, amrex::GpuBndryFuncFab<NoFill>(NoFill{}));
        PhysBC fbc(Geom(lev), bcs, amrex::GpuBndryFuncFab<NoFill>(NoFill{}));
        amrex::InterpFromCoarseLevel(F[0], 0.0, C[0], 0, 0, C.nq, Geom(lev - 1), Geom(lev),
                                     cbc, 0, fbc, 0, refRatio(lev - 1),
                                     pc ? static_cast<amrex::Interpolater*>(&amrex::pc_interp)
                                        : &amrex::cell_cons_interp, bcs, 0);
        return F;
    }

    /// Level lev's state (interpolated in W = amr_w(Q), with Q's pointwise aux) and aux from level lev - 1 (src_q / src_a, default Q / A); only the aux when `state` is false.
    std::pair<Field, Field> from_coarse(int lev, const amrex::BoxArray& ba,
                                        const amrex::DistributionMapping& dm, bool state)
    {
        const Field& Qc = src_q ? (*src_q)[lev - 1] : Q[lev - 1];
        const Field& Ac = src_a ? (*src_a)[lev - 1] : A[lev - 1];
        Field a = interp(lev, Ac, ba, dm, pc);
        if (!state) return {Field(), a};
        const Mesh& M = *mesh[lev - 1];
        const Field As = std::get<0>(amr_stage_aux(Qc, Ac, p, std::cref(M), 0L));
        const Field Wc = std::get<0>(amr_w(Qc, As, p, std::cref(M), 0L));
        Field q = std::get<0>(amr_q(interp(lev, Wc, ba, dm, pc), a, p, std::cref(*mesh[lev]), 0L));
        q.ngv = q.ng();
        return {q, a};
    }

    /// The halo of a finer level: its own and periodic cells exact, the rest from the coarser level's current state or aux (by the field's rows).
    Field fill_halo(int lev, const Field& X)
    {
        BL_PROFILE("zs::amr_fill_halo");
        const Synced scope;
        if (X.nq == 0) { Field Y = X; Y.ngv = Y.ng(); return Y; }
        const bool state = X.nq == System::n_state;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(state != (X.nq == System::n_aux),
            "AMR halo: a field with neither (or both) the state's and the aux's rows");
        auto qa = from_coarse(lev, X[0].boxArray(), X[0].DistributionMap(), state);
        Field F = state ? qa.first : qa.second;
        amrex::MultiFab::Copy(F[0], X[0], 0, 0, X.nq, 0);
        F[0].FillBoundary(Geom(lev).periodicity());
        F.ngv = F.ng();
        return F;
    }

    /// AmrCore hook: build level lev from the initial-data files (level 0) or interpolated from the coarser level (optional per-level files).
    void MakeNewLevelFromScratch(int lev, Real, const amrex::BoxArray& ba,
                                 const amrex::DistributionMapping& dm) override
    {
        setup_level(lev, ba, dm);
        const Mesh& M = *mesh[lev];
        if (lev > 0)
        {
            auto qa = from_coarse(lev, ba, dm, true);
            Q[lev] = qa.first; A[lev] = qa.second;
            if (lev < int(q0.size()) && !q0[lev].empty()) load(q0[lev], Q[lev], M);
            return;
        }
        Q[0] = cells_of(M, System::n_state);
        A[0] = cells_of(M, System::n_aux);
        load(q0[0], Q[0], M);
        load(qaux0, A[0], M);
    }

    /// AmrCore hook: build a new level lev by interpolation from level lev - 1.
    void MakeNewLevelFromCoarse(int lev, Real, const amrex::BoxArray& ba,
                                const amrex::DistributionMapping& dm) override
    {
        setup_level(lev, ba, dm);
        auto qa = from_coarse(lev, ba, dm, true);
        Q[lev] = qa.first; A[lev] = qa.second;
    }

    /// AmrCore hook: regrid level lev, keeping its own cells exact and interpolating the rest from the coarser level.
    void RemakeLevel(int lev, Real, const amrex::BoxArray& ba,
                     const amrex::DistributionMapping& dm) override
    {
        const Field Qo = Q[lev], Ao = A[lev];
        setup_level(lev, ba, dm);
        auto qa = from_coarse(lev, ba, dm, true);
        qa.first[0].ParallelCopy(Qo[0], 0, 0, Qo.nq);      // the level's own cells exact
        if (Ao.nq > 0) qa.second[0].ParallelCopy(Ao[0], 0, 0, Ao.nq);
        qa.first.ngv = qa.second.ngv = 0;
        Q[lev] = qa.first; A[lev] = qa.second;
    }

    /// AmrCore hook: drop level lev's state, aux and flux register.
    void ClearLevel(int lev) override
    {
        Q[lev] = Field(); A[lev] = Field(); fr[lev].reset();
    }

    /// AmrCore hook: tag the cells of level lev by the Tagging rule.
    void ErrorEst(int lev, amrex::TagBoxArray& tags, Real, int) override
    {
        const Mesh& M = *mesh[lev];
        const Field q = need(Q[lev], 1, M);
        const auto qa = q[0].const_arrays();
        const auto ta = tags.arrays();
        const Tagging t = tag;
        const Geo g = M.g;
        amrex::ParallelFor(tags, [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
        {
            ZS_CELL;
            bool hit = true;
            if (t.mode == 2)
            {
                const Col<3> x = cell_centre(g, c);
                for (int a = 0; a < D; ++a) hit = hit && x(a, 0) >= t.lo[a] && x(a, 0) <= t.hi[a];
            }
            else if (t.mode == 0) hit = qa[bx](c, t.row) > t.thr;
            else
            {
                Real s = 0.0;
                for (int d = 0; d < D; ++d)
                    for (int o = -1; o <= 1; o += 2)
                    {
                        const IV n = c + o*unit(d);
                        if (inside(g, n)) s = amrex::max(s, std::abs(qa[bx](n, t.row) - qa[bx](c, t.row)));
                    }
                hit = s > t.thr;
            }
            if (hit) ta[bx](c) = amrex::TagBox::SET;
        });
    }

    /// What a sync refluxes: the step's RK stages with their weights rk_weights (and dlog), one stage (the last log entry), or the implicit fluxes of dlog alone.
    enum class Part { step, stage, implicit };

    /// The face terms logged by divergence (with the stage's diffusive fluxes) on the cell below (Wm) or above (-Wp) each face, integrated over `part`, plus the implicit diffusion's face fluxes by their share of dt (dlog), times mult.
    amrex::MultiFab face_integral(int lev, int d, bool below, Real mult, Part part)
    {
        const auto& log = *mesh[lev]->log;
        const auto& dlog = *mesh[lev]->dlog;
        const auto b = part == Part::step ? rk_weights()
                       : std::vector<double>(part == Part::stage ? 1 : 0, 1.0);
        const int ns = int(b.size()), n = int(log.size());
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(n >= ns, "AMR reflux: fewer face terms than RK stages");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ns > 0 || !dlog.empty(), "AMR reflux: no face terms");
        const amrex::MultiFab& f0 = ns > 0 ? log[n - 1][0][d] : dlog[0].second[d];
        amrex::MultiFab F(f0.boxArray(), f0.DistributionMap(), System::n_state, 0);
        F.setVal(0.0);
        for (int s = 0; s < ns; ++s)
            amrex::MultiFab::Saxpy(F, (below ? 1.0 : -1.0)*b[s]*mult,
                                   log[n - ns + s][below ? 0 : 1][d], 0, 0, System::n_state, 0);
        if (part != Part::stage)
            for (const auto& [w, Fd] : dlog)     // a flux: the same on both sides
                amrex::MultiFab::Saxpy(F, w*mult, Fd[d], 0, 0, System::n_state, 0);
        return F;
    }

    /// Level lev - 1 of X from level lev: reflux the coarse cells next to level lev with the fine faces' terms of `part` integrated over dt (a refluxed cell keeps the reconstruction variables of the rows not refluxed: amr_keep), then average down and clean round-off.
    /// Ref: Berger, Colella 1989, https://doi.org/10.1016/0021-9991(89)90035-1
    void sync(int lev, Real dt, std::vector<Field>& X, Part part)
    {
        BL_PROFILE("zs::amr_sync");
        const int nq = System::n_state;
        const Synced scope;
        Field Qc = like(X[lev - 1], nq);
        amrex::MultiFab::Copy(Qc[0], X[lev - 1][0], 0, 0, nq, 0);
        if (reflux && (part != Part::implicit || !mesh[lev]->dlog->empty()))
        {
            amrex::FluxRegister& R = *fr[lev];
            R.setVal(0.0);
            for (int d = 0; d < D; ++d)
            {
                Real ac = dt, af = dt;                    // face areas x dt
                for (int a = 0; a < D; ++a)
                    if (a != d) { ac *= Geom(lev - 1).CellSize(a); af *= Geom(lev).CellSize(a); }
                for (int up = 0; up < 2; ++up)
                {
                    const amrex::Orientation o(d, up ? amrex::Orientation::high
                                                     : amrex::Orientation::low);
                    R[o].copyFrom(face_integral(lev - 1, d, !up, -ac, part), 0, 0, 0, nq,
                                  Geom(lev - 1).periodicity());
                }
                // a fine box's lower faces: the coarse cell below; upper: above
                amrex::MultiFab lo = face_integral(lev, d, true, 1.0, part);
                const amrex::MultiFab hi = face_integral(lev, d, false, 1.0, part);
                const auto la = lo.arrays();
                const auto ha = hi.const_arrays();
                amrex::ParallelFor(lo, amrex::IntVect(0), nq,
                    [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k, int n) noexcept
                {
                    ZS_CELL;
                    const auto b0 = amrex::lbound(la[bx]);
                    if (c[d] != (d == 0 ? b0.x : (d == 1 ? b0.y : b0.z)))
                        la[bx](c, n) = ha[bx](c, n);
                });
                R.FineAdd(lo, d, 0, 0, nq, af);
            }
            for (const int r : reflux_rows()) R.Reflux(Qc[0], 1.0, r, r, 1, Geom(lev - 1));
            Qc = keep(lev - 1, X[lev - 1], Qc);
        }
        amrex::average_down(X[lev][0], Qc[0], 0, nq, refRatio(lev - 1));
        Qc.ngv = 0;
        X[lev - 1] = std::get<0>(amr_clean(Qc, std::cref(*mesh[lev - 1]), 0L));
        X[lev - 1].ngv = 0;
    }

    /// Qn (Qo refluxed on level lev) with amr_keep(Qo, Qn) in the cells whose refluxed rows changed: the refluxed mass carries the cell's velocity.
    Field keep(int lev, const Field& Qo, const Field& Qn)
    {
        Field Qo0 = Qo, Qn0 = Qn;
        Qo0.ngv = Qn0.ngv = 0;
        const Field K = std::get<0>(amr_keep(Qo0, Qn0, A[lev], p, std::cref(*mesh[lev]), 0L));
        Field out = like(Qn, Qn.nq);
        out.ngv = 0;
        const auto rr = reflux_rows();
        amrex::Gpu::DeviceVector<int> rows(rr.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, rr.begin(), rr.end(), rows.begin());
        const int nr = int(rr.size()), nq = Qn.nq;
        const int* r = rows.data();
        const auto o = out[0].arrays();
        const auto qo = Qo[0].const_arrays();
        const auto qn = Qn[0].const_arrays();
        const auto k = K[0].const_arrays();
        amrex::ParallelFor(out[0], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int kk) noexcept
        {
            const IV c(AMREX_D_DECL(i, j, kk));
            bool changed = false;
            for (int m = 0; m < nr; ++m) changed = changed || qn[bx](c, r[m]) != qo[bx](c, r[m]);
            for (int n = 0; n < nq; ++n) o[bx](c, n) = changed ? k[bx](c, n) : qn[bx](c, n);
        });
        amrex::Gpu::streamSynchronize();
        return out;
    }

    /// Every level's halo and aux at the current state.
    void refresh()
    {
        for (int l = 0; l <= finest_level; ++l)
        {
            Q[l].ngv = 0;
            Q[l] = halo(Q[l], *mesh[l]);
            A[l] = std::get<0>(initial_aux(Q[l], A[l], p, std::cref(*mesh[l]), 0L));
        }
    }

#ifdef ZS_AMR_STAGES
    /// One step of every level from t to t + dt by composite stages (`pass`), MOOD on the composite (the cells inadmissible on any level, then every cell, at first order), Lie's implicit slots level by level (refluxed), then the step's hygiene and aux.
    /// Ref: Clain, Diot, Loubere 2011, https://doi.org/10.1016/j.jcp.2011.02.026
    void advance(double t, double dt)
    {
        const int nl = finest_level + 1;
        std::vector<Field> Y = pass(t, dt, 0, std::vector<Field>(nl));
        if constexpr (amr_mood)
        {
            bool any = false;
            const auto bad = demoted(Y, any);
            if (any)
            {
                Y = pass(t, dt, 1, bad);
                bool still = false;
                demoted(Y, still);
                if (still) Y = pass(t, dt, 2, bad);
            }
        }
        for (int l = 0; l < nl; ++l) Q[l] = Y[l];
        if constexpr (amr_lie)
        {
            for (int l = 0; l < nl; ++l)        // the halo from the coarser level's new state
            {
                mesh[l]->log->clear();
                mesh[l]->dlog->clear();
                the_mesh() = mesh[l].get();
                Q[l] = std::get<0>(amr_implicit(Q[l], A[l], p, t, dt, std::cref(*mesh[l]), 0L));
            }
            the_mesh() = nullptr;
            for (int l = nl - 1; l > 0; --l) sync(l, dt, Q, Part::implicit);
        }
        for (int l = 0; l < nl; ++l)
        {
            Q[l].ngv = 0;
            const auto r = amr_post(Q[l], A[l], p, t, dt, std::cref(*mesh[l]), 0L);
            Q[l] = std::get<0>(r);
            A[l] = std::get<1>(r);
        }
    }

    /// The step's RK stages on every level from Q (its halo and aux A at t): stage i on each level coarse to fine, its face terms refluxed (share 1 - a_i) and the levels averaged down fine to coarse, then the halos of the stage states coarse to fine, so every finer level's stage reads the coarser level's state of the same stage (the composite stage: the coarse cells next to a finer level pay what its halo gives).  mode 1 demotes the cells of `bad` to first order, mode 2 every cell, with the halos piecewise constant: the first-order composite step is then the first-order scheme on the composite grid, positive under the levels' CFL step (ANALYSIS 52b, 55).
    std::vector<Field> pass(double t, double dt, int mode, const std::vector<Field>& bad)
    {
        BL_PROFILE("zs::amr_pass");
        const int nl = finest_level + 1;
        const auto w = amr_stage_weights();
        std::vector<Field> Y(Q.begin(), Q.begin() + nl), As(nl);
        src_q = &Y;
        src_a = &A;
        pc = mode == 2;
        if (pc)
            for (int l = 1; l < nl; ++l) { Y[l].ngv = 0; Y[l] = halo(Y[l], *mesh[l]); }
        for (int i = 0; i < int(w.size()); ++i)
        {
            for (int l = 0; l < nl; ++l)
            {
                mesh[l]->log->clear();
                mesh[l]->dlog->clear();
                the_mesh() = mesh[l].get();        // the level's linear solves
                // stage 0: the aux of the halo's states (an interpolated aux is not that of
                // the interpolated state); later stages take their states' own
                const Field& Ai = i == 0 && l > 0 ? (As[l] = std::get<0>(amr_stage_aux(
                                      Y[l], A[l], p, std::cref(*mesh[l]), 0L))) : A[l];
                Y[l] = amr_stage(i, mode, Q[l], Ai, Y[l], p, t, dt, *mesh[l], bad[l]);
                Y[l].ngv = 0;
            }
            the_mesh() = nullptr;
            for (int l = nl - 1; l > 0; --l) sync(l, w[i]*dt, Y, Part::stage);
            for (int l = 0; l < nl; ++l) Y[l] = halo(Y[l], *mesh[l]);
        }
        src_q = src_a = nullptr;
        pc = false;
        return Y;
    }

    /// The cells MOOD demotes after a pass: the inadmissible cells of every level but those covered by a finer level (any: whether there are some), with their halo: on a finer level its own cells' and, in the ghost cells of a coarse cell, that cell's.
    std::vector<Field> demoted(const std::vector<Field>& Y, bool& any)
    {
        const int nl = finest_level + 1;
        std::vector<Field> m(nl);
        any = false;
        for (int l = 0; l < nl; ++l)
        {
            m[l] = std::get<0>(amr_bad(Y[l], A[l], p, std::cref(*mesh[l]), 0L));
            m[l].ngv = 0;
            if (l + 1 < nl)
            {
                const Synced scope;
                const amrex::iMultiFab fm = amrex::makeFineMask(boxArray(l), DistributionMap(l),
                                                                boxArray(l + 1), refRatio(l));
                const auto f = fm.const_arrays();
                const auto a = m[l][0].arrays();
                amrex::ParallelFor(m[l][0], [=] AMREX_GPU_DEVICE (int bx, int i, int j, int k) noexcept
                {
                    if (f[bx](i, j, k)) a[bx](i, j, k) = 0.0;
                });
            }
            any = reduce_any(m[l]) || any;
        }
        for (int l = 0; l < nl; ++l) m[l] = l == 0 ? halo(m[0], *mesh[0]) : mask_halo(l, m[l], m[l - 1]);
        return m;
    }

    /// The halo of level lev's mask: its own cells exact, the rest piecewise constant from the coarser level's mask C.
    Field mask_halo(int lev, const Field& X, const Field& C)
    {
        const Synced scope;
        Field F = make_like(X[0].boxArray(), X[0].DistributionMap(), 1, false, mesh[lev]->NG);
        PhysBC cbc(Geom(lev - 1), bcs, amrex::GpuBndryFuncFab<NoFill>(NoFill{}));
        PhysBC fbc(Geom(lev), bcs, amrex::GpuBndryFuncFab<NoFill>(NoFill{}));
        amrex::InterpFromCoarseLevel(F[0], 0.0, C[0], 0, 0, 1, Geom(lev - 1), Geom(lev), cbc, 0,
                                     fbc, 0, refRatio(lev - 1), &amrex::pc_interp, bcs, 0);
        amrex::MultiFab::Copy(F[0], X[0], 0, 0, 1, 0);
        F[0].FillBoundary(Geom(lev).periodicity());
        F.ngv = F.ng();
        return F;
    }
#else
    /// One step of every level from t to t + dt: each level's whole step coarse to fine (a finer level's halo from the coarser level's new state), then fine to coarse the step's face terms refluxed and the levels averaged down.
    void advance(double t, double dt)
    {
        for (int l = 0; l <= finest_level; ++l)
        {
            mesh[l]->log->clear();
            mesh[l]->dlog->clear();
            the_mesh() = mesh[l].get();            // the level's linear solves
            const auto r = step(Q[l], A[l], p, t, dt, std::cref(*mesh[l]), 0L);
            Q[l] = std::get<0>(r);
            A[l] = std::get<1>(r);
        }
        the_mesh() = nullptr;
        for (int l = finest_level; l > 0; --l) sync(l, dt, Q, Part::step);
    }
#endif

    /// The time loop over all levels: one dt per step (min of the levels' CFL steps), regrid every regrid_int steps, write every dt_write; returns the end time.
    double march(double time_end, double dt_write, int regrid_int,
                 const std::function<void(double)>& write)
    {
        double t = 0.0;
        long n = 0, nw = 1;
        while (t < time_end)
        {
            if (regrid_int > 0 && n > 0 && n % regrid_int == 0 && max_level > 0)
            {
                refresh();
                const Synced scope;
                regrid(0, t);
            }
            refresh();
            double dt = time_end - t;
            if (dt_write > 0) dt = minimum(dt, nw*dt_write - t);
            for (int l = 0; l <= finest_level; ++l)
                dt = minimum(dt, std::get<0>(dt_of(Q[l], A[l], p, std::cref(*mesh[l]), 0L)));
            if (!(t + dt > t))          // NaN, <= 0 or below the resolution of t: check.march
                check_march(0L, t, time_end, dt, n + 1, q_max());
            advance(t, dt);
            for (int l = 0; l <= finest_level; ++l) { mesh[l]->log->clear(); mesh[l]->dlog->clear(); }
            t += dt; ++n; ++loop_iterations();
            if (dt_write > 0 && t >= (nw - 1e-9)*dt_write) { const Synced scope; write(t); ++nw; }
        }
        check_march(0L, t, time_end, 0.0, n, q_max());     // a non-finite final state fails
        return t;
    }

    /// max |Q| over all levels (NaN if any entry is NaN).
    double q_max()
    {
        double m = 0.0;
        for (int l = 0; l <= finest_level; ++l) m = maximum(m, reduce_max(abs(Q[l])));
        return m;
    }
};

/// The AMR driver (zs.amr = 1): inputs as the single-level run plus zs.lsq / zs.q0 per level, zs.dt_write, zs.regrid_int, zs.reflux, zs.tag / zs.tag_lo / zs.tag_hi; writes plt<k> and <k>_l<lev>.bin per level and final_l<lev>.bin.
inline void run_amr()
{
    amrex::ParmParse pp("zs");
    amrex::Vector<int> branch(6, -1);
    amrex::Vector<std::string> lsq, q0;
    std::string qaux0, out = ".";
    double time_end = 0.0, dt_write = 0.0;
    int regrid_int = 0, reflux = 1;
    pp.get("time_end", time_end);
    pp.queryarr("branch", branch);
    branch.resize(6, -1);
    pp.getarr("lsq", lsq);
    pp.queryarr("q0", q0);
    pp.query("qaux0", qaux0);
    pp.query("out", out);
    pp.query("dt_write", dt_write);
    pp.query("regrid_int", regrid_int);
    pp.query("reflux", reflux);
    const int br[6] = {branch[0], branch[1], branch[2], branch[3], branch[4], branch[5]};
    Col<NPAR> p{};
    const auto pv = System::parameters();
    for (int i = 0; i < NPAR; ++i) p(i, 0) = pv[i];
    for (auto& s : q0) if (s == "-") s.clear();
    configure();

    Amr amr({lsq.begin(), lsq.end()}, {q0.begin(), q0.end()}, qaux0, br, p);
    amr.reflux = reflux != 0;
    amrex::Vector<amrex::Real> tg, lo, hi;
    if (pp.queryarr("tag", tg))
    { amr.tag.mode = int(tg[0]); amr.tag.row = int(tg[1]); amr.tag.thr = tg[2]; }
    if (pp.queryarr("tag_lo", lo) && pp.queryarr("tag_hi", hi))
    {
        amr.tag.mode = 2;
        for (int d = 0; d < D; ++d) { amr.tag.lo[d] = lo[d]; amr.tag.hi[d] = hi[d]; }
    }
    the_mesh() = nullptr;
    amr.InitFromScratch(0.0);
    for (int l = amr.finestLevel(); l > 0; --l)
    {
        Field Qc = like(amr.Q[l - 1], System::n_state);
        amrex::MultiFab::Copy(Qc[0], amr.Q[l - 1][0], 0, 0, System::n_state, 0);
        amrex::average_down(amr.Q[l][0], Qc[0], 0, System::n_state, amr.refRatio(l - 1));
        amr.Q[l - 1] = Qc;
    }
    amr.refresh();

    amrex::UtilCreateDirectory(out, 0755);
    dump(out + "/initial.bin", 0.0, amr.Q[0], *amr.mesh[0]);    // the composite at t = 0
    int nsnap = 0;
    amrex::Vector<std::string> names;
    for (int i = 0; i < System::n_state; ++i) names.push_back("Q" + std::to_string(i));
    auto write = [&](double t, const std::string& tagname)
    {
        const int nl = amr.finestLevel() + 1;
        amrex::Vector<amrex::MultiFab> v(nl);
        for (int l = 0; l < nl; ++l)
        {
            v[l].define(amr.boxArray(l), amr.DistributionMap(l), System::n_state, 0);
            amrex::MultiFab::Copy(v[l], amr.Q[l][0], 0, 0, System::n_state, 0);
            dump(out + "/" + tagname + "_l" + std::to_string(l) + ".bin", t, amr.Q[l], *amr.mesh[l]);
        }
        amrex::WriteMultiLevelPlotfile(out + "/plt" + std::to_string(nsnap), nl,
                                       amrex::GetVecOfConstPtrs(v), names, amr.Geom(), t,
                                       amrex::Vector<int>(nl, nsnap), amr.refRatio());
        ++nsnap;
    };
    if (dt_write > 0) write(0.0, "0");
    amrex::Print() << "symAmrex AMR: " << D << "-D, levels " << amr.finestLevel() + 1
                   << ", reflux " << amr.reflux << "\n";
    amrex::ParallelDescriptor::Barrier();
    const auto t0 = std::chrono::steady_clock::now();
    const double t = [&]
    {
        amrex::Gpu::NoSyncRegion no_sync;
        return amr.march(time_end, dt_write, regrid_int,
                         [&](double tw) { write(tw, std::to_string(nsnap)); });
    }();
    amrex::Gpu::streamSynchronize();
    amrex::ParallelDescriptor::Barrier();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    amrex::Print() << "symAmrex: march wall time " << wall << " s\n"
                   << "symAmrex: loop iterations (steps) " << loop_iterations() << "\n";
    for (int l = 0; l <= amr.finestLevel(); ++l)
    {
        amrex::Print() << "symAmrex AMR: level " << l << " cells " << amr.boxArray(l).numPts() << "\n";
        dump(out + "/final_l" + std::to_string(l) + ".bin", t, amr.Q[l], *amr.mesh[l]);
    }
    dump(out + "/final.bin", t, amr.Q[0], *amr.mesh[0]);
    amrex::Print() << "symAmrex: t = " << t << "\n";
}
} // namespace zs
