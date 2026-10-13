"""Printers of a frozen `SystemModel` for the AMReX C++ drivers: `Model.H` (source,
Jacobians, aux update, eigenvalues, BC and Chorin kernels) and `Numerics.H` (core's
`AmrexNumerics`).  Opaque calls (`eigensystem`, `eigenvalues`, `solve`, ...) are routed to
the hand-written `UserFunctions.H`, as foam does.
"""

import functools

import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase
from zoomy_core.transformation.to_amrex import AmrexCore, AmrexNumerics


# Cast both branches of a ``conditional`` to ``amrex::Real`` so a dry-gate
# ternary whose branches are integer 0 (e.g. ``gate_eigenvalues_dry``) does not
# produce an ``int`` that breaks ``amrex::max(int, double)``.  Core's default
# emits raw ``(c ? t : f)``; we override only the type.
def _conditional_real(p, c, t, f):
    """Print `conditional` with both branches cast to `amrex::Real` (else `amrex::max` sees an
    int).
    """
    return (f"(({p.doprint(c)}) ? (amrex::Real({p.doprint(t)})) "
            f": (amrex::Real({p.doprint(f)})))")


# sympy emits the 2-arg ``Heaviside(x, h0)`` from a subgradient, e.g.
# ``d/dh Max(h, c) = Heaviside(h - c, 1/2)`` (the friction h-floor). C++ has no
# Heaviside; lower it to a branch with the same value-at-zero ``h0``.
def _heaviside_real(p, x, *h0):
    """Print the two-argument `Heaviside(x, h0)` as a branch with value `h0` at zero."""
    xv = p.doprint(x)
    h0v = p.doprint(h0[0]) if h0 else "0.5"
    return (f"(({xv}) > 0.0 ? amrex::Real(1.0) "
            f": (({xv}) < 0.0 ? amrex::Real(0.0) : amrex::Real({h0v})))")


def _emit_user_call(name):
    """Return a `c_functions` printer emitting the opaque call `amrex_user::<name>(args)`."""

    def _printer(p, *args):
        """Print the opaque call `amrex_user::<name>(args)`."""
        return f"amrex_user::{name}({', '.join(p.doprint(a) for a in args)})"

    return _printer


# Numerics printer = core's AmrexNumerics, extended only with the real-valued
# ``conditional`` / ``Heaviside`` C helpers.  The NDimArray-in-scalar-context
# unwrap that used to live here is gone: core now guarantees the lowering seam
# hands C-family printers pure scalars (shared conditional expansion + flatten,
# core@9f0f2fc), so a singleton array never reaches the printer.  Otherwise core's
# AmrexNumerics is used verbatim (REQ-58: real local_max_abs_eigenvalue, no
# NumericsUser.H indirection).
#
# The opaque kernels MUST be re-mapped here as well as on the Model printer.
# Core's ``generic_c.py`` map emits a BARE ``eigensystem(...)`` and has no entry
# at all for ``eigenvalues`` / ``solve`` (they fall through to the raw-name
# fallback in ``zoomy_core/transformation/to_amrex.py::AmrexCore._print_Function``),
# and this printer is what writes ``Numerics.H`` / ``NumericsPred.H``.  ADL cannot
# rescue the unqualified call: every argument is ``double`` or ``int``, a
# fundamental type with no associated namespace, so ``amrex_user`` is never
# searched.  Without these three entries ``--model vam`` fails to compile with
# "'eigenvalues' was not declared in this scope".
class AmrexNumericsPrinter(AmrexNumerics):
    """Core's `AmrexNumerics` with real-valued `conditional` / `Heaviside` and the opaque
    eigen- and solve calls routed to `amrex_user`.
    """
    c_functions = {**AmrexNumerics.c_functions, "conditional": _conditional_real,
                   "Heaviside": _heaviside_real,
                   "eigenvalues": _emit_user_call("eigenvalues"),
                   "eigensystem": _emit_user_call("eigensystem"),
                   "spectral_radius": _emit_user_call("spectral_radius"),
                   "solve": _emit_user_call("solve")}


class AmrexSystemModelPrinter(AmrexCore, GenericCppBase):
    """Emit `Model.H` (operators, BC kernels, state-row map, Chorin metadata) from a frozen
    `SystemModel`.  Called by `generate_headers` and `write_chorin_headers`.
    """

    _wrapper_name = "Model"
    _is_template_class = False
    gpu_enabled = True

    # Placeholder-function map: base C kernels + the opaque ones (foam parity).
    c_functions = {
        **GenericCppBase.c_functions,
        "conditional": _conditional_real,
        "Heaviside": _heaviside_real,
        "eigenvalues": _emit_user_call("eigenvalues"),
        "eigensystem": _emit_user_call("eigensystem"),
        "solve": _emit_user_call("solve"),
        "compute_derivative": _emit_user_call("compute_derivative"),
    }

    def _detect_has_diffusion(self):
        """Whether the model's `diffusion_matrix` is non-zero (sets `Model::has_diffusion`)."""
        import sympy as _sp
        expr = getattr(self.sm, "diffusion_matrix", None)
        if expr is None:
            return False
        if hasattr(expr, "__iter__"):
            return not all(_sp.simplify(e) == 0 for e in _sp.flatten(expr))
        return _sp.simplify(expr) != 0

    def __init__(self, sm, *, analytical_eigenvalues=True, wrapper_name="Model",
                 time_position_ops=False):
        """Bind the SystemModel `sm` and register the Q, Qaux, n, p, X and time symbol maps;
        `dt` is appended to `p` when an operator uses it.

        Args:
            analytical_eigenvalues: print the model's eigenvalues; else zeros (the backend then uses
                a numerical spectral radius).
            wrapper_name: name of the emitted struct.
            time_position_ops: `source` / `update_aux_variables` take `(time, X)` (hyperbolic
                `Model.H` only).
        """
        super().__init__()
        self.sm = sm
        self.model = sm  # AmrexCore signature helpers read ``self.model.*``
        self.analytical_eigenvalues = analytical_eigenvalues
        self._wrapper_name = wrapper_name
        # REQ-185: append (time, X) to source / update_aux_variables (+ its
        # jacobian) for the STANDARD hyperbolic Model.H only.  The Chorin split
        # sub-models (write_chorin_headers) keep the 3-arg source — their driver
        # carries dt via a parameter and does not thread cell time/position.
        self._t185 = time_position_ops
        # n_state = length of the Q vector; n_eq = number of evolution equations
        # (operator rows).  Equal for square models (SWE/SME/full VAM); for a
        # Chorin split sub-model n_eq < n_state (it evolves only its own rows).
        self.n_state = len(sm.state)
        self.n_eq = int(getattr(sm, "n_equations", len(sm.state)))
        self.n_aux = len(sm.aux_state)
        self.dim = int(sm.dimension)
        # Parameter list: the model's parameters, plus the time-step symbol ``dt``
        # appended as the LAST parameter IF the operators reference it (the Chorin
        # pressure/corrector sub-models carry it).  The driver writes dt into that
        # slot each step ("dt rides in p(last)"), mirroring the foam ChorinDriver.
        import sympy as sp
        from zoomy_core.transformation.to_numpy import DT_SYMBOL
        self._param_syms = list(sm.parameters.values())
        self._param_names = list(sm.parameters.keys())
        if self._uses_symbol(DT_SYMBOL):
            self._param_syms = self._param_syms + [DT_SYMBOL]
            self._param_names = self._param_names + ["dt"]
        self.n_param = len(self._param_syms)
        # Symbol → accessor maps (order matters: most specific first is fine since
        # the symbol sets are disjoint).
        self.register_map("Q", list(sm.state))
        self.register_map("Qaux", list(sm.aux_state))
        self.register_map("n", list(sm.normal.values()))
        self.register_map("p", self._param_syms)
        # REQ-185: bind the scalar time ``t`` and the length-3 position vector
        # ``x/y/z`` so a ``source`` / ``update_aux`` referencing them lowers
        # (same convention the BC kernel uses).  ``x/y/z`` → ``X(i, 0)``;
        # ``time`` is a by-value scalar.  The in-scope hydrostatic models never
        # reference them, so their bodies are unchanged.
        self.register_map("X", list(sm._position_struct().values()))
        self.symbol_maps.append({sm.time: "time"})

    def _uses_symbol(self, sym):
        """True if `sym` appears in any printed operator (decides whether `dt` is appended to
        `p`).
        """
        import sympy as sp
        for attr in ("flux", "source", "nonconservative_matrix", "eigenvalues",
                     "update_variables", "update_aux_variables", "state_update",
                     "hydrostatic_pressure"):
            M = getattr(self.sm, attr, None)
            if M is None:
                continue
            try:
                flat = sp.flatten(M)
            except Exception:
                flat = [M]
            for e in flat:
                if sym in sp.sympify(e).free_symbols:
                    return True
        return False

    # ── type / signature helpers ────────────────────────────────────────────
    def _t(self, n):
        """Array type of `n` entries."""
        return self.get_array_type((n,))

    def _args(self, *names):
        """C++ argument list for the named arguments."""
        tbl = {
            "Q": f"{self._t(self.n_state)} const& Q",
            "Qaux": f"{self._t(self.n_aux)} const& Qaux",
            "p": f"{self._t(self.n_param)} const& p",
            "n": f"{self._t(self.dim)} const& n",
            "X": f"{self._t(3)} const& X",
            "gradQ": f"{self._t(self.n_state * self.dim)} const& gradQ",
            "time": "amrex::Real const time",
            "dt": "amrex::Real const dt",
            "dX": "amrex::Real const dX",
            "bc_idx": "const int bc_idx",
        }
        return ",\n        ".join(tbl[a] for a in names)

    def _registry_argnames(self, name):
        """Argument names of operator `name` from `Function.args`, minus `dt` (it rides in
        `p`).
        """
        return tuple(k for k in self._operator_arg_keys(name) if k != "dt")

    def _kernel(self, name, expr, shape, argnames):
        """Print one operator as a device function of the arguments `argnames`."""
        # Bed/surface-trace ``Subs(f(ζ),ζ,0|1)`` nodes are resolved centrally at
        # the SystemModel → NumericalSystemModel seam (REQ-130,
        # numerical_system_model._resolve_boundary_traces) and guarded there, so
        # by the time an operator reaches a printer it is Subs-free — no
        # printer-local resolve is needed.
        body = self.convert_expression_body(expr, shape)
        return self.wrap_function_signature(name, self._args(*argnames), body, shape)

    # ── expression builders ─────────────────────────────────────────────────
    def _vec(self, elements):
        """Column `sp.Array` of `elements`."""
        return sp.Array(list(elements)).reshape(len(elements), 1)

    def _zeros(self, n):
        """Zero column of length `n`."""
        return self._vec([sp.Integer(0)] * n)

    def _identity_flat(self, n):
        """The `n x n` identity, flattened to a column."""
        return self._vec([sp.Integer(1) if r == c else sp.Integer(0)
                          for r in range(n) for c in range(n)])

    def _flatten_dir_major(self, arr, n_rows):
        """Flatten an `(n_rows, ..., dim)` tensor in direction-major order to a column."""
        flat = [arr[(*idx, d)]
                for d in range(self.dim)
                for idx in self._row_indices(arr, n_rows)]
        return self._vec(flat)

    def _row_indices(self, arr, n_rows):
        """All index tuples of `arr` without its trailing direction axis."""
        import itertools
        inner = arr.shape[:-1]  # drop the trailing dim axis
        return list(itertools.product(*(range(s) for s in inner)))

    # ── per-slot expressions, falling back to stubs when sm lacks a slot ─────
    def _expr_flux(self):
        """The flux, direction-major, as a column and its length."""
        f = self.sm.flux  # (n_eq, dim)
        flat = [f[i, d] for d in range(self.dim) for i in range(self.n_eq)]
        return self._vec(flat), self.n_eq * self.dim

    def _expr_matrix(self, mat):
        """An `(n_eq, n_state, dim)` matrix flattened direction-major, as a column and its length."""
        # (n_eq, n_state, dim) -> dir-major flat (dim * n_eq * n_state)
        flat = [mat[i, j, d]
                for d in range(self.dim)
                for i in range(self.n_eq)
                for j in range(self.n_state)]
        return self._vec(flat), self.dim * self.n_eq * self.n_state

    def _expr_source(self):
        """The source as a column and its length."""
        s = self.sm.source
        return self._vec([s[i, 0] for i in range(self.n_eq)]), self.n_eq

    def _expr_eigenvalues(self):
        """The eigenvalues as a column and its length (zeros when the model has none)."""
        # eigenvalues are length n_state (the spectrum of the full quasilinear
        # operator).  None (declarative / hand-built VAM) -> zeros: the backend
        # falls back to a numerical spectral radius (Chorin predictor path).
        e = getattr(self.sm, "eigenvalues", None)
        if e is None:
            return self._zeros(self.n_state), self.n_state
        return self._vec([e[i, 0] for i in range(self.n_state)]), self.n_state

    def _expr_source_jac(self):
        """The source Jacobian w.r.t. the state, flattened (zeros when absent)."""
        # core renamed source_jacobian -> source_jacobian_wrt_variables (n_eq, n_state)
        J = getattr(self.sm, "source_jacobian_wrt_variables", None)
        if J is None:
            J = getattr(self.sm, "source_jacobian", None)  # legacy fallback
        if J is None:
            return self._zeros(self.n_eq * self.n_state), self.n_eq * self.n_state
        return self._vec([J[r, c] for r in range(self.n_eq)
                          for c in range(self.n_state)]), self.n_eq * self.n_state

    def _expr_source_jac_aux(self):
        """The source Jacobian w.r.t. the aux variables, flattened (zeros when absent)."""
        J = getattr(self.sm, "source_jacobian_wrt_aux_variables", None)  # (n_eq, n_aux)
        if J is None:
            return self._zeros(self.n_eq * self.n_aux), self.n_eq * self.n_aux
        return self._vec([J[r, c] for r in range(self.n_eq)
                          for c in range(self.n_aux)]), self.n_eq * self.n_aux

    def _expr_update_aux(self):
        """The aux update as a column (identity when absent)."""
        ua = getattr(self.sm, "update_aux_variables", None)
        if ua is None:
            # identity: return Qaux unchanged
            return self._vec(list(self.sm.aux_state)), self.n_aux
        return self._vec([ua[i, 0] for i in range(self.n_aux)]), self.n_aux

    def _bc_definition(self, attr):
        """The Piecewise body of the SystemModel BC `Function` slot `attr`, or None."""
        fn = getattr(self.sm, attr, None)
        if fn is None:
            return None
        # The framework BC slot is a custom sp.Function carrying ``.definition``
        # (a nested array / Piecewise).  Fall back to the function's single arg.
        defn = getattr(fn, "definition", None)
        if defn is None and getattr(fn, "args", None):
            defn = fn
        return defn

    # ── header / footer ─────────────────────────────────────────────────────
    def get_includes(self):
        """AMReX includes of `Model.H`."""
        return ('#include <AMReX_Array4.H>\n'
                '#include <AMReX_Vector.H>\n'
                '#include <AMReX_SmallMatrix.H>')

    def _file_header(self, bc_tags):
        """Top of `Model.H`: sizes, state-row map (b, h, depth-scaled, discharge), BC tags,
        parameters.
        """
        # parameter_names/default_parameters follow self._param_names (which may
        # include the appended "dt" slot); dt defaults to 0 (the driver sets it).
        pv = dict(self.sm.parameter_values)
        p_names = ", ".join(f'"{k}"' for k in self._param_names)
        p_vals = ", ".join(str(pv.get(k, 0.0)) for k in self._param_names)
        bc_str = ", ".join(f'"{t}"' for t in bc_tags)
        # ── the model's own STATE ROW MAP ───────────────────────────────────
        # The driver must never infer "which row is the y-momentum" from the
        # state ORDER.  Contiguity after ``h`` holds only for the four-row SWE
        # state ``[b, h, q_x_0, q_y_0]``; SME(level=N, dimension=3) lays out
        # ``[b, h, q_x_0..q_x_N, q_y_0..q_y_N]``, so ``idx_h + 2`` is q_x_1.
        # READ from the SystemModel (the ONE answer-holder); a printer
        # translates, it never decides.
        #
        #   idx_b / idx_h        bathymetry / depth rows, -1 when absent
        #   depth_scaled_rows    every row carrying one factor of the depth
        #                        (q = h * intensive) — exactly the rows the
        #                        Audusse hydrostatic reconstruction must
        #                        rescale by h*/h so the reconstructed pair is
        #                        a STATE (q* = h* u).  Rescaling only some of
        #                        them lets a face with h* = 0 export mass.
        #   discharge_rows[d]    the row whose value IS the mass flux in
        #                        direction d — the row that carries the
        #                        bed-slope force and therefore receives the
        #                        Audusse well-balancing source.  -1 where the
        #                        system has no unique such row (multilayer);
        #                        ZoomyAmr refuses solver.well_balanced then.
        #                        Padded to two entries so the driver's
        #                        `if constexpr (dimension == 2)` branch stays
        #                        in bounds on a 1-D model, where it is dead.
        #
        # The two lists are published as constexpr ACCESSORS, not as static
        # member arrays: nvcc rejects a `static constexpr int a[N]` member
        # subscripted from device code ("identifier is undefined in device
        # code") because the array is an object needing device storage, while
        # a function-local constexpr array inside an inlined accessor folds to
        # an immediate.
        names = [str(s) for s in self.sm.state]
        scaled = list(self.sm.depth_scaled_state_indices) or [-1]
        discharge = list(self.sm.discharge_state_indices)
        discharge = discharge + [-1] * (2 - len(discharge))
        row_map = [
            f"    static constexpr int idx_b = "
            f"{names.index('b') if 'b' in names else -1};",
            f"    static constexpr int idx_h = "
            f"{names.index('h') if 'h' in names else -1};",
            f"    static constexpr int n_depth_scaled_rows = "
            f"{len(self.sm.depth_scaled_state_indices)};",
            "    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "    static constexpr int depth_scaled_row(int m) noexcept {",
            f"        constexpr int rows[{len(scaled)}] = "
            f"{{ {', '.join(str(i) for i in scaled)} }};",
            "        return rows[m]; }",
            "    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE",
            "    static constexpr int discharge_row(int d) noexcept {",
            f"        constexpr int rows[{len(discharge)}] = "
            f"{{ {', '.join(str(i) for i in discharge)} }};",
            "        return rows[d]; }",
        ]
        L = [
            "#pragma once",
            self.get_includes(),
            "#include <vector>",
            "#include <string>",
            "#include <algorithm>",
            "",
            "#ifdef __CUDACC__",
            "#define PORTABLE_FN __host__ __device__",
            "#else",
            "#define PORTABLE_FN",
            "#endif",
            "",
            '#include "UserFunctions.H"',
            "",
            f"struct {self._wrapper_name} {{",
            "    using T = amrex::Real;",
            f"    static constexpr int n_dof_q    = {self.n_state};",
            f"    static constexpr int n_dof_eq   = {self.n_eq};",
            f"    static constexpr int n_dof_qaux = {self.n_aux};",
            f"    static constexpr int n_parameters = {self.n_param};",
            f"    static constexpr int dimension  = {self.dim};",
            f"    static constexpr int n_dof_gradQ = {self.n_state * self.dim};",
            f"    static constexpr bool has_diffusion = {'true' if self._detect_has_diffusion() else 'false'};",
            *row_map,
            f"    static constexpr int n_boundary_tags = {len(bc_tags)};",
            f"    static const std::vector<std::string> get_boundary_tags() {{ return {{ {bc_str} }}; }}",
            f"    static const std::vector<std::string> parameter_names() {{ return {{ {p_names} }}; }}",
            f"    static const std::vector<T> default_parameters() {{ return {{ {p_vals} }}; }}",
        ]
        return "\n".join(L)

    # ── top-level emission ───────────────────────────────────────────────────
    def create_code(self):
        """Print the whole `Model.H` struct: every operator kernel (zero stubs for the slots the
        driver does not read), the BC kernels and, on a Chorin sub-model, the pressure-operator
        kernels and metadata.
        """
        sm = self.sm
        bc_src = getattr(sm, "_bc_source", None)
        bc_tags = sorted(bc_src.boundary_conditions_list_dict.keys()) \
            if bc_src is not None else []
        blocks = [self._file_header(bc_tags)]

        ne, ns, na, d = self.n_eq, self.n_state, self.n_aux, self.dim

        # flux / NCP / quasilinear (emitted for completeness; the C++ driver
        # reads these from Numerics.H, not from Model.H)
        fexpr, fshape = self._expr_flux()
        blocks.append(self._kernel("flux", fexpr, (fshape, 1), ("Q", "Qaux", "p")))
        blocks.append(self._kernel("dflux", self._zeros(ne * d), (ne * d, 1),
                                   ("Q", "Qaux", "p")))
        blocks.append(self._kernel("hydrostatic_pressure", self._zeros(ne * d),
                                   (ne * d, 1), ("Q", "Qaux", "p")))
        ncp, nshape = self._expr_matrix(sm.nonconservative_matrix)
        blocks.append(self._kernel("nonconservative_matrix", ncp, (nshape, 1),
                                   ("Q", "Qaux", "p")))
        ql, qshape = self._expr_matrix(sm.quasilinear_matrix)
        blocks.append(self._kernel("quasilinear_matrix", ql, (qshape, 1),
                                   ("Q", "Qaux", "p")))

        # diffusive_flux: zero stub (has_diffusion=false → never called, but the
        # symbol must exist: make_rhs.H references it in a non-template
        # `if constexpr` branch which is still fully type-checked).
        blocks.append(self._kernel("diffusive_flux", self._zeros(ne * d),
                                   (ne * d, 1), ("Q", "Qaux", "gradQ", "p")))

        # eigenvalues (length n_state; depend on the face normal)
        eexpr, _ = self._expr_eigenvalues()
        if self.analytical_eigenvalues:
            blocks.append(self._kernel("eigenvalues", eexpr, (ns, 1),
                                       ("Q", "Qaux", "p", "n")))
        else:
            blocks.append(self._kernel("eigenvalues", self._zeros(ns), (ns, 1),
                                       ("Q", "Qaux", "p", "n")))
        blocks.append(self._kernel("left_eigenvectors", self._zeros(ns * ns),
                                   (ns * ns, 1), ("Q", "Qaux", "p", "n")))
        blocks.append(self._kernel("right_eigenvectors", self._zeros(ns * ns),
                                   (ns * ns, 1), ("Q", "Qaux", "p", "n")))

        # source + jacobians (n_eq rows; jacobian (n_eq, n_state)).
        # REQ-185: argnames READ from the declared Function.args registry (time +
        # position X; dt rides in the p-slot so ``_registry_argnames`` drops it).
        # Only on the standard hyperbolic Model.H (self._t185); Chorin sub-models
        # keep the 3-arg source (their driver threads no cell time/position).
        sexpr, _ = self._expr_source()
        src_args = (self._registry_argnames("source") if self._t185
                    else ("Q", "Qaux", "p"))
        blocks.append(self._kernel("source", sexpr, (ne, 1), src_args))
        jexpr, _ = self._expr_source_jac()
        blocks.append(self._kernel("source_jacobian_wrt_variables", jexpr,
                                   (ne * ns, 1), ("Q", "Qaux", "p")))
        jauxexpr, _ = self._expr_source_jac_aux()
        blocks.append(self._kernel("source_jacobian_wrt_aux_variables",
                                   jauxexpr, (ne * na, 1), ("Q", "Qaux", "p")))

        # 3d projection / misc (stubs; ZoomyAmr writes var0..varN directly)
        blocks.append(self._kernel("project_2d_to_3d", self._zeros(6), (6, 1),
                                   ("X", "Q", "Qaux", "p")))
        blocks.append(self._kernel("project_3d_to_2d", self._zeros(ns), (ns, 1),
                                   ("Q", "Qaux", "p")))
        blocks.append(self._kernel("residual", self._zeros(ns), (ns, 1),
                                   ("time", "X", "dX", "Q", "Qaux", "p")))
        blocks.append(self._kernel("interpolate", self._vec(list(sm.state)),
                                   (ns, 1), ("Q", "Qaux", "p")))
        blocks.append(self._kernel("initial_condition", self._zeros(ns), (ns, 1),
                                   ("X", "p")))
        blocks.append(self._kernel("initial_aux_condition", self._zeros(na),
                                   (na, 1), ("X", "p")))

        # state_update — the Chorin corrector operator (rank-1, len = #corrected
        # modes).  Only present on a corrector sub-model (SM_corr from
        # chorin_split); emitted when the slot is populated.  dt rides in p(last).
        su = getattr(sm, "state_update", None)
        if su is not None:
            su_list = list(su)
            blocks.append(self._kernel("state_update", self._vec(su_list),
                                       (len(su_list), 1), ("Q", "Qaux", "p")))

        # update variables / aux + jacobians.  Emit the MODEL's real
        # update_variables (state hygiene: e.g. MalpassetSWE's |u| cap + dry-
        # momentum zeroing, or the Chorin corrector's pressure impulse), falling
        # back to identity only when the model has none.  Previously this was
        # hardcoded to identity, which silently dropped both.
        uv = getattr(sm, "update_variables", None)
        # update_variables may be square (n_state rows, e.g. SWE/SME state
        # hygiene) or RECTANGULAR (n_eq rows, e.g. the Chorin corrector which
        # only rewrites its e2s rows); emit it at its own length and let the
        # driver scatter via e2s. Identity fallback when the model has none.
        if uv is not None:
            n_uv = int(uv.shape[0])
            uv_expr = self._vec([uv[i, 0] for i in range(n_uv)])
        else:
            n_uv = ns
            uv_expr = self._vec(list(sm.state))
        blocks.append(self._kernel("update_variables", uv_expr,
                                   (n_uv, 1), ("Q", "Qaux", "p")))
        uaexpr, _ = self._expr_update_aux()
        # REQ-185: the algebraic aux closure (+ its jacobian) gain time +
        # position X from the registry (no dt); standard hyperbolic Model.H only.
        ua_args = (self._registry_argnames("update_aux_variables") if self._t185
                   else ("Q", "Qaux", "p"))
        blocks.append(self._kernel("update_aux_variables", uaexpr, (na, 1),
                                   ua_args))
        blocks.append(self._kernel("update_variables_jacobian_wrt_variables",
                                   self._identity_flat(ns), (ns * ns, 1),
                                   ("Q", "Qaux", "p")))
        ua_jac_args = (self._registry_argnames(
            "update_aux_variables_jacobian_wrt_variables")
            if self._t185 else ("Q", "Qaux", "p"))
        blocks.append(self._kernel("update_aux_variables_jacobian_wrt_variables",
                                   self._zeros(na * ns), (na * ns, 1),
                                   ua_jac_args))

        # boundary conditions — the real per-tag dispatch: ``sm.boundary_conditions``
        # is a framework ``Function`` whose ``.definition`` is a Piecewise over
        # ``bc_idx`` (the sorted-tag index) with a ghost-state vector per branch.
        # ``convert_expression_body`` lowers the Piecewise to an if/else-if chain;
        # ``bc_idx`` prints as itself (matches the C++ arg), Q/Qaux/n via the maps.
        bc_def = self._bc_definition("boundary_conditions")
        bc_expr = bc_def if bc_def is not None else self._vec(list(sm.state))
        blocks.append(self._kernel("boundary_conditions", bc_expr, (ns, 1),
                                   ("bc_idx", "Q", "Qaux", "n", "X", "time", "dX")))
        # aux BCs are extrapolation (ghost aux = interior) for SWE/SME/VAM — and
        # the aux at ghosts are recomputed (FD gradients / closure) downstream
        # anyway.  Emit identity at the full aux count: the source BC definition
        # may cover only a subset (a Chorin split adds pressure-gradient aux the
        # original aux-BC kernel never knew about), so always size it to n_aux.
        abc_expr = self._vec(list(sm.aux_state)) if na else self._zeros(0)
        blocks.append(self._kernel("aux_boundary_conditions", abc_expr, (na, 1),
                                   ("bc_idx", "Q", "Qaux", "n", "X", "time", "dX")))

        # Chorin pressure operator (REQ-94): explicit affine coefficients of the
        # pressure system  A·P = RHS,  A = A0 + Ax∂x + Ay∂y + Axx∂xx + Ayy∂yy.
        # Emitted ONLY on the pressure sub-model (SM_press carries pressure_operator);
        # lets the C++ driver assemble MLABecLaplacian (β from Axx/Ayy) + a block
        # smoother (A0) analytically instead of probing the matrix-free residual.
        po = getattr(sm, "pressure_operator", None)
        if po is not None:
            try:
                op = po()
            except Exception:
                op = None
            if op is not None:
                blocks.extend(self._pressure_op_kernels(op))

        blocks.append(self._chorin_meta())
        blocks.append("};\n")
        return "\n".join(blocks)

    def _pressure_op_kernels(self, op):
        """Print `pressure_{A0,Axx,Ayy,Ax,Ay}`, `pressure_rhs` and `n_press` of operator
        `op`.
        """
        import sympy as sp
        NP = len(op.P_modes)
        def flat(M):
            """Row-major flattening of the `NP x NP` coefficient `M` (zeros if None)."""
            if M is None: return [sp.Integer(0)] * (NP * NP)
            return [sp.sympify(M[k][l]) for k in range(NP) for l in range(NP)]
        out = []
        for name, M in (("pressure_A0", op.A0), ("pressure_Axx", op.Axx),
                        ("pressure_Ayy", op.Ayy), ("pressure_Ax", op.Ax),
                        ("pressure_Ay", op.Ay)):
            out.append(self._kernel(name, self._vec(flat(M)), (NP * NP, 1),
                                    ("Q", "Qaux", "p")))
        rhs = [sp.sympify(op.RHS[k]) for k in range(NP)]
        out.append(self._kernel("pressure_rhs", self._vec(rhs), (NP, 1),
                                ("Q", "Qaux", "p")))
        out.append(f"    static constexpr int n_press = {NP};")
        return out

    def _chorin_meta(self):
        """Chorin sub-model metadata: evolved rows `e2s` and aux tables `deriv_aux`,
        `input_aux`.
        """
        sm = self.sm
        e2s = list(getattr(sm, "equation_to_state_index", range(self.n_eq)))
        def _rows(reg):
            """Table rows `(aux_row, state_index, x_order, y_order)` of an aux registry."""
            # (aux_row, state_index, x_order, y_order): the full spatial multi-
            # index so the driver can FD-fill x- and y-derivatives generically
            # (1-horizontal models carry a 1-tuple multi_index -> y_order 0).
            out = []
            for e in (reg or []):
                mi = e.get("multi_index", (0,))
                xo = int(mi[0]) if len(mi) > 0 else 0
                yo = int(mi[1]) if len(mi) > 1 else 0
                out.append((e["row"], e["state_index"], xo, yo))
            return out
        live = _rows(getattr(sm, "aux_registry", None))
        frozen = _rows(getattr(sm, "aux_input_registry", None))
        def _arr(name, rows):
            """C++ table `n_<name>` / `<name>` for `rows`."""
            if not rows:
                return (f"    static constexpr int n_{name} = 0;\n"
                        f"    static constexpr int {name}[1][4] = {{{{0,0,0,0}}}};")
            body = ", ".join(f"{{{r},{s},{xo},{yo}}}" for r, s, xo, yo in rows)
            return (f"    static constexpr int n_{name} = {len(rows)};\n"
                    f"    static constexpr int {name}[{len(rows)}][4] = {{{body}}};")
        L = [
            f"    static constexpr int n_e2s = {len(e2s)};",
            f"    static constexpr int e2s[{max(len(e2s),1)}] = {{ {', '.join(map(str, e2s)) or '0'} }};",
            _arr("deriv_aux", live),
            _arr("input_aux", frozen),
        ]
        return "\n".join(L)


# ── user-functions helper header (opaque kernel bodies) ─────────────────────
# These are HAND-WRITTEN C++, not generated code, and they now live where C++
# belongs: ``Source/UserFunctions.H`` (the 4 kernel entry points) and
# ``Source/ZoomyEig.H`` (the device eigensolver they call).  They used to be a
# Python string literal here, which meant the checked-in Source/UserFunctions.H
# was dead — silently overwritten by this constant on every run.  A real
# implementation was landed in the .H and had no effect at all until this
# indirection was removed, so: ONE source of truth, and it is the .H file.
from pathlib import Path as _Path

_USER_FN_SOURCES = ("UserFunctions.H", "ZoomyEig.H")
_SOURCE_DIR = _Path(__file__).resolve().parents[2] / "Source"


def _stage_user_functions(out, write):
    """Copy the hand-written kernel headers into `out` through the write-if-changed helper
    `write`.
    """
    for name in _USER_FN_SOURCES:
        write(out / name, (_SOURCE_DIR / name).read_text())


def generate_headers(sm, out_dir, *, riemann=None, analytical_eigenvalues=True):
    """Generate `Model.H`, `Numerics.H` and the user-function headers into `out_dir` from a
    frozen SystemModel.  Files are rewritten only when their text changes, so `make` does not
    recompile.

    Args:
        sm: frozen SystemModel.
        riemann: `zoomy_core.fvm.riemann_solvers.Numerics` subclass; default
            `NonconservativeRusanov`.
    Returns:
        `out_dir` as a Path.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if riemann is None:
        from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
        riemann = NonconservativeRusanov
    num = riemann(model=sm)

    # WRITE-IF-DIFFERENT.  These headers are regenerated on every run_case call,
    # and an unconditional write_text() refreshes the mtime even when the emitted
    # code is byte-identical -- which makes `make` rebuild every translation unit
    # that includes Model.H/Numerics.H, i.e. the whole driver.  On the CUDA build
    # that is a full nvcc/ptxas recompile (~2 min) on EVERY run, dwarfing a solve
    # that takes 2 s (60 s sim) to 26 s (600 s sim).  Skipping the write when the
    # content is unchanged lets make no-op, so a re-run is just the solve.
    def _write_if_changed(path, code):
        """Write `code` to `path` unless identical (keeps the mtime, so `make` does not
        rebuild).
        """
        try:
            if path.exists() and path.read_text() == code:
                return
        except OSError:
            pass
        path.write_text(code)

    _stage_user_functions(out, _write_if_changed)
    _write_if_changed(out / "Model.H",
        AmrexSystemModelPrinter(sm, analytical_eigenvalues=analytical_eigenvalues,
                                time_position_ops=True)   # REQ-185 (hyperbolic path)
        .create_code())
    _write_if_changed(out / "Numerics.H", AmrexNumericsPrinter(num).create_code())
    return out


def write_chorin_headers(split, out_dir):
    """Generate `ModelPred.H` / `ModelPress.H` / `ModelCorr.H`, `NumericsPred.H` and the
    user-function headers for the C++ Chorin driver (predictor, pressure solve, corrector).  An
    unresolved boundary trace in the emitted C++ raises `ValueError`.

    Args:
        split: the model's own `chorin_split(system_model=sm)` result (`SM_pred`, `SM_press`,
            `SM_corr`).
    Returns:
        `out_dir` as a Path.
    References:
        Chorin 1968, https://doi.org/10.1090/S0025-5718-1968-0242392-2
    """
    from pathlib import Path
    from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    def _write_if_changed(path, code):
        """Write `code` to `path` unless identical (keeps the mtime, so `make` does not
        rebuild).
        """
        try:
            if path.exists() and path.read_text() == code:
                return
        except OSError:
            pass
        path.write_text(code)

    _stage_user_functions(out, _write_if_changed)
    for sub, name in [(split.SM_pred, "ModelPred"),
                      (split.SM_press, "ModelPress"),
                      (split.SM_corr, "ModelCorr")]:
        # REQ-140 defense-in-depth: the splitter resolves bed/surface-trace
        # ``Subs(f(ζ), ζ, 0|1)`` nodes at construction (core `e12393b`), but
        # the collapse is environment-sensitive (sympy cache/hashseed governs
        # birth-collapse), so re-resolve HERE at the printer boundary and then
        # assert the emitted C++ is clean — an unresolved ``Subs``/``zeta``
        # prints as an undeclared symbol and the whole case fails to build.
        from zoomy_core.model.splitter import _resolve_subs
        for _attr in ("flux", "nonconservative_matrix", "source",
                      "source_explicit", "boundary_source", "mass_matrix",
                      "eigenvalues", "update_variables",
                      "update_aux_variables"):
            _arr = getattr(sub, _attr, None)
            if _arr is not None:
                try:
                    setattr(sub, _attr, _resolve_subs(_arr))
                except Exception:
                    # best-effort: some containers (ZArray) reject applyfunc
                    # reconstruction / frozen attrs reject set.  The post-emit
                    # assertion below is the authoritative guard.
                    pass
        code = AmrexSystemModelPrinter(sub, analytical_eigenvalues=False,
                                       wrapper_name=name).create_code()
        if "Subs(" in code or "zeta" in code:
            bad = [ln.strip() for ln in code.splitlines()
                   if "Subs(" in ln or "zeta" in ln][:5]
            raise ValueError(
                f"REQ-140: unresolved boundary trace leaked into {name}.H — "
                f"the emitted C++ contains Subs/zeta and will not compile. "
                f"Offending lines: {bad}. Fix the trace at its birthplace "
                "(model derivation) or extend splitter._resolve_subs coverage.")
        (out / f"{name}.H").write_text(code)
    # predictor flux: Audusse PositiveNonconservativeRusanov (hydrostatic
    # reconstruction = well-balanced over the bed), matching the numpy reference
    # ChorinSplitVAMSolver.  Plain Rusanov is NOT WB -> spurious pressure spike +
    # surface hump at the steepest bed slope (the bump crest).
    npred = PositiveNonconservativeRusanov(model=split.SM_pred)
    prn = AmrexNumericsPrinter(npred)
    prn._wrapper_name = "NumericsPred"
    (out / "NumericsPred.H").write_text(
        prn.create_code().replace('#include "Model.H"', '#include "ModelPred.H"'))
    return out
