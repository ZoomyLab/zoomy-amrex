"""Device kernels of the symbolic solvers on AMReX: one `struct <system>_t` per system with a
`static constexpr` functor per kernel the printed solver references (`nsm.source` is
`nsm_t::source`).  The kernels are the system's `call` table (the functions JAX lambdifies and
foam prints) in AMReX syntax, `AMREX_GPU_HOST_DEVICE` operators on `amrex::SmallMatrix`.
"""
import re

import numpy as np
import sympy as sp

from zoomy_core.transformation.generic_c import GenericCppBase, get_nested_shape
from zoomy_core.transformation.to_amrex import AmrexCore
from zoomy_core.symsolver.discretization.residual import check_derivative_aux
from zoomy_amrex.transformation.to_amrex import AmrexNumericsPrinter

FAMILY = {"face": 0, "bc": 1, "normal": 2, "recon": 3, "cell": 4}
_KEYS = {           # family -> its full argument list (group keys)
    "face": ("q_minus", "q_plus", "aux_minus", "aux_plus", "p", "normal"),
    "bc": ("idx", "time", "position", "distance", "variables",
           "aux_variables", "parameters", "normal"),
    "normal": ("variables", "aux_variables", "parameters", "normal"),
    "recon": ("reconstruction", "aux_variables", "parameters"),
    "cell": ("variables", "aux_variables", "parameters", "time", "dt",
             "position"),
}
_NAME = {"variables": "Q", "aux_variables": "Qaux", "parameters": "p",
         "p": "p", "normal": "n", "position": "X", "time": "time", "dt": "dt",
         "distance": "dX", "idx": "bc_idx", "reconstruction": "W",
         "boundary_data": "Q",
         "q_minus": "Q_minus", "q_plus": "Q_plus", "aux_minus": "Qaux_minus",
         "aux_plus": "Qaux_plus", "Q": "Q", "Qaux": "Qaux", "n": "n"}
_SCALAR = {"time", "dt", "distance"}


def family(f):
    """Kernel family of the `Function` `f` (face, bc, recon, normal or cell), from its argument
    groups.
    """
    keys = set(f.args.keys())
    if "q_minus" in keys:
        return "face"
    if "idx" in keys:
        return "bc"
    if "reconstruction" in keys:
        return "recon"
    if "normal" in keys or "n" in keys:
        return "normal"
    return "cell"


def _m(n):
    """`amrex::SmallMatrix` column type of `n` entries."""
    return f"amrex::SmallMatrix<amrex::Real, {max(int(n), 1)}, 1>"


class AmrexKernelPrinter(AmrexCore, GenericCppBase):
    """AMReX syntax of the kernel bodies (`AmrexCore` plus the opaque calls of
    `AmrexNumericsPrinter`), with symbol maps from each kernel's own argument groups.
    """
    c_functions = AmrexNumericsPrinter.c_functions

    def kernel(self, f, sizes):
        """Print `struct <name>_k` and its constexpr instance for the `Function` `f`.  Every
        kernel of a family takes the family's full argument list and ignores what it does not
        declare.

        Args:
            sizes: entries per argument group of the system.
        Returns:
            The C++ text.
        """
        self.symbol_maps = []
        fam = family(f)
        keys = _KEYS[fam]
        if fam == "normal" and "n" in f.args.keys():
            keys = ("Q", "Qaux", "p", "n")
        for key, group in f.args.items():
            name = _NAME[key]
            if isinstance(group, sp.Symbol):
                self.symbol_maps.append({group: name})
            else:
                vals = list(group.values()) if hasattr(group, "values") else list(group)
                self.register_map(name, vals)
        expr = f.definition
        if getattr(expr, "shape", None) is None:          # piecewise BC
            shape = (sizes["n_state"],)
        else:
            expr = self._expand_vector_conditionals(expr)
            shape, expr = get_nested_shape(expr)
        n_out = int(np.prod(shape))
        body = self.convert_expression_body(expr, shape) if n_out else (
            f"    {_m(1)} res{{}};\n    return res;")
        sig = []
        for k in keys:
            nm = _NAME[k]
            if k == "idx":
                sig.append("int bc_idx")
            elif k in _SCALAR:
                sig.append(f"amrex::Real {nm}")
            else:
                sig.append(f"{_m(sizes[k])} const& {nm}")
        n_in = sizes["n_state"] if fam != "recon" else sizes["reconstruction"]
        return (f"    struct {f.name}_k\n    {{\n"
                f"        static constexpr int family = {FAMILY[fam]}, "
                f"n_out = {max(n_out, 1)}, n_in = {n_in}, "
                f"n_aux = {max(sizes['aux_variables'], 1)}, "
                f"n_nrm = {max(sizes['normal'], 1)};\n"
                f"        AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE {_m(n_out)} "
                f"operator()({', '.join(sig)}) const noexcept\n"
                f"        {{\n{body}\n        }}\n    }};\n"
                f"    static constexpr {f.name}_k {f.name}{{}};\n")


def referenced(source, system):
    """Kernel names `<system>.<name>` used by the printed solver."""
    return sorted(set(re.findall(rf"\b{system}\.(\w+)", source)))


def write_kernels(systems, source, path, dimension, classes=None):
    """Write `Kernels.H` to `path`: `struct <name>_t` and its constexpr instance for each
    `(name, nsm)` in `systems`, with the kernels `source` uses.

    Args:
        classes: `{(name, "state"): closure class per state row}` (the topology's
            `lsq_closure_class`), the 6th int of each `aux_derivatives` entry; -1 without a
            closure.
    Returns:
        `path`.
    """
    # the derivative aux of a state row only: anything else would read 0
    check_derivative_aux([nsm for _, nsm in systems], ("state",), "amrex")
    pr = AmrexKernelPrinter()
    first = systems[0][1]
    n_par = len(first.parameters.keys())
    out = ["#pragma once", "#include <array>", "#include <AMReX_REAL.H>",
           "#include <AMReX_SmallMatrix.H>", '#include "UserFunctions.H"', "",
           "namespace zs", "{",
           f"constexpr int DIM_MODEL = {int(dimension)};",
           f"constexpr int NPAR = {max(n_par, 1)};", ""]
    for name, nsm in systems:
        call, n_state, n_aux = nsm.call, len(nsm.state), len(nsm.aux_state)
        if len(nsm.parameters.keys()) != n_par:
            raise ValueError(f"{name}: {len(nsm.parameters.keys())} parameters, "
                             f"the first system {n_par}: one p for all systems")
        tags = list(getattr(getattr(nsm, "_bc_source", None),
                            "list_sorted_function_names", []) or [])
        pv = nsm.parameter_values
        params = ", ".join(repr(float(getattr(pv, str(k)))) for k in nsm.parameters.keys())
        n_normal = len(list(nsm.normal.values())) if hasattr(nsm.normal, "values") else len(nsm.normal)
        sizes = {"n_state": n_state, "variables": n_state, "Q": n_state,
                 "q_minus": n_state, "q_plus": n_state,
                 "aux_variables": n_aux, "Qaux": n_aux, "aux_minus": n_aux,
                 "aux_plus": n_aux, "parameters": n_par, "p": n_par,
                 "normal": n_normal, "n": n_normal, "position": 3,
                 "reconstruction": n_state}
        out += [f"struct {name}_t", "{",
                f"    static constexpr int n_state = {n_state};",
                f"    static constexpr int n_aux = {n_aux};",
                f"    static constexpr int n_normal = {n_normal};",
                f"    static std::array<double, {max(n_par, 1)}> parameters() "
                f"{{ return {{{params or '0.0'}}}; }}",
                f"    static constexpr std::array<const char*, {len(tags)}> bc_tags"
                f" = {{{', '.join(chr(34) + t + chr(34) for t in tags)}}};"]
        for k in referenced(source, name):
            if k == "aux_derivatives":
                cls = (classes or {}).get((name, "state"), ())
                rows = [(e["row"], e["state_index"], *_axes(e["multi_index"]),
                         cls[e["state_index"]] if cls else -1)
                        for e in nsm.aux_derivatives
                        if e["kind"] == "derivative" and e["target_kind"] == "state"]
                items = ", ".join("{" + ", ".join(map(str, r)) + "}" for r in rows)
                out.append(f"    static constexpr std::array<std::array<int, 6>, "
                           f"{len(rows)}> aux_derivatives = {{{{{items}}}}};")
            elif call[k] is None:
                raise ValueError(f"{name}.{k} is used but empty")
            else:
                out.append(pr.kernel(call[k].functions[k], sizes))
        out += ["};", f"inline constexpr {name}_t {name}{{}};", ""]
    out += [f"using System = {systems[0][0]}_t;",
            f"inline constexpr const System& sys = {systems[0][0]};",
            "} // namespace zs", ""]
    path.write_text("\n".join(out))
    return path


def _axes(multi_index):
    """(d/dx, d/dy, d/dz) orders from the registry multi-index."""
    m = list(multi_index) + [0, 0, 0]
    return m[0], m[1], m[2]
