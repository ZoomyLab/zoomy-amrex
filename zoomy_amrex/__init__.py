from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zoomy_amrex")
except PackageNotFoundError:
    # Package not installed, e.g. running from source
    __version__ = "0.0.0"


def __getattr__(name):
    """`zoomy_amrex.run_case` (the legacy structured-grid entry, REQ-89) on first use only, so
    importing the symbolic driver (`zoomy_amrex.symsolver`) never loads the legacy solvers."""
    if name == "run_case":
        from zoomy_amrex.run_case import run_case
        globals()["run_case"] = run_case     # over the submodule the import bound
        return run_case
    raise AttributeError(f"module 'zoomy_amrex' has no attribute {name!r}")
