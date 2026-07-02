from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zoomy_amrex")
except PackageNotFoundError:
    # Package not installed, e.g. running from source
    __version__ = "0.0.0"

from zoomy_amrex.run_case import run_case  # noqa: F401,E402  (server entry point, REQ-89)
