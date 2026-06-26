# AMREX_HOME defines the directory in which we will find all the AMReX code.
# Inside the zoomy_amrex container AMReX lives at /opt/amrex.
AMREX_HOME ?= $(if $(ZOOMY_AMREX_HOME),$(ZOOMY_AMREX_HOME),/opt/amrex)

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 3

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source
# Generated Model.H / Numerics.H / UserFunctions.H now live in ../Source
# (refreshed by ``build.py``); the external $(ZOOMY_AMREX_MODEL) include is
# only needed when generating headers out-of-tree.
INCLUDE_LOCATIONS += $(ZOOMY_AMREX_MODEL)

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/MLMG/Make.package
INCLUDE_LOCATIONS += $(AMREX_HOME)/Src/LinearSolvers

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
