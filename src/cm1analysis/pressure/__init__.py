"""Vertical momentum forcing: dynamic pressure-gradient force and effective buoyancy.

The vertical acceleration of CM1 output is split as

    Dw/Dt = -(1/rho0) dp'_d/dz + B + F,    B = b - (1/rho0) dp'_b/dz

where p'_d and p'_b solve Poisson problems forced by the advective
acceleration and by buoyancy (Xu et al. 2015a form, with the top and bottom
conditions of Klemp and Rotunno 1983). The same p'_d and p'_b optionally give
the horizontal forces HDPGF and HBPGF.

Modules:

- :mod:`.grid`     stretched vertical grid, z-face interpolation, index boxes and the ROI rule
- :mod:`.poisson`  transform-plus-tridiagonal Poisson solver (periodic, Neumann, Dirichlet sides)
- :mod:`.forcing`  base state, buoyancy, advective forcing and flux-form right-hand sides
- :mod:`.vforce`   one decomposition, the variants, splits and sensitivity tests
- :mod:`.hforce`   horizontal pressure-gradient forces
- :mod:`.source`   validated solver inputs read from a :class:`cm1analysis.io.CM1File`
- :mod:`.writer`   the ``wforce_*.nc`` output file
- :mod:`.compute`  :func:`compute_file`, the whole pipeline for one input file
"""

from .compute import ComputeOptions, compute, compute_file, output_path, stored_options
from .source import REQUIRED, SolverSource
from .writer import WForceWriter

__all__ = [
    "REQUIRED",
    "ComputeOptions",
    "SolverSource",
    "WForceWriter",
    "compute",
    "compute_file",
    "output_path",
    "stored_options",
]
