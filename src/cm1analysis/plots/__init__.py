"""Plot modules, one per category.

Importing this package registers every plot with :mod:`cm1analysis.registry`.
Add each new module to the imports below when it is ported.
"""

from . import (
    cross_section,  # noqa: F401
    planar,  # noqa: F401
    timeseries,  # noqa: F401
)
