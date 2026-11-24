"""Thread counts for scipy.fft that respect the job's CPU allocation.

scipy.fft reads a negative ``workers`` against ``os.cpu_count()``, which on a
shared HPC node is every core of the node, not the cores the job was given.
Inside a SLURM allocation that oversubscribes the job's cores and slows the
transforms. :func:`fft_workers` counts negative values against the CPUs this
process may actually use instead.
"""

import os


def available_cpus() -> int:
    """CPUs this process may use: SLURM_CPUS_PER_TASK, else the CPU affinity mask, else the CPU count."""
    slurm = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def fft_workers(workers: int | None) -> int:
    """Resolve a scipy.fft ``workers`` value: -1 is every available CPU, -2 all but one, and so on."""
    if workers is None:
        return 1
    workers = int(workers)
    if workers == 0:
        raise ValueError("workers must not be zero")
    if workers > 0:
        return workers
    return max(1, available_cpus() + 1 + workers)
