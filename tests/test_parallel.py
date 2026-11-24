import os

import pytest

from cm1analysis.parallel import available_cpus, fft_workers


def test_slurm_allocation_wins(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    assert available_cpus() == 4
    assert fft_workers(-1) == 4
    assert fft_workers(-2) == 3


def test_without_slurm_uses_affinity(monkeypatch):
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    expected = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    assert fft_workers(-1) == expected


@pytest.mark.parametrize("value", ["", "abc", "0"])
def test_bad_slurm_value_is_ignored(monkeypatch, value):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)
    assert available_cpus() >= 1


def test_explicit_and_edge_values(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert fft_workers(3) == 3  # an explicit count is kept as given
    assert fft_workers(None) == 1
    assert fft_workers(-10) == 1  # never below one
    with pytest.raises(ValueError):
        fft_workers(0)
