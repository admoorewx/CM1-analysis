"""Registry of available plots and the CM1 fields each one needs.

Every plot registers itself with the fields it reads, so the pipeline can check
a file once and run whatever that file supports, instead of failing partway
through on a field the output namelist did not write::

    @register("cref", Category.PLANAR, requires=["cref"])
    def plot_cref(...): ...

A requirement is either a field name or a tuple of alternatives, any one of
which is enough, e.g. ``("u10", "uinterp")`` for a low-level wind.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .io import CM1File

Requirement = str | tuple[str, ...]


class Category(StrEnum):
    PLANAR = "planar"
    TIMESERIES = "timeseries"
    STATISTICS = "statistics"
    CROSS_SECTION = "cross_section"
    PRESSURE = "pressure"
    WIND = "wind"


@dataclass(frozen=True)
class PlotSpec:
    name: str
    category: Category
    func: Callable
    requires: tuple[Requirement, ...] = ()
    description: str = ""

    def unmet(self, cm1: CM1File) -> list[str]:
        """Requirements the file does not satisfy, written as 'a' or 'a or b'."""
        unmet = []
        for req in self.requires:
            options = (req,) if isinstance(req, str) else req
            if not any(cm1.has(name) for name in options):
                unmet.append(" or ".join(options))
        return unmet


_REGISTRY: dict[str, PlotSpec] = {}


def register(
    name: str,
    category: Category | str,
    requires: Sequence[Requirement] = (),
    description: str = "",
) -> Callable[[Callable], Callable]:
    """Decorator that adds a plot function to the registry under a unique name."""

    def decorator(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"plot {name!r} is already registered")
        _REGISTRY[name] = PlotSpec(
            name=name,
            category=Category(category),
            func=func,
            requires=tuple(r if isinstance(r, str) else tuple(r) for r in requires),
            description=description or (func.__doc__ or "").strip().split("\n")[0],
        )
        return func

    return decorator


def get(name: str) -> PlotSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no plot named {name!r}; known plots: {sorted(_REGISTRY)}") from None


def plots(category: Category | str | None = None) -> list[PlotSpec]:
    """Registered plots, optionally in one category, sorted by category then name."""
    selected = _REGISTRY.values()
    if category is not None:
        selected = [spec for spec in selected if spec.category == Category(category)]
    return sorted(selected, key=lambda s: (list(Category).index(s.category), s.name))


def check(
    cm1: CM1File, category: Category | str | None = None
) -> tuple[list[PlotSpec], dict[str, list[str]]]:
    """Split registered plots into those the file supports and those it cannot (with reasons)."""
    runnable, skipped = [], {}
    for spec in plots(category):
        unmet = spec.unmet(cm1)
        if unmet:
            skipped[spec.name] = unmet
        else:
            runnable.append(spec)
    return runnable, skipped
