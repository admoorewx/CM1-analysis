"""Registry of available plots, the CM1 fields each needs, and its options.

Every plot registers itself with the fields it reads, so the pipeline can check
a file once and run whatever that file supports, instead of failing partway
through on a field the output namelist did not write::

    @register("cref", Category.PLANAR, requires=["cref"])
    def plot_cref(ctx): ...

A requirement is a raw field name, a derived quantity from
:data:`cm1analysis.diagnostics.RECIPES` (e.g. ``"low_level_wind"``), or a
tuple of alternatives, any one of which is enough.

Scope says what a plot function receives (see :mod:`cm1analysis.pipeline`):

- ``TIME``: one output time of one run; called once per time.
- ``RUN``: every file of one run, for plots that accumulate through time.
- ``EXPERIMENT``: several runs, for side-by-side comparisons.

``options`` holds the plot's adjustable settings and their defaults; the
config file may override them, and any other key is an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from . import diagnostics
from .io import CM1File

Requirement = str | tuple[str, ...]


class Category(StrEnum):
    PLANAR = "planar"
    TIMESERIES = "timeseries"
    STATISTICS = "statistics"
    CROSS_SECTION = "cross_section"
    PRESSURE = "pressure"
    WIND = "wind"


class Scope(StrEnum):
    TIME = "time"
    RUN = "run"
    EXPERIMENT = "experiment"


@dataclass(frozen=True)
class PlotSpec:
    name: str
    category: Category
    func: Callable
    requires: tuple[Requirement, ...] = ()
    description: str = ""
    scope: Scope = Scope.TIME
    options: Mapping[str, Any] = field(default_factory=dict)

    def unmet(self, cm1: CM1File) -> list[str]:
        """Requirements the file does not satisfy, written as 'a' or 'a or b'."""
        unmet = []
        for req in self.requires:
            options = (req,) if isinstance(req, str) else req
            if not any(satisfied(cm1, name) for name in options):
                unmet.append(" or ".join(_describe(name) for name in options))
        return unmet


def satisfied(cm1: CM1File, name: str) -> bool:
    """True if the file has the raw field, or can compute the derived quantity."""
    if name in diagnostics.RECIPES:
        return diagnostics.recipe(cm1, name) is not None
    return cm1.has(name)


def _describe(name: str) -> str:
    if name in diagnostics.RECIPES:
        return f"{name} ({' or '.join('+'.join(r) for r in diagnostics.RECIPES[name])})"
    return name


_REGISTRY: dict[str, PlotSpec] = {}
_COMMON_OPTIONS: dict[Category, dict[str, Any]] = {}


def register(
    name: str,
    category: Category | str,
    requires: Sequence[Requirement] = (),
    description: str = "",
    scope: Scope | str = Scope.TIME,
    options: Mapping[str, Any] | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator that adds a plot function to the registry under a unique name."""

    def decorator(func: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"plot {name!r} is already registered")
        cat = Category(category)
        clash = set(options or {}) & set(_COMMON_OPTIONS.get(cat, {}))
        if clash:
            raise ValueError(f"plot {name!r} redefines common {cat} options {sorted(clash)}")
        _REGISTRY[name] = PlotSpec(
            name=name,
            category=cat,
            func=func,
            requires=tuple(r if isinstance(r, str) else tuple(r) for r in requires),
            description=description or (func.__doc__ or "").strip().split("\n")[0],
            scope=Scope(scope),
            options=dict(options or {}),
        )
        return func

    return decorator


def common_options(category: Category | str, defaults: Mapping[str, Any]) -> None:
    """Declare options shared by every plot in a category (e.g. a plan-view extent)."""
    _COMMON_OPTIONS.setdefault(Category(category), {}).update(defaults)


def category_options(category: Category | str, category_config: Mapping[str, Any]) -> dict[str, Any]:
    """A category's common options: defaults, overridden by the config's category-wide keys.

    Lets one category reuse another's settings, e.g. timeseries of storm
    motion placed exactly as the cross sections are.
    """
    common = dict(_COMMON_OPTIONS.get(Category(category), {}))
    common.update({k: v for k, v in category_config.items() if k in common})
    return common


def resolve_options(spec: PlotSpec, category_config: Mapping[str, Any]) -> dict[str, Any]:
    """Merge defaults with the config's ``plots.<category>`` block for one plot.

    In that block, a key naming a plot holds that plot's options; any other
    key is a common option for the whole category. Unknown keys are errors.
    """
    common = dict(_COMMON_OPTIONS.get(spec.category, {}))
    plot_names = {s.name for s in plots(spec.category)}
    unknown = set(category_config) - plot_names - set(common)
    if unknown:
        raise ValueError(
            f"plots.{spec.category}: unknown key(s) {sorted(unknown)}; "
            f"common options: {sorted(common)}, plots: {sorted(plot_names)}"
        )
    merged = common | dict(spec.options)
    for key, value in category_config.items():
        if key in common:
            merged[key] = value
    own = category_config.get(spec.name) or {}
    bad = set(own) - set(spec.options) - set(common)
    if bad:
        raise ValueError(
            f"plots.{spec.category}.{spec.name}: unknown option(s) {sorted(bad)}; "
            f"allowed: {sorted(set(spec.options) | set(common))}"
        )
    merged.update(own)
    return merged


def get(name: str) -> PlotSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no plot named {name!r}; known plots: {sorted(_REGISTRY)}") from None


def plots(category: Category | str | None = None, scope: Scope | str | None = None) -> list[PlotSpec]:
    """Registered plots, optionally filtered, sorted by category then name."""
    selected = list(_REGISTRY.values())
    if category is not None:
        selected = [s for s in selected if s.category == Category(category)]
    if scope is not None:
        selected = [s for s in selected if s.scope == Scope(scope)]
    return sorted(selected, key=lambda s: (list(Category).index(s.category), s.name))


def check(
    cm1: CM1File,
    category: Category | str | None = None,
    specs: Sequence[PlotSpec] | None = None,
) -> tuple[list[PlotSpec], dict[str, list[str]]]:
    """Split plots into those the file supports and those it cannot (with reasons)."""
    runnable, skipped = [], {}
    for spec in plots(category) if specs is None else specs:
        unmet = spec.unmet(cm1)
        if unmet:
            skipped[spec.name] = unmet
        else:
            runnable.append(spec)
    return runnable, skipped
