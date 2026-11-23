"""Pipeline configuration from a YAML file.

Replaces the paths and experiment tables that were hard-coded in each script.
See ``config/example.yaml`` for every key. Paths may use ``~`` and environment
variables (``$SCRATCH``); relative paths are taken from the config file's
directory. Unknown keys are errors, so a typo cannot silently fall back to a
default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .io import DEFAULT_PATTERN
from .registry import Category
from .style import DEFAULT_DPI, FONT_DEFAULTS

_TOP_KEYS = {"output_dir", "file_pattern", "dpi", "fonts", "experiments", "plots", "reference"}
_RUN_KEYS = {"path", "color", "marker", "label"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Run:
    """One simulation: a directory of CM1 output plus how to draw it."""

    name: str
    path: Path
    color: str = "k"
    marker: str = "o"
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or self.name


@dataclass(frozen=True)
class Config:
    output_dir: Path
    file_pattern: str = DEFAULT_PATTERN
    dpi: int = DEFAULT_DPI
    fonts: dict[str, int] = field(default_factory=dict)
    experiments: dict[str, dict[str, Run]] = field(default_factory=dict)
    plots: dict[str, dict[str, Any]] = field(default_factory=dict)
    reference: str | None = None  # "group/run" compared against in experiment plots

    def experiment(self, name: str) -> dict[str, Run]:
        try:
            return self.experiments[name]
        except KeyError:
            raise ConfigError(f"no experiment {name!r}; defined: {sorted(self.experiments)}") from None

    def run(self, ref: str) -> Run:
        """Look up a run written as 'group/run'."""
        group, sep, name = ref.partition("/")
        if not sep:
            raise ConfigError(f"run reference {ref!r} must be written 'group/run'")
        runs = self.experiment(group)
        if name not in runs:
            raise ConfigError(f"experiment {group!r} has no run {name!r}; runs: {sorted(runs)}")
        return runs[name]

    def plot_options(self, category: Category | str) -> dict[str, Any]:
        """Options for one plot category (the `plots:` block), empty if none are set."""
        return dict(self.plots.get(str(Category(category)), {}))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    base = path.parent

    _reject_unknown(raw, _TOP_KEYS, f"{path}")
    if "output_dir" not in raw:
        raise ConfigError(f"{path}: 'output_dir' is required")

    fonts = raw.get("fonts") or {}
    _reject_unknown(fonts, set(FONT_DEFAULTS), f"{path}: fonts")

    plots = raw.get("plots") or {}
    _reject_unknown(plots, {str(c) for c in Category}, f"{path}: plots")

    experiments = {}
    for group, runs in (raw.get("experiments") or {}).items():
        if not isinstance(runs, dict) or not runs:
            raise ConfigError(f"{path}: experiment {group!r} must map run names to settings")
        experiments[group] = {}
        for name, spec in runs.items():
            where = f"{path}: experiments.{group}.{name}"
            if not isinstance(spec, dict) or "path" not in spec:
                raise ConfigError(f"{where} needs a 'path'")
            _reject_unknown(spec, _RUN_KEYS, where)
            experiments[group][name] = Run(
                name=name,
                path=_resolve(spec["path"], base),
                color=str(spec.get("color", "k")),
                marker=str(spec.get("marker", "o")),
                label=spec.get("label"),
            )

    config = Config(
        output_dir=_resolve(raw["output_dir"], base),
        file_pattern=str(raw.get("file_pattern", DEFAULT_PATTERN)),
        dpi=int(raw.get("dpi", DEFAULT_DPI)),
        fonts=dict(fonts),
        experiments=experiments,
        plots={k: dict(v or {}) for k, v in plots.items()},
        reference=raw.get("reference"),
    )
    if config.reference is not None:
        config.run(config.reference)  # fail now, not after the per-run plots
    return config


def _resolve(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if "$" in str(expanded):
        raise ConfigError(f"unset environment variable in path {value!r}")
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def _reject_unknown(mapping: dict, allowed: set[str], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{where} must be a mapping")
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
