from pathlib import Path

import pytest

from cm1analysis.config import ConfigError, load_config

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "example.yaml"


def write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    return path


def test_example_config_loads(monkeypatch):
    monkeypatch.setenv("CM1_DATA", "/data/cm1")
    monkeypatch.setenv("CM1_IMAGES", "/images")
    cfg = load_config(EXAMPLE)
    assert cfg.output_dir == Path("/images")
    control = cfg.experiment("control")["control"]
    assert control.path == Path("/data/cm1/control")
    assert control.color == "k"
    assert all(run.path.is_absolute() for runs in cfg.experiments.values() for run in runs.values())


def test_defaults_and_relative_paths(tmp_path):
    cfg = load_config(write(tmp_path, "output_dir: images\nexperiments:\n  g:\n    r1: {path: runs/r1}\n"))
    assert cfg.output_dir == tmp_path / "images"
    run = cfg.experiment("g")["r1"]
    assert run.path == tmp_path / "runs" / "r1"
    assert (run.color, run.marker, run.display_name) == ("k", "o", "r1")
    assert cfg.file_pattern == "cm1out_*.nc"
    assert cfg.plot_options("planar") == {}


def test_plot_options(tmp_path):
    cfg = load_config(write(tmp_path, "output_dir: out\nplots:\n  cross_section: {window_km: 100}\n"))
    assert cfg.plot_options("cross_section") == {"window_km": 100}


@pytest.mark.parametrize(
    "text, match",
    [
        ("experiments: {}\n", "output_dir"),
        ("output_dir: out\noutptu_dir: x\n", "outptu_dir"),
        ("output_dir: out\nfonts: {titel: 12}\n", "titel"),
        ("output_dir: out\nplots: {maps: {}}\n", "maps"),
        ("output_dir: out\nexperiments: {g: {r: {color: red}}}\n", "needs a 'path'"),
        ("output_dir: out\nexperiments: {g: {r: {path: x, colour: red}}}\n", "colour"),
        ("output_dir: $CM1_SURELY_UNSET_VAR/out\n", "unset environment variable"),
    ],
)
def test_invalid_configs(tmp_path, text, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write(tmp_path, text))


def test_unknown_experiment(tmp_path):
    cfg = load_config(write(tmp_path, "output_dir: out\n"))
    with pytest.raises(ConfigError, match="no experiment"):
        cfg.experiment("lapse_rate")
