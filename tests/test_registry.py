import pytest
from conftest import write_cm1_file

from cm1analysis.io import CM1File
from cm1analysis.registry import Category


def test_register_and_lookup(clean_registry):
    @clean_registry.register("cref", Category.PLANAR, requires=["cref"])
    def plot_cref():
        """Composite reflectivity."""

    spec = clean_registry.get("cref")
    assert spec.func is plot_cref
    assert spec.category is Category.PLANAR
    assert spec.description == "Composite reflectivity."
    with pytest.raises(KeyError, match="known plots"):
        clean_registry.get("nope")


def test_duplicate_name_rejected(clean_registry):
    clean_registry.register("a", "planar")(lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        clean_registry.register("a", "timeseries")(lambda: None)


def test_unknown_category_rejected(clean_registry):
    with pytest.raises(ValueError):
        clean_registry.register("a", "maps")(lambda: None)


def test_check_splits_by_available_fields(clean_registry, tmp_path):
    reg = clean_registry
    reg.register("refl", "planar", requires=["cref"])(lambda: None)
    reg.register("wind", "planar", requires=[("u10", "sws")])(lambda: None)
    reg.register("pressure", "cross_section", requires=["prspert", ("dbz", "cref")])(lambda: None)

    path = write_cm1_file(tmp_path / "f.nc", fields=("cref", "sws"))
    with CM1File(path) as cm1:
        runnable, skipped = reg.check(cm1)
        assert [s.name for s in runnable] == ["refl", "wind"]
        assert skipped == {"pressure": ["prspert"]}

        runnable, skipped = reg.check(cm1, category="cross_section")
        assert runnable == [] and list(skipped) == ["pressure"]


def test_unmet_alternatives_are_reported_together(clean_registry, tmp_path):
    clean_registry.register("w", "planar", requires=[("winterp", "w")])(lambda: None)
    path = write_cm1_file(tmp_path / "f.nc", fields=("cref",))
    with CM1File(path) as cm1:
        _, skipped = clean_registry.check(cm1)
    assert skipped == {"w": ["winterp or w"]}


def test_plots_sorted_by_category_then_name(clean_registry):
    for name, cat in (("z", "planar"), ("b", "wind"), ("a", "planar"), ("m", "timeseries")):
        clean_registry.register(name, cat)(lambda: None)
    assert [s.name for s in clean_registry.plots()] == ["a", "z", "m", "b"]
    assert [s.name for s in clean_registry.plots("planar")] == ["a", "z"]
