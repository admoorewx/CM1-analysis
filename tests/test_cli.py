from conftest import NX, NY, NZ

from cm1analysis import cli


def test_inventory(cm1_path, capsys, clean_registry):
    clean_registry.register("refl", "planar", requires=["cref"])(lambda: None)
    clean_registry.register("ppert", "cross_section", requires=["prspert"])(lambda: None)

    assert cli.main(["inventory", str(cm1_path), "--fields"]) == 0
    out = capsys.readouterr().out
    assert f"{NZ} x {NY} x {NX}" in out
    assert "dx=500 m" in out
    assert "hydrometeors: qr" in out
    assert "ok    planar/refl" in out
    assert "skip  cross_section/ppert: needs prspert" in out


def test_inventory_missing_file(tmp_path, capsys):
    assert cli.main(["inventory", str(tmp_path / "nope.nc")]) == 1
    assert "cm1plot: error" in capsys.readouterr().err


def test_list(capsys, clean_registry):
    assert cli.main(["list"]) == 0
    assert "no plots registered" in capsys.readouterr().out

    clean_registry.register("swath", "planar", requires=[("u10", "sws")], description="Wind swath")(
        lambda: None
    )
    assert cli.main(["list", "--category", "planar"]) == 0
    assert "planar/swath: Wind swath [needs: u10 or sws]" in capsys.readouterr().out
