"""A view written with fewer dimensions and as int8 (generation/export.py), carried by diffs and read as int8."""
import io
import json

import numpy as np
import pytest

from octantvector.common import dbformat
from octantvector.generation import GenerationError, calibration, config, dbrelease, export
from octantvector.generation.encoders import get
from octantvector.runtime.db import DBError, Int8Vectors, apply_diff, open_db, search

from test_ledger import cfg  # noqa: F401
from test_release import AT, CP, MT, VIEWS, change_world, next_day, same_content, world  # noqa: F401
from datetime import timedelta

SMALL = {"meta_tags": {"dim": 8, "dtype": "int8"}, "chart_patterns": {"dtype": "int8"}}


def small_release(tmp_path, cfg, days=1, **kw):
    return next_day(tmp_path, cfg, days=days, export=SMALL, **kw)


def test_a_view_is_written_with_the_export_dimensions_and_type(world, cfg):
    tmp_path, cal, pool, base = world
    out, diff = small_release(tmp_path, cfg)
    assert diff is None  # a new export is a new calibration: a full
    m = json.loads((out / "manifest.json").read_text())
    assert m["schemaVersion"] == dbformat.SCHEMA_VERSION
    mt, cp = m["views"][MT], m["views"][CP]
    assert (mt["dim"], mt["dtype"], mt["encoderDim"], len(mt["scale"])) == (8, "int8", get("meta_tags").dim, 8)
    assert (cp["dim"], cp["dtype"], len(cp["scale"])) == (get("chart_patterns").dim, "int8", cp["dim"])
    assert "encoderDim" not in cp
    assert np.load(out / mt["file"]).dtype == np.int8
    c = json.loads((out / m["calibrationFile"]).read_text())
    assert c["views"][MT]["export"]["basis"] and c["views"][CP]["export"]["basis"] is None
    assert export.basis_path(tmp_path / "store", c["views"][MT]["export"]["basis"]).exists()

    db = open_db(out)
    x, _ = db.views[MT]
    assert isinstance(x, Int8Vectors) and x.shape[1] == 8
    full = np.asarray(x)
    assert np.allclose(np.linalg.norm(full, axis=1), 1, atol=0.03)  # unit vectors, up to the rounding
    q = full[3]
    assert np.allclose(x @ q, full @ q, atol=1e-5) and np.allclose(x[[1, 2]] @ q, full[[1, 2]] @ q, atol=1e-5)
    assert np.allclose(x[3], q) and np.allclose(x[:, 2], full[:, 2])
    h = db.hashes[0]
    r = search(db, h, MT, k=3)["results"]
    assert len(r) == 3 and all(-1.05 <= x["score"] <= 1.05 for x in r)
    assert db.info()["views"][MT]["dim"] == 8


def test_the_export_stays_fixed_across_diffs(world, cfg):
    tmp_path, cal, pool, base = world
    first, _ = small_release(tmp_path, cfg)
    change_world(tmp_path)
    new, diff = small_release(tmp_path, cfg, days=2)
    assert diff is not None  # same export: the calibration is the same, so a diff
    applied = apply_diff(first, diff, tmp_path / "host" / new.name)
    same_content(dbformat.read_content(applied), dbformat.read_content(new))
    a, b = (json.loads((p / "manifest.json").read_text())["views"][MT]["scale"] for p in (first, new))
    assert a == b


def test_changing_or_dropping_the_export_is_a_full(world, cfg):
    tmp_path, cal, pool, base = world
    small_release(tmp_path, cfg)
    log = io.StringIO()
    out, diff = dbrelease.release(config.load(), VIEWS, built_at=AT + timedelta(days=2), log=log,
                                  export={"meta_tags": {"dim": 4}})
    assert diff is None and "with the export" in log.getvalue()
    m = json.loads((out / "manifest.json").read_text())
    assert (m["views"][MT]["dim"], m["views"][MT]["dtype"]) == (4, "float16") and "scale" not in m["views"][MT]
    assert m["views"][CP]["dtype"] == "float16"  # not named: as the encoder made it
    out, diff = dbrelease.release(config.load(), VIEWS, built_at=AT + timedelta(days=3), log=io.StringIO(), export={})
    assert diff is None and json.loads((out / "manifest.json").read_text())["views"][MT]["dim"] == get("meta_tags").dim
    _, diff = dbrelease.release(config.load(), VIEWS, built_at=AT + timedelta(days=4), log=io.StringIO())
    assert diff is not None  # None keeps what the calibration has


@pytest.mark.parametrize("spec, why", [({"dim": 0}, "dim must be"), ({"dim": 10 ** 4}, "dim must be"),
                                       ({"dim": 4.0}, "dim must be"), ({"dtype": "int4"}, "dtype"),
                                       ({"size": 4}, "give"), ([4], "give")])
def test_an_export_that_cannot_be_written_stops(world, cfg, spec, why):
    with pytest.raises(GenerationError, match=why):
        dbrelease.release(config.load(), VIEWS, built_at=AT + timedelta(days=1), log=io.StringIO(),
                          export={"meta_tags": spec})


def test_an_export_names_views_of_the_db_only(world, cfg):
    with pytest.raises(GenerationError, match="not one of the views"):
        calibration.with_export(world[1], world[0] / "store", ["meta_tags"], {"chart_patterns": {"dtype": "int8"}})


def test_a_genre_view_keeps_its_dimensions(monkeypatch):
    e = get("meta_tags")
    monkeypatch.setattr(type(e), "role_name", lambda self: "genre")
    with pytest.raises(GenerationError, match="genre"):
        export.check({"dim": 4}, e)
    assert export.check({"dtype": "int8"}, e) == {"dim": e.dim, "dtype": "int8"}


def test_open_db_refuses_vectors_that_are_not_what_the_manifest_says(world, cfg):
    tmp_path, cal, pool, base = world
    out, _ = small_release(tmp_path, cfg)
    m = json.loads((out / "manifest.json").read_text())
    for change in ({"scale": None}, {"scale": [1.0]}, {"dtype": "float16"}, {"dtype": "int4"}, {"dim": 9}):
        (out / "manifest.json").write_text(json.dumps({**m, "views": {**m["views"], MT: {**m["views"][MT], **change}}}))
        with pytest.raises(DBError):
            open_db(out)
