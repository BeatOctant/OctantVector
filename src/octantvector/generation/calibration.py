"""較正: 各ビューの正規化の統計。プールで一度計算して固定するので、後で足すベクトルも同じ座標の空間に入る
（OctantSpecs 法 2.3）。

較正は JSON のファイル <store>/calibration/<id>.json:
  {"id": "<中身の sha256>", "pool": ..., "created_at": ..., "normalisation": <版。1 なら書かない>,
   "views": {"<id>@<version>": {"norm", "fingerprint", "n", "mean": [...], "std": [...] | null,
                                "export": {"dim", "dtype", "basis": <sha> | null, "scale": [...] | null}}}}
"export" はビューを DB にどう書くか（export.py）。ないビューはエンコーダーが作ったまま書く。統計は、作ったときの
正規化の版（normalisation.py）でだけ使え、使うときに確かめる。
検索では Index(..., calibration=load(path)) で使う。比べるのは、id@version と較正の id が同じベクトルどうしだけ。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..common.charts import key_hash
from . import export as export_mod
from . import normalisation
from .encoders import get
from .errors import GenerationError
from .normalisation import normalise, stats
from .store import Store


def compute(store_root, view_ids, hashes=None, pool=None):
    """`hashes`（既定はストアのすべてのベクトル）での各ビューの統計。譜面のビューは、そのマップの版のすべての譜面を使う。"""
    views = {}
    for i in view_ids:
        enc = get(i)
        hs, X = Store(enc, store_root).load()
        if hashes is not None:
            want = set(hashes)
            X = X[[n for n, h in enumerate(hs) if key_hash(h) in want]]
        if len(X) == 0:
            raise GenerationError(f"{i}: no vectors to calibrate on")
        st = stats(enc, X)
        views[f"{enc.id}@{enc.version}"] = _plain_stats({
            "norm": st["norm"], "fingerprint": enc.fingerprint(), "n": int(len(X)), "mean": st["mean"], "std": st["std"]})
    return _make(pool, views)


def extend(cal, store_root, view_ids, hashes=None, pool=None):
    """`cal` に、持っていないビューの統計を足したもの。持っているビューの統計はそのまま残すので、DB はほかを動かさずに
    ビューを足せる（ビューのパック）。-> 新しい較正（新しい id）。足りないものがなければ `cal` そのもの。"""
    missing = [i for i in view_ids if f"{get(i).id}@{get(i).version}" not in cal["views"]]
    if not missing:
        return cal
    normalisation.check(cal)  # 残す統計は、新しい統計と同じ意味であること
    views = plain(cal)["views"]
    views.update(compute(store_root, missing, hashes, pool)["views"])
    return load_dict({**_make(cal.get("pool"), views), "extends": cal["id"]})


def with_export(cal, store_root, view_ids, export, hashes=None, pool=None):
    """`cal` の各ビューの export の項目を `export`（{view id: 指定}、export.py。名前のないビューは なし）に合わせて
    作ったもの。`hashes`（既定はストアのすべてのベクトル）のベクトルで合わせる。項目がもう合っているビューはそのまま。
    -> 新しい較正（新しい id）。全部合っていれば `cal` そのもの。"""
    encs = {f"{get(i).id}@{get(i).version}": get(i) for i in view_ids}
    want = {}
    for i, spec in (export or {}).items():
        key = f"{get(i).id}@{get(i).version}"
        if key not in encs:
            raise GenerationError(f"export names {i}, which is not one of the views")
        want[key] = export_mod.check(spec, encs[key])
    views, changed = plain(cal)["views"], False
    for key, e in encs.items():
        if export_mod.same(views[key].get("export"), want.get(key)):
            continue
        changed = True
        views[key].pop("export", None)
        if key in want:
            hs, X = Store(e, store_root).load()
            if hashes is not None:
                keep = set(hashes)
                X = X[[n for n, h in enumerate(hs) if key_hash(h) in keep]]
            if len(X) == 0:
                raise GenerationError(f"{key}: no vectors to fit its export on")
            views[key]["export"] = export_mod.fit(normalise(e, X, views[key]), want[key], e, store_root)
    if not changed:
        return cal
    normalisation.check(cal)  # 残す統計は、export を合わせるものと同じ意味であること
    return load_dict({**_make(cal.get("pool"), views), "extends": cal["id"]})


def _make(pool, views):
    body = {"pool": pool, "views": views}
    if normalisation.VERSION != 1:  # id に含めるので、新しい正規化は新しい座標系になる
        body["normalisation"] = normalisation.VERSION
    cid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {"id": cid, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **body}


def plain(cal):
    """`cal` を JSON の値にしたもの: 統計は配列ではなく list（load_dict が読み戻す形）。"""
    return {**cal, "views": {k: _plain_stats(st) for k, st in cal["views"].items()}}


def _plain_stats(st):
    return {**st, "mean": np.asarray(st["mean"]).tolist(),
            "std": None if st["std"] is None else np.asarray(st["std"]).tolist()}


def save(cal, store_root):
    p = Path(store_root) / "calibration" / f"{cal['id']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plain(cal)))
    return p


def load(path):
    return load_dict(json.loads(Path(path).read_text()))


def load_dict(cal):
    for st in cal["views"].values():
        st["mean"] = np.asarray(st["mean"])
        st["std"] = None if st["std"] is None else np.asarray(st["std"])
    return cal


def drift(store_root, cal, view_ids, hashes):
    """`cal` の平均と `hashes` の今のベクトルとの {view id@version: ずれ}: 各ビューの平均がどれだけ動いたかを、較正の
    平均のまわりの広がりを単位にして表す（0 = 動いていない）。`cal` にないビューは外す。"""
    want, out = set(hashes), {}
    for i in view_ids:
        e = get(i)
        key = f"{e.id}@{e.version}"
        st = cal["views"].get(key)
        if st is None:
            continue
        hs, X = Store(e, store_root).load()
        X = np.nan_to_num(np.asarray(X[[n for n, h in enumerate(hs) if key_hash(h) in want]], np.float64))
        if not len(X):
            continue
        mu = np.asarray(st["mean"], np.float64)
        spread = np.sqrt(((X - mu) ** 2).sum(1).mean()) + 1e-12
        out[key] = float(np.linalg.norm(X.mean(0) - mu) / spread)
    return out
