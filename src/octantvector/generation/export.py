"""ビューのベクトルを DB にどう書くか: 次元の数と型（OctantSpecs contracts/、契約2（DB build to recommendation core）
ビューの型）。ストアはどのベクトルもエンコーダーが作ったまま持つので、ベクトルを計算し直さずに、いつでもどの
大きさでも DB を書ける。

ビューの書き出しの指定:
  {"dim": <次元の数。エンコーダーの次元以下>, "dtype": "float16" | "int8"}
どちらのキーも省ける: "dim" がなければエンコーダーの次元のまま、"dtype" がなければ float16。指定のないビューは
これまでどおり（エンコーダーの次元、float16）に書き、較正に "export" を持たない。

次元を減らすときは、プールでの正規化したベクトルの右特異ベクトルの上位を使う（平均は引かない。保ちたいのは内積
なので）。射影したベクトルはそれぞれ単位ベクトルに戻す。int8 は各次元を尺度（プールでの絶対値の最大 / 127）で
割って丸めて持ち、範囲の外は切る。どちらも作ったときの較正で固定する（calibration.py の "export"）ので、後の版で
足すベクトルも同じ座標に入り、差分で運べる。指定を変えると新しい較正になり、次の版は全量になる。射影の行列は
ストアに置く（<store>/calibration/basis/<sha>.npy。名前はその SHA-256）: 要るのは生成側だけで、DB は射影した
ベクトルを持つ。

ジャンルのビューは射影しない: 各次元が 1 つの語だから（契約2）。
"""
import hashlib
from pathlib import Path

import numpy as np

from .errors import GenerationError

DTYPES = ("float16", "int8")
SAMPLE = 40000  # 射影を合わせる行（seed は固定）: 数百次元には足りる


def check(spec, enc):
    """エンコーダー enc の指定を {"dim", "dtype"} にする。書けないものなら止める。"""
    key = f"{enc.id}@{enc.version}"
    if not isinstance(spec, dict) or set(spec) - {"dim", "dtype"}:
        raise GenerationError(f"export of {key}: give {{\"dim\": <int>, \"dtype\": \"float16\" | \"int8\"}}, not {spec!r}")
    dim, dtype = spec.get("dim", enc.dim), spec.get("dtype", "float16")
    if dtype not in DTYPES:
        raise GenerationError(f"export of {key}: dtype {dtype!r} is not one of {', '.join(DTYPES)}")
    if type(dim) is not int or not 1 <= dim <= enc.dim:
        raise GenerationError(f"export of {key}: dim must be a whole number from 1 to {enc.dim}, not {dim!r}")
    if dim != enc.dim and enc.role_name() == "genre":
        raise GenerationError(f"export of {key}: a genre view keeps its dimensions (each one is a word)")
    return {"dim": dim, "dtype": dtype}


def same(entry, spec):
    """較正の export の項目（None: なし）が、指定（{"dim", "dtype"}。None: なし）のために作られたものか。"""
    if entry is None or spec is None:
        return entry is None and spec is None
    return entry["dim"] == spec["dim"] and entry["dtype"] == spec["dtype"]


def basis_path(store_root, sha):
    return Path(store_root) / "calibration" / "basis" / f"{sha}.npy"


def fit(x, spec, enc, store_root):
    """正規化したベクトル x（プール）と指定からの export の項目: 射影はストアに保存する。"""
    entry = {"dim": spec["dim"], "dtype": spec["dtype"], "basis": None, "scale": None}
    if spec["dim"] != enc.dim:
        rng = np.random.default_rng(0)
        s = x[rng.choice(len(x), SAMPLE, replace=False)] if len(x) > SAMPLE else x
        _, _, vt = np.linalg.svd(np.asarray(s, np.float64), full_matrices=False)
        if len(vt) < spec["dim"]:
            raise GenerationError(f"export of {enc.id}@{enc.version}: {len(s)} vectors cannot give {spec['dim']} "
                                  "dimensions")
        b = np.ascontiguousarray(vt[:spec["dim"]].T, np.float32)
        sha = hashlib.sha256(b.tobytes() + str(b.shape).encode()).hexdigest()[:16]
        p = basis_path(store_root, sha)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            np.save(p, b)
        entry["basis"] = sha
    if spec["dtype"] == "int8":
        z = _project(x, entry, store_root)
        entry["scale"] = (np.maximum(np.abs(z).max(0), 1e-12) / 127).astype(np.float32).tolist()
    return entry


def _project(x, entry, store_root):
    x = np.asarray(x, np.float32)
    if entry.get("basis") is None:
        return x
    p = basis_path(store_root, entry["basis"])
    if not p.exists():
        raise GenerationError(f"{p} is missing: the projection of this calibration is kept in the store; calibrate "
                              "again with the export settings")
    z = x @ np.load(p)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def apply(x, entry, store_root):
    """正規化したベクトル x を、export の項目に従って DB の持つ形にしたもの（None: そのまま float16）。"""
    if entry is None:
        return np.asarray(x).astype(np.float16)
    z = _project(x, entry, store_root)
    if entry["dtype"] == "int8":
        return np.clip(np.rint(z / np.asarray(entry["scale"], np.float32)), -127, 127).astype(np.int8)
    return z.astype(np.float16)
