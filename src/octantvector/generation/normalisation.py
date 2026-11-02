"""各ビューの正規化: エンコーダーのベクトルを、配布 DB（dbrelease.py）と実験用の索引（search.py）の値にする。
norm が "center" なら中心を引いて L2 正規化し、"zscore" なら z スコアにして L2 正規化する。統計は較正
（calibration.py）のもの、なければ手元の行のもの。"none" は L2 正規化だけ（統計はずれを見るために持つが使わない）。

このコードはエンコーダーの定義と同じくらい値の意味を決めるが、fingerprint では覆われないので、独自の VERSION を
持つ: stats() か normalise() が違う数を出すようになったら上げる（OctantSpecs 法 2.2）。較正は作ったときの版を
記録し、その id にも含まれるので、上げた後は古い較正を断り、較正し直すと新しい較正の id になる。別の正規化をした
DB は前の版の差分にならない（dbformat.diffable）。tests/test_search_views.py が版ごとの出力を固定している。
"""
import numpy as np

from ..common.dbformat import normalisation
from .errors import GenerationError

VERSION = 1  # 最初の版。較正には書かないので、較正の id はこれができる前のまま


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def stats(enc, x):
    """1 つのビューの正規化の統計: {"norm", "mean", "std"}（std は z スコアのビューだけ）。"""
    x = np.nan_to_num(np.asarray(x, np.float64))
    return {"norm": enc.norm, "mean": x.mean(0), "std": x.std(0) if enc.norm == "zscore" else None}


def normalise(enc, x, st=None):
    x = np.nan_to_num(np.asarray(x, np.float64))
    st = st or stats(enc, x)
    if enc.norm == "zscore":
        return l2((x - st["mean"]) / (np.asarray(st["std"]) + 1e-6))
    if enc.norm == "none":
        return l2(x)
    return l2(x - st["mean"])  # 学習した埋め込みは大きな共通の成分を持つ


def check(calibration):
    """`calibration` がこの正規化で作られたものでなければ止める。"""
    if normalisation(calibration) != VERSION:
        raise GenerationError(f"calibration {calibration['id']} was made with normalisation version "
                              f"{normalisation(calibration)}, this OctantVector normalises with version {VERSION}: "
                              "calibrate again")


def calibrated(calibration, enc):
    """`calibration` の中の enc の統計。正規化の版、id@version、fingerprint、norm が同じであること。違えば止める。"""
    check(calibration)
    key = f"{enc.id}@{enc.version}"
    st = calibration["views"].get(key)
    if st is None:
        raise GenerationError(f"calibration {calibration['id']} has no statistics for {key}")
    if st["fingerprint"] != enc.fingerprint():
        raise GenerationError(f"calibration {calibration['id']} was made for another definition of {key}")
    if st["norm"] != enc.norm:
        raise GenerationError(f"calibration {calibration['id']} normalises {key} by {st['norm']}, "
                              f"the encoder by {enc.norm}")
    return st
