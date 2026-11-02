"""すべての設定とその既定値（正本はここ。説明は OctantSpecs octantvector/config.md）。

値は 環境変数 > 設定ファイル > 下の既定値 の順で決まる。設定ファイルは TOML で、$BV_CONFIG か
./octantvector.toml にある。キーは変数名から BV_ を外して小文字にしたもの（例 `archive = "/path/to/archive"`）。

`given(values)` の中では、すべての設定と表を `values` だけから取り、ファイルも環境変数も読まない（OctantSpecs
contracts/、契約1 生成側の関数）。
"""
import contextvars
import os
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import GenerationError

DEFAULTS = {
    "archive": None,              # BV_ARCHIVE: maps/<hash[:2]>/<hash>.zip と download.journal.jsonl のあるフォルダ
    "catalogue": None,            # BV_CATALOGUE: BeatSaver のカタログのスナップショット（catalogue.jsonl.gz）
    "artists": None,              # BV_ARTISTS: なくてもよい、map_artists.jsonl.gz と similar_artists.jsonl.gz のあるフォルダ
                                  # （関係のビュー listenbrainz_artists@<YYYYMMDD>、役割 `artist`。generation/artists.py）
    "catalogue_sha256": None,     # BV_CATALOGUE_SHA256: なくてもよい、期待する hash。`octantvector check` で確かめる
    "store": "store",             # BV_STORE: ベクトルのストアの根（エンコーダーの id@version ごとにサブフォルダ）
    "models_dir": None,           # BV_MODELS_DIR: エンコーダーが読むモデルのファイル。<models_dir>/hf/<repo>/<revision>/ の形
                                  # （OctantVector はダウンロードしない）
    "device": "auto",             # BV_DEVICE: "auto"（使えれば CUDA）か "cpu"
    "workers": 12,                # BV_WORKERS: 埋め込みに使う CPU のワーカーのプロセス数
    "worker_memory_gb": 6,        # BV_WORKER_MEMORY_GB: 自分の分でこれより多くのメモリを使ったワーカーは止め、そのマップを失敗として
                                  # 記録する（0: 上限なし。Linux だけ）
    "full_every": 30,             # BV_FULL_EVERY: 2 つの全量のリリースの間の差分の数（octantvector release）
    "swing_model": None,          # BV_SWING_MODEL: `octantvector swing-train` が書くスタイルのモデルのファイル
}


_GIVEN = contextvars.ContextVar("given", default=None)


@contextmanager
def given(values: dict):
    """ブロックの間、設定と表を `values` だけから取る（ファイルも環境変数も読まない）。"""
    token = _GIVEN.set(dict(values))
    try:
        yield
    finally:
        _GIVEN.reset(token)


def current():
    """`given` で設定した値（その外では None）。ワーカーのプロセスに渡す。"""
    return _GIVEN.get()


def _file_values():
    g = _GIVEN.get()
    if g is not None:
        return g
    p = Path(os.environ.get("BV_CONFIG", "octantvector.toml"))
    if p.exists():
        with open(p, "rb") as f:
            return tomllib.load(f)
    return {}


def section(name) -> dict:
    """設定ファイルの表 1 つ（例 [swing]）。キーと既定値は、それを読むモジュールが持つ。"""
    v = _file_values().get(name, {})
    if not isinstance(v, dict):
        raise GenerationError(f"'{name}' in the config file must be a table ([{name}])")
    return v


@dataclass(frozen=True)
class Config:
    archive: Path | None
    catalogue: Path | None
    artists: Path | None
    catalogue_sha256: str | None
    store: Path
    models_dir: Path | None
    device: str
    workers: int
    worker_memory_gb: float
    full_every: int
    swing_model: Path | None

    def require(self, name):
        v = getattr(self, name)
        if v is None:
            raise GenerationError(f"setting '{name}' is not configured: set BV_{name.upper()} or '{name}' in "
                                  f"octantvector.toml (see https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)")
        return v


def load():
    vals = dict(DEFAULTS)
    vals.update({k: v for k, v in _file_values().items() if k in DEFAULTS})
    unknown = set(_GIVEN.get() or {}) - set(DEFAULTS) - {"pools", "swing", "swing_tags"}
    if unknown:
        raise GenerationError(f"unknown setting(s): {', '.join(sorted(unknown))} (OctantSpecs octantvector/config.md)")
    for k in DEFAULTS if _GIVEN.get() is None else ():
        env = os.environ.get(f"BV_{k.upper()}")
        if env is not None:
            vals[k] = env
    for k in ("workers", "full_every"):
        vals[k] = int(vals[k])
    vals["worker_memory_gb"] = float(vals["worker_memory_gb"])
    for k in ("archive", "catalogue", "artists", "store", "models_dir", "swing_model"):
        vals[k] = Path(vals[k]).expanduser() if vals[k] else None
    return Config(**vals)
