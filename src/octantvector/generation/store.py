"""エンコーダーごとのベクトルのストア: <root>/<id>@<version>/{encoder.json, shard_NNNN.npz}。キーは版の hash、
譜面の行を持つビューでは譜面のキー（charts.py）。

エンコーダーを足してもほかのエンコーダーのフォルダには触れない。encoder.json にはエンコーダーの定義と
fingerprint（encoders/base.py）を記録する。版を上げずにどちらかが変わったら（例: 派生エンコーダーの元の
ビューが新しい版になった）、出力を混ぜずにストアを開くのを失敗させる。
"""
import json
from pathlib import Path

import numpy as np

from .errors import GenerationError


class Store:
    def __init__(self, enc, root, flush_every=250, on_flush=None):
        """on_flush(keys) は各 shard を書いた後に走る（台帳は結果がディスクに載ってから記録する）。"""
        self.enc, self.on_flush = enc, on_flush
        self.dir = Path(root) / f"{enc.id}@{enc.version}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        info, self.fingerprint = enc.definition(), enc.fingerprint()
        p = self.dir / "encoder.json"
        if p.exists():
            saved = json.loads(p.read_text())
            if {k: v for k, v in saved.items() if k != "fingerprint"} != info:
                raise GenerationError(f"{p} differs from the current definition of {enc.id}; bump its version")
            if saved.get("fingerprint") != self.fingerprint:
                raise GenerationError(f"{p} records another fingerprint than {enc.id} has now (a source changed: "
                                      f"{', '.join(enc.sources()) or 'none'}); bump its version")
        else:
            p.write_text(json.dumps({**info, "fingerprint": self.fingerprint}, indent=1))
        self._h, self._v = [], []

    def shards(self):
        return sorted(self.dir.glob("shard_*.npz"))

    def done(self) -> set:
        return {h for s in self.shards() for h in np.load(s)["hashes"].tolist()}

    def add(self, version_hash, vec):
        self.add_many({version_hash: vec})

    def add_many(self, items):
        """1 マップ（その譜面）の {key: vector}。1 つの shard に入れ、マップが書きかけで記録されないようにする。"""
        vecs = {k: np.asarray(v, np.float32) for k, v in items.items()}
        bad = next((v.shape for v in vecs.values() if v.shape != (self.enc.dim,)), None)
        if bad is not None:
            raise ValueError(f"{self.enc.id}: expected shape ({self.enc.dim},), got {bad}")
        self._h.extend(vecs); self._v.extend(vecs.values())
        if len(self._h) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._h:
            n = len(self.shards())
            np.savez(self.dir / f"shard_{n:04d}.npz", hashes=np.array(self._h), vecs=np.stack(self._v))
            hs, self._h, self._v = self._h, [], []
            if self.on_flush:
                self.on_flush(hs)

    def load(self):
        """-> (hash の list, (n, dim) の float32 配列)。hash が重なったら後の shard が勝つ。"""
        d = {}
        for s in self.shards():
            z = np.load(s)
            d.update(zip(z["hashes"].tolist(), z["vecs"]))
        hs = list(d)
        return hs, (np.stack([d[h] for h in hs]) if hs else np.zeros((0, self.enc.dim), np.float32))
