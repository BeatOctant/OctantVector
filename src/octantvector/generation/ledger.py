"""台帳: OctantVector 自身の状態を 1 つの SQLite のファイル（<store>/ledger.sqlite）に持つ。アーカイブとカタログは
読むだけで、OctantVector がそれについて決めたことはすべてここに書く。

  versions   BeatSaver の版の hash ごとに 1 行: マップの key、状態（current / old / deleted）、アーカイブにある zip の
             commit_seq（NULL = まだ zip がない）、zip の SHA-256（zip を最初に読んだときに記録）、最初と最後に見た時刻
  meta_revs  版とマップの種類のエンコーダーごと: エンコーダーが宣言するカタログの欄（入力 "meta:<field>"）の
             要約と、その要約を最初に見た時刻。一番新しい行が今のもの
  computed   版とエンコーダーの fingerprint ごと: 結果（ok / error / skip）、計算した元の入力の要約、エラーの文と時刻
  pools      名前の付いたプールと版ごと: プールに入った時刻と、出たならその時刻

どの段も台帳だけを読んで残りを決めるので、止まった実行は止まったところから続く。
"""
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..common.charts import key_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (hash TEXT PRIMARY KEY, key TEXT, state TEXT NOT NULL, seq INTEGER,
                                     zip_sha256 TEXT, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta_revs (hash TEXT NOT NULL, encoder TEXT NOT NULL, digest TEXT NOT NULL,
                                      first_seen TEXT NOT NULL, PRIMARY KEY (hash, encoder, digest));
CREATE TABLE IF NOT EXISTS computed (hash TEXT NOT NULL, fingerprint TEXT NOT NULL, encoder TEXT NOT NULL,
                                     result TEXT NOT NULL, input_digest TEXT, error TEXT, at TEXT NOT NULL,
                                     PRIMARY KEY (hash, fingerprint));
CREATE TABLE IF NOT EXISTS pools (pool TEXT NOT NULL, hash TEXT NOT NULL, entered TEXT NOT NULL, left_at TEXT,
                                  PRIMARY KEY (pool, hash, entered));
CREATE INDEX IF NOT EXISTS pools_open ON pools (pool, hash) WHERE left_at IS NULL;
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def field(m, v, name):
    """宣言したカタログの欄の値: "versions.<x>" は版の記録を、それ以外はマップの記録を読む。ドットで入れ子の中へ進む。"""
    obj, path = (v, name.split(".")[1:]) if name.startswith("versions.") else (m, name.split("."))
    for p in path:
        obj = obj.get(p) if isinstance(obj, dict) else None
    return obj


def meta_digest(m, v, fields):
    blob = json.dumps({f: field(m, v, f) for f in sorted(fields)}, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def meta_fields(enc):
    return [i[5:] for i in enc.inputs if i.startswith("meta:")]


class Ledger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self):
        self.db.close()

    @classmethod
    @contextmanager
    def of(cls, store):
        """作業用ストア `store` の台帳（<store>/ledger.sqlite）。ブロックが終わると閉じる。"""
        led = cls(Path(store) / "ledger.sqlite")
        try:
            yield led
        finally:
            led.close()

    # ---- 取り込み -----------------------------------------------------------------------------------------------

    def ingest(self, maps, available, encoders, pools):
        """maps: カタログのマップの記録の iterable。available: アーカイブにある zip の {hash: commit_seq}。
        encoders: メタの要約を記録する {id: エンコーダー}（マップの種類だけ）。pools: {name: 定義}（pools.py）。
        -> 変わったものの数。"""
        from .catalogue import published_version
        from .pools import matches, members
        t = now()
        seen, eligible, current = set(), {p: [] for p in pools}, {}
        meta_encs = {i: meta_fields(e) for i, e in encoders.items() if e.kind == "map"}
        prev = {h: (k, s) for h, k, s in self.db.execute("SELECT hash, key, state FROM versions")}
        keys = set()
        n = {"new": 0, "old": 0, "deleted": 0, "meta_changed": 0}

        def changed(h, state):
            if h not in prev:
                n["new"] += 1
            elif state != prev[h][1] and state in ("old", "deleted"):
                n[state] += 1
        with self.db:
            for m in maps:
                pub = published_version(m)
                gone = bool(m.get("deletedAt")) or pub is None
                keys.add(m.get("id"))
                for v in m.get("versions", []):
                    h = v["hash"]
                    state = "deleted" if gone else ("current" if v is pub else "old")
                    seen.add(h)
                    changed(h, state)
                    self.db.execute(
                        "INSERT INTO versions (hash, key, state, seq, first_seen, last_seen) VALUES (?,?,?,?,?,?) "
                        "ON CONFLICT(hash) DO UPDATE SET key=excluded.key, state=excluded.state, seq=excluded.seq, "
                        "last_seen=excluded.last_seen", (h, m.get("id"), state, available.get(h), t, t))
                    if state != "current":
                        continue
                    current[m.get("id")] = h
                    for i, fields in meta_encs.items():
                        d = meta_digest(m, v, fields)
                        cur = self.db.execute("SELECT 1 FROM meta_revs WHERE hash=? AND encoder=? AND digest=?",
                                              (h, i, d)).fetchone()
                        if cur is None:
                            n["meta_changed"] += self.db.execute(
                                "SELECT 1 FROM meta_revs WHERE hash=? AND encoder=?", (h, i)).fetchone() is not None
                            self.db.execute("INSERT INTO meta_revs VALUES (?,?,?,?)", (h, i, d, t))
                    for p, definition in pools.items():
                        if matches(definition, m, v):
                            eligible[p].append((m.get("id"), h))
            for h, (k, s) in prev.items():
                if h in seen or s == "deleted":
                    continue
                # もう載っていない: カタログに残るマップの新しい版に置き換わったか、そうでなければマップがなくなった
                state = "old" if k in keys else "deleted"
                if state != s:
                    changed(h, state)
                    self.db.execute("UPDATE versions SET state=? WHERE hash=?", (state, h))
            for p, definition in pools.items():
                n[f"pool:{p}"] = self._set_pool(p, members(definition, eligible[p], current), t)
        return n

    def _set_pool(self, pool, hashes, t):
        inside = {h for (h,) in self.db.execute("SELECT hash FROM pools WHERE pool=? AND left_at IS NULL", (pool,))}
        self.db.executemany("INSERT INTO pools (pool, hash, entered) VALUES (?,?,?)",
                            [(pool, h, t) for h in hashes - inside])
        self.db.executemany("UPDATE pools SET left_at=? WHERE pool=? AND hash=? AND left_at IS NULL",
                            [(t, pool, h) for h in inside - hashes])
        return {"size": len(hashes), "entered": len(hashes - inside), "left": len(inside - hashes)}

    # ---- 計画 ---------------------------------------------------------------------------------------------

    def pool(self, name):
        """プールの今のメンバー。アーカイブの順（zip のない版は最後）-> [(hash, seq)]。"""
        return self.db.execute(
            "SELECT v.hash, v.seq FROM pools p JOIN versions v ON v.hash = p.hash "
            "WHERE p.pool=? AND p.left_at IS NULL AND v.state='current' "
            "ORDER BY v.seq IS NULL, v.seq, v.hash", (name,)).fetchall()

    def pool_hashes(self, name):
        """pool(name) の hash。同じ順。"""
        return [h for h, _ in self.pool(name)]

    def old_versions(self):
        """同じマップの新しい版に置き換わった版の [(hash, マップの key)]。"""
        return self.db.execute("SELECT hash, key FROM versions WHERE state='old'").fetchall()

    def current_digest(self, h, encoder_id):
        r = self.db.execute("SELECT digest FROM meta_revs WHERE hash=? AND encoder=? ORDER BY first_seen DESC, rowid DESC "
                            "LIMIT 1", (h, encoder_id)).fetchone()
        return r[0] if r else None

    def todo(self, enc, fingerprint, members, retry_errors=False):
        """`members`（[(hash, seq)]）のうち `enc` がまだ計算すべき hash: この fingerprint の結果がまだない、やり直すときの
        エラー、または（マップの種類で）計算した後にカタログの要約が変わったもの。音声、画像、譜面のエンコーダーは zip が
        要るので、zip のない版は待つ。"""
        done = {h: (r, d) for h, r, d in self.db.execute(
            "SELECT hash, result, input_digest FROM computed WHERE fingerprint=?", (fingerprint,))}
        out = []
        for h, seq in members:
            if enc.kind in ("audio", "image", "zip") and seq is None:
                continue
            r = done.get(h)
            if r is None or (retry_errors and r[0] == "error"):
                out.append(h)
            elif enc.kind == "map" and r[1] != self.current_digest(h, enc.id):
                out.append(h)
        return out

    # ---- 結果 ----------------------------------------------------------------------------------------------

    def record(self, enc, fingerprint, rows):
        """rows: [(hash, result, input_digest, error)]。ok のベクトルをストアに書き出してから呼ぶこと。"""
        t = now()
        with self.db:
            self.db.executemany(
                "INSERT INTO computed VALUES (?,?,?,?,?,?,?) ON CONFLICT(hash, fingerprint) DO UPDATE SET "
                "result=excluded.result, input_digest=excluded.input_digest, error=excluded.error, at=excluded.at",
                [(h, fingerprint, f"{enc.id}@{enc.version}", r, d, e, t) for h, r, d, e in rows])

    def adopt(self, enc, fingerprint, keys):
        """ストアにもうあるのに台帳に ok のないベクトル（落ちる直前に書き出したか、派生エンコーダーが書いた）は ok と数える。
        マップの種類の行には今の要約を付ける: 計算してからカタログは変わっていないとみなす。譜面のキーはそのマップの版に
        数える（結果は版ごとに記録する）。"""
        known = {h for (h,) in self.db.execute("SELECT hash FROM computed WHERE fingerprint=? AND result='ok'",
                                               (fingerprint,))}
        new = [h for h in dict.fromkeys(key_hash(k) for k in keys) if h not in known]
        self.record(enc, fingerprint, [(h, "ok", self.current_digest(h, enc.id) if enc.kind == "map" else None, None)
                                       for h in new])
        return len(new)

    def zip_sha(self, h, sha):
        """zip を最初に読んだときに、その SHA-256 を記録する。-> この hash で別の zip を見ていれば False。"""
        r = self.db.execute("SELECT zip_sha256 FROM versions WHERE hash=?", (h,)).fetchone()
        if r is None or r[0] is None:
            with self.db:
                self.db.execute("UPDATE versions SET zip_sha256=? WHERE hash=?", (sha, h))
            return True
        return r[0] == sha

    # ---- 報告 --------------------------------------------------------------------------------------------

    def results(self):
        """計算したすべての行を (key "<hash>:<fingerprint>", encoder, result, input digest, error, at) で。"""
        return set(self.db.execute("SELECT hash || ':' || fingerprint, encoder, result, input_digest, error, at "
                                   "FROM computed"))

    def summary(self):
        states = dict(self.db.execute("SELECT state, count(*) FROM versions GROUP BY state"))
        pools = dict(self.db.execute("SELECT pool, count(*) FROM pools WHERE left_at IS NULL GROUP BY pool"))
        results = {}
        for enc, r, c in self.db.execute("SELECT encoder, result, count(*) FROM computed GROUP BY encoder, result"):
            results.setdefault(enc, {})[r] = c
        return {"versions": states, "pools": pools, "computed": results}
