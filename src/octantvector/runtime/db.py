"""配布 DB を読む側（OctantSpecs contracts/: 契約1（OctantCompass to OctantVector）の `open_db`、契約2（DB build to
recommendation core）の形式。`search` はこの推薦側のもので、契約にはない）。

open_db は、何かを使う前にマニフェストを確かめる: schemaVersion のメジャーがこのコードの読めるものであること、
全量の DB であること（差分は先に apply_diff で当てる）、どのファイルもマニフェストの SHA-256 と合うこと。知らない
マニフェストのキーと、この版の知らない行のビューは無視する（新しいマイナー版）。署名はここでは確かめない:
apply_update が外から来るものを確かめ、差分を当てて作った全量は署名を持たない（OctantSpecs contracts/、
契約5（OctantHarvest to OctantCompass）署名）。ベクトルはもう DB の較正で正規化してあるので、検索は内積になる。
索引は初めて開いたときにここで作り、配らない。float16 のビューは float32 で持ち、int8 のビューは int8 のまま
（Int8Vectors）で float32 の 4 分の 1。

ビューは、マップの行（マップの版ごとに 1 つのベクトル: 曲、メタデータ）か譜面の行（譜面ごとに 1 つのベクトル）を
持つ。譜面のビューは譜面で検索し、同じプレイスタイルの譜面だけを返す（OctantSpecs constitution.md、原則 2。
common/charts.py）。

関係のビュー（契約2、ビューの種類）はベクトルを持たない: 各マップが何に属するか（そのアーティスト）と、メンバーの
対の点を持つ。基準のマップとの近さは、メンバーを共有すれば 1、そうでなければメンバーの対の一番よい点（2 つの向きの
大きいほう）、それもなければ 0（Relation）。`search` はベクトルのビューだけ。
"""
from pathlib import Path

import numpy as np

from ..common import charts as chartids
from ..common.dbformat import (ROW_KINDS, VIEW_KINDS, FormatError, chart_keys, listed, read_charts, read_hashes,
                               read_manifest, read_relation, read_superseded, read_view, view_kind)
from ..common.dbformat import VersionError, apply_diff  # noqa: F401  （API の一部）


DBError = FormatError  # DB を使えない
# VersionError（DBError の一種）: マニフェストか索引が、この OctantVector の読めない版を宣言している（OctantSpecs
# contracts/、契約の版）


class Relation:
    """配列にした関係のビュー: メンバーのあるマップの行（並べたもの）、各行のメンバーの添字（CSR: `ptr`、`idx`）、
    メンバーごとの、2 つの向きの大きいほうの点での近所（CSR: `nptr`、`nidx`、`nscore`）。"""

    def __init__(self, members, pairs):
        names = sorted(set(members["member"]) | set(pairs["a"]) | set(pairs["b"]))
        self.member_pos = {m: n for n, m in enumerate(names)}
        n = len(names)
        r = np.asarray(members["row"], np.int64)
        m = np.array([self.member_pos[x] for x in members["member"]], np.int64)
        order = np.lexsort((m, r))
        r, m = r[order], m[order]
        self.rows, starts = np.unique(r, return_index=True)
        self.ptr = np.append(starts, len(r)).astype(np.int64)
        self.idx = m
        self.row_pos = {int(x): k for k, x in enumerate(self.rows)}
        a = np.array([self.member_pos[x] for x in pairs["a"]], np.int64)
        b = np.array([self.member_pos[x] for x in pairs["b"]], np.int64)
        sc = np.asarray(pairs["score"], np.float32)
        i, j, s = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([sc, sc])
        keep = i != j
        i, j, s = i[keep], j[keep], s[keep]
        order = np.lexsort((-s, j, i))  # (i, j) ごとに、大きい点を先に
        i, j, s = i[order], j[order], s[order]
        first = np.ones(len(i), bool)
        first[1:] = (i[1:] != i[:-1]) | (j[1:] != j[:-1])
        i, j, s = i[first], j[first], s[first]
        self.nptr = np.searchsorted(i, np.arange(n + 1)).astype(np.int64)
        self.nidx, self.nscore = j, s
        self.n_members = n

    def members_of(self, row):
        k = self.row_pos.get(row)
        return () if k is None else tuple(int(x) for x in self.idx[self.ptr[k]:self.ptr[k + 1]])

    def closeness(self, row):
        """メンバーのある各行（`rows` の順）とマップ `row` との近さ。メンバーがなければ None。"""
        ms = self.members_of(row)
        if not ms:
            return None
        per_member = np.zeros(self.n_members, np.float32)
        for a in ms:
            nb = slice(self.nptr[a], self.nptr[a + 1])
            per_member[self.nidx[nb]] = np.maximum(per_member[self.nidx[nb]], self.nscore[nb])
        per_member[list(ms)] = 1.0
        return np.maximum.reduceat(per_member[self.idx], self.ptr[:-1])

    def reference(self, rows):
        """メンバーのある基準のマップについての近さの平均（基準が複数: ベクトルのビューが平均のベクトルを取るのと同じく
        平均）。どれもなければ None。"""
        found = [c for c in (self.closeness(r) for r in rows) if c is not None]
        return np.mean(found, 0) if found else None


class Int8Vectors:
    """int8 のまま持つ int8 のビュー（契約2: 値にその次元の尺度を掛けたものがベクトル）。ベクトルの float32 の配列の
    代わりに使う。`x @ q` は行を塊ごとに通るので、float32 の写しが丸ごとできることはない。行を選ぶ添字（マスク、
    list）なら Int8Vectors を返し、ほかの添字なら float32 の値を返す。"""

    BLOCK = 16384

    def __init__(self, q, scale):
        self.q, self.scale = q, np.asarray(scale, np.float32)
        self.shape, self.size, self.dtype = q.shape, q.size, np.dtype(np.float32)
        self.nbytes = q.nbytes + self.scale.nbytes  # float32 の値の分ではなく、実際に使う分

    def __len__(self):
        return len(self.q)

    def __matmul__(self, other):
        w = np.asarray(other, np.float32)
        w = w * (self.scale if w.ndim == 1 else self.scale[:, None])
        out = np.empty((len(self.q),) + w.shape[1:], np.float32)
        for a in range(0, len(self.q), self.BLOCK):
            out[a:a + self.BLOCK] = self.q[a:a + self.BLOCK].astype(np.float32) @ w
        return out

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            r, c = idx
            return self.q[r, c].astype(np.float32) * self.scale[c]
        if isinstance(idx, (int, np.integer)):
            return self.q[idx].astype(np.float32) * self.scale
        return Int8Vectors(self.q[idx], self.scale)

    def __array__(self, dtype=None, copy=None):
        x = self.q.astype(np.float32) * self.scale
        return x if dtype is None else x.astype(dtype)


def _vectors(x, meta):
    """推薦側が持つ形のビューのベクトル: float32（DB では float16）か Int8Vectors。"""
    return Int8Vectors(x, meta["scale"]) if x.dtype == np.int8 else x.astype(np.float32)


class DB:
    def __init__(self, path, manifest, hashes, chart_keys, play_styles, superseded, views, skipped, chart_metrics=None,
                 relations=None):
        self.path, self.manifest = path, manifest
        self.hashes = hashes
        self.pos = {h: n for n, h in enumerate(hashes)}
        self.chart_keys = chart_keys  # charts.parquet の順
        self.chart_pos = {k: n for n, k in enumerate(chart_keys)}
        self.play_styles = play_styles  # 譜面の行ごと
        self.superseded = superseded
        self.views = views  # {id@version: (ベクトル (n, dim): float32 か Int8Vectors, maps か charts の行の番号)}
        self.relations = relations or {}  # {id@version: Relation}
        self.rows = {k: manifest["views"][k]["rows"] for k in list(views) + list(self.relations)}
        self.skipped = skipped
        self.chart_metrics = chart_metrics or {}  # charts.parquet の数の列を float の配列で（空なら NaN）
        self._chart_maps = self._chart_map_array = self._hash_array = self._style_array = None
        self._row_pos, self._styles = {}, {}

    @property
    def version(self):
        return self.manifest["dbVersion"]

    def info(self):
        return _info(self.manifest, list(self.views) + list(self.relations), self.skipped,
                     {**{k: (len(r), int(x.shape[1])) for k, (x, r) in self.views.items()},
                      **{k: (len(rel.rows), 0) for k, rel in self.relations.items()}})

    def current(self, h):
        """マップの版 `h` の現行版: h そのものか、superseded.parquet が指す版。DB がどちらも持たなければ None（それが何を
        意味するかは呼ぶ側が決める）。"""
        if h in self.pos:
            return h
        cur = self.superseded.get(h)
        return cur if cur in self.pos else None

    def chart_map_rows(self):
        """各譜面の行のマップの行（譜面のマップの版が maps.parquet になければ None）。"""
        if self._chart_maps is None:
            self._chart_maps = [self.pos.get(chartids.key_hash(k)) for k in self.chart_keys]
        return self._chart_maps

    def chart_map_array(self):
        """chart_map_rows を配列にしたもの。譜面のマップの版が maps.parquet になければ -1。"""
        if self._chart_map_array is None:
            self._chart_map_array = np.array([-1 if m is None else m for m in self.chart_map_rows()], np.int64)
        return self._chart_map_array

    def hash_array(self):
        """マップの hash の配列。マップの行を hash で並べるため。"""
        if self._hash_array is None:
            self._hash_array = np.array(self.hashes)
        return self._hash_array

    def views_of_role(self, role):
        """役割のベクトルのビュー。"""
        return [k for k in self.views if self.manifest["views"][k].get("role") == role]

    def roles(self):
        """この推薦側が読むすべてのビュー（ベクトルと関係）の {view: role}。"""
        return {k: self.manifest["views"][k].get("role") for k in list(self.views) + list(self.relations)}

    def artists(self):
        """役割 `artist` の関係のビュー（キーで最初のもの）。なければ None（OctantSpecs 法 3.4）。"""
        keys = sorted(k for k in self.relations if self.manifest["views"][k].get("role") == "artist")
        return self.relations[keys[0]] if keys else None

    def genres(self):
        """この DB で絞り込めるジャンルの語: 役割 `genre` のビューからの {語: (view, 次元)}（契約2。語の一覧そのものは
        まだ決まっていないので、DB が持たないこともある）。"""
        out = {}
        for k in self.views_of_role("genre"):
            for i, word in enumerate(self.manifest["views"][k].get("genres", [])):
                out.setdefault(word, (k, i))
        return out

    def play_style_array(self):
        """play_styles を配列にしたもの（譜面の行ごとに 1 つ）。一度だけ作る。"""
        if self._style_array is None:
            self._style_array = np.array(self.play_styles, dtype=object)
        return self._style_array

    def styles_in_view(self, view):
        """譜面のビューの各行のプレイスタイル。"""
        if view not in self._styles:
            self._styles[view] = self.play_style_array()[self.views[view][1]]
        return self._styles[view]

    def row_in_view(self, view, row):
        if view not in self._row_pos:
            self._row_pos[view] = {int(r): n for n, r in enumerate(self.views[view][1])}
        return self._row_pos[view].get(row)


def _reads(v):
    """この推薦側がマニフェストのビューを読むか。open_db はほかを飛ばす（skippedViews）: 新しいマイナー版の行や種類。"""
    return v.get("rows") in ROW_KINDS and view_kind(v) in VIEW_KINDS and (view_kind(v) == "vector" or v["rows"] == "maps")


def _info(m, read, skipped, sizes=None):
    """DB.info() と read_info() が DB について言うこと: マニフェスト m から、読んだビュー（ベクトルのビュー、続いて
    関係のビュー）と飛ばしたビュー。sizes: 読んだときの {view: (rows, dim)}。なければ（read_info）各ビューは "rows" を
    持たず、dim はマニフェストにあるもの（なければ None）。"""
    views = {}
    for k in read:
        v = m["views"][k]
        given = v.get("dim") if type(v.get("dim")) is int else None
        rows, dim = sizes[k] if sizes is not None else (None, 0 if view_kind(v) == "relation" else given)
        views[k] = {"role": v.get("role"), "kind": view_kind(v), "rowKind": v["rows"],
                    **({} if rows is None else {"rows": rows}), "dim": dim, "credits": _credits(v)}
    # ジャンルの語: 役割 `genre` のベクトルのビューのもの（DB.genres）
    genres = {w for k in read if view_kind(m["views"][k]) == "vector" and m["views"][k].get("role") == "genre"
              for w in m["views"][k].get("genres", [])}
    return {"schemaVersion": m["schemaVersion"], "dbVersion": m["dbVersion"], "builtAt": m.get("builtAt"),
            "calibrationId": m["calibrationId"], "counts": m.get("counts", {}), "views": views,
            "skippedViews": skipped, "genres": sorted(genres),
            "notice": m["notice"] if isinstance(m.get("notice"), str) else None}


def _credits(view):
    """ビューのクレジット。文字の行だけを残す（DB は誰から来るか分からない: 契約5 署名）。"""
    c = view.get("credits")
    return [x for x in c if isinstance(x, str)] if isinstance(c, list) else []


TEXT_COLS = ("hash", "characteristic", "difficulty", "play_style")


def _numbers(values):
    """charts.parquet の列が数を持つ（か何も持たない）なら True: 指標で、知っているもの（nps、njs、pass、tech と、測った列 njs_play、jd、reaction_s、nps_peak）か、
    新しい DB が足したもの（ほかの列はそのまま運ぶ、dbformat.py）。"""
    return all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in values)


def open_db(path):
    path = Path(path)
    manifest = read_manifest(path, "full")  # schemaVersion、kind、各ファイルの SHA-256

    hashes = read_hashes(path, manifest)
    ch = read_charts(listed(path, manifest, "charts.parquet"))
    keys = chart_keys(ch)
    superseded = read_superseded(path, manifest)
    n_rows = {"maps": len(hashes), "charts": len(keys)}
    views, relations, skipped = {}, {}, []
    for key, v in manifest.get("views", {}).items():
        if not _reads(v):
            skipped.append(key)
            continue
        if view_kind(v) == "relation":
            relations[key] = Relation(*read_relation(path, manifest, key, len(hashes)))
            continue
        x, rows = read_view(path, manifest, key, n_rows[v["rows"]])
        views[key] = (_vectors(x, v), rows)
    metrics = {c: np.array([np.nan if v is None else v for v in ch[c]], np.float64)
               for c in ch if c not in TEXT_COLS and _numbers(ch[c])}
    return DB(path, manifest, hashes, keys, list(ch["play_style"]), superseded, views, skipped, metrics, relations)


def read_info(path):
    """path にある全量の DB について、マニフェストだけから DB.info() と同じことを言う: 各ビューは "rows" を除く（ビューを
    読んだときに数える）。ほかのファイルは読みも確かめもしない（open_db がすべて確かめる）ので速く、開いていない DB を
    自分の言うとおりに見せる。読める全量の DB のマニフェストがなければ DBError、schemaVersion がこの OctantVector の
    読めないものなら VersionError。"""
    m = read_manifest(path, "full", check_files=False)
    try:
        views = m.get("views", {})
        read = [k for kind in VIEW_KINDS for k, v in views.items() if _reads(v) and view_kind(v) == kind]
        return _info(m, read, [k for k in views if k not in read])
    except (KeyError, TypeError, ValueError, AttributeError) as ex:
        raise FormatError(f"{path}: the manifest is not one of a full DB ({type(ex).__name__}: {ex})") from None


def _current(db, h):
    """-> (現行版の hash, それが置き換えた古い hash か None)。"""
    cur = db.current(h)
    if cur is None:
        raise KeyError(f"{h} is not in DB {db.version}")
    return cur, (h if cur != h else None)


def search(db, query, view, k=10):
    """1 つのビュー（`id@version`）の上位 k。

    マップのビューは版の hash を取り、{"hash", "rank", "score"} を返す。譜面のビューは譜面を、譜面のキーか
    (hash, characteristic, difficulty) で取り、同じプレイスタイルの譜面だけの {"hash", "characteristic",
    "difficulty", "rank", "score"} を返す。旧い版は同じマップの現行版に置き換える（譜面なら、現行版の同じ
    characteristic と difficulty。作り直された譜面のこともある。"replaced" がそれを示す）。-> {"view", "query",
    "replaced", "results"}。クエリがそのビューにベクトルを持たなければ results は空。"""
    if view not in db.views:
        raise KeyError(f"{view} is not in DB {db.version}; views: {', '.join(db.views)}")
    x, rows = db.views[view]
    if db.rows[view] == "charts":
        h, c, d = chartids.split(query) if isinstance(query, str) else tuple(query)
        cur, replaced = _current(db, h)
        q_key = chartids.key(cur, c, d)
        if q_key not in db.chart_pos:
            raise KeyError(f"{chartids.key(h, c, d)} is not in DB {db.version}")
        out = {"view": view, "query": q_key, "replaced": replaced and chartids.key(h, c, d), "results": []}
        q_row = db.chart_pos[q_key]
        style = db.play_styles[q_row]
        allowed = db.styles_in_view(view) == style
    else:
        cur, replaced = _current(db, query)
        out = {"view": view, "query": cur, "replaced": replaced, "results": []}
        q_row = db.pos[cur]
        allowed = None
    q = db.row_in_view(view, q_row)
    if q is None:
        return out
    s = x @ x[q]
    if allowed is not None:
        s[~allowed] = -np.inf
    s[q] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return out
    top = np.argpartition(-s, k - 1)[:k]
    top = top[np.argsort(-s[top], kind="stable")]
    for n, j in enumerate(top, 1):
        if db.rows[view] == "charts":
            h, c, d = chartids.split(db.chart_keys[int(rows[j])])
            out["results"].append({"hash": h, "characteristic": c, "difficulty": d, "rank": n, "score": float(s[j])})
        else:
            out["results"].append({"hash": db.hashes[int(rows[j])], "rank": n, "score": float(s[j])})
    return out
