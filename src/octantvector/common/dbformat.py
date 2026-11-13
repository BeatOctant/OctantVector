"""配布 DB のファイル（OctantSpecs contracts/、契約2（DB build to recommendation core））: 全量の DB を書く、前の DB との
差分を書く、差分を当てる。生成側（generation/dbrelease.py）と推薦側（apply_diff）がこのコードを共有するので、
「全量」と「前の全量 + 差分」は同じ関数で書かれる。

ファイルによらない DB の中身:
  maps        {hash: (artist, mapper, duration_s, published_at)}
  charts      {hash: [(characteristic, difficulty, nps, njs, pass, tech, play_style, njs_play, jd, reaction_s, nps_peak), ...]}。カタログの順
  superseded  {旧い版の hash: 同じマップの現行版の hash}
  views       {"id@version": {"meta": {...}, "vecs": {key: vector}}}。キーは "maps" の行のビューなら版の hash、
              "charts" の行のビューなら譜面のキー（charts.py）。ベクトルは meta の "dtype" の型: float16 か、次元ごとに
              meta の "scale" を持つ int8（値に尺度を掛ける）
  relations   {"id@version": {"meta": {...}, "members": {hash: (member, ...)}, "pairs": [(a, b, score), ...]}}:
              関係のビュー（契約2、ビューの種類）: マップの各版が何に属するか（アーティスト ID）と、メンバーの対の点
              （向き付き、並べたもの）。行はいつも "maps"
  calibration 較正の JSON（配列ではなく list）
  presets     マニフェストに入れる手札、または None。差分はいつも一覧全体を運ぶ（なければ null）ので、当てると
              新しい版の手札になる
  notice      DB の条件とクレジットの 1 つの Markdown の文（マニフェストの `notice`）、または None。差分は手札と同じく
              運ぶ。1 つのビューが出所に負うものは、そのビューの `credits`（meta にある素の文の行の list）なので、
              ビューはどこへ行ってもクレジットを持つ
  chart_metrics  マニフェストの chartMetrics（charts.parquet の指標の列をどの計算が埋めたか）
  extra_cols  {"maps": [field], "charts": [field]}: この版が知らない、その表の列。値は各行で知っている列の後に続く
  extra       この版が知らないマニフェストの欄
  schema      書く schemaVersion。この版のものでないとき（新しいマイナー版の差分を当てた）
行は hash の順に書くので、行の番号は中身だけで決まる。"charts" のビューの行の番号は charts.parquet を、"maps" の
ビューのものは maps.parquet を指す。関係のビューはベクトルの代わりにメンバーの表 (row, member) と対の表
(a, b, score) を持ち、較正しない。

差分（マニフェストの "kind": "diff"）は、足したか変えた行（changed.parquet が挙げ、maps.parquet、charts.parquet、
各ビューのベクトルが運ぶ）、出ていった hash（deleted.parquet）、superseded の表全体（小さく、置き換えるほうが継ぎ
当てるより簡単）を持つ。ビューを足す（その全部の行: ビューのパック）、同じエンコーダーの別の版で置き換える、外す
こともできる（契約2、差分更新）。関係のビューは、メンバーがマップとともに変わり、小さいので、どの差分でも丸ごと
運ぶ（"change": "added"。base の同じ id のビューを置き換える）。だから関係のビューを持つ差分は、マップのすべての行を
運ぶ。差分を作るのは、両方の版にあるどのビューも fingerprint と較正の統計を保ち、値の正規化が同じ（較正の正規化の
版）ときだけ。そうでなければ全量が要る。

「前の全量 + 差分」は、DB がこの OctantVector より新しくても新しい全量にならなければならない（OctantSpecs 法 2.4）
ので、この版の知らないものはそのまま運ぶ: maps.parquet と charts.parquet の列、マニフェストの欄（差分の
マニフェストは、2 つで違う FIELDS を除き、全量の持つすべての欄を運ぶ）、ビューごとの欄。運べないものを黙って
落とすことはしない: 行を知らないビュー（番号を振り直せない）、知らないファイルや superseded.parquet の列、base だけに
あるマニフェストの欄、base と差分で知らない列が違う表があれば、apply_diff は断り、呼ぶ側は代わりに全量を取る。
前の版がこの版の知らないものを持つリリースは全量になる（dbrelease.py）。
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import charts as chartids

SCHEMA_VERSION = "2.0"  # 2.0: int8 のビュー（meta の "dtype"、"scale"）。1.x の読み手は読み違える
INDEX_VERSION = "1.1"  # 契約5（OctantHarvest to OctantCompass）の索引（indexVersion）
MAP_COLS = (("artist", "string"), ("mapper", "string"), ("duration_s", "float64"), ("published_at", "string"))
CHART_COLS = (("characteristic", "string"), ("difficulty", "string"), ("nps", "float64"),
              ("njs", "float64"), ("pass", "float64"), ("tech", "float64"), ("play_style", "string"),
              ("njs_play", "float64"), ("jd", "float64"), ("reaction_s", "float64"),
              ("nps_peak", "float64"))
ROW_KINDS = ("maps", "charts")  # ビューの行が指せる表
FIELDS = ("schemaVersion", "kind", "dbVersion", "builtAt", "baseDbVersion", "diffsSinceFull", "calibrationId",
          "calibrationFile", "counts", "views", "removedViews", "chartMetrics", "presets", "notice", "files")  # 知っているもの
VIEW_FILE_KEYS = ("file", "rowsFile", "membersFile", "pairsFile", "change", "replaces")  # ビューの項目のデータがどこにあるかを示すキー
# （ビューが何かは示さない）
VIEW_KINDS = ("vector", "relation")  # 契約2 ビューの種類。"kind" のないビューはベクトルのビュー


DTYPES = ("float16", "int8")  # ベクトルのビューの "dtype"（契約2）。int8 のビューは "scale" も持つ


def vector_dtype(v):
    """ベクトルのビューの値の numpy の型。この版が使えない dtype か int8 の尺度なら FormatError。"""
    dtype = v.get("dtype", "float16")
    if dtype not in DTYPES:
        raise FormatError(f"vector dtype {dtype!r} is not one of {', '.join(DTYPES)}")
    if dtype == "int8":
        sc = v.get("scale")
        if not isinstance(sc, list) or len(sc) != v.get("dim") or not all(isinstance(x, (int, float)) and x > 0 for x in sc):
            raise FormatError("an int8 view needs a positive scale for each of its dimensions")
    return np.dtype(dtype)


def view_kind(v):
    return v.get("kind", "vector")


class FormatError(Exception):
    pass


class VersionError(FormatError):
    """ファイルが、この OctantVector の読めない形式の版を宣言している（OctantSpecs contracts/、契約の版）。
    `what` は形式、`found` は宣言された版、`reads` はこの OctantVector が読む版。"""

    def __init__(self, what, found, reads, where=""):
        self.what, self.found, self.reads = what, found, reads
        super().__init__(f"{where}{': ' if where else ''}{what} {found!r} (this OctantVector reads {reads})")


def major(version):
    return int(str(version).split(".")[0])


def sha256(p):
    with open(p, "rb") as f:
        return "sha256:" + hashlib.file_digest(f, "sha256").hexdigest()


def empty():
    return {"maps": {}, "charts": {}, "superseded": {}, "views": {}, "relations": {}, "calibration": None, "presets": None,
            "notice": None, "chart_metrics": {}, "extra_cols": {"maps": [], "charts": []}, "extra": {}}


def row_keys(content, hashes, rows):
    """ビューのあり得る行のキー。ファイルの順: hash（"maps"）か、その譜面のキー（"charts"）。"""
    if rows == "charts":
        return [chartids.key(h, r[0], r[1]) for h in hashes for r in content["charts"].get(h, [])]
    return list(hashes)


def view_keys_of(content, h, rows):
    return row_keys(content, [h], rows)


# ---- 書く ------------------------------------------------------------------------------------------------------

def _table(cols, rows, hashes, extra=()):
    import pyarrow as pa
    schema = pa.schema([("hash", pa.string())] + [(c, getattr(pa, t)()) for c, t in cols] + list(extra))
    data = [[h] + list(r) for h, r in zip(hashes, rows)]
    return pa.table({n: [d[i] for d in data] for i, n in enumerate(schema.names)}, schema=schema)


def _write_rows(tmp, content, hashes):
    """`hashes`（並べたもの）の maps、charts、ビューのファイル。-> ビューごとのマニフェストの項目。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    extra = content.get("extra_cols", {})
    pq.write_table(_table(MAP_COLS, [content["maps"][h] for h in hashes], hashes, extra.get("maps", ())),
                   tmp / "maps.parquet")
    ch = [(h, r) for h in hashes for r in content["charts"].get(h, [])]
    pq.write_table(_table(CHART_COLS, [r for _, r in ch], [h for h, _ in ch], extra.get("charts", ())),
                   tmp / "charts.parquet")
    sup = sorted(content["superseded"].items())
    pq.write_table(pa.table({"hash": [a for a, _ in sup], "current": [b for _, b in sup]},
                            schema=pa.schema([("hash", pa.string()), ("current", pa.string())])),
                   tmp / "superseded.parquet")
    (tmp / "views").mkdir()
    views = {}
    keys_of = {kind: row_keys(content, hashes, kind) for kind in ROW_KINDS}
    for key, v in content["views"].items():
        vecs = v["vecs"]
        keys = keys_of[v["meta"].get("rows", "maps")]
        rows = np.array([n for n, k in enumerate(keys) if k in vecs], np.int32)
        dim = v["meta"]["dim"]
        dtype = vector_dtype(v["meta"])
        x = np.stack([vecs[keys[n]] for n in rows]).astype(dtype) if len(rows) else np.zeros((0, dim), dtype)
        np.save(tmp / "views" / f"{key}.npy", x)
        np.save(tmp / "views" / f"{key}.rows.npy", rows)
        views[key] = {**v["meta"], "file": f"views/{key}.npy", "rowsFile": f"views/{key}.rows.npy"}
    pos = {h: n for n, h in enumerate(hashes)}
    for key, r in content.get("relations", {}).items():
        members = sorted((pos[h], m) for h, ms in r["members"].items() if h in pos for m in ms)
        pq.write_table(pa.table({"row": pa.array([n for n, _ in members], pa.int32()),
                                 "member": pa.array([m for _, m in members], pa.string())}),
                       tmp / "views" / f"{key}.members.parquet")
        pairs = sorted(r["pairs"])
        pq.write_table(pa.table({"a": pa.array([a for a, _, _ in pairs], pa.string()),
                                 "b": pa.array([b for _, b, _ in pairs], pa.string()),
                                 "score": pa.array([x for _, _, x in pairs], pa.float32())}),
                       tmp / "views" / f"{key}.pairs.parquet")
        views[key] = {**r["meta"], "kind": "relation", "rows": "maps",
                      "membersFile": f"views/{key}.members.parquet", "pairsFile": f"views/{key}.pairs.parquet"}
    return views


def _finish(tmp, out, manifest):
    manifest["files"] = {str(p.relative_to(tmp)).replace("\\", "/"): sha256(p)
                         for p in sorted(tmp.rglob("*")) if p.is_file()}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.rename(out)  # フォルダはそろってから現れる
    return out


def _start(out):
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FormatError(f"{tmp} is left from an interrupted write; remove it first")
    tmp.mkdir(parents=True)
    return out, tmp


def write_full(out, content, db_version, built_at, diffs_since_full=0):
    """全量の DB。diffs_since_full: 最後に出した全量からこの中身までの差分の数（全量なら 0）。"""
    out, tmp = _start(out)
    hashes = sorted(content["maps"])
    views = _write_rows(tmp, content, hashes)
    cal = content["calibration"]
    cal_file = f"calibration/{cal['id']}.json"
    (tmp / "calibration").mkdir()
    (tmp / cal_file).write_text(json.dumps(cal))
    manifest = {"schemaVersion": content.get("schema") or SCHEMA_VERSION, "kind": "full", "dbVersion": db_version,
                "builtAt": built_at, "baseDbVersion": None, "diffsSinceFull": diffs_since_full,
                "calibrationId": cal["id"], "calibrationFile": cal_file,
                "counts": {"maps": len(hashes), "charts": sum(len(content["charts"].get(h, [])) for h in hashes),
                           "superseded": len(content["superseded"])},
                "views": views, "chartMetrics": content.get("chart_metrics") or {}}
    if content.get("presets"):
        manifest["presets"] = content["presets"]
    if content.get("notice"):
        manifest["notice"] = content["notice"]
    manifest.update(content.get("extra") or {})
    return _finish(tmp, out, manifest)


def normalisation(cal):
    """較正を作ったときの正規化の版（generation/normalisation.py）。最初の 1 は書かないので、版ができる前に作った較正は
    id が変わらない。"""
    return cal.get("normalisation", 1)


def diffable(base_manifest, base_cal, content):
    """base から content への差分が作れない理由（作れれば None）: 較正の正規化の版が同じで、両方にあるどのビューも
    fingerprint と較正の統計を保つこと。ビューは足す、別の版で置き換える、外すことができる。"""
    new_cal = content["calibration"]
    if normalisation(base_cal) != normalisation(new_cal):
        return f"the normalisation changed (version {normalisation(base_cal)} to {normalisation(new_cal)})"
    for key, v in content["views"].items():
        b = base_manifest.get("views", {}).get(key)
        if b is None:
            continue
        if b.get("fingerprint") != v["meta"]["fingerprint"]:
            return f"{key} has another fingerprint"
        if base_cal["views"].get(key) != new_cal["views"].get(key):
            return f"the calibration of {key} changed"
    return None


def view_changes(base_keys, new_keys):
    """-> ({足したキー: それが置き換えるキーか None}, [外したキー])。キーは同じエンコーダーの id の base のキーを置き換える。"""
    by_id = {k.split("@")[0]: k for k in base_keys if k not in new_keys}
    added = {k: by_id.get(k.split("@")[0]) for k in new_keys if k not in base_keys}
    removed = sorted(k for k in base_keys if k not in new_keys and k not in added.values())
    return added, removed


def changes(base, new):
    """-> (足した, 置き換えた, 消した) hash。行は両方の中身にあるビューで比べる。"""
    shared = [k for k in new["views"] if k in base["views"]]

    def same(h):
        if base["maps"][h] != new["maps"][h] or base["charts"].get(h, []) != new["charts"].get(h, []):
            return False
        for key in shared:
            for k in view_keys_of(new, h, new["views"][key]["meta"].get("rows", "maps")):
                a, b = base["views"][key]["vecs"].get(k), new["views"][key]["vecs"].get(k)
                if (a is None) != (b is None) or (a is not None and a.tobytes() != np.asarray(b, a.dtype).tobytes()):
                    return False
        return True
    added = sorted(set(new["maps"]) - set(base["maps"]))
    deleted = sorted(set(base["maps"]) - set(new["maps"]))
    replaced = sorted(h for h in set(new["maps"]) & set(base["maps"]) if not same(h))
    return added, replaced, deleted


def write_diff(out, base, base_manifest, new, db_version, built_at):
    """足したか変えた行、消した hash、そして足した（その全部の行: ビューのパック）、同じエンコーダーの別の版で置き換えた、
    外したビュー。新しい較正（古いものに足したビューの統計を加えたもの）を差分と一緒に運ぶ。"""
    why = diffable(base_manifest, base["calibration"], new)
    if why:
        raise FormatError(f"no diff from {base_manifest['dbVersion']}: {why}; a full is needed")
    added, replaced, deleted = changes(base, new)
    new_views, removed_views = view_changes(list(base["views"]), list(new["views"]))
    new_rel, removed_rel = view_changes(list(base.get("relations", {})), list(new.get("relations", {})))
    new_rel.update({k: None for k in new.get("relations", {}) if k not in new_rel})  # 残す関係も丸ごと運ぶ
    removed_views = sorted(removed_views + removed_rel)
    out, tmp = _start(out)
    changed = sorted(added + replaced)
    # 新しいビューと、どの関係のビューも、持つ行のすべてが要る
    rows = sorted(set(changed) | (set(new["maps"]) if new_views or new_rel else set()))
    rows_set, changed_set = set(rows), set(changed)
    sub = {**new, "maps": {h: new["maps"][h] for h in rows},
           "views": {k: {**v, "vecs": {h: x for h, x in v["vecs"].items()
                                       if chartids.key_hash(h) in (rows_set if k in new_views else changed_set)}}
                     for k, v in new["views"].items()}}
    views = _write_rows(tmp, sub, rows)
    for k, v in views.items():
        v["change"] = "added" if k in new_views or k in new_rel else "rows"
        if new_views.get(k) or new_rel.get(k):
            v["replaces"] = new_views.get(k) or new_rel[k]
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"hash": pa.array(deleted, pa.string())}), tmp / "deleted.parquet")
    pq.write_table(pa.table({"hash": pa.array(changed, pa.string())}), tmp / "changed.parquet")
    cal = new["calibration"]
    manifest = {"schemaVersion": new.get("schema") or SCHEMA_VERSION, "kind": "diff", "dbVersion": db_version,
                "builtAt": built_at, "baseDbVersion": base_manifest["dbVersion"], "calibrationId": cal["id"],
                "counts": {"added": len(added), "replaced": len(replaced), "deleted": len(deleted),
                           "superseded": len(new["superseded"])},
                "views": views, "removedViews": removed_views, "chartMetrics": new.get("chart_metrics") or {},
                "presets": new.get("presets") or None, "notice": new.get("notice") or None, **(new.get("extra") or {})}
    if cal["id"] != base["calibration"]["id"]:
        manifest["calibrationFile"] = f"calibration/{cal['id']}.json"
        (tmp / "calibration").mkdir()
        (tmp / manifest["calibrationFile"]).write_text(json.dumps(cal))
    return _finish(tmp, out, manifest)


# ---- 読む ------------------------------------------------------------------------------------------------------

def read_manifest(path, kind=None, *, check_files=True):
    """path にある DB の版のマニフェスト。schemaVersion のメジャーがこの版のものでないか、`kind` でなければ断る。
    check_files: 挙げたどのファイルも SHA-256 が合ってそこにあるかも確かめる（False は、版が何を言うかを使わずに
    見せるときだけ: read_info）。"""
    path = Path(path)
    try:
        m = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError) as ex:
        raise FormatError(f"{path}: no readable manifest.json ({ex})") from None
    try:
        found = major(m["schemaVersion"])
    except (KeyError, ValueError):
        raise FormatError(f"{path}: manifest has no valid schemaVersion") from None
    if found != major(SCHEMA_VERSION):
        raise VersionError("schemaVersion", m["schemaVersion"], f"{major(SCHEMA_VERSION)}.x", where=str(path))
    if kind and m.get("kind", "full") != kind:
        raise FormatError(f"{path} is a {m.get('kind')}, not a {kind}")
    for name, digest in m.get("files", {}).items() if check_files else ():
        p = path / name
        if not p.is_file():
            raise FormatError(f"{path}: {name} is missing")
        if sha256(p) != digest:
            raise FormatError(f"{path}: {name} does not match the manifest")
    return m


def listed(path, m, name):
    if name not in m.get("files", {}):
        raise FormatError(f"{path}: {name} is not listed in the manifest's files")
    return Path(path) / name


def read_view(path, m, key, n_rows):
    """保存したままのビューのベクトルとその行の番号。表の n_rows 行と照らして確かめる。"""
    v = m["views"][key]
    x = np.load(listed(path, m, v["file"]))
    rows = np.load(listed(path, m, v["rowsFile"]))
    if len(x) != len(rows) or (len(rows) and (rows.min() < 0 or rows.max() >= n_rows)):
        raise FormatError(f"{path}: {key} rows do not match its vectors or the {v['rows']} table")
    try:
        dtype = vector_dtype(v)
    except FormatError as ex:
        raise FormatError(f"{path}: {key}: {ex}") from None
    if x.dtype != dtype or x.ndim != 2 or x.shape[1] != v.get("dim"):
        raise FormatError(f"{path}: {key} holds {x.dtype} {x.shape}, the manifest says {dtype} with {v.get('dim')} "
                          "dimensions")
    return x, rows


def read_relation(path, m, key, n_rows):
    """関係のビューのメンバー（{"row", "member"} の列。行は n_rows のマップの行と照らして確かめる）と対
    （{"a", "b", "score"}）。"""
    import pyarrow.parquet as pq
    v = m["views"][key]
    members = pq.read_table(listed(path, m, v["membersFile"]), columns=["row", "member"]).to_pydict()
    pairs = pq.read_table(listed(path, m, v["pairsFile"]), columns=["a", "b", "score"]).to_pydict()
    rows = members["row"]
    if rows and (min(rows) < 0 or max(rows) >= n_rows):
        raise FormatError(f"{path}: {key} members point outside the maps table")
    return members, pairs


def read_hashes(path, m):
    """maps.parquet のマップの hash。行の順。"""
    import pyarrow.parquet as pq
    return pq.read_table(listed(path, m, "maps.parquet"), columns=["hash"]).column("hash").to_pylist()


def read_superseded(path, m):
    """superseded.parquet を {旧い版の hash: 現行版の hash} にしたもの。"""
    import pyarrow.parquet as pq
    sup = pq.read_table(listed(path, m, "superseded.parquet"), columns=["hash", "current"]).to_pydict()
    return dict(zip(sup["hash"], sup["current"]))


def chart_keys(ch):
    """列として読んだ charts.parquet の各行の譜面のキー（read_charts）。"""
    return [chartids.key(h, c, d) for h, c, d in zip(ch["hash"], ch["characteristic"], ch["difficulty"])]


def read_relations(path, m=None):
    """全量の DB の関係のビューだけ。read_content と同じ形（アーティストのファイルが渡されないとき、リリースはこれを残す）。"""
    m = m or read_manifest(path)
    hashes = read_hashes(path, m)
    out = {}
    for key, v in m.get("views", {}).items():
        if view_kind(v) != "relation" or v.get("rows") != "maps":
            continue
        out[key] = _relation_content(path, m, key, hashes)
    return out


def _relation_content(path, m, key, hashes):
    members, pairs = read_relation(path, m, key, len(hashes))
    ms = {}
    for r, member in zip(members["row"], members["member"]):
        ms.setdefault(hashes[r], []).append(member)
    return {"meta": {k: a for k, a in m["views"][key].items() if k not in VIEW_FILE_KEYS + ("kind",)},
            "members": {h: tuple(x) for h, x in ms.items()},
            "pairs": list(zip(pairs["a"], pairs["b"], pairs["score"]))}


def _columns(p, cols):
    """列にした表と、この版が知らない列の欄。ファイルの順。"""
    import pyarrow.parquet as pq
    t = pq.read_table(p)
    known = {"hash"} | {c for c, _ in cols}
    return t.to_pydict(), [f for f in t.schema if f.name not in known]


def _files(m):
    """マニフェストのファイルのうち、この版が扱い方を知っているもの。"""
    names = {"maps.parquet", "charts.parquet", "superseded.parquet", m.get("calibrationFile")}
    names |= {v[k] for v in m.get("views", {}).values() for k in ("file", "rowsFile", "membersFile", "pairsFile")
              if k in v and view_kind(v) in VIEW_KINDS}
    return names | ({"deleted.parquet", "changed.parquet"} if m.get("kind") == "diff" else set())


def read_content(path, m=None):
    """全量の DB か差分の中身（先に確かめる）。この版が運べないものは外し、"unread" に (何か, 名前) で挙げる: 行を知らない
    ビュー、知らないファイル、superseded.parquet の列。"""
    m = m or read_manifest(path)
    c = empty()
    c["unread"] = [("file", n) for n in sorted(set(m.get("files", {})) - _files(m))]
    maps, c["extra_cols"]["maps"] = _columns(listed(path, m, "maps.parquet"), MAP_COLS)
    hashes = maps["hash"]
    names = [k for k, _ in MAP_COLS] + [f.name for f in c["extra_cols"]["maps"]]
    c["maps"] = {h: tuple(maps[k][n] for k in names) for n, h in enumerate(hashes)}
    ch, c["extra_cols"]["charts"] = _chart_columns(listed(path, m, "charts.parquet"))
    names = [k for k, _ in CHART_COLS] + [f.name for f in c["extra_cols"]["charts"]]
    for n, h in enumerate(ch["hash"]):
        c["charts"].setdefault(h, []).append(tuple(ch[k][n] for k in names))
    sup, extra = _columns(listed(path, m, "superseded.parquet"), (("current", "string"),))
    c["superseded"] = dict(zip(sup["hash"], sup["current"]))
    c["unread"] += [("column", f"superseded.parquet {f.name}") for f in extra]
    keys_of = {"maps": hashes, "charts": chart_keys(ch)}
    for key, v in m.get("views", {}).items():
        if not _readable(v):
            c["unread"].append(("view", key))
            continue
        if view_kind(v) == "relation":
            c["relations"][key] = _relation_content(path, m, key, hashes)
            continue
        keys = keys_of[v["rows"]]
        x, rows = read_view(path, m, key, len(keys))
        c["views"][key] = {"meta": {k: a for k, a in v.items() if k not in VIEW_FILE_KEYS},
                           "vecs": {keys[int(r)]: x[n] for n, r in enumerate(rows)}}
    if m.get("calibrationFile"):
        c["calibration"] = json.loads(listed(path, m, m["calibrationFile"]).read_text())
    c["presets"] = m.get("presets")
    c["notice"] = m.get("notice")
    c["chart_metrics"] = m.get("chartMetrics") or {}
    c["extra"] = {k: v for k, v in m.items() if k not in FIELDS}
    return c


def unknown(content):
    """read_content が読んだ中身のうち、この OctantVector が知らないもの。運べるかによらず（説明）。"""
    return ([f"{what} {name}" for what, name in content.get("unread", [])]
            + [f"column {table}.parquet {f.name}" for table, fields in content["extra_cols"].items() for f in fields]
            + [f"manifest field {k}" for k in content["extra"]])


def _chart_columns(path):
    ch, extra = _columns(path, CHART_COLS)
    missing = [c for c, _ in CHART_COLS if c not in ch]
    if missing:  # 列が足りない DB（0.14.0 より前に作った DB）は読まない
        raise FormatError(f"{path}: no column {', '.join(missing)} (a DB built before OctantVector 0.14.0; "
                          "get a DB built for this version)")
    return ch, extra


def read_charts(path):
    """列にした charts.parquet。"""
    return _chart_columns(path)[0]


def _newer(a, b):
    try:
        return max(a, b, key=lambda v: tuple(int(x) for x in str(v).split(".")))
    except ValueError:
        raise FormatError(f"cannot compare schemaVersion {a!r} with {b!r}") from None


def _readable(v):
    """この版がマニフェストのビューの項目を読めるか（read_content はほかを unread として外す）。"""
    if v.get("rows") not in ROW_KINDS or view_kind(v) not in VIEW_KINDS:
        return False
    return view_kind(v) != "relation" or v["rows"] == "maps"


def _schema_extra(p, cols):
    """この版が知らない表の列の欄。ファイルの順で、スキーマだけから読む。"""
    import pyarrow.parquet as pq
    known = {"hash"} | {c for c, _ in cols}
    return [f for f in pq.read_schema(p) if f.name not in known]


def _cannot_carry(base_path, bm, diff_path, dm):
    """apply_diff が失うもの（説明）。マニフェストと表のスキーマから: 両方の read_content を比べるのと同じ断り方。"""
    extra = {}
    for path, m in ((base_path, bm), (diff_path, dm)):
        extra[path] = {t: _schema_extra(listed(path, m, f"{t}.parquet"), MAP_COLS if t == "maps" else CHART_COLS)
                       for t in ROW_KINDS}
    unread_files = lambda m: sorted(set(m.get("files", {})) - _files(m))  # noqa: E731
    sup = _schema_extra(listed(diff_path, dm, "superseded.parquet"), (("current", "string"),))
    unread = [("file", n) for n in unread_files(dm)] + [("view", k) for k, v in dm.get("views", {}).items()
                                                         if not _readable(v)]
    unread += [("column", f"superseded.parquet {f.name}") for f in sup]
    base_extra = [k for k in bm if k not in FIELDS]
    vector = lambda m: {k for k, v in m.get("views", {}).items() if _readable(v) and view_kind(v) == "vector"}  # noqa: E731
    return ([f"{what} {name}" for what, name in unread]
            + [f"file {name} of the base" for name in unread_files(bm)]
            + [f"manifest field {k} of the base" for k in base_extra if k not in dm]
            + [f"the columns of {t}.parquet changed" for t in ROW_KINDS if extra[base_path][t] != extra[diff_path][t]]
            + [f"rows of view {k}, which the base lacks" for k in sorted(vector(dm) - vector(bm))
               if dm["views"][k].get("change") != "added"]), extra[diff_path]


def _merged_table(base_path, bm, diff_path, dm, name, cols, extra, keep_base, take_diff):
    """hash が keep_base を通る base の行と、hash が take_diff を通る差分の行を、hash の順に（安定なので、マップの譜面は
    カタログの順を保つ）、write_full と同じ列で。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    schema = pa.schema([("hash", pa.string())] + [(c, getattr(pa, t)()) for c, t in cols] + list(extra))
    parts = []
    for path, m, keep in ((base_path, bm, keep_base), (diff_path, dm, take_diff)):
        t = pq.read_table(listed(path, m, name))
        missing = [c for c in schema.names if c not in t.column_names]
        if missing:
            raise FormatError(f"{path}: no column {', '.join(missing)} in {name}")
        t = t.select(schema.names).cast(schema)
        parts.append(t.filter(keep(t.column("hash"))))
    t = pa.concat_tables(parts)
    return t.take(pc.sort_indices(t, [("hash", "ascending")]))


def _relation_tables(path, m, key, hashes, new_hashes):
    """マップ new_hashes（arrow の配列、並べたもの）について write_full が書くのと同じ形の、差分の関係のビュー: メンバー
    （new_hashes での行, member）を並べたものと、並べた対。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    v = m["views"][key]
    members = pq.read_table(listed(path, m, v["membersFile"]), columns=["row", "member"])
    pairs = pq.read_table(listed(path, m, v["pairsFile"]), columns=["a", "b", "score"])
    rows = members.column("row")
    if len(rows) and (pc.min(rows).as_py() < 0 or pc.max(rows).as_py() >= len(hashes)):
        raise FormatError(f"{path}: {key} members point outside the maps table")
    pos = pc.index_in(hashes.take(rows), value_set=new_hashes)
    keep = pc.is_valid(pos)
    members = pa.table({"row": pc.cast(pos.filter(keep), pa.int32()),
                        "member": pc.cast(members.column("member").filter(keep), pa.string())})
    members = members.take(pc.sort_indices(members, [("row", "ascending"), ("member", "ascending")]))
    pairs = pa.table({"a": pc.cast(pairs.column("a"), pa.string()), "b": pc.cast(pairs.column("b"), pa.string()),
                      "score": pc.cast(pairs.column("score"), pa.float32())})
    pairs = pairs.take(pc.sort_indices(pairs, [("a", "ascending"), ("b", "ascending"), ("score", "ascending")]))
    return members, pairs


def _row_key_array(t, rows):
    """maps か charts の表の行のキーを arrow の配列で: hash か、譜面のキー（charts.key）。"""
    import pyarrow.compute as pc
    if rows == "charts":
        return pc.binary_join_element_wise(t.column("hash"), t.column("characteristic"), t.column("difficulty"),
                                           chartids.SEP)
    return t.column("hash")


def apply_diff(base_path, diff_path, out):
    """base + 差分を、`out` に新しい全量の DB として書く（契約5: 別のフォルダで用意し、そろってから切り替える）。先に両方を
    確かめる。差分は base の dbVersion から始まること。この OctantVector が知らないものは運ぶ。運べなければ FormatError
    で、呼ぶ側は代わりに全量を取る。

    結果は、read_content が返す中身について write_full が書くもの（消しも変えもしない base の行、差分の変えた行、差分の
    ビュー、差分の関係のビューを丸ごと）と同じだが、表ごと、ビューごとに配列で計算するので、メモリにあるのは一度に
    1 つのビューのベクトルだけ。base の関係のビューは読まない。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    bm, dm = read_manifest(base_path, "full"), read_manifest(diff_path, "diff")
    if dm["baseDbVersion"] != bm["dbVersion"]:
        raise FormatError(f"the diff {dm['dbVersion']} starts from {dm['baseDbVersion']}, not {bm['dbVersion']}")
    cant, extra_cols = _cannot_carry(base_path, bm, diff_path, dm)
    if cant:
        raise FormatError(f"this OctantVector cannot apply the diff {dm['dbVersion']} without losing what it does not "
                          f"know ({'; '.join(cant)}); take the full of {dm['dbVersion']} instead")
    base_cal = json.loads(listed(base_path, bm, bm["calibrationFile"]).read_text())
    cal = json.loads(listed(diff_path, dm, dm["calibrationFile"]).read_text()) if dm.get("calibrationFile") else base_cal
    if cal["id"] != dm["calibrationId"]:
        raise FormatError(f"the diff {dm['dbVersion']} needs calibration {dm['calibrationId']}")
    vector_keys = [k for k, v in dm.get("views", {}).items() if view_kind(v) == "vector"]
    meta = {k: {a: x for a, x in dm["views"][k].items() if a not in VIEW_FILE_KEYS} for k in vector_keys}
    why = diffable(bm, base_cal, {"calibration": cal, "views": {k: {"meta": meta[k]} for k in vector_keys}})
    if why:
        raise FormatError(f"the diff does not fit the base: {why}")
    column = lambda name: pq.read_table(listed(diff_path, dm, name), columns=["hash"]).column("hash").combine_chunks()  # noqa: E731
    changed = pc.unique(column("changed.parquet"))
    gone = pc.unique(pa.concat_arrays([column("deleted.parquet"), changed]))
    keep_base = lambda h: pc.invert(pc.is_in(h, value_set=gone))  # noqa: E731
    take_diff = lambda h: pc.is_in(h, value_set=changed)  # noqa: E731
    out, tmp = _start(out)
    try:
        tables = {t: _merged_table(base_path, bm, diff_path, dm, f"{t}.parquet", MAP_COLS if t == "maps" else CHART_COLS,
                                   extra_cols[t], keep_base, take_diff) for t in ROW_KINDS}
        hashes = tables["maps"].column("hash").combine_chunks()
        charts = tables["charts"].filter(pc.is_in(tables["charts"].column("hash"), value_set=hashes))
        pq.write_table(tables["maps"], tmp / "maps.parquet")
        pq.write_table(charts, tmp / "charts.parquet")
        n_charts = len(charts)
        new_keys = {"maps": hashes, "charts": _row_key_array(charts, "charts").combine_chunks()}
        del tables, charts
        sup = pq.read_table(listed(diff_path, dm, "superseded.parquet"), columns=["hash", "current"])
        sup = sup.cast(pa.schema([("hash", pa.string()), ("current", pa.string())]))
        sup = sup.take(pc.sort_indices(sup, [("hash", "ascending")]))
        pq.write_table(sup, tmp / "superseded.parquet")
        (tmp / "views").mkdir()
        sides = {}
        for side, path, m in (("base", base_path, bm), ("diff", diff_path, dm)):
            cols = {"maps": ["hash"], "charts": ["hash", "characteristic", "difficulty"]}
            sides[side] = (path, m, {t: _row_key_array(pq.read_table(listed(path, m, f"{t}.parquet"), columns=c), t)
                                     .combine_chunks() for t, c in cols.items()})
        views = {}
        for key in vector_keys:
            v = dm["views"][key]
            rows_kind, dtype = v["rows"], vector_dtype(v)
            sources = [("diff", None)] if v.get("change") == "added" else [("diff", None), ("base", keep_base)]
            xs, ks = [], []
            for side, keep in sources:
                path, m, keys_of = sides[side]
                keys = keys_of[rows_kind]
                x, rows = read_view(path, m, key, len(keys))
                k = keys.take(pa.array(rows, pa.int32()))
                if keep is not None:
                    sel = keep(k if rows_kind == "maps" else  # 譜面のキーのマップ: chartids.key_hash
                               pc.list_element(pc.split_pattern(k, chartids.SEP, max_splits=1), 0))
                    sel = np.asarray(sel.to_numpy(zero_copy_only=False), bool)
                    x, k = x[sel], k.filter(pa.array(sel))
                xs.append(x)
                ks.append(k)
            cand = pa.concat_arrays(ks)
            idx = pc.index_in(new_keys[rows_kind], value_set=cand)  # 差分の行を先に: 差分が勝つ
            found = np.asarray(pc.is_valid(idx).to_numpy(zero_copy_only=False), bool)
            rows = np.flatnonzero(found).astype(np.int32)
            pick = np.asarray(idx.filter(pa.array(found)).to_numpy(zero_copy_only=False), np.int64)
            x = np.empty((len(rows), v["dim"]), dtype)
            start = 0
            for part in xs:
                sel = (pick >= start) & (pick < start + len(part))
                x[sel] = part[pick[sel] - start].astype(dtype, copy=False)
                start += len(part)
            del xs, ks, cand, idx, pick
            np.save(tmp / "views" / f"{key}.npy", x)
            np.save(tmp / "views" / f"{key}.rows.npy", rows)
            del x
            views[key] = {**meta[key], "file": f"views/{key}.npy", "rowsFile": f"views/{key}.rows.npy"}
        diff_hashes = sides["diff"][2]["maps"]
        for key, v in dm.get("views", {}).items():
            if view_kind(v) != "relation":
                continue
            members, pairs = _relation_tables(diff_path, dm, key, diff_hashes, hashes)
            pq.write_table(members, tmp / "views" / f"{key}.members.parquet")
            pq.write_table(pairs, tmp / "views" / f"{key}.pairs.parquet")
            views[key] = {**{a: x for a, x in v.items() if a not in VIEW_FILE_KEYS + ("kind",)}, "kind": "relation",
                          "rows": "maps", "membersFile": f"views/{key}.members.parquet",
                          "pairsFile": f"views/{key}.pairs.parquet"}
        cal_file = f"calibration/{cal['id']}.json"
        (tmp / "calibration").mkdir()
        (tmp / cal_file).write_text(json.dumps(cal))
        presets = dm.get("presets") if "presets" in dm else bm.get("presets")
        notice = dm.get("notice") if "notice" in dm else bm.get("notice")
        manifest = {"schemaVersion": _newer(SCHEMA_VERSION, dm["schemaVersion"]), "kind": "full",
                    "dbVersion": dm["dbVersion"], "builtAt": dm["builtAt"], "baseDbVersion": None,
                    "diffsSinceFull": bm.get("diffsSinceFull", 0) + 1, "calibrationId": cal["id"],
                    "calibrationFile": cal_file,
                    "counts": {"maps": len(hashes), "charts": n_charts, "superseded": len(sup)},
                    "views": views, "chartMetrics": dm.get("chartMetrics") or {}}
        if presets:
            manifest["presets"] = presets
        if notice:
            manifest["notice"] = notice
        manifest.update({k: x for k, x in dm.items() if k not in FIELDS})
        return _finish(tmp, out, manifest)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
