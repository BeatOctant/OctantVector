"""作業用ストアからの、次の配布 DB の版。全量か、前の版からの差分として作る（契約2（DB build to recommendation core）
と契約5（OctantHarvest to OctantCompass））。

行は台帳にある配布プールの現行版。旧い版は superseded.parquet に、同じマップの現行版を指す行だけを持つ。各ビューは
自分が持つ行を持ち（音声に失敗したマップは、そのビューに行がないだけ）、較正で正規化し（normalisation.py）、
エンコーダーの次元のまま float16 で、または較正にあるそのビューの export のとおりに（次元を減らす、int8:
export.py）持つ。譜面から計算したビューは譜面ごとに 1 行（"charts" の行）。マップごとに持つ譜面のビューは断る:
DB は各譜面のことをマップにまとめてはならないため（OctantSpecs constitution.md、原則 2）。

すべて、呼ぶ側が名指すフォルダ `out` に書く。承認とアップロードは呼ぶ側の仕事:
  <dbVersion>/       その版の完全な DB（いつも書く。次の差分の元）
  <dbVersion>.diff/  前の版 `base` からの差分。作れるとき
差分を作るのは、前の版にもあるどのビューも fingerprint と較正の統計を保ち、正規化の版が同じで、前の版がこの
OctantVector の知らないものを持たず（差分ではそれを失う: dbformat.py）、最後の全量から作った差分が `full_every`
より少ないとき。そうでなければ全量だけを出す。差分はビューを足し（ビューのパック: 新しく名指したビュー）、置き換え
（エンコーダーの版が上がった）、外せる（もう名指さない）。そのときの較正は、前の較正を新しいビューで広げたもの
（calibration.extend）。ファイルの形式は dbformat.py。

charts.parquet の Pass と Tech は、`swing_flow` がビューの 1 つかによらず、スイングのシミュレーション（`swing_flow`、
METRICS）の保存した値。値の保存されていない譜面（別のプレイスタイル、ノーツが少なすぎるか多すぎる、zip に失敗した）
は空のままにし、manifest.chartMetrics に各列を埋めたビューを書く。版には、承認されたときに後から署名する（api.sign）。
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import artists as artist_files, calibration as calibrations, catalogue, export, mappers
from ..common import charts, dbformat
from .encoders import get
from .errors import GenerationError
from .ledger import Ledger
from .normalisation import calibrated, normalise
from .store import Store


# charts.parquet の指標の列と、各列を同じ名前の保存した（正規化しない）特徴で埋めるビュー
METRICS = {"pass": "swing_flow", "tech": "swing_flow"}


def metric_views():
    """リリースが譜面の指標を読むビュー（compute は、呼ぶ側が何を名指してもこれを計算する）。"""
    return sorted(set(METRICS.values()))


def build(cfg, view_ids, calibration, log=sys.stderr, relations=()):
    """台帳、カタログ、ストアからの DB の中身（dbformat.py）。relations: カタログから作って入れる関係のビュー（mappers.NAME）。"""
    encs = {i: get(i) for i in view_ids}
    for i, e in encs.items():
        if e.role_name() == "chart" and e.row_kind() != "charts":
            raise GenerationError(f"{i} describes charts but stores one vector per map; only views with chart rows "
                                  "can go into the DB (OctantSpecs constitution.md, principle 2)")
    sts = {i: calibrated(calibration, e) for i, e in encs.items()}  # 正規化と定義が同じであること。違えば止める
    with Ledger.of(cfg.store) as led:
        hashes = sorted(led.pool_hashes("distribution"))
        old = led.old_versions()
    if not hashes:
        raise GenerationError("the distribution pool is empty; run `octantvector update` first")
    recs = catalogue.records_for(cfg.require("catalogue"), hashes)
    hashes = [h for h in hashes if h in recs]  # 台帳にあるが、このカタログのスナップショットにはもうない: 外す
    c = dbformat.empty()
    for h in hashes:
        m, v = recs[h]
        md = m.get("metadata", {})  # 曲名は入れない: 特徴にしないため
        c["maps"][h] = (md.get("songAuthorName"), md.get("levelAuthorName"), _num(md.get("duration")),
                        v.get("createdAt") or m.get("uploaded"))
        c["charts"][h] = [(d.get("characteristic"), d.get("difficulty"), _num(d.get("nps")),
                           _num(d.get("njs")), None, None, charts.play_style(d.get("characteristic")))
                          for d in v.get("diffs", [])]
    c["chart_metrics"] = _metrics(cfg, c["charts"], log)
    current_of_key = {recs[h][0]["id"]: h for h in hashes}
    c["superseded"] = {h: current_of_key[k] for h, k in old if k in current_of_key}
    want = set(hashes)
    chart_keys = {charts.key(h, r[0], r[1]) for h in hashes for r in c["charts"][h]}
    for i, e in encs.items():
        key = f"{e.id}@{e.version}"
        hs, X = Store(e, cfg.store).load()
        keep = [n for n, h in enumerate(hs) if charts.key_hash(h) in want]
        ex = sts[i].get("export")  # ビューをどう書くか（export.py）。None: エンコーダーが作ったまま、float16
        dim, dtype = (ex["dim"], ex["dtype"]) if ex else (e.dim, "float16")
        x = export.apply(normalise(e, X[keep], sts[i]), ex, cfg.store) if keep else np.zeros((0, dim), dtype)
        meta = {"role": e.role_name(), "rows": e.row_kind(), "dim": int(x.shape[1]), "dtype": dtype,
                "norm": e.norm, "fingerprint": e.fingerprint(),
                "sources": [f"{get(s).id}@{get(s).version}" for s in e.sources()]}
        if dim != e.dim:
            meta["encoderDim"] = e.dim
        if dtype == "int8":  # 値にその次元の尺度を掛けたものがベクトル（契約2）
            meta["scale"] = list(ex["scale"])
        if e.all_credits():
            meta["credits"] = e.all_credits()
        if meta["role"] == "genre":  # 1 次元に 1 語: 要求の `genre` が名指せる語（契約2）
            meta["genres"] = list(getattr(e, "names", ()))
        c["views"][key] = {"meta": meta,
                           "vecs": {charts.catalogue_spelling(hs[n]): x[j] for j, n in enumerate(keep)}}
        print(f"{key}: {len(keep)} of {len(hashes)} rows", file=log)
        if e.row_kind() == "charts":  # カタログにない譜面は charts.parquet に行を持たない
            unlisted = sorted(k for k in c["views"][key]["vecs"] if k not in chart_keys)
            if unlisted:
                print(f"{key}: {len(unlisted)} charts left out, their difficulty is not in the catalogue: "
                      f"{', '.join(unlisted[:20])}{' ...' if len(unlisted) > 20 else ''}", file=log)
    if cfg.artists is not None:
        rel = artist_files.relation(cfg.artists, hashes)
        if rel is None:
            print(f"{cfg.artists}: no {artist_files.MAP_FILE} and {artist_files.SIMILAR_FILE}; no artist view", file=log)
        else:
            key, c["relations"][key] = rel
            print(f"{key}: {len(rel[1]['members'])} of {len(hashes)} maps have artists, {len(rel[1]['pairs'])} pairs",
                  file=log)
    if mappers.NAME in relations:
        key, c["relations"][key] = mappers.relation(recs, hashes)
        print(f"{key}: {len(c['relations'][key]['members'])} of {len(hashes)} maps have accounts", file=log)
    c["calibration"] = calibrations.plain(calibration)
    return c


def _metrics(cfg, rows, log):
    """譜面の行 {hash: [row]} の METRICS の列を、保存したビューからその場で埋める。
    -> manifest.chartMetrics: {列: それを埋めたビューの id@version}。"""
    cols = {"pass": 4, "tech": 5}  # 譜面の行の中の位置（dbformat.py）
    filled = {}
    for i in metric_views():
        e = get(i)
        key = f"{e.id}@{e.version}"
        names = [m for m, v in METRICS.items() if v == i]
        if not (Path(cfg.store) / key).exists():
            print(f"{key}: not computed; {', '.join(names)} stay empty", file=log)
            continue
        hs, X = Store(e, cfg.store).load()
        at = [e.names.index(m) for m in names]
        got = {charts.catalogue_spelling(k): X[n, at] for n, k in enumerate(hs)}
        n = 0
        for h, rs in rows.items():
            for j, r in enumerate(rs):
                v = got.get(charts.key(h, r[0], r[1]))
                if v is None:
                    continue
                r = list(r)
                for m, x in zip(names, v):
                    r[cols[m]] = float(x)
                rs[j] = tuple(r)
                n += 1
        print(f"{key}: {', '.join(names)} of {n} of {sum(len(rs) for rs in rows.values())} charts", file=log)
        if n:
            filled.update({m: key for m in names})
    return filled


def latest(root):
    """`root` の下の一番新しい完全な全量の版のマニフェストとフォルダ（なければ None）。"""
    best = None
    for p in Path(root).glob("*/manifest.json"):
        if p.parent.name.endswith(".partial"):
            continue
        m = json.loads(p.read_text())
        if m.get("kind", "full") == "full" and (best is None or m["builtAt"] > best[0]["builtAt"]):
            best = (m, p.parent)
    return best


def release(cfg, view_ids, calibration=None, full=False, built_at=None, log=sys.stderr, out=None, base=...,
            presets=None, notice=None, relations=(), export=None):
    """ビュー `view_ids` と、カタログから作る関係のビュー `relations`（mappers.NAME）の次の版。`out`（既定
    <store>/releases）の下に書く。base: 前の全量の版のフォルダ。なければ None（既定: `out` の下の一番新しい全量）。
    calibration: 既定は base のものを、配布プールで足りないビューの分だけ広げたもの（<store>/calibration に保存）。
    presets: マニフェストに書く手札（list か、それを持つ JSON のファイルのパス）。None なら base のものを残す。残す
    ものも含めてどの手札も、推薦側がこの DB で出せるものであること。でなければリリースを止める。
    notice: マニフェストに書く DB の条件とクレジット（Markdown の文）。None なら base のものを残す。
    export: 各ビューをどう書くか（{view id: {"dim", "dtype"}}、export.py。名指さないビューはエンコーダーが作ったまま
    書く）。None なら較正にあるものを残す。変えると新しい較正になり（<store>/calibration に保存）、その版は全量になる。
    -> (全量のフォルダ, 差分のフォルダか None)。"""
    root = Path(out) if out is not None else Path(cfg.store) / "releases"
    if base is ...:
        prev = latest(root)
    elif base is not None:
        prev = (json.loads((Path(base) / "manifest.json").read_text()), Path(base))
        if prev[0].get("kind", "full") != "full":
            raise GenerationError(f"{base} is a diff; the base of a release is a full version")
    else:
        prev = None
    if not view_ids:
        raise GenerationError("name the views to put in the DB")
    if calibration is None:
        if prev is None:
            raise GenerationError("the first release needs a calibration")
        calibration = calibrations.load(prev[1] / prev[0]["calibrationFile"])
        with Ledger.of(cfg.store) as led:
            pool = led.pool_hashes("distribution")
        extended = calibrations.extend(calibration, cfg.store, view_ids, pool, "distribution")
        if extended is not calibration:
            print(f"calibration {extended['id']} = {calibration['id']} + new views; wrote "
                  f"{calibrations.save(extended, cfg.store)}", file=log)
            calibration = extended
    if export is not None:
        with Ledger.of(cfg.store) as led:
            pool = led.pool_hashes("distribution")
        made = calibrations.with_export(calibration, cfg.store, view_ids, export, pool, "distribution")
        if made is not calibration:
            print(f"calibration {made['id']} = {calibration['id']} with the export "
                  f"{json.dumps(export, sort_keys=True)}; wrote {calibrations.save(made, cfg.store)}", file=log)
            calibration = made
    content = build(cfg, view_ids, calibration, log, relations)
    artist = any(r["meta"].get("role") == "artist" for r in content["relations"].values())
    if not artist and prev is not None:  # アーティストのファイルを渡されなかった: 前の版のアーティストのビュー
        for key, r in dbformat.read_relations(prev[1], prev[0]).items():
            if str(r["meta"].get("source", "")).startswith("listenbrainz:"):
                r["meta"]["credits"] = list(artist_files.CREDITS)  # 今日のクレジットで
                content["relations"][key] = r
    if presets is None and prev is not None:
        presets = prev[0].get("presets")
    content["presets"] = _hands(presets, content) if presets is not None else None
    if notice is None and prev is not None:
        notice = prev[0].get("notice")
    if notice is not None and not (isinstance(notice, str) and notice.strip()):
        raise GenerationError("the notice must be a non-empty text")
    content["notice"] = notice

    built_at = built_at or datetime.now(timezone.utc).replace(microsecond=0)
    stamp = built_at.isoformat().replace("+00:00", "Z")
    version = built_at.strftime("%Y.%m.%d")
    n = 1
    while (root / version).exists() or (root / f"{version}.diff").exists():
        n += 1
        version = f"{built_at.strftime('%Y.%m.%d')}.{n}"

    base = dbformat.read_content(prev[1]) if prev and not full else None
    why = "--full" if full else "first release" if prev is None else dbformat.diffable(prev[0], base["calibration"], content)
    if why is None and dbformat.unknown(base):  # この OctantVector が書き直せないものは、差分では失われる
        why = f"the previous version holds what this OctantVector does not know: {'; '.join(dbformat.unknown(base))}"
    if why is None and prev[0].get("diffsSinceFull", 0) + 1 > cfg.full_every:
        why = f"{cfg.full_every} diffs since the last full (setting full_every)"
    diff = None
    if why is None:
        diff = dbformat.write_diff(root / f"{version}.diff", base, prev[0], content, version, stamp)
        dm = json.loads((diff / "manifest.json").read_text())
        packs = [k + (f" (replaces {v['replaces']})" if v.get("replaces") else "")
                 for k, v in dm["views"].items() if v.get("change") == "added"]
        print(f"wrote {diff}: diff from {prev[0]['dbVersion']}: " + ", ".join(f"{k} {v}" for k, v in dm["counts"].items())
              + (f"; views added {', '.join(packs)}" if packs else "")
              + (f"; views removed {', '.join(dm['removedViews'])}" if dm["removedViews"] else ""), file=log)
    out = dbformat.write_full(root / version, content, version, stamp,
                              0 if diff is None else prev[0].get("diffsSinceFull", 0) + 1)
    print(f"wrote {out}: {len(content['maps'])} maps, views {', '.join(list(content['views']) + list(content['relations']))}"
          + ("" if diff else f" (publish as a full: {why})"), file=log)
    return out, diff


def _hands(presets, content):
    """中身と照らした手札: 推薦側が外すものは、ここでリリースを止める。"""
    from ..common.hands import check_hands
    hands = json.loads(Path(presets).read_text()) if isinstance(presets, (str, Path)) else presets
    if not isinstance(hands, list):
        raise GenerationError("the presets must be a JSON list of hands")
    views = {k: v["meta"].get("role") for part in ("views", "relations") for k, v in content[part].items()}
    genres = {g for v in content["views"].values() for g in v["meta"].get("genres", [])}
    bad = check_hands(hands, views, genres)
    if bad:
        raise GenerationError("hands this DB cannot serve: " + "; ".join(f"{k}: {why}" for k, why in bad.items()))
    return hands or None


def _num(x):
    return None if x is None else float(x)

