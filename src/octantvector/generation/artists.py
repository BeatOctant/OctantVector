"""関係のビュー listenbrainz_artists@<YYYYMMDD>（fetchedAt の日付）、役割 `artist`（OctantSpecs contracts/、
契約2（DB build to recommendation core）ビューの種類）。クロール（契約4（OctantHarvest crawl to OctantVector））の
フォルダ `artists` にある、なくてもよい 2 つのファイルから作る:
  map_artists.jsonl.gz      {hash, mbids}: マップの版の MusicBrainz のアーティスト ID（名前は持たない: OctantSpecs 法 3.2）
  similar_artists.jsonl.gz  見出し {algorithm, fetchedAt}、続いて ListenBrainz の
                            {mbid, similar: [[mbid, score], ...]}
メンバーは DB にあるマップの ID。対の点は ListenBrainz の点をその一覧の最大の点で割ったもので、各 ID の一番近い
相手が 1 になる。2 つの ID がどちらもメンバーの対だけを、ID ごとに `per_member` まで（よいものから）残す。
ビューの `credits` には両方の出所を書く。
"""
from pathlib import Path

from .catalogue import read_jsonl

MAP_FILE, SIMILAR_FILE = "map_artists.jsonl.gz", "similar_artists.jsonl.gz"
PER_MEMBER = 30
CREDITS = ("MusicBrainz artist IDs, https://musicbrainz.org/, core data under CC0",
           "ListenBrainz similar artists, https://listenbrainz.org/, data under CC0 (checked 2026-10-17)")


def relation(folder, hashes, per_member=PER_MEMBER):
    """マップ `hashes` の (ビューのキー, common/dbformat.py の形の関係の中身)。フォルダにファイルがなければ None。"""
    folder = Path(folder)
    if not (folder / MAP_FILE).is_file() or not (folder / SIMILAR_FILE).is_file():
        return None
    want = {h.lower() for h in hashes}
    members = {}
    for r in read_jsonl(folder / MAP_FILE):
        h = str(r.get("hash", "")).lower()
        ids = tuple(sorted(set(r.get("mbids") or ())))
        if h in want and ids:
            members[h] = ids
    used = {m for ids in members.values() for m in ids}
    lines = read_jsonl(folder / SIMILAR_FILE)
    header = next(lines, {}) or {}
    pairs = []
    for r in lines:
        a = r.get("mbid")
        sims = [(b, float(s)) for b, s in r.get("similar") or () if b != a]
        if a not in used or not sims:
            continue
        top = max(s for _, s in sims)
        if top <= 0:
            continue
        kept = sorted(((b, s / top) for b, s in sims if b in used), key=lambda t: (-t[1], t[0]))[:per_member]
        pairs += [(a, b, s) for b, s in kept]
    algorithm = header.get("algorithm") or "unknown"
    fetched = header.get("fetchedAt")
    key = f"listenbrainz_artists@{(fetched or 'unknown')[:10].replace('-', '')}"
    meta = {"role": "artist", "rows": "maps", "source": f"listenbrainz:{algorithm}", "fetchedAt": fetched,
            "perMember": per_member, "fingerprint": f"listenbrainz:{algorithm}", "sources": [],
            "credits": list(CREDITS)}
    return key, {"meta": meta, "members": members, "pairs": sorted(pairs)}
