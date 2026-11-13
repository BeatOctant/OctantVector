"""BeatSaver のカタログのスナップショット（catalogue.jsonl.gz、1 行 1 マップ）-> 記録と表示用のメタデータ。"""
import gzip
import json
from pathlib import Path


def read_jsonl(path: Path):
    """クロールの gzip された JSON lines のファイル（カタログ、アーティストのファイル）の記録。空行は飛ばす。"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def published_version(m):
    return next((v for v in m["versions"] if v["state"] == "Published"), None)


def standard_diffs(v):
    return [d for d in v.get("diffs", []) if d["characteristic"] == "Standard"]


def records_for(catalogue: Path, hashes) -> dict:
    """指定した hash の 版の hash -> (マップの記録, 版の記録)（1 回のストリーミングで読む）。"""
    want, out = set(hashes), {}
    for m in read_jsonl(catalogue):
        for v in m["versions"]:
            if v["hash"] in want:
                out[v["hash"]] = (m, v)
    return out


def load_meta(catalogue: Path, cache: Path) -> dict:
    """カタログのすべての版の 版の hash -> 表示用のメタデータ。ストアの隣に JSON でキャッシュする。
    曲名とアーティストは表示と記録の照合にだけ使い、似ているかの特徴にはしない（OctantSpecs 法 3.2）。"""
    if cache.exists() and cache.stat().st_mtime >= Path(catalogue).stat().st_mtime:
        return json.loads(cache.read_text())
    out = {}
    for m in read_jsonl(catalogue):
        md = m["metadata"]
        for v in m["versions"]:
            out[v["hash"]] = {"key": m["id"], "song": md["songName"], "artist": md["songAuthorName"],
                              "mapper": md["levelAuthorName"], "bpm": md["bpm"], "up": m["stats"]["upvotes"],
                              "tags": m.get("tags", [])}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False))
    return out
