"""カタログの記録だけから計算するエンコーダー（zip は開かない）: メタデータとタグ。"""
import numpy as np

from .base import Encoder, register


TAGS = ["accuracy", "balanced", "challenge", "dance-style", "fitness", "speed", "tech", "alternative", "ambient",
        "anime", "classical-orchestral", "comedy-meme", "dance", "drum-and-bass", "dubstep", "electronic", "folk-acoustic",
        "funk-disco", "hardcore", "hip-hop-rap", "holiday", "house", "indie", "instrumental", "j-pop", "j-rock", "jazz",
        "k-pop", "kids-family", "metal", "nightcore", "pop", "punk", "rb", "rock", "soul", "speedcore",
        "swing", "tv-movie-soundtrack", "trance", "video-game-soundtrack", "vocaloid", "poodle", "techno", "ai"]


@register
class MetaTags(Encoder):
    """BeatSaver のタグと、年、長さ、BPM、印。人気と譜面の作者は意図して入れず、譜面から計算したものも入れない
    （版 2 で has_chroma_ne_me と n_std_diffs を外した: 譜面の情報は譜面ごとに持つ）。"""
    id = "meta_tags"
    version = "2"
    inputs = tuple(f"meta:{f}" for f in ["tags", "uploaded", "createdAt", "metadata.duration", "metadata.bpm", "ranked",
                                         "blRanked", "qualified", "blQualified", "curator", "curatedAt", "automapper",
                                         "declaredAi"])
    names = [f"tag:{t}" for t in TAGS] + ["year", "log_duration", "bpm", "ranked_any", "curated", "automapper",
                                          "declared_ai"]
    dim = len(names)

    def params(self):
        return {"source": "catalogue", "features": self.names, "note": "no popularity, no mapper identity"}

    def embed_map(self, m, v):
        md, tags = m["metadata"], set(m.get("tags", []))
        up = m.get("uploaded") or m.get("createdAt") or "2018-06"
        f = [float(t in tags) for t in TAGS]
        f += [int(up[:4]) + int(up[5:7]) / 12, np.log1p(md.get("duration") or 0), md.get("bpm") or 0,
              float(bool(m.get("ranked") or m.get("blRanked") or m.get("qualified") or m.get("blQualified"))),
              float(bool(m.get("curator") or m.get("curatedAt"))), float(bool(m.get("automapper"))),
              float(m.get("declaredAi", "None") != "None")]
        return np.array(f, np.float32)
