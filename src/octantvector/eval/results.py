"""保存した評価の実行: クエリごとの結果を、それを出したもの（ビューの id@version、fingerprint、重み、設定）と一緒に
持つ。あとで計算し直さずにエンコーダーを比べられる。形式は JSON lines（パスが .gz で終われば gzip）。1 行目は
{"run": ...}、ほかの行はクエリ 1 つずつ。"""
import gzip
import json
from datetime import datetime, timezone

from .. import __version__


def run_info(index, command, summary=None, **settings):
    return {"command": command, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "octantvector": __version__, "n_items": len(index), "calibration": index.calibration_id,
            "missing": index.missing,
            "views": {f"{e.id}@{e.version}": {"weight": index.views[i][1], "fingerprint": e.fingerprint()}
                      for i, e in index.encoders.items()},
            "settings": settings, "summary": summary}


def write(path, info, rows):
    with (gzip.open(path, "wt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "w", encoding="utf-8")) as f:
        f.write(json.dumps({"run": info}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")

