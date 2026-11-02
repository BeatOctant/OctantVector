"""契約5 の索引（OctantSpecs contracts/、契約5（OctantHarvest to OctantCompass）: DBの配布と更新）の、生成側。

  build(folders, out, part_size_mb, publish_url, allow_unsigned, trusted_keys)
      folders: 索引に載せてよい版。それぞれ `release` が出したフォルダ（差分か全量）。`out` に、載せる版ごとに
      `part_size_mb` 以下の部分に分けた tar を 1 つと、index.json を書く。索引が持つのは、全量として出した中で
      一番新しいものと、それに続く差分（それぞれが 1 つ前から始まる限り）。渡されなかったものは載せない。
アップロードはここではしない。各部分の URL は `publish_url` + その名前（未設定なら null）。`publish_url` には
`{dbVersion}` を置け、部分の版に置き換わるので、版ごとに別の場所に置ける（版ごとに GitHub Release を 1 つ:
https://github.com/<owner>/<repo>/releases/download/{dbVersion}）。
載せる版は、どれも知っている鍵で署名されていること（契約5、署名。`sign` が manifest.sig を書く）: 同梱の鍵
（common/trusted_keys.py）か `trusted_keys` の鍵で、apply_update と同じく signing.inspect で引く。渡されたが
外れる古い版は見ない。allow_unsigned（開発用）なら署名によらず載せる。索引の項目ごとの `signature` は使わず、
null のまま。

  write_tar(folder, path)
      1 つの版のフォルダを 1 つの tar に: 部分を切り出す元のファイルで、`pack` が丸ごと渡すもの。

索引のホスト側（何を取り、どう確かめて展開するか）は runtime/update.py。
"""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from ..common import signing
from ..common.dbformat import FormatError, INDEX_VERSION, read_manifest
from .errors import GenerationError


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def chain(folders, allow_unsigned=False, trusted_keys=None):
    """索引に載せる [(manifest, folder)]: `folders` の中で一番新しい全量と、それに続く差分。"""
    items = []
    for f in map(Path, folders):
        m = read_manifest(f)  # そろっていて壊れていない版だけ
        items.append((m, f))
    items.sort(key=lambda t: t[0]["builtAt"])
    fulls = [i for i, (m, _) in enumerate(items) if m["kind"] == "full"]
    if not fulls:
        return []
    out = [items[fulls[-1]]]
    for m, f in items[fulls[-1] + 1:]:
        if m["kind"] == "diff" and m["baseDbVersion"] == out[-1][0]["dbVersion"]:
            out.append((m, f))
        else:
            break  # 間の版がない: その後はまだ当てられない
    if not allow_unsigned:  # 載せるものだけ。どれも知っている鍵で署名されている（受け取る側でもう一度確かめる）
        for _, f in out:
            sig = signing.inspect(f, {"keys": trusted_keys})
            if sig["status"] != "signed":
                raise FormatError(f"not indexed: {sig['detail']}")
    return out


def part_url(publish_url, db_version, name):
    """版 db_version の部分の置き場所: publish_url（{dbVersion} を置き換える）+ "/" + name。publish_url がなければ None。"""
    if not publish_url:
        return None
    rest = publish_url.replace("{dbVersion}", "")
    if "{" in rest or "}" in rest:
        raise GenerationError(f"publish url {publish_url!r}: the only placeholder is {{dbVersion}}")
    return publish_url.replace("{dbVersion}", db_version).rstrip("/") + "/" + name


def build(folders, out, part_size_mb, publish_url=None, allow_unsigned=False, trusted_keys=None):
    """-> index.json のパス。"""
    part_url(publish_url, "", "")  # publish_url が壊れていたら、何も書く前に止める
    pub = Path(out)
    pub.mkdir(parents=True, exist_ok=True)
    limit = max(1, int(float(part_size_mb) * (1 << 20)))
    entries = []
    for m, folder in chain(folders, allow_unsigned, trusted_keys):
        name = f"{m['dbVersion']}.{m['kind']}.tar"
        if not (pub / f"{name}.done").exists():
            _write_parts(folder, pub, name, limit)
        info = json.loads((pub / f"{name}.done").read_text())
        entries.append({"dbVersion": m["dbVersion"], "schemaVersion": m["schemaVersion"], "kind": m["kind"],
                        "baseDbVersion": m.get("baseDbVersion"), "calibrationId": m["calibrationId"],
                        "builtAt": m["builtAt"], "size": info["size"], "sha256": info["sha256"],
                        "parts": [{**p, "url": part_url(publish_url, m["dbVersion"], p["name"])}
                                  for p in info["parts"]],
                        "signature": None})
    if not entries:
        raise GenerationError("nothing to index: no version published as a full among the approved ones")
    index = {"indexVersion": INDEX_VERSION, "generatedAt": _now(), "entries": entries}
    p = pub / "index.json"
    p.write_text(json.dumps(index, indent=1))
    return p


def write_tar(folder, path):
    """版のフォルダを、圧縮しない 1 つの tar として path に書く。ファイルはフォルダの名前の下。決定的（名前は並べ、
    メタデータは固定）なので、同じ版はいつも同じバイト列になる。"""
    with tarfile.open(path, mode="w") as tar:
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            ti = tarfile.TarInfo(f"{folder.name}/{f.relative_to(folder).as_posix()}")
            ti.size, ti.mtime, ti.mode = f.stat().st_size, 0, 0o644
            with open(f, "rb") as fh:
                tar.addfile(ti, fh)


def _write_parts(folder, pub, name, limit):
    tmp = pub / f"{name}.tmp"
    write_tar(folder, tmp)
    whole, parts, n = hashlib.sha256(), [], 0
    with open(tmp, "rb") as src:
        while True:
            pn, h, size = f"{name}.part{n:03d}", hashlib.sha256(), 0
            with open(pub / pn, "wb") as dst:
                while size < limit:
                    chunk = src.read(min(1 << 20, limit - size))
                    if not chunk:
                        break
                    dst.write(chunk); h.update(chunk); whole.update(chunk)
                    size += len(chunk)
            if size == 0 and parts:
                (pub / pn).unlink()
                break
            parts.append({"name": pn, "size": size, "sha256": "sha256:" + h.hexdigest()})
            n += 1
    total = tmp.stat().st_size
    tmp.unlink()
    if len(parts) == 1:  # 分けずに済むときは、手渡しの tar（pack）と同じ名前の 1 ファイル
        (pub / parts[0]["name"]).replace(pub / name)
        parts[0]["name"] = name
    (pub / f"{name}.done").write_text(json.dumps({"size": total, "sha256": "sha256:" + whole.hexdigest(), "parts": parts}))
