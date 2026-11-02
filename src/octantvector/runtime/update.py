"""契約5 の索引（OctantSpecs contracts/、契約5（OctantHarvest to OctantCompass）: DBの配布と更新）のホスト側。
OctantVector 自身は何もダウンロードしない: ファイルは渡されたものを使う。

  plan_update(index, local_version)  取る項目: 手元の版が連なりにあればその後の差分、なければ全量とその後の
                                     すべての差分（契約1（OctantCompass to OctantVector））
  apply_update(files, base, out, trust)
                                     取った項目を順に確かめ、展開して当てる。新しい全量の DB は `out` に書き、
                                     `base` の手元の DB は変えない（契約1）。この OctantVector が知らないものを
                                     失う差分は断る（DBError、dbformat.apply_diff）。その版の全量なら同じ DB になる
  unpack                             1 つの項目を確かめて展開する
  import_archive(path, out, trust)   1 つの tar で手渡しされた全量の版（索引なし: 契約5 非公開（探求中）の扱い）を
                                     確かめて `out` に展開する（契約1）

索引を書くのは generation/index.py。
"""
import hashlib
import shutil
import tarfile
import tempfile
import warnings
from pathlib import Path

from ..common import signing
from ..common.dbformat import INDEX_VERSION, SCHEMA_VERSION, FormatError, VersionError, apply_diff, major, read_manifest

READS_MAJOR = major(SCHEMA_VERSION)
KINDS = ("full", "diff")


MAX_NAME = 100  # dbVersion のバイト数: resolve() はファイルシステムに長すぎる名前でも失敗せず、後で使うときに失敗する


def _folder_name(entry):
    """項目の dbVersion。ホストのフォルダとファイルの名前になる。索引は署名されないので、1 つの素のパスの要素以外
    （置いたフォルダから出られる名前）は、使う前に断る。"""
    v = entry.get("dbVersion")
    if not isinstance(v, str) or v in ("", ".", "..") or "/" in v or "\\" in v or "\0" in v or ":" in v \
            or Path(v).is_absolute() or len(v.encode()) > MAX_NAME:
        raise FormatError(f"the index names a version that is not a plain folder name: {v!r}")
    if entry.get("kind") not in KINDS:
        raise FormatError(f"{v}: unknown kind {entry.get('kind')!r}")
    return v


def _inside(parent, name):
    """parent / name。parent の中に解決しなければ断る（上の名前の確認を、実際のパスでもう一度する）。"""
    path = Path(parent) / name
    if not path.resolve().is_relative_to(Path(parent).resolve()):
        raise FormatError(f"{name!r} would be written outside {parent}")
    return path


def plan_update(index, local_version=None):
    """この OctantVector が読める一番新しい版に届くために、順に取る項目。最新なら []。
    索引そのものがこの OctantVector の読めない形式（indexVersion）なら VersionError。"""
    found = index.get("indexVersion")
    try:
        ok_index = major(found) == major(INDEX_VERSION)
    except (TypeError, ValueError):
        ok_index = False
    if not ok_index:
        raise VersionError("indexVersion", found, f"{major(INDEX_VERSION)}.x")
    ok = [e for e in index["entries"] if major(e["schemaVersion"]) == READS_MAJOR]
    for e in ok:
        _folder_name(e)
    if not ok or ok[-1]["dbVersion"] == local_version:
        return []
    versions = [e["dbVersion"] for e in ok]
    if local_version in versions:
        return ok[versions.index(local_version) + 1:]
    start = max(i for i, e in enumerate(ok) if e["kind"] == "full")
    return ok[start:]


def unpack(entry, part_paths, out_dir, trust=None):
    """1 つの項目の部分を（索引の順に）確かめ、out_dir に展開する。manifest.sig は同梱の鍵と照らし、正しい署名でなければ
    失敗ではなく SignatureWarning。-> 展開した DB のフォルダ。"""
    version = _folder_name(entry)
    if len(part_paths) != len(entry["parts"]):
        raise FormatError(f"{version}: {len(entry['parts'])} parts expected, got {len(part_paths)}")
    out_dir = Path(out_dir)
    joined = _inside(out_dir, f"{version}.{entry['kind']}.tar.partial")
    out_dir.mkdir(parents=True, exist_ok=True)
    whole = hashlib.sha256()
    try:
        with open(joined, "wb") as dst:
            for p, want in zip(part_paths, entry["parts"]):
                h, size = hashlib.sha256(), 0
                with open(p, "rb") as src:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        h.update(chunk); whole.update(chunk); dst.write(chunk)
                        size += len(chunk)
                if size != want["size"] or "sha256:" + h.hexdigest() != want["sha256"]:
                    raise FormatError(f"{p} does not match the index ({want['name']})")
        if "sha256:" + whole.hexdigest() != entry["sha256"]:
            raise FormatError(f"{entry['dbVersion']}: the joined parts do not match the index")
        folder = _extract(joined, out_dir, entry["dbVersion"])
    finally:
        joined.unlink(missing_ok=True)
    m = read_manifest(folder, entry["kind"])
    _warn_unsigned(folder, trust, entry["dbVersion"])
    if m["dbVersion"] != entry["dbVersion"]:
        raise FormatError(f"the archive holds {m['dbVersion']}, the index says {entry['dbVersion']}")
    return folder


def _extract(tar_path, out_dir, label):
    """tar_path の tar にある 1 つの版のフォルダを out_dir に展開する。1 つの最上位のフォルダの下の素のファイル以外があれば、
    何も書く前に断る。受け付けるのは圧縮しない tar だけ（生成側が書くもの）: 圧縮したものは際限なく膨らんでディスクを
    埋め得る（tar 爆弾）が、素の tar ではどのメンバーのバイトもそのまま入っているので、取って確かめた分より多くは
    書かれない。同じ理由で、アーカイブが持つより多くを宣言するメンバー（疎なメンバー）も断る。"""
    try:
        tar = tarfile.open(tar_path, mode="r:")
    except tarfile.TarError as ex:
        raise FormatError(f"{label}: not an uncompressed tar archive ({ex})") from None
    with tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        top = {n.split("/")[0] for n in names}
        if len(top) != 1 or any(n.startswith("/") or ".." in Path(n).parts for n in names) or \
                any(not m.isfile() for m in members):
            raise FormatError(f"{label}: unexpected paths in the archive")
        if sum(m.size for m in members) > Path(tar_path).stat().st_size:
            raise FormatError(f"{label}: the archive declares more content than it holds")
        if (Path(out_dir) / next(iter(top))).exists():
            raise FormatError(f"{Path(out_dir) / next(iter(top))} already exists")
        tar.extractall(out_dir, filter="data")
    return Path(out_dir) / top.pop()


def _warn_unsigned(folder, trust, label):
    sig = signing.inspect(folder, trust)  # 署名は作り手を示すだけで、止めはしない（契約5、署名）
    if sig["status"] != "signed" and not (sig["status"] == "unsigned" and signing.check_trust(trust)[1]):
        warnings.warn(signing.SignatureWarning(f"{label}: {sig['detail']}"), stacklevel=3)


def import_archive(path, out, trust=None):
    """path: 1 つの tar にした全量の版（生成側の `pack`。分けない索引のファイルと同じバイト列）。索引なしで手渡しされる。
    -> `out`、全量の DB のフォルダ。ファイルがマニフェストと合ってから現れ、どこかで失敗すれば `out` には何も残らない。
    trust は apply_update と同じ。知っている鍵で署名されていない版も SignatureWarning 付きで使う。"""
    signing.check_trust(trust)
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    work = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent if out.parent.exists() else None))
    try:
        try:
            folder = _extract(path, work, Path(path).name)
        except tarfile.TarError as ex:
            raise FormatError(f"{Path(path).name}: not a DB archive ({ex})") from None
        m = read_manifest(folder, "full")
        if folder.name != _folder_name(m):
            raise FormatError(f"{Path(path).name}: the folder {folder.name!r} holds {m['dbVersion']}")
        _warn_unsigned(folder, trust, m["dbVersion"])
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def apply_update(files, base, out, trust=None):
    """files: plan_update の順の [(項目, 部分のパス)]。base: 手元の全量の DB のフォルダ（plan_update がまず全量を頼んだ
    ときは None）。-> `out`、すべての段を通ってから現れる全量の DB のフォルダ。どこかで失敗すれば `out` には何も残らず、
    `base` は触れないまま。trust: {"keys": 同梱のほかに知っている {keyId: 公開鍵の base64}、"allowUnsigned":
    manifest.sig のない版に警告しない（開発用）}。None = 同梱の鍵だけ。知っている鍵で署名されていない版も
    SignatureWarning 付きで当てる。"""
    signing.check_trust(trust)  # 壊れた trust は、何かを展開する前に失敗する
    out = Path(out)
    if out.exists():
        raise FormatError(f"{out} already exists; a DB is never overwritten")
    if not files:
        raise FormatError("nothing to apply: plan_update returned no entries")
    for entry, _ in files:
        _folder_name(entry)
    work = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent if out.parent.exists() else None))
    try:
        cur = Path(base) if base is not None else None
        for n, (entry, parts) in enumerate(files):
            folder = unpack(entry, parts, work / f"{n}.unpacked", trust)
            if entry["kind"] == "full":
                cur = folder
            elif cur is None:
                raise FormatError(f"{entry['dbVersion']}: a diff needs a base DB")
            else:
                cur = apply_diff(cur, folder, _inside(work / f"{n}.applied", entry["dbVersion"]))
        read_manifest(cur, "full")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cur), str(out))
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
