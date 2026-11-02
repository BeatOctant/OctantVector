"""契約1 の生成側（OctantSpecs contracts/、契約1（OctantHarvest to OctantVector）生成側の関数）。

入力はすべて引数で受け取る: クロール（契約4（OctantHarvest crawl to OctantVector））を {"archive": フォルダ,
"catalogue": catalogue.jsonl.gz} で、なくてもよい "artists": map_artists.jsonl.gz と similar_artists.jsonl.gz の
フォルダ（関係のビュー listenbrainz_artists@<YYYYMMDD>、役割 `artist`）。
`views`、計算するか DB に入れるビュー: エンコーダーは id@version（か版のない id）で、カタログから作る関係のビューは
名前（RELATIONS。計算も較正もせず、リリースに入るだけ）で。
作業用ストアのフォルダ（台帳を持つ）と、`params`、OctantSpecs octantvector/config.md の設定を名前で（数、`device`、
`swing_model`、`pools`・`swing`・`swing_tags` の表）。`params` で省いたものは既定値で埋める。ここでは設定ファイルも
環境変数も読まず、ネットワークにもつながない: エンコーダーが読むモデルは `models(views)` が一覧にし、呼ぶ側が
取って `models_dir` として渡す。

いつ実行するか、どの版を承認するか、ファイルをどこに置くかは呼ぶ側が決める。
"""
import functools
import io
import json
from pathlib import Path

from ..common import dbformat, signing
from ..common.dbformat import read_manifest
from . import calibration as calibrations, config, dbrelease, embed, index as indexes, mappers
from .encoders import get, registry
from .encoders.base import not_provided, plugin_errors
from .errors import GenerationError, VersionError
from .ledger import Ledger

PATHS = ("archive", "catalogue", "artists", "store", "models_dir")
RELATIONS = (mappers.NAME,)  # `views` が名指したときに、リリースがカタログから作る関係のビュー
# この OctantVector が読む契約4 の形式（OctantSpecs contracts/、契約の版）: catalogue.jsonl.gz の隣にある
# カタログの取得の manifest.json の `schema` と、アーカイブの archive_meta.json の `format`
CATALOGUE_SCHEMAS = ("octantharvest-beatsaver-catalogue-capture/v1",)
ARCHIVE_FORMATS = ("beatmap-music-identity-beatsaver-archive-v2",)


def _contract(f):
    """契約1 の関数: 生成側の読めない形式の版は、むき出しの dbformat のものではなく生成側の VersionError（GenerationError の
    一種）として呼ぶ側に届き、ほかの形式のエラー（版のフォルダがない、壊れている、知っている鍵で署名されていない）は
    GenerationError として届く。"""
    @functools.wraps(f)
    def call(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except VersionError:
            raise
        except dbformat.VersionError as ex:
            raise VersionError.of(ex) from ex
        except dbformat.FormatError as ex:
            raise GenerationError(str(ex)) from ex
    return call


def _declared(path, key):
    try:
        return json.loads(Path(path).read_text()).get(key)
    except (OSError, ValueError):
        return None


def _check_input(input):
    """クロールがこの OctantVector の読める形式を宣言していなければ VersionError（manifest.json のない取得はそろっておらず、
    何も宣言しない）。"""
    cat = Path(input["catalogue"]).parent / "manifest.json"
    found = _declared(cat, "schema")
    if found not in CATALOGUE_SCHEMAS:
        raise VersionError("catalogue schema", found, ", ".join(CATALOGUE_SCHEMAS), where=str(cat))
    meta = Path(input["archive"]) / "archive_meta.json"
    found = _declared(meta, "format")
    if found not in ARCHIVE_FORMATS:
        raise VersionError("archive format", found, ", ".join(ARCHIVE_FORMATS), where=str(meta))


def _values(input, store, params, models_dir=None):
    v = dict(params or {})
    passed = set(v) & set(PATHS)
    if passed:
        raise GenerationError(f"{', '.join(sorted(passed))}: pass paths as the arguments, not in params")
    if input is not None:
        _check_input(input)
        v["archive"], v["catalogue"] = str(input["archive"]), str(input["catalogue"])
        if input.get("artists"):
            v["artists"] = str(input["artists"])
    v["store"] = str(store)
    if models_dir is not None:
        v["models_dir"] = str(models_dir)
    return v


def _ids(views):
    """`id@version`（か版のない id）-> エンコーダーの id。今の版でない版は断る。ビューはいつも呼ぶ側が名指す。
    関係のビュー（RELATIONS）は外す: 計算しないため。"""
    return _split(views)[0]


def _split(views):
    """-> `views` の (エンコーダーの id, 関係のビューの名前)。"""
    if not views or isinstance(views, str):
        raise GenerationError(f"views: a non-empty list of id@version is required, got {views!r}")
    out, rel = [], []
    for view in views:
        i, _, ver = str(view).partition("@")
        if i in RELATIONS:
            if ver and f"{i}@{ver}" != mappers.KEY:
                raise GenerationError(f"{view}: this OctantVector builds {mappers.KEY}")
            rel.append(i)
            continue
        if i not in registry():
            raise GenerationError(f"{view}: no such encoder (OctantSpecs octantvector/encoders.md){not_provided()}")
        if ver and ver != str(registry()[i].version):
            raise GenerationError(f"{view}: this OctantVector computes {i}@{registry()[i].version}")
        out.append(i)
    return out, rel


def _pool(store):
    with Ledger.of(store) as led:
        return led.pool_hashes("distribution")


def _results(store):
    with Ledger.of(store) as led:
        return led.results()


@_contract
def compute(input, store, views, params=None, pool="distribution", models_dir=None, log=None):
    """クロールを台帳に取り込み、`views`（と派生のものの元、譜面の指標の元のビュー: dbrelease.METRICS）について、台帳が
    足りないと言うものを、プール `pool`（params["pools"]、OctantSpecs octantvector/config.md）で計算する。models_dir:
    `models(views)` が挙げるモデルを持つフォルダ。続きからやり直せ、直後にもう一度呼べば何も計算しない。GPU に空きが
    なければ、モデルは一度に扱うマップを減らす。1 つでも入らないマップは計算も記録もせず残し（"deferred"）、次の
    呼び出しで計算する。
    -> この呼び出しの {"computed": n, "failed": n, "skipped": n, "deferred": n,
        "views": {id@version: {"ok"|"error"|"skip"|"deferred": n}}}。"""
    ids = _ids(views)
    ids += [i for i in dbrelease.metric_views() if i not in ids]  # リリースはこれで譜面の指標を埋める
    with config.given(_values(input, store, params, models_dir)):
        before = _results(store)
        no_room = embed.run(config.load(), ids, pool=pool, log=log or io.StringIO())
        rows = _results(store) - before  # この呼び出しが書いたか変えた結果
    per = {}
    for _, enc, r, *_ in rows:
        per.setdefault(enc, {})[r] = per.get(enc, {}).get(r, 0) + 1
    for i, k in no_room.items():
        per.setdefault(f"{i}@{get(i).version}", {})["deferred"] = k
    total = {r: sum(v.get(r, 0) for v in per.values()) for r in ("ok", "error", "skip", "deferred")}
    return {"computed": total["ok"], "failed": total["error"], "skipped": total["skip"], "deferred": total["deferred"],
            "views": per}


@_contract
def models(views):
    """`views` が（その元のものも）読むモデル。呼ぶ側が `compute` の前に models_dir へ取る。
    -> [{"repo", "revision", "files": [...], "path": "hf/<repo>/<revision>"}]（どれも 1 回、並べたもの）。"""
    with config.given({}):  # 作るときに設定を読むエンコーダーがある。ここではファイルからは何も来ない
        found = {m.key: m for i in _ids(views) for m in get(i).required_models()}
    return [{"repo": m.repo, "revision": m.revision, "files": list(m.files), "path": m.path}
            for _, m in sorted(found.items())]


@_contract
def drift(store, base, params=None):
    """全量の DB `base` の較正と、ストアの今の配布プールとの {view id@version: ずれ}: 各ビューの平均がどれだけ動いたかを、
    較正の平均のまわりの広がりを単位にして表す（0 = 動いていない）。`base` のベクトルのビューで、入っているどの
    エンコーダーも計算しないものがあれば止める（プラグインがないのを見過ごしてはならない）。"""
    try:
        m = json.loads((Path(base) / "manifest.json").read_text())
    except (OSError, ValueError) as ex:
        raise GenerationError(f"{base}: no readable manifest.json ({ex})") from ex
    vec = [k for k, v in m["views"].items() if dbformat.view_kind(v) != "relation"]  # 関係は与えられるもので、計算しない
    views = [k.split("@")[0] for k in vec]
    unknown = [k for k, i in zip(vec, views) if i not in registry()]
    if unknown:
        raise GenerationError(f"{base} has views no installed encoder computes: {', '.join(unknown)}; install the "
                              f"plugin that provides them{plugin_errors()}")
    with config.given(_values(None, store, params)):
        cal = calibrations.load(Path(base) / m["calibrationFile"])
        return calibrations.drift(store, cal, views, _pool(store))


@_contract
def calibrate(store, views, params=None):
    """配布プールでの `views` の新しい較正。<store>/calibration にも保存する。
    -> 較正（素の JSON: id、pool、created_at、views）。"""
    with config.given(_values(None, store, params)):
        cal = calibrations.compute(store, _ids(views), _pool(store), "distribution")
        return json.loads(calibrations.save(cal, store).read_text())


@_contract
def release(input, store, views, calibration, base, out, params=None, full=False, built_at=None, log=None, presets=None,
            notice=None, export=None):
    """フォルダ `out` の下の次の DB の版（契約2（DB build to recommendation core））: いつも全量と、作れれば `base`（前の
    全量のフォルダか None）からの差分。views: 入れるビュー（必須）。calibration: `calibrate` が返したもの。None なら
    base のもの（新しいビューの分を広げる）。
    presets: マニフェストに書く手札（list か JSON のファイルのパス）。None なら base のものを残す。どちらでも新しい
    DB と照らして確かめる。
    notice: マニフェストに書く、DB の条件とクレジットの 1 つの Markdown の文。None なら base のものを残す。各ビューの
    クレジットはエンコーダーから来る。
    export: 各ビューをどう書くか（{view id: {"dim": <int>, "dtype": "float16" | "int8"}}。名指さないビューはエンコーダーの
    次元のまま float16）。None なら較正にあるものを残す。変えたら全量として出す。
    -> {"dbVersion", "full": フォルダ, "diff": フォルダか None, "published": 承認して索引に載せるもの}。"""
    ids, rel = _split(views)
    with config.given(_values(input, store, params)):
        cal = calibrations.load_dict(json.loads(json.dumps(calibration))) if calibration is not None else None
        full_dir, diff_dir = dbrelease.release(config.load(), ids, cal, full=full,
                                               built_at=built_at, log=log or io.StringIO(), out=out, base=base,
                                               presets=presets, notice=notice, relations=rel,
                                               export=export)
    return {"dbVersion": full_dir.name, "full": full_dir, "diff": diff_dir, "published": diff_dir or full_dir}


@_contract
def keygen(path):
    """新しい Ed25519 の秘密鍵を `path` に書く（あるファイルは上書きしない）。-> {"publicKey": base64, "keyId"}。公開鍵は
    リリースで OctantVector の同梱の一覧（common/trusted_keys.py）に入れる。"""
    pub, kid = signing.keygen(path)
    return {"publicKey": pub, "keyId": kid}


@_contract
def sign(folder, key_path):
    """`release` が "published" として返した版のフォルダに、秘密鍵のファイルで署名する。-> manifest.sig のパス。"""
    read_manifest(folder)  # そろっていて壊れていない版だけ
    return signing.sign(folder, key_path)


@_contract
def pack(folder, out):
    """`release` が "full" として返した全量の版のフォルダを、フォルダ `out` に <dbVersion>.full.tar という 1 つの tar に
    する: 索引なしで手渡しするもの（OctantSpecs contracts/distribution.md、非公開（探求中）の扱い）で、分ける前の索引の
    その版のファイルと同じバイト列。受け取る側が作り手を見られるよう、先にフォルダに署名する（`sign`）。あるファイルは
    上書きしない。-> tar のパス。"""
    folder = Path(folder)
    m = read_manifest(folder, "full")  # そろっていて壊れていない全量だけ: 受け取る側には差分を当てる元がない
    if folder.name != m["dbVersion"]:
        raise GenerationError(f"{folder}: the folder of {m['dbVersion']} must be named after it")
    path = Path(out) / f"{m['dbVersion']}.full.tar"
    if path.exists():
        raise GenerationError(f"{path} already exists; it is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        indexes.write_tar(folder, tmp)
        tmp.rename(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


@_contract
def build_index(releases, out, params=None):
    """承認した版 `releases`（どれも `release` が "published" として返したフォルダ）の、契約5（OctantHarvest to
    OctantCompass）の索引と分けたファイルを、フォルダ `out` に書く。params: part_size_mb（書くファイルの最大。必須で、
    呼ぶ側の設定）、publish_url、trusted_keys（同梱のほかに知っている {keyId: 公開鍵の base64}。apply_update の
    trust["keys"] と同じ: この OctantVector が同梱していない作り手の鍵）、allow_unsigned（署名によらず載せる。開発用）。
    載せる版は、どれも知っている鍵で署名されていること。
    -> index.json のパス。"""
    p = dict(params or {})
    unknown = set(p) - {"part_size_mb", "publish_url", "trusted_keys", "allow_unsigned"}
    if unknown:
        raise GenerationError(f"build_index: unknown params {', '.join(sorted(unknown))}")
    if p.get("part_size_mb") is None:
        raise GenerationError("build_index: params.part_size_mb is required")
    try:
        signing.check_trust({"keys": p.get("trusted_keys")})
    except (ValueError, TypeError) as ex:
        raise GenerationError(f"build_index: params.trusted_keys: {ex}") from None
    return indexes.build(releases, out, p["part_size_mb"], p.get("publish_url"), bool(p.get("allow_unsigned", False)),
                         p.get("trusted_keys"))
