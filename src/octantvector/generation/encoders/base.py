"""エンコーダーの型と登録簿。エンコーダーの一覧と既知の問題は OctantSpecs octantvector/encoders.md にある。

どのエンコーダーも、変わらない `id`、`version`（出力が変わるなら上げる）、決まった `dim`、`kind`、`norm` を持つ:
  kind "audio"   prep(y, sr) は CPU のワーカーのプロセスで走る（結果は pickle できること）。embed(prepped) はメインの
                 プロセス（使えれば GPU）で走り、(dim,) のベクトルを返す。曲にこのビューが表すものがなければ None
                 （skip として記録する）。`batch` が 1 より大きいエンコーダーは、GPU を休ませないよう、embed_many で
                 その数のマップをまとめて受け取る（それぞれに 1 つの出力）。
  kind "image"   prep(data) は CPU のワーカーで、マップのカバー画像（そのファイルのバイト列）について走る。
                 embed(prepped) は音声のエンコーダーと同じくメインのプロセスで走る（`batch` と embed_many も同じ）。
                 読めるカバーのないマップは skip として記録する。
  kind "zip"     embed_zip(mapzip) は CPU のワーカーで、開いたマップの zip（generation/archive.py の MapZip）について走る。
  kind "map"     embed_map(m, v) はカタログのマップの記録と版の記録を受け取る。
  kind "derived" derive(store_root) はほかのストアからベクトルを計算する。
  norm "center"  検索でビューの中心を引いて L2 正規化する（学習した埋め込み）
  norm "zscore"  検索で z スコアにして L2 正規化する（手で作った、読み取れる特徴）
  norm "none"    検索で L2 正規化だけする（コサインをそのままの意味で使うビュー。例 one-hot のカテゴリ: 2 つのマップの
                 コサインは、カテゴリを共有すれば 1、しなければ 0）

各エンコーダーは `inputs`（値が何に依るか）も宣言するので、まさにその入力が変わったときに計算し直す:
  "audio"            曲のファイル          "chart"          Info.dat と難易度のファイル
  "cover"            カバー画像のファイル
  "meta:<field>"     カタログの欄          "view:<id>"      ほかのエンコーダーのベクトル（派生のエンコーダー）
  "file:<setting>"   設定が名指すファイル（例 学習したモデル）
どのビューを DB に入れるかは呼ぶ側が決める（契約1、`views`）。`role` はビューが何を表すか（music、chart、meta）を
名付ける: 推薦側が既定と絞り込みに使うラベル（手札はビューを名指し、役割は名指さない）。DB のマニフェストに書き、
定義には含めない（変えても版を上げなくてよい）。既定は kind に従う（ROLES）。派生のビューは元の役割を取る。

`rows` は 1 つのベクトルが何を表すかを言う（OctantSpecs constitution.md、原則 2）: "maps"（マップの 1 つの版。キーは
その hash: 曲とメタデータ）か "charts"（1 つの譜面。キーは charts.key: 譜面から計算するすべて）。"charts" の
エンコーダーは、マップについて 1 つのベクトルの代わりに {譜面のキー: ベクトル} を返す（embed_zip / embed_map）。
使える譜面のないマップは空の dict か None を返す。`play_styles` は計算するプレイスタイルの一覧（None: すべての
プレイスタイル。charts.play_style）。ほかのプレイスタイルの譜面は、ベクトルを持たないだけ。派生のビューは行を元から
取る（row_kind）。
配布 DB で譜面を表してよいのは "charts" のビューだけ（dbrelease.py）。

fingerprint は、定義（id、version、dim、kind、params）を元のビューの fingerprint と一緒にハッシュしたものなので、
上流のどこが変わっても変わる。ストアは 1 つの版の下に違う fingerprint を混ぜない。

`models` はエンコーダーが読むモデルのファイルを宣言する（Model: コミットに固定した Hugging Face のリポジトリ）。
OctantVector はダウンロードしない: 呼ぶ側が設定 `models_dir` の下に置き（契約1 の関数 `models(views)` から）、
エンコーダーはそこからだけ読む（model_dir）。各モデルの repo@revision は定義の一部なので、fingerprint を変えずに
モデルは変えられない。

`credits` は、ビューが出所（モデル、移したコード、データセット）に負うものを素の文の行で挙げる: 何か、誰のものか、
どこにあるか、どの条件か。リリースはそれを、ステムと元のビューのもの（all_credits）と一緒に DB のマニフェストの
そのビューの項目に書くので、クレジットはビューと一緒に運ばれる（契約2）。`role` と同じく定義には含めない: 変えても
版を上げなくてよい。BeatSaver のデータだけから作るビューには要らない（DB 自体の表記が BeatSaver を挙げる）。

プラグイン（ほかのパッケージ）も、エントリーポイントのグループ `octantvector.encoders` から、`octantvector.plugin`
だけを使って同じようにエンコーダーを登録する。別の PLUGIN_API 向けに作ったプラグインは読み込まない（load_plugins）。
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...common.dbformat import ROW_KINDS
from .. import config
from ..errors import GenerationError
from ..store import Store

REGISTRY = {}
INPUT_KINDS = ("audio", "cover", "chart", "meta:", "view:", "file:")
NORMS = ("center", "zscore", "none")
PLUGIN_GROUP = "octantvector.encoders"
# 不安定なプラグインの API（octantvector.plugin）: 公開するものを変えたら上げる。別の番号を宣言するプラグインは
# 読み込まない
PLUGIN_API = "0.3"  # 0.2: Encoder.credits。0.3: ImageEncoder、norm "none"
PLUGIN_ERRORS = {}  # エントリーポイントの名前 -> そのプラグインのエンコーダーを登録しない理由
PLUGINS = set()  # エンコーダーを登録したプラグインの、エントリーポイントの名前
ROLES = {"audio": "music", "image": "cover", "zip": "chart", "map": "meta"}
_staged = None  # プラグインのモジュールを import しているあいだ: それが登録するクラス。API が合うときだけ確定する
_plugins_loaded = False


def _plain(name):
    return isinstance(name, str) and name not in ("", ".", "..") and not any(c in name for c in "/\\:\0")


@dataclass(frozen=True)
class Model:
    """エンコーダーが読むモデルのファイル: Hugging Face のリポジトリ `repo` のコミット `revision` の `files`。
    <models_dir>/hf/<repo>/<revision>/ に置く（path）。"""
    repo: str
    revision: str
    files: tuple

    @property
    def key(self):
        return f"{self.repo}@{self.revision}"

    @property
    def path(self):
        return f"hf/{self.repo}/{self.revision}"

    def check(self):
        """-> この宣言が使えない理由。使えれば None。repo と files はフォルダとファイルの名前になる。"""
        parts = self.repo.split("/") if isinstance(self.repo, str) else []
        if len(parts) != 2 or not all(map(_plain, parts)):
            return f"repo must be 'owner/name', got {self.repo!r}"
        if not (isinstance(self.revision, str) and len(self.revision) == 40
                and set(self.revision) <= set("0123456789abcdef")):
            return f"{self.repo}: revision must be a full 40-hex commit, got {self.revision!r}"
        if not isinstance(self.files, tuple) or not self.files or not all(map(_plain, self.files)):
            return f"{self.key}: files must be a non-empty tuple of file names, got {self.files!r}"
        return None


def register(cls):
    if cls.id in REGISTRY or any(c.id == cls.id for c in _staged or ()):
        raise ValueError(f"duplicate encoder id {cls.id}")
    if not cls.inputs or any(not i.startswith(INPUT_KINDS) or i.endswith(":") for i in cls.inputs):
        raise ValueError(f"{cls.id}: inputs must be a non-empty tuple of {INPUT_KINDS}, got {cls.inputs!r}")
    if (cls.kind == "derived") != any(i.startswith("view:") for i in cls.inputs):
        raise ValueError(f"{cls.id}: derived encoders, and only they, read other views ('view:<id>' inputs)")
    if cls.rows not in ROW_KINDS or (cls.rows == "charts" and cls.kind in ("audio", "image")):
        raise ValueError(f"{cls.id}: rows must be one of {ROW_KINDS} (audio and image views describe maps), "
                         f"got {cls.rows!r}")
    if cls.norm not in NORMS:
        raise ValueError(f"{cls.id}: norm must be one of {NORMS}, got {cls.norm!r}")
    if not isinstance(cls.models, tuple) or not all(isinstance(m, Model) for m in cls.models):
        raise ValueError(f"{cls.id}: models must be a tuple of Model, got {cls.models!r}")
    bad = next((why for why in (m.check() for m in cls.models) if why), None)
    if bad:
        raise ValueError(f"{cls.id}: {bad}")
    if not isinstance(cls.credits, tuple) or not all(isinstance(c, str) and c.strip() for c in cls.credits):
        raise ValueError(f"{cls.id}: credits must be a tuple of non-empty strings, got {cls.credits!r}")
    if _staged is not None:
        _staged.append(cls)
    else:
        REGISTRY[cls.id] = cls
    return cls


def load_plugins(eps=None):
    """ほかのパッケージがエントリーポイントのグループ `octantvector.encoders` に挙げるモジュールを import する。モジュールが
    import のあいだに登録したものは、OCTANTVECTOR_PLUGIN_API == PLUGIN_API を宣言するときだけ残す。失敗したプラグインは
    例外を投げずに PLUGIN_ERRORS に記録するので、ほかのビューは動き続ける。get() がそれを名指す。"""
    global _staged
    if eps is None:
        from importlib.metadata import entry_points
        eps = entry_points(group=PLUGIN_GROUP)
    for ep in eps:
        _staged = []
        try:
            module = ep.load()
        except Exception as ex:
            PLUGIN_ERRORS[ep.name] = f"{ep.value} failed to import: {ex!r}"
            PLUGINS.discard(ep.name)
            continue
        else:
            staged = _staged
        finally:
            _staged = None
        api = getattr(module, "OCTANTVECTOR_PLUGIN_API", None)
        if api != PLUGIN_API:
            # 前に import された（ここを通らずに直接）: そのとき登録したものは、そのまま REGISTRY に入った
            for i in [i for i, c in REGISTRY.items() if _from(c, module)]:
                del REGISTRY[i]
            PLUGIN_ERRORS[ep.name] = (f"{ep.value} is built for plugin API {api!r}, this OctantVector has {PLUGIN_API!r}"
                                      if api is not None else f"{ep.value} declares no OCTANTVECTOR_PLUGIN_API")
            PLUGINS.discard(ep.name)
            continue
        PLUGIN_ERRORS.pop(ep.name, None)
        PLUGINS.add(ep.name)
        REGISTRY.update((c.id, c) for c in staged)


def _from(cls, module):
    return cls.__module__ == module.__name__ or cls.__module__.startswith(module.__name__ + ".")


def registry():
    """プラグインを読み込んだ REGISTRY（import のときではなく初めて使うときに読み込む: プラグインは octantvector.plugin を
    import し、それがこのパッケージを import するので、import の途中で読み込むと作りかけのものを見つけてしまう）。"""
    global _plugins_loaded
    if not _plugins_loaded:
        _plugins_loaded = True
        load_plugins()
    return REGISTRY


def plugin_errors():
    """'' か、読み込めなかったプラグインがある理由。ビューがないというメッセージに足す。"""
    if not PLUGIN_ERRORS:
        return ""
    return "; plugins not loaded: " + "; ".join(f"{n} ({why})" for n, why in sorted(PLUGIN_ERRORS.items()))


def not_provided():
    """登録したどのエンコーダーも持たないビューの id についてのメッセージに足すもの: 読み込んだプラグイン（それを出す
    プラグインが入っていなければなし。例 別の OctantVector の版向けに作ったもの）と、ほかを読み込まなかった理由。"""
    return f"; no loaded plugin provides it (loaded: {', '.join(sorted(PLUGINS)) or 'none'}){plugin_errors()}"


def get(encoder_id):
    try:
        return registry()[encoder_id]()
    except KeyError:
        raise GenerationError(f"unknown encoder {encoder_id!r}; available: {', '.join(sorted(REGISTRY))}"
                              + plugin_errors()) from None


def clips(y, sr, clip_s, n):
    """曲の 15-75% に広げた clip_s 秒の窓を n 個（前奏と後奏を飛ばす）。短い曲は埋める。"""
    L = int(clip_s * sr)
    if len(y) <= L:
        return [np.pad(y, (0, L - len(y)))]
    return [y[int(s):int(s) + L] for s in np.linspace(0.15, 0.75, n) * (len(y) - L)]


def resample(y, sr, target):
    if sr == target:
        return y
    import soxr
    return soxr.resample(y, sr, target)


class Encoder:
    kind = "map"
    norm = "zscore"
    version = "1"
    inputs = ()
    role = None
    rows = "maps"
    play_styles = None
    models = ()
    credits = ()

    def role_name(self):
        if self.role:
            return self.role
        return get(self.source).role_name() if self.kind == "derived" else ROLES[self.kind]

    def row_kind(self):
        return get(self.source).row_kind() if self.kind == "derived" else self.rows

    def params(self):
        return {}

    def own_models(self):
        """このエンコーダー自身が読むモデル（ClipAudioEncoder はステムのものを足す）。"""
        return self.models

    def own_credits(self):
        """このエンコーダー自身のクレジット（ClipAudioEncoder はステムのものを足す）。"""
        return self.credits

    def all_credits(self):
        """ビューが負うもの: 自分のクレジットと元のもの（重ねず、順に）。DB のマニフェストに書く。"""
        out = list(self.own_credits())
        for s in self.sources():
            out += [c for c in get(s).all_credits() if c not in out]
        return out

    def required_models(self):
        """このビューの計算に要るすべてのモデル: 自分のものと元のもの（重ねず、順に）。"""
        out = list(self.own_models())
        for s in self.sources():
            out += [m for m in get(s).required_models() if m not in out]
        return out

    def model_dir(self, model=None):
        """設定 models_dir の下の `model`（既定: 最初に宣言したもの）の手元のフォルダ。宣言したどのファイルもあることを
        確かめてから返す。OctantVector はモデルをダウンロードしない。"""
        m = model or self.own_models()[0]
        root = config.load().models_dir
        d = Path(root) / m.path if root else None
        if d is None or not all((d / f).is_file() for f in m.files):
            raise GenerationError(f"{m.key} is not in {root or 'models_dir (not set)'}: run `octantharvest models`")
        return d

    def sources(self):
        """このエンコーダーを計算する元のビューの id。"""
        return [i[5:] for i in self.inputs if i.startswith("view:")]

    @property
    def source(self):
        """派生のエンコーダーのただ 1 つの元のビュー。"""
        (s,) = self.sources()
        return s

    def definition(self):
        """ストアが encoder.json に記録し、開くときに比べるもの（JSON に正規化したもの）。"""
        params = self.params()
        if self.own_models():
            params = {**params, "models": [m.key for m in self.own_models()]}
        d = {"id": self.id, "version": self.version, "dim": self.dim, "kind": self.kind, "params": params,
             "rows": self.rows, "play_styles": list(self.play_styles) if self.play_styles else None}
        return json.loads(json.dumps(d))

    def fingerprint(self):
        up = {s: get(s).fingerprint() for s in self.sources()}
        blob = json.dumps({"definition": self.definition(), "sources": up}, sort_keys=True)
        return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()



class DerivedEncoder(Encoder):
    """1 つの元のビューの保存したベクトルから、行ごとに計算するビュー: values(X, device) は (rows, 元の dim) を
    (rows, dim) にする。derive は元のストアを読み、このビューがまだ持たない行だけを書く。そういう行がなければフックは
    呼ばない（読み込むモデルを無駄に読み込まないように）。"""
    kind = "derived"

    def values(self, X, device="cpu"):
        raise NotImplementedError

    def derive(self, store_root, device="cpu"):
        dst = Store(self, store_root)
        done = dst.done()
        hs, X = Store(get(self.source), store_root).load()
        new = [i for i, h in enumerate(hs) if h not in done]
        if new:
            for i, v in zip(new, self.values(X[new], device)):
                dst.add(hs[i], v)
            dst.flush()
        return len(dst.done())


class AudioEncoder(Encoder):
    kind = "audio"
    norm = "center"
    inputs = ("audio",)
    sr = None

    def setup_worker(self):  # CPU の側の物（特徴抽出器）。ワーカーのプロセスごとに一度
        pass

    def setup_model(self, device):  # モデル。メインのプロセスだけ
        pass

    batch = 1  # > 1: 重い処理が embed_many にこの数のマップをまとめて渡す

    def embed_many(self, prepped):
        """-> 用意したマップごとに 1 つの出力（ベクトルか None）。"""
        return [self.embed(x) for x in prepped]


class ImageEncoder(Encoder):
    """マップのカバー画像のビュー: CPU のワーカーで prep(data) が画像のファイルのバイト列を受け取り、pickle できる値
    （デコードして縮めたもの）を返す。メインのプロセス（使えれば GPU）で embed(prepped) が (dim,) のベクトルか None を
    返す。`batch` が 1 より大きければ、音声と同じく embed_many がその数のマップをまとめて受け取る。"""
    kind = "image"
    norm = "center"
    inputs = ("cover",)

    def setup_worker(self):
        pass

    def setup_model(self, device):
        pass

    batch = 1

    def prep(self, data):
        raise NotImplementedError

    def embed(self, prepped):
        raise NotImplementedError

    def embed_many(self, prepped):
        return [self.embed(x) for x in prepped]
