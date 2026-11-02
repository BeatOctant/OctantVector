"""エンコーダー（ビュー）を OctantVector に足すパッケージのための、ただ 1 つの入口。不安定: 公開するものを変えたら
PLUGIN_API を上げ、別の番号向けに作ったプラグインは読み込まない（理由はそのビューを頼んだときに示す）。
固まったら OctantSpecs octantvector/ に書く。契約1 には含まない。

プラグインは、エントリーポイントのグループ `octantvector.encoders` に名前のあるモジュールで、
`OCTANTVECTOR_PLUGIN_API = "<書いたときの PLUGIN_API>"` を置き、エンコーダーのクラスに `register` を付ける。
import するのはこのモジュールだけ。モデルは宣言し（Model）、models_dir から読む。OctantVector はモデルを
ダウンロードしない。
"""
from .generation.encoders.audio import ClipAudioEncoder, HiddenMean
from .generation.encoders.base import (PLUGIN_API, AudioEncoder, DerivedEncoder, Encoder, ImageEncoder, Model, clips,
                                       register, resample)
from .generation.errors import GenerationError

__all__ = ["PLUGIN_API", "Encoder", "AudioEncoder", "ImageEncoder", "DerivedEncoder", "ClipAudioEncoder", "HiddenMean",
           "Model", "register", "clips", "resample", "GenerationError"]
