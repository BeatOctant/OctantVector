"""エンコーダーの登録簿: 組み込みのエンコーダーはこのパッケージを import したときに、プラグインのエンコーダーは
登録簿を初めて使うとき（registry()、get()）に登録される。一覧、既知の問題、足し方は OctantSpecs
octantvector/encoders.md。"""
from .base import (REGISTRY, AudioEncoder, Encoder, ImageEncoder, Model, get, load_plugins, register,  # noqa: F401
                   registry)
from .audio import ClipAudioEncoder, HiddenMean  # noqa: F401
from . import chart_patterns, catalogue_features, swing  # noqa: F401,E402
