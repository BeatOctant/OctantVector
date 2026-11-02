"""DB の生成: クロール（契約4（OctantHarvest crawl to OctantVector））から、台帳、エンコーダー、作業用ストアを経て、
配布 DB（契約2（DB build to recommendation core））とその索引（契約5（OctantHarvest to OctantCompass））まで。
公開する関数は OctantSpecs contracts/engine-generation.md にある。

`common` は import するが、推薦側は import しない: ここから推薦側に渡るのは DB だけ（OctantSpecs
constitution.md、原則 8。tests/test_layers.py で確かめる）。
"""

# 生成 API の版（OctantSpecs contracts/engine-generation.md、版の付け方）。パッケージの版とも推薦側の版とも別
API_VERSION = "1.2"  # 1.2: release が `export` を受け取る

from .api import build_index, calibrate, compute, drift, keygen, models, pack, release, sign  # noqa: E402,F401
from .errors import GenerationError, VersionError  # noqa: E402,F401  （VersionError: 知らないクロールか DB の形式）
