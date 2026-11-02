"""推薦側（OctantSpecs contracts/engine-runtime.md）。

渡された DB だけを読み、数はすべて引数で受け取る: 設定ファイルを読まず、ユーザーの状態を持たず、アーカイブ、
カタログ、作業用ストア、エンコーダーには触れない。import するのは `common` だけ（tests/test_layers.py で確かめる）
なので、numpy と pyarrow だけで動く。

契約1（OctantCompass to OctantVector）:
  open_db(path)                                db.py: 全量の DB を確かめて読む（ビュー、ジャンルの語、表記、各ビューの
                                               クレジット: DB.info()）。使えなければ DBError、schemaVersion がこの
                                               OctantVector の読めないものなら VersionError（DBError の一種）
  read_info(path)                              db.py: DB を開かずに、マニフェストだけから DB.info() と同じことを言う
                                               （各ビューの行は除く。ファイルは確かめない）: 開いていない DB を見せる
                                               ため（DBError、VersionError は open_db と同じ）
  recommend(db, history, request, params)      recommend.py: 理由付きで順位を付けた譜面（要求が DB に合わなければ
                                               RequestError。起点が DB になければ RequestError の一種の
                                               SeedNotInDBError。載っているが比べるベクトルがなければ
                                               SeedNotInDBError の一種の SeedNotEmbeddedError）
  presets(db)                                  presets.py: この DB とこの OctantVector が出せる手札。DB のマニフェスト
                                               のものか、同梱の既定
  expand_preset(db, request)                   presets.py: 手札を名指す要求を契約1 の条件にする（一覧にない id は
                                               RequestError の一種の UnknownPresetError）
  describe(db, history, params)                recommend.py: プレイヤーの表示用の要約
  plan_update(index, local_version)            update.py: 契約5（OctantHarvest to OctantCompass）の索引から取る項目
                                               （索引の indexVersion がこの OctantVector の読めないものなら
                                               VersionError）
  apply_update(files, base, out, trust)        update.py: 取った項目を確かめて新しい DB のフォルダに当てる（知って
                                               いる鍵で署名されていない版は失敗ではなく SignatureWarning）
  import_archive(path, out, trust)             update.py: 1 つの tar で手渡しされた全量の版を確かめ、新しい DB の
                                               フォルダに展開する（SignatureWarning は apply_update と同じ）
推薦側の中のもの（契約にはない。道具とテストが使う）:
  search(db, query, view, k)                   db.py: 1 つのビューの上位 k
  apply_diff, unpack                           apply_update を作る段階
  history.py, skill.py, likes.py               読んだ履歴、距離としての実力、好意のマップ（OctantSpecs
                                               octantvector/law.md）。数の既定は common/params.py
"""
# 推薦 API の版（OctantSpecs contracts/engine-runtime.md、版の付け方）。パッケージの版とは別なので、API を
# 保つ実装の変更ではそのまま
API_VERSION = "1.1"

from .db import DB, DBError, VersionError, apply_diff, open_db, read_info, search  # noqa: F401
from .recommend import RequestError, SeedNotEmbeddedError, SeedNotInDBError, describe, recommend  # noqa: F401
from .presets import UnknownPresetError, expand_preset, presets  # noqa: F401
from .update import apply_update, import_archive, plan_update, unpack  # noqa: F401
from ..common.signing import SignatureWarning  # noqa: F401
