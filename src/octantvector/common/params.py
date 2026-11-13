"""`recommend` と `describe` が `params` で受け取る数の既定値（契約1）。

`params` で省いたキーはここの既定値を使い、知らないキーは無視する。難易度のキーは OctantSpecs の既定値（契約1 の
params のスキーマの `x-defaultIn`）を使う。「仮」と書いたものは、この推薦側の仮の値。
"""

DEFAULTS = {
    "minViewWeight": 0.5,          # 候補がベクトルを持つべきビューの重みの割合（OctantSpecs 法 3.1）
    "includeKnown": False,         # 仮: プレイヤーがもう遊んだ、お気に入りにした、投票したマップも推薦する
    "revisitDays": 0,              # 0 より大きければ、好意のマップのうち最後のプレイがこの日数より前のものだけを出す
                                   # （「久しぶり」。2026-11-03）。0: 使わない
    "recentPlays": 200,            # 仮: ジャンルの入口で「最近」と数えるプレイの数
    "genreTopShare": 0.1,          # 仮: ジャンルの絞り込みで残す、語に一番近いマップの割合
    "narrowTopMaps": 300,          # 仮: `weights` で並べる前に `narrow` で残すマップ数
    "narrowWiden": 3,              # 仮: 残りが `count` より少ないあいだ、narrowTopMaps にこれを掛ける
    "seedTasteWeight": 0.0,        # 仮: 起点のある推薦に、プレイヤーの好みをどれだけ混ぜるか
    "chainEvery": 0,               # 仮: 起点があるとき、何マップごとに起点を動かすか。0: 動かさない
    "chainFirstWeight": 0.0,       # 仮: 起点が動いた後も残す、最初の起点の点の割合
    "mapsPerSong": 1,              # 仮: 一覧に入れる 1 曲のマップ（ほかの作者のマップ）の数。0: 上限なし
    "mapsPerArtist": 3,            # 一覧に入れる 1 アーティスト（同じアーティストの組）のマップの数。0: 上限なし（OctantSpecs 法 3.4）
    "sameSongCosine": 0.9,         # 仮: 2 つのマップを同じ曲とみなす、音楽のビューの平均コサインの下限
    "mixPriorDrafts": 20,          # 仮: ミックスの手札で、手札ごとの残した率を、手札に共通の率へ縮める重み（下見の数）
                                   # （1 人の下見での前向きの試しで、20 から 50 が一番損が少なかった）
    # 難易度
    "features": {"nps": 1.0, "jd": 1.0},  # 距離に使う譜面の特徴と、その重み。譜面のファイルから測った NPS と
                                   # ジャンプ距離（chart_timing）: 開発用の記録で後の失敗を一番よく分け（距離の AUC
                                   # 0.70 -> 0.78）、どのプレイスタイルにも値がある（pass は Standard だけ）
    "k": 5,                        # 距離として平均する、一番近い得意な譜面の数
    "normalPercentile": 80,        # クリアした譜面の距離のうち、この百分位までを「普通」とする
    "strongRecentClears": 200,     # 得意な譜面の枠になる、直近のクリアの数
    "minClears": 50,               # プレイスタイルのクリアがこれより少なければ、そのプレイスタイルは難易度を合わせない
    "minFailsForCurve": 30,        # 失敗がこれより少なければクリアの見込みの曲線を作らない。得意な譜面の一番古いクリア以降の
                                   # 失敗がこれより少なければ苦手な方向を作らない
    "directionShrinkPlays": 100,   # 仮: 学習した苦手な方向の重みと設定の重みが半々になる、プレイ（クリアと失敗）の数
    "toleranceSd": 1.0,            # 目標から許す距離。得意な譜面どうしの距離の広がり（四分位範囲 / 1.349、skill.spread）を単位にする
    "toleranceMaxSd": 2.0,         # 候補が少なすぎるとき、ここまで広げる
    "penaltyWeight": 1.0,          # 目標からの距離の罰の重み（`beta`）
    "challengeClearChance": 0.5,   # 挑戦の狙いが目指すクリアの見込み
    "warmupScale": 0.5,            # ウォームアップの目標 = これ x 普通の目標
    # 足切り（ceiling）: 狙いとは別に、プレイヤーが入れる上限
    "ceiling": False,              # ceilingFeatures のどれかでプレイヤーの上限を超える譜面と、floorFeatures の
                                   # どれかで下限を下回る譜面を外す
    "ceilingFeatures": ["pass"],   # 仮: 上限を決める譜面の指標（1 人の失敗を一番よく当てたのは Pass。Movement は
                                   # ほとんど足さず、同時に画面にあるノーツ（nps x ジャンプの時間）は、何も外さない
                                   # ほど高くしない限り遊びたい譜面を外した）。DB にない指標は使わない
    "ceilingPercentile": 97,       # 仮: 上限は直近のクリアのこの百分位（90 では遊んでいる譜面の半分が外れた）
    "floorFeatures": [],           # 下限を決める譜面の指標（NJS の下限は、同時に画面にあるノーツより失敗の当たりが
                                   # 悪く、多く外した）
    "floorBelowMedian": 3.0,       # 仮: 下限は直近のクリアの中央値からこれを引いたもの
    "ceilingMinClears": 20,        # 仮: プレイスタイルのクリア（自分の素のもの、続いて strongRecentClears までの
                                   # PlayerData のもの）がこれより少なければ、そこには上限も下限も付けない
}

AIMS = ("normal", "challenge", "warmup")

# 各数が取れる値: (下限, 上限, 整数か)。範囲はこの推薦側の守りで、OctantSpecs では決めていない: まともな設定は
# どれも入るほど広く、どの値でもホストを暴走させないほど狭い（recommend.py の許容幅を広げるループは 0.5 ずつ
# toleranceMaxSd まで進み、範囲のない百分位や割合は numpy を壊し、DB の手札も params を運ぶ）。整数かが真なら
# 小数を許さない。300.0 は通る。
BOUNDS = {
    "minViewWeight": (0, 1, False),
    "revisitDays": (0, 3650, True),
    "recentPlays": (0, 10**6, True),
    "genreTopShare": (0, 1, False),
    "narrowTopMaps": (1, 10**7, True),
    "narrowWiden": (2, 1000, True),
    "seedTasteWeight": (0, 1, False),
    "chainEvery": (0, 10**4, True),
    "chainFirstWeight": (0, 1, False),
    "mapsPerSong": (0, 10**4, True),
    "mapsPerArtist": (0, 10**4, True),
    "sameSongCosine": (-1, 1, False),
    "mixPriorDrafts": (0, 10**6, False),
    "k": (1, 10**4, True),
    "normalPercentile": (0, 100, False),
    "strongRecentClears": (1, 10**6, True),
    "minClears": (0, 10**6, True),
    "minFailsForCurve": (1, 10**6, True),
    "directionShrinkPlays": (0, 10**12, False),
    "toleranceSd": (0, 100, False),
    "toleranceMaxSd": (0, 100, False),
    "penaltyWeight": (0, 10**6, False),
    "challengeClearChance": (0.001, 0.999, False),  # log(c / (1 - c)) が有限のままであること（skill.py）
    "warmupScale": (0, 100, False),
    "ceilingPercentile": (0, 100, False),
    "floorBelowMedian": (0, 1e6, False),
    "ceilingMinClears": (1, 10**6, True),
}


def _number(value, low, high, whole):
    import math
    import numbers
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        return False
    if whole and value != int(value):
        return False
    return low <= value <= high


def why_invalid(params):
    """この params を使えない理由。使えれば None。知らないキーは resolve と同じく無視する（契約1（OctantCompass/
    OctantHarvest to OctantVector））。知っているキーで、値が BOUNDS の外か型が違うものは、扱い方に届く前に断る。
    recommend と describe（要求が失敗する）と、common/hands.py の手札の確認（手札が一覧から外れる）が確かめる。"""
    for key, value in (params or {}).items():
        if key not in DEFAULTS:
            continue
        if key in ("includeKnown", "ceiling"):
            if not isinstance(value, bool):
                return f"{key} must be true or false"
        elif key in ("ceilingFeatures", "floorFeatures"):
            if not isinstance(value, list) or not all(isinstance(f, str) for f in value):
                return f"{key} must be a list of chart metric names"
        elif key == "features":
            if not isinstance(value, dict) or not all(isinstance(f, str) and _number(w, 0, 10**6, False)
                                                      for f, w in value.items()):
                return "features must map feature names to finite weights between 0 and 1000000"
        else:
            low, high, whole = BOUNDS[key]
            kind = "a whole number" if whole else "a number"
            if not _number(value, low, high, whole):
                return f"{key} must be {kind} between {low} and {high}"
    return None


def resolve(params):
    """DEFAULTS に、`params` の知っているキーを上書きしたもの。"""
    out = dict(DEFAULTS)
    for k, v in (params or {}).items():
        if k in DEFAULTS:
            out[k] = v
    return out
