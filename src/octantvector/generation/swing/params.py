"""スイングのシミュレーションのすべての数の定数と、その既定値（正本はここ。モデルの説明は OctantSpecs
octantvector/swing.md）。どれも octantvector.toml の `[swing]` の表で上書きできる。例

    [swing]
    reset_cost = 0.4
    slider_max_gap_s = 0.06

エンコーダーは実際に使った値を encoder.json に記録するので、エンコーダーの版を上げずに値を変えると、出力を混ぜる
代わりにストア（store.py）が気づく。
"""
from .. import config
from ..errors import GenerationError

DEFAULTS = {
    # --- ノーツをスイングにまとめる
    "simultaneous_s": 0.001,      # これより近い同じ色のノーツは、同じスイングで打つ
    "slider_max_gap_s": 0.07,     # これより近い同じ色のノーツは、まだ 1 回のスイングになり得る（スライダー、ドットの連打）
    "slider_max_beats": 0.125,    # ... かつ矢印がこの拍（1/8）以内。さらに ... 以内なら
    "slider_max_gap_long_s": 0.1,  # ... この秒
    "slider_max_span_s": 0.3,     # スライダー（つながったノーツのチェーン）は長くてもこの長さ ...
    "slider_max_turn": 180.0,     # ... で、矢印の向きの変わりは合わせてこの度まで
    "slider_max_angle": 90.0,     # ... 矢印の違いがこの度以内なら（曲がったスライダー）
    "slider_geom_angle": 67.5,    # ... か、間の一歩が矢印からこの範囲を向くなら
    "dot_max_gap_s": 0.06,        # ドットが隣のノーツと 1 回のスイングになるのは、ここまで近いときだけ（リプレイ: 65 ms は 2 回）
    "swing_merge_s": 0.04,        # これより近い同じ色のノーツは、矢印によらず 1 回のスイング
    "min_swing_dt_s": 0.05,       # 片手のスイングの間は、少なくともこれとみなす（EBPM の上限が 600）
    # --- プレイヤーのモデル（位置はメートル。グリッドは BeatLeader の分析器と同じ）
    "lane_spacing_m": 0.6,
    "row_y_m": [0.0, 0.55, 1.05],
    "note_radius_m": 0.3,         # 入りと出の点は、スイングに沿ってノーツのこれだけ前と後にある
    "neutral_right_forehand": 247.5,  # 一番楽なスイングの向き（度、0 = 右、90 = 上）
    "neutral_right_backhand": 67.5,
    "neutral_left_forehand": 292.5,
    "neutral_left_backhand": 112.5,
    "dot_geom_blend": 0.5,        # 1 つだけのドットは前のスイングの逆に振り、一歩のほうへこれだけ向ける
    # --- パリティの探索（手ごとのフォアハンド/バックハンドの動的計画法）
    "parity_strain_weight": 0.8,  # パリティの手首の負担の重み
    "parity_rotation_weight": 0.75,  # （見込む向きからの度 / 180）^2 の重み
    "reset_cost": 0.6,            # 同じパリティを保つ費用（打たずにセーバーを戻す）
    "parity_long_gap_s": 0.65,    # 片手のスイングの間がこれ以上なら、*_long の重みを使う:
    "parity_strain_weight_long": 1.0,  # 近いスイングの間では、プレイヤーはモデルより気軽にリセットした
    "parity_rotation_weight_long": 1.0,  # が、長い間の後ではしなかった（BeatLeader のリプレイ）
    "reset_cost_long": 0.5,
    "bomb_reset_cost": 0.05,      # ... 前のスイングの後にセーバーが止まるところに爆弾があるとき
    "bomb_block_cost": 1.0,       # そういう爆弾を交互で通り抜ける費用
    "bomb_rest_dist_m": 0.75,     # 前の出の点からこの距離以内で、スイングの先にある爆弾
    "strain_falloff_start_s": 0.25,   # 間がこれより長いと、角度の負担は弱まる ...
    "strain_falloff_base": 4.0,       # ... 1 秒ごとにこの割合で
    "inverted_roll_deg": 90.0,    # これを超える手首のひねりは、逆さ（裏返し）の打ちと数える
    "linear_tolerance_deg": 22.5,
    # --- スイングごとの難しさ（BeatLeader の pass/tech の式。OctantSpecs octantvector/swing.md を参照）
    "strain_scale": 4.0,
    "distance_falloff": 2.668,
    "speed_falloff_base": 1.4,
    "stress_falloff": 2.0,
    "reset_speed_mult": 2.0,
    "stream_bonus": 1.05,
    "stress_w_strain": 0.05,
    "stress_w_reposition": 0.3,
    "stress_w_rotation": 0.2,
    "reposition_perp_scale": 0.6875,
    "reposition_avg_scale": 0.0625,
    "njs_buff_from": 24.0,
    "njs_buff_per": 0.01,
    "pass_windows": [8, 16, 32, 64, 128],
    "one_saber_nerf": 0.5,
    "pass_calibration": 0.825,
    "tech_skip_frac": 0.25,       # tech は、一番難しい（1 - これ）の割合のスイングを平均する
    "tech_scale": 14.0,
    "tech_pass_base": 1.4,
    "wall_buff_dodge": 1.1,       # 壁を避けながらのスイングの難しさの倍率（BeatLeader）...
    "wall_buff_crouch": 1.2,      # ... か、壁の下でしゃがみながら
    "wall_buff_margin_s": 0.5,    # ... 壁のこれだけ前から、これだけ後まで
    # --- 壁（hazards.py）。位置は列（グリッドを横切って 0..4）と行（0 床 .. 3+ 一番上の行より上）
    "head_x_lanes": 2.0,          # 立った頭: グリッドの中央
    "head_half_width_lanes": 0.15,
    "head_stand_row": 2.5,        # 立ったときの頭の高さ（頭上の壁は行 2 から始まる）...
    "head_crouch_row": 1.5,       # ... としゃがんだとき
    "head_reach_lanes": 1.5,      # 頭が横に踏み出すか傾ける距離
    "head_step_lanes": 0.25,      # 試す頭の位置の細かさ
    "head_time_step_s": 0.05,
    "head_move_cost": 0.5,        # 前の頭の位置から 1 列動くごとの費用 ...
    "head_crouch_cost": 1.0,      # ... としゃがむ費用（中央から外れた列で）。中央から外れると 1 列ごとに 1
    "crouch_depth_m": 0.5,        # しゃがむか立つときに数える頭の移動
    "art_wall_min_s": 0.02,       # これより短い壁は邪魔にならない: 背景 ...
    "wall_join_gap_s": 0.1,       # ... 同じ箱でこれだけ近く続く壁をまとめた後で（トンネル）
    "head_crouch_margin_s": 0.3,  # プレイヤーはしゃがむ壁のこれだけ前にしゃがみ、後もこれだけしゃがんだまま
    "side_wall_max_row": 3.0,     # 頭に当たらない壁のうち、この行より下で始まるものは side、高いものは art
    "art_wall_max_onsets_ps": 8.0,  # 1 秒あたりこれより多い異なる出だしの中で始まる壁 ...
    "art_wall_max_together": 6,     # ... か、同じ瞬間にこれより多い壁は、壁のアート
    # --- 爆弾（hazards.py）
    "swing_lead_s": 0.08,         # セーバーはスイングのこれだけ前に入り、これだけ後に出にある
    "bomb_window_s": 0.1,         # 爆弾が通る前後のこれだけの間のセーバーの位置を確かめる
    "bomb_near_m": 0.65,          # これより近く（およそ 1 列）を通るセーバーは、爆弾を避けて振る必要がある
    "bomb_far_m": 1.2,            # どちらのセーバーからも少なくともこれだけ離れた爆弾は背景
    "bomb_cluster_n": 4,          # 同時にこれだけの爆弾があると塊になる（爆弾の壁、形）
    "art_bomb_max_onsets_ps": 12.0,  # 1 秒あたりこれより多い異なる出だしの中の爆弾 ...
    "art_bomb_max_together": 12,     # ... か、同じ瞬間にこれより多い爆弾は、爆弾の絵
    "hazard_min_length_s": 30.0,  # これより短い譜面（ネタ、ジングル）は壁と爆弾の特徴を持たない
    "hazard_min_count": 10,       # 遊びに関わる爆弾（体が反応する壁）がこれより少なければ、爆弾（壁のリズム）の特徴を持たない
    # --- 壁と爆弾のリズム
    "rhythm_grid_beats": 0.25,    # このグリッドに乗る出だし ...
    "rhythm_max_ioi_beats": 2.0,  # ... で、出だしの間がこの拍以下の等間隔の並びにあるものはリズミカル
    # --- まとめの特徴
    "gap_bins_s": [0.15, 0.25, 0.4, 0.7],  # 同じ手のスイングの間の時間、ヒストグラムの境
    "full_swing_s": 0.4,          # スイングの間が少なくともこれだけある手は、大きな（精度の）スイングができる
    "wide_jump_m": 1.0,           # 入りから入りまでの手の移動を、跳びと数える ...
    "wide_jump_max_s": 0.3,       # ... 手の持ち時間がこれより短いとき
    "sustained_swings": 8,        # 続けて振る速さの最大（EBPM）の窓
    "snap_tolerance_beats": 0.02,
    "min_notes": 16,              # 色ノーツがこれより少ない譜面は特徴を持たない
    "max_njs": 50.0,              # ... NJS がこれより速い譜面も（ネタのマップは 100 や 1e20 と書く: 誰もノーツを見られない）
    "play_lanes": [-4.0, 7.0],    # この列か行の外の色ノーツ（Mapping Extensions か Noodle Extensions の
    "play_rows": [-3.0, 6.0],     # 999 列離れたアート）は外す: 誰も届かない ...
    "max_unreachable_frac": 0.1,  # ... そして、そこにあるノーツがこれより多い譜面は特徴を持たない
}

LIST_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, list)}


def load(overrides: dict | None = None) -> dict:
    """既定値 < 設定ファイルの `[swing]` の表 < 上書き。"""
    p = dict(DEFAULTS)
    for src in (config.section("swing"), overrides or {}):
        unknown = set(src) - set(DEFAULTS)
        if unknown:
            raise GenerationError(f"unknown [swing] setting(s): {', '.join(sorted(unknown))} (see swing/params.py)")
        for k, v in src.items():
            p[k] = [float(x) for x in v] if k in LIST_KEYS else type(DEFAULTS[k])(v)
    p["pass_windows"] = [int(w) for w in p["pass_windows"]]
    if len(p["gap_bins_s"]) != len(DEFAULTS["gap_bins_s"]) or len(p["row_y_m"]) != 3:
        raise GenerationError("[swing] gap_bins_s needs 4 edges and row_y_m 3 heights (they size the feature vector)")
    if any(len(p[k]) != 2 or p[k][0] > p[k][1] for k in ("play_lanes", "play_rows")):
        raise GenerationError("[swing] play_lanes and play_rows are [lowest, highest]")
    return p
