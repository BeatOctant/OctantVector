"""1 つの難易度のスイングのシミュレーション -> 決まった形の、読み取れる特徴ベクトル（名前は NAMES）。

まとまり: 速さ、リズム、手首の角度、パリティ、動き、複数ノーツ、壁、爆弾、評価、精度。各まとまりの意味と、
その上に作るスタイルタグは OctantSpecs octantvector/swing.md にある。
"""
import math
from dataclasses import replace

import numpy as np

from .chart import Chart, windows
from . import hazards
from .simulate import angdiff, simulate_all

SNAPS = [1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1, 1.5, 2]  # 同じ手の間隔の区分（拍。">2" も）

NAMES = [
    # 速さ
    "sps", "sps_p90", "ebpm_median", "ebpm_p90", "ebpm_sustained", "notes_per_swing",
    # リズム
    "rhythm_entropy", "rhythm_change", "offbeat_frac", "triplet_frac", "unsnapped_frac", "alternation", "double_frac",
    "gap_lt_0", "gap_lt_1", "gap_lt_2", "gap_lt_3", "gap_ge_3",
    # 手首の角度
    "roll_abs_mean", "roll_abs_p90", "inverted_frac", "roll_change_mean", "roll_speed_p90",
    "strain_mean", "strain_p90", "diag_frac", "horiz_frac", "offgrid_angle_frac",
    # パリティ
    "reset_frac", "bomb_reset_frac", "rebound_reset_frac", "dot_frac", "linear_frac", "rotation_mean", "rotation_p90",
    # 動き
    "reposition_mean", "reposition_p90", "hit_distance_mean", "hit_distance_p90", "wide_jump_frac",
    "crossover_frac", "deep_crossover_frac", "outer_lane_frac", "top_row_frac", "bottom_row_frac",
    "hand_x_std", "hand_y_std",
    # 複数ノーツ
    "stack_frac", "window_frac", "slider_frac", "chain_frac", "arc_ps",
    # 壁（hazards.py。HAZARD_NAMES は wall_ps から bomb_cluster_frac まで）
    "wall_ps", "art_wall_ps", "art_wall_frac", "side_wall_ps", "art_wall_offgrid_frac", "dodge_wall_ps", "crouch_wall_ps",
    "unavoidable_wall_ps", "head_move_mps", "dodge_time_frac", "crouch_time_frac", "dodge_switch_ps",
    "wall_rhythm_frac", "wall_sync_frac", "art_wall_rhythm_frac", "vision_wall_frac",
    # 爆弾（hazards.py）
    "bomb_ps", "bomb_dodge_ps", "bomb_art_frac", "bomb_rhythm_frac", "bomb_sync_frac", "bomb_cluster_frac",
    # 評価と尺度
    "pass", "tech", "pass_short", "pass_long", "sustain_ratio", "hands_ratio", "njs", "nps", "length_s",
    # 精度
    "full_swing_frac", "center_frac",
]

HAZARD_NAMES = NAMES[NAMES.index("wall_ps"):NAMES.index("bomb_cluster_frac") + 1]


def _p(a, q):
    return float(np.percentile(a, q)) if len(a) else 0.0


def _m(a):
    return float(np.mean(a)) if len(a) else 0.0


def _snap_class(db, tol):
    for i, s in enumerate(SNAPS):
        if abs(db - s) <= tol * max(1.0, s):
            return i
    return len(SNAPS) if db > SNAPS[-1] else -1  # -1: 普通でない（よくある刻みのどれにも当たらない）


def chart_features(chart: Chart, P):
    """-> (ベクトル (len(NAMES),) float32, スイング, 評価)。書かれたとおりには誰も遊べない譜面は (None, [], {}):
    ノーツが少なすぎる、NJS が max_njs を超える（誰もノーツを見られない）、またはノーツのうち max_unreachable_frac
    より多くが play_lanes / play_rows の外にある。ほかの譜面でその外にあるノーツは外す（誰も届かない）。"""
    (x0, x1), (y0, y1) = P["play_lanes"], P["play_rows"]
    reach = [n for n in chart.notes if x0 <= n.x <= x1 and y0 <= n.y <= y1]
    if (len(chart.notes) < P["min_notes"] or chart.njs > P["max_njs"]
            or len(chart.notes) - len(reach) > P["max_unreachable_frac"] * len(chart.notes)):
        return None, [], {}
    if len(reach) < len(chart.notes):
        chart = replace(chart, notes=reach)
        if len(reach) < P["min_notes"]:
            return None, [], {}
    both, rat, hz = simulate_all(chart, P)
    span = max(chart.length_s, 1.0)
    sw = [s for s in both if s.dt > 0]  # 同じ手に前のスイングがあるスイング
    t = np.array([s.t for s in both])
    f = {}

    # 速さ
    f["sps"] = len(both) / span
    win = np.histogram(t, bins=windows(span, 2))[0] / 2.0
    f["sps_p90"] = _p(win, 90)
    ebpm = np.array([60.0 / (2 * s.dt) for s in sw])
    f["ebpm_median"], f["ebpm_p90"] = _p(ebpm, 50), _p(ebpm, 90)
    k = int(P["sustained_swings"])
    best = 0.0
    for h in (0, 1):
        e = np.array([60.0 / (2 * s.dt) for s in sw if s.hand == h])
        if len(e) >= k:
            best = max(best, float(np.convolve(e, np.ones(k) / k, "valid").max()))
    f["ebpm_sustained"] = best
    f["notes_per_swing"] = len(chart.notes) / max(len(both), 1)

    # リズム
    tol = P["snap_tolerance_beats"]
    cls = [_snap_class(s.dbeat, tol) for s in sw]
    hist = np.bincount([c + 1 for c in cls], minlength=len(SNAPS) + 2).astype(float)
    pr = hist[hist > 0] / max(hist.sum(), 1)
    f["rhythm_entropy"] = float(abs((pr * np.log2(pr)).sum())) if len(pr) else 0.0
    per_hand = {h: [c for s, c in zip(sw, cls) if s.hand == h] for h in (0, 1)}
    ch = [a != b for h in (0, 1) for a, b in zip(per_hand[h], per_hand[h][1:])]
    f["rhythm_change"] = _m(ch)
    on_grid = hazards.on_grid
    # シンコペーション: 連打していない（前のスイングが 1/2 拍以上前の）手が 1/4 の裏拍に当たる
    f["offbeat_frac"] = _m([on_grid(s.b, 0.25, tol) and not on_grid(s.b, 0.5, tol) for s in sw if s.dbeat >= 0.5 - tol])
    f["unsnapped_frac"] = _m([not (on_grid(s.b, 1 / 8, tol) or on_grid(s.b, 1 / 6, tol)) for s in both])
    trip = [min((s.b * 3) % 1, 1 - (s.b * 3) % 1) <= tol * 3 and min((s.b * 4) % 1, 1 - (s.b * 4) % 1) > tol * 4
            for s in both]
    f["triplet_frac"] = _m(trip)
    seq = [s for i, s in enumerate(both) if i == 0 or s.t - both[i - 1].t >= P["simultaneous_s"]]
    f["alternation"] = _m([a.hand != b.hand for a, b in zip(seq, seq[1:])])
    f["double_frac"] = 1.0 - len(seq) / max(len(both), 1)
    edges = [0.0] + list(P["gap_bins_s"]) + [math.inf]
    dts = np.array([s.dt for s in sw])
    for i in range(len(edges) - 1):
        name = f"gap_lt_{i}" if i < len(edges) - 2 else f"gap_ge_{i - 1}"
        f[name] = _m((dts >= edges[i]) & (dts < edges[i + 1])) if len(dts) else 0.0

    # 手首の角度
    roll = np.array([abs(s.roll) for s in both])
    f["roll_abs_mean"], f["roll_abs_p90"] = _m(roll), _p(roll, 90)
    f["inverted_frac"] = _m(roll > P["inverted_roll_deg"])
    prev = {}
    rc, rs = [], []
    for s in both:
        q = prev.get(s.hand)
        if q is not None:
            d = abs(angdiff(q.roll, s.roll))
            rc.append(d); rs.append(d / max(s.dt, 1e-3))
        prev[s.hand] = s
    f["roll_change_mean"], f["roll_speed_p90"] = _m(rc), _p(rs, 90)
    strain = np.array([s.strain for s in sw])
    f["strain_mean"], f["strain_p90"] = _m(strain), _p(strain, 90)
    dirs = np.array([s.dir % 90 for s in both])
    axis = np.minimum(dirs, 90 - dirs)
    f["diag_frac"] = _m(np.abs(axis - 45) < 10)
    f["horiz_frac"] = _m([min(abs(angdiff(s.dir, 0)), abs(angdiff(s.dir, 180))) < 10 for s in both])
    f["offgrid_angle_frac"] = _m(np.minimum(np.array([s.dir % 45 for s in both]), 45 - np.array([s.dir % 45 for s in both])) > 5)

    # パリティ
    rest = [s for i, s in enumerate(both) if s.dt > 0]
    f["reset_frac"] = _m([s.reset for s in rest])
    f["bomb_reset_frac"] = _m([s.bomb_reset for s in rest])
    f["rebound_reset_frac"] = _m([s.reset and not s.bomb_reset for s in rest])
    f["dot_frac"] = _m([s.dot for s in both])
    f["linear_frac"] = _m([s.linear for s in rest])
    rot = np.array([s.rotation for s in rest])
    f["rotation_mean"], f["rotation_p90"] = _m(rot), _p(rot, 90)

    # 動き
    rep = np.array([s.reposition for s in rest]); hd = np.array([s.hit_distance for s in rest])
    f["reposition_mean"], f["reposition_p90"] = _m(rep), _p(rep, 90)
    f["hit_distance_mean"], f["hit_distance_p90"] = _m(hd), _p(hd, 90)
    f["wide_jump_frac"] = _m([s.hit_distance >= P["wide_jump_m"] and s.dt < P["wide_jump_max_s"] for s in rest])
    xs = np.array([n.x for n in chart.notes]); ys = np.array([n.y for n in chart.notes])
    cs = np.array([n.c for n in chart.notes])
    f["crossover_frac"] = _m(np.where(cs == 0, xs >= 2, xs <= 1))
    f["deep_crossover_frac"] = _m(np.where(cs == 0, xs >= 3, xs <= 0))
    f["outer_lane_frac"] = _m((xs <= 0) | (xs >= 3))
    f["top_row_frac"], f["bottom_row_frac"] = _m(ys >= 2), _m(ys <= 0)
    ex = np.array([s.entry[0] for s in both]); ey = np.array([s.entry[1] for s in both])
    hx = [ex[[s.hand == h for s in both]] for h in (0, 1)]
    f["hand_x_std"] = _m([a.std() for a in hx if len(a)])
    f["hand_y_std"] = float(ey.std()) if len(ey) else 0.0

    # 複数ノーツ
    for kind in ("stack", "window", "slider", "chain"):
        f[f"{kind}_frac"] = _m([s.kind == kind for s in both])
    f["arc_ps"] = chart.arcs / span

    # 壁と爆弾
    f.update(hazards.summary(chart, both, hz["walls"], hz["wall_roles"], hz["bomb_roles"], span, P))

    # 評価と尺度
    for k_ in ("pass", "tech", "pass_short", "pass_long", "hands_ratio"):
        f[k_] = rat[k_]
    f["sustain_ratio"] = rat["pass_short"] / max(rat["pass_long"], 1e-3) if rat["pass_long"] else 0.0
    f["njs"], f["nps"], f["length_s"] = chart.njs, len(chart.notes) / span, span

    # 精度
    f["full_swing_frac"] = _m([s.dt >= P["full_swing_s"] for s in sw])
    f["center_frac"] = _m(((xs == 1) | (xs == 2)) & (ys <= 1))

    v = np.array([f[n] for n in NAMES], np.float32)
    return np.nan_to_num(v), both, rat
