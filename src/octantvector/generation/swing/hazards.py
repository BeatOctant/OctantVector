"""壁と爆弾を、プレイヤーにとっての意味で分ける（OctantSpecs octantvector/swing.md、「壁と爆弾」）。

壁が関わるのは頭だけ（セーバーは通り抜ける）なので、各壁はシミュレーションした頭と照らして決める:
  art          背景: Noodle の偽の壁、どのプレイヤーも反応できない群れの一部（壁のアート）、短すぎる、大きさが 0、
               または遊ぶ空間の外
  side         遊ぶ空間の中だが、中央に立つ頭には当たらない（外の列と低い壁）: 遊びの枠になり、動きは求めない
  dodge        立った頭に届く。プレイヤーは横に踏み出すか傾く
  crouch       立った頭には届くが、しゃがんだ頭には届かない（頭上の壁）
  unavoidable  届く範囲のどの頭の位置も覆う（偽の壁か MOD 前提の壁）
頭そのものも時間を追ってシミュレーションする: 各時点で、中央とそれまでの位置に一番近い空いた位置を取り、傾くより
安いときだけしゃがむ。その移動、中央を外れた時間としゃがんだ時間、左右の切り替えで、壁がどれだけ体を動かすかを測る。

爆弾が関わるのはセーバーだけなので、各爆弾は両手のシミュレーションしたセーバーの道と照らして決める:
  reset        爆弾のリセットを強いる（simulate.assign_parity）
  dodge        セーバーが通るころに、その道の近くにある: プレイヤーは避けて振る必要がある
  art          どちらのセーバーの道からも遠い、Noodle の偽の爆弾、または大きな塊（爆弾の絵）の一部: 背景
リズムは、遊びに関わる（art でない）壁と爆弾の出だしから決める: 1/4 拍のグリッドに乗って等間隔の並びの一部
（rhythmic）か、ノーツと同時（synced）。art の壁は、それだけでリズムの割合を出す。
"""
import bisect

import numpy as np


def on_grid(b, step, tol):
    """拍 b が `step` 拍のグリッドから tol 拍以内なら True。"""
    r = (b / step) % 1
    return min(r, 1 - r) * step <= tol


def rhythm(onsets, note_beats, P):
    """出だしの拍（並べたもの）-> (rhythmic の印, synced の印)。異なる出だしごとに 1 つ。異なる出だしも返す。"""
    tol = P["snap_tolerance_beats"]
    ons = []
    for b in onsets:
        if not ons or b - ons[-1] > tol:
            ons.append(b)
    n = len(ons)
    ioi = np.diff(ons) if n > 1 else np.zeros(0)
    even = np.zeros(n, bool)
    for i in range(1, n - 1):  # 前の間隔と後の間隔が同じなら、出だし i は等間隔の並びにある
        if ioi[i - 1] <= P["rhythm_max_ioi_beats"] and abs(ioi[i] - ioi[i - 1]) <= tol:
            even[i - 1:i + 2] = True
    snapped = np.array([on_grid(b, P["rhythm_grid_beats"], tol) for b in ons], bool)
    sync = np.array([_near(note_beats, b, tol) for b in ons], bool)
    return even & snapped, sync, ons


def _near(sorted_vals, v, tol):
    i = bisect.bisect_left(sorted_vals, v - tol)
    return i < len(sorted_vals) and sorted_vals[i] <= v + tol


# ---------------------------------------------------------------- 壁
class Head:
    """頭の位置の候補: 中央のまわりの列（x）x 立つ / しゃがむ（行の高さ）。"""

    def __init__(self, P):
        c, r, st = P["head_x_lanes"], P["head_reach_lanes"], P["head_step_lanes"]
        k = int(round(r / st))
        xs = c + st * np.arange(-k, k + 1)
        self.x = np.concatenate([xs, xs])
        self.y = np.concatenate([np.full(len(xs), P["head_stand_row"]), np.full(len(xs), P["head_crouch_row"])])
        self.crouch = self.y == P["head_crouch_row"]
        self.hw, self.c = P["head_half_width_lanes"], c
        self.neutral = int(np.argmin(np.abs(xs - c)))  # (中央, 立つ) の添字
        self.neutral_crouch = self.neutral + len(xs)

    def blocked(self, walls):
        """walls: x, y, w, h の (k, 4) 配列 -> (k, 候補) の bool。"""
        if not len(walls):
            return np.zeros((0, len(self.x)), bool)
        x, y, w, h = (walls[:, i:i + 1] for i in range(4))
        lo, hi = np.minimum(x, x + w), np.maximum(x, x + w)  # Mapping Extensions では大きさが負になり得る
        bot, top = np.minimum(y, y + h), np.maximum(y, y + h)
        return (lo < self.x + self.hw) & (hi > self.x - self.hw) & (bot < self.y) & (top > self.y)


def _crowded(times, max_ps, max_together):
    """物ごと: その前後 1 秒で、プレイヤーが反応できるより多い異なる出だし/秒の中で始まるか、同じ瞬間に多すぎる物が
    あるなら True: 障害ではなく壁か爆弾の絵。"""
    t = np.asarray(times, float)
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    starts = np.r_[True, np.diff(ts) > 1e-3]
    on = ts[starts]
    together = np.diff(np.r_[np.where(starts)[0], len(ts)])  # 異なる出だしあたりの物の数
    group = np.cumsum(starts) - 1
    per_s = np.searchsorted(on, on + 0.5, "right") - np.searchsorted(on, on - 0.5, "left")
    bad = (per_s > max_ps) | (together > max_together)
    out = np.zeros(len(t), bool)
    out[order] = bad[group]
    return out


def join_walls(walls, P):
    """同じ箱（と偽の印）で wall_join_gap_s 以内に続く壁 -> 1 つの壁: 0.02 秒の欠片を並べたしゃがむトンネルは、
    プレイヤーには 1 本の長い壁。"""
    out, last = [], {}
    for w in sorted(walls, key=lambda w: w[0]):
        key = (round(w[2], 3), round(w[3], 3), round(w[4], 3), round(w[5], 3), w[7])
        k = last.get(key)
        if k is not None and w[0] <= out[k][0] + out[k][1] + P["wall_join_gap_s"]:
            o = out[k]
            out[k] = (o[0], max(o[1], w[0] + w[1] - o[0])) + tuple(o[2:])
            continue
        last[key] = len(out)
        out.append(tuple(w))
    return out


def classify_walls(walls, P):
    """-> 壁ごとに 1 つの役割（"art"、"side"、"dodge"、"crouch"、"unavoidable"）。Noodle の偽の壁と、どのプレイヤーも
    反応できない群れの壁（壁のアート、ライトショー）は、位置によらず art。"""
    if not walls:
        return []
    head = Head(P)
    W = np.array([[w[2], w[3], w[4], w[5]] for w in walls], float)
    B = head.blocked(W)
    dur = np.array([w[1] for w in walls])
    size = np.abs(W[:, 2] * W[:, 3])
    crowd = _crowded([w[0] for w in walls], P["art_wall_max_onsets_ps"], P["art_wall_max_together"])
    roles = []
    for i in range(len(walls)):
        if walls[i][7] or crowd[i] or dur[i] < P["art_wall_min_s"] or size[i] == 0:
            roles.append("art")
        elif not B[i, head.neutral]:
            lo, hi = sorted((W[i, 0], W[i, 0] + W[i, 2]))
            bot = min(W[i, 1], W[i, 1] + W[i, 3])
            inside = lo < 4 and hi > 0 and bot < P["side_wall_max_row"]
            roles.append("side" if inside else "art")
        elif B[i].all():
            roles.append("unavoidable")
        elif not B[i, head.neutral_crouch]:
            roles.append("crouch")
        else:
            roles.append("dodge")
    return roles


def head_path(walls, roles, P):
    """体が反応する壁（dodge / crouch）の上でシミュレーションした頭 -> dict(travel_m, off_centre_s, crouch_s, switches)。"""
    out = {"travel_m": 0.0, "off_centre_s": 0.0, "crouch_s": 0.0, "switches": 0}
    body = [w for w, r in zip(walls, roles) if r in ("dodge", "crouch")]
    if not body:
        return out
    head = Head(P)
    step = P["head_time_step_s"]
    starts = np.array([w[0] for w in body]); ends = starts + np.array([max(w[1], 0.0) for w in body])
    crouch = np.array([r == "crouch" for r in roles if r in ("dodge", "crouch")])
    starts = starts - crouch * P["head_crouch_margin_s"]; ends = ends + crouch * P["head_crouch_margin_s"]
    boxes = np.array([[w[2], w[3], w[4], w[5]] for w in body], float)
    B = head.blocked(boxes)
    order = np.argsort(starts)
    t0, t1 = starts.min(), ends.max() + 2 * step
    cur = head.neutral
    side = 0
    active = []
    nxt = 0
    lane_m, depth_m = P["lane_spacing_m"], P["crouch_depth_m"]
    for t in np.arange(t0, t1, step):
        while nxt < len(order) and starts[order[nxt]] <= t + step:
            active.append(order[nxt]); nxt += 1
        active = [i for i in active if ends[i] >= t]
        free = ~B[active].any(0) if active else np.ones(len(head.x), bool)
        if not free.any():
            continue  # 行き場がない: そのまま
        cost = (P["head_move_cost"] * np.abs(head.x - head.x[cur]) + np.abs(head.x - head.c)
                + P["head_crouch_cost"] * head.crouch)
        cost[~free] = np.inf
        new = int(np.argmin(cost))
        out["travel_m"] += abs(head.x[new] - head.x[cur]) * lane_m + (head.crouch[new] != head.crouch[cur]) * depth_m
        cur = new
        off = head.x[cur] - head.c
        if abs(off) > 1e-9:
            out["off_centre_s"] += step
            s = 1 if off > 0 else -1
            if side and s != side:
                out["switches"] += 1
            side = s
        if head.crouch[cur]:
            out["crouch_s"] += step
    return out


def wall_spans(walls, roles, P):
    """スイングが BeatLeader の壁による上乗せを受ける窓をまとめた [(start, end, buff)]（dodge / crouch の壁）。"""
    out = []
    m = P["wall_buff_margin_s"]
    for w, r in zip(walls, roles):
        if r in ("dodge", "crouch"):
            out.append((w[0] - m, w[0] + max(w[1], 0.0) + m, P[f"wall_buff_{r}"]))
    return sorted(out)


def wall_buffs(times, spans):
    """並べたスイングの時刻 -> 各時刻を覆う区間の一番大きい上乗せ（なければ 1.0）。"""
    out, active, k = [], [], 0
    for t in times:
        while k < len(spans) and spans[k][0] <= t:
            active.append(spans[k]); k += 1
        active = [sp for sp in active if sp[1] >= t]
        out.append(max((sp[2] for sp in active), default=1.0))
    return out


# ---------------------------------------------------------------- 爆弾
def _hand_track(swings, P):
    """片手の要の時刻とセーバーの位置: スイングの少し前の入りと、少し後の出。"""
    T, X, Y = [], [], []
    for i, s in enumerate(swings):
        lead = P["swing_lead_s"]
        if i > 0:
            lead = min(lead, max(s.t - swings[i - 1].t_end, 0.0) / 2)
        lag = P["swing_lead_s"]
        if i + 1 < len(swings):
            lag = min(lag, max(swings[i + 1].t - s.t_end, 0.0) / 2)
        T += [s.t - lead, s.t_end + lag]; X += [s.entry[0], s.exit[0]]; Y += [s.entry[1], s.exit[1]]
    T = np.maximum.accumulate(np.array(T))
    return T, np.array(X), np.array(Y)


def inert_bombs(bombs, P):
    """爆弾ごと: Noodle の偽の爆弾と、どのプレイヤーも反応できない群れの爆弾（爆弾の絵）なら True。リセットを強いず、
    art になる。"""
    if not bombs:
        return []
    crowd = _crowded([b[0] for b in bombs], P["art_bomb_max_onsets_ps"], P["art_bomb_max_together"])
    return [bool(b[4] or c) for b, c in zip(bombs, crowd)]


def classify_bombs(bombs, both, grid, P, inert=None):
    """-> 爆弾ごとに 1 つの役割（"art"、"near"、"dodge"、"reset"。near: 遠くもなく、避けて振るほど近くもない）
    （chart.bombs の順）。Noodle の偽の爆弾と大きな塊（絵）の爆弾は art（`inert_bombs`）。"""
    if not bombs:
        return []
    inert = inert_bombs(bombs, P) if inert is None else inert
    reset = set(k for s in both if s.bomb_reset for k in s.reset_bombs)
    bt = np.array([b[0] for b in bombs])
    bx = np.array([grid.x(b[1]) for b in bombs]); by = np.array([grid.y(b[2]) for b in bombs])
    w = P["bomb_window_s"]
    ts = bt[:, None] + np.linspace(-w, w, 5)[None, :]
    dist = np.full(len(bombs), np.inf)
    for h in (0, 1):
        ss = [s for s in both if s.hand == h]
        if not ss:
            continue
        T, X, Y = _hand_track(ss, P)
        hx, hy = np.interp(ts, T, X), np.interp(ts, T, Y)
        dist = np.minimum(dist, np.hypot(hx - bx[:, None], hy - by[:, None]).min(1))
    roles = []
    for k in range(len(bombs)):
        if k in reset:
            roles.append("reset")
        elif inert[k]:
            roles.append("art")
        elif dist[k] <= P["bomb_near_m"]:
            roles.append("dodge")
        elif dist[k] >= P["bomb_far_m"]:
            roles.append("art")
        else:
            roles.append("near")
    return roles


# ---------------------------------------------------------------- まとめ
def _frac(a):
    return float(np.mean(a)) if len(a) else 0.0


def _offgrid(v):
    return abs(v - round(v)) > 1e-6


def summary(chart, both, walls, wall_roles, bomb_roles, span, P):
    """まとめた壁（analyse）からの壁と爆弾の特徴（features.NAMES の hazard のまとまり）。hazard_min_length_s より短い
    譜面ではどれも 0。"""
    if span < P["hazard_min_length_s"]:
        from .features import HAZARD_NAMES
        return dict.fromkeys(HAZARD_NAMES, 0.0)
    f = {}
    note_beats = sorted(n.b for n in chart.notes)
    bombs = chart.bombs
    roles = np.array(wall_roles)
    n_w = len(walls)
    art = roles == "art" if n_w else np.zeros(0, bool)
    f["wall_ps"] = float((~art).sum()) / span  # 遊びに関わる壁
    f["art_wall_ps"] = float(art.sum()) / span
    f["art_wall_frac"] = _frac(art)
    f["side_wall_ps"] = float((roles == "side").sum()) / span if n_w else 0.0
    f["art_wall_offgrid_frac"] = _frac([any(_offgrid(v) for v in w[2:6]) for w, a in zip(walls, art) if a])
    for r in ("dodge", "crouch", "unavoidable"):
        f[f"{r}_wall_ps"] = float((roles == r).sum()) / span if n_w else 0.0
    hp = head_path(walls, wall_roles, P)
    f["head_move_mps"] = hp["travel_m"] / span
    f["dodge_time_frac"] = min(hp["off_centre_s"] / span, 1.0)
    f["crouch_time_frac"] = min(hp["crouch_s"] / span, 1.0)
    f["dodge_switch_ps"] = hp["switches"] / span
    body = sorted(w[6] for w, r in zip(walls, wall_roles) if r in ("dodge", "crouch"))
    rb, sy, _ = rhythm(body, note_beats, P) if len(body) >= P["hazard_min_count"] else ([], [], [])
    f["wall_rhythm_frac"], f["wall_sync_frac"] = _frac(rb), _frac(sy)  # 体が反応する壁
    ra, _, _ = rhythm(sorted(w[6] for w, a in zip(walls, art) if a), note_beats, P)
    f["art_wall_rhythm_frac"] = _frac(ra)  # 音楽に合わせて動く背景
    f["vision_wall_frac"] = _vision(chart)

    br = np.array(bomb_roles)
    f["bomb_art_frac"] = _frac(br == "art") if len(br) else 0.0
    beats = sorted(b[3] for b, r in zip(bombs, bomb_roles) if r != "art")  # 遊びに関わる爆弾。絵は除く
    if len(beats) < P["hazard_min_count"]:  # 爆弾が数個では、マップのスタイルについて何も言えない
        beats = []
    f["bomb_ps"] = len(beats) / span
    f["bomb_dodge_ps"] = float((br == "dodge").sum()) / span if beats else 0.0
    rh, sy, ons = rhythm(beats, note_beats, P)
    f["bomb_rhythm_frac"], f["bomb_sync_frac"] = _frac(rh), _frac(sy)
    tol = P["snap_tolerance_beats"]
    real = sorted(b[3] for b in bombs if not b[4])  # 遊びに関わる塊は、その瞬間の本物の爆弾をすべて数える
    sizes = [bisect.bisect_right(real, o + tol) - bisect.bisect_left(real, o - tol) for o in ons]
    f["bomb_cluster_frac"] = _frac([s >= P["bomb_cluster_n"] for s in sizes])
    return f


def _vision(chart):
    """ノーツのうち、壁の箱の中に届くもの（近づくあいだ、壁の後ろか中に隠れる）の割合。"""
    if not chart.walls or not chart.notes:
        return 0.0
    nt = [n.t for n in chart.notes]
    hidden = np.zeros(len(nt), bool)
    for t, d, x, y, w, h, *_ in chart.walls:
        if d <= 0:
            continue
        lo, hi = min(x, x + w), max(x, x + w)
        bot, top = min(y, y + h), max(y, y + h)
        for k in range(bisect.bisect_left(nt, t), bisect.bisect_right(nt, t + d)):
            n = chart.notes[k]
            if lo < n.x + 1 and hi > n.x and bot < n.y + 1 and top > n.y:
                hidden[k] = True
    return float(hidden.mean())


def analyse(chart, both, grid, P, inert=None):
    """-> dict(walls: 譜面の壁をまとめたもの（join_walls）, wall_roles: まとめた壁ごとに 1 つ, bomb_roles)。"""
    walls = join_walls(chart.walls, P)
    return {"walls": walls, "wall_roles": classify_walls(walls, P),
            "bomb_roles": classify_bombs(chart.bombs, both, grid, P, inert)}

