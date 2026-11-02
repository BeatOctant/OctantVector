"""スイングのシミュレーション: プレイヤーが譜面をどう振ると見込まれるか（OctantSpecs octantvector/swing.md）。

手ごとに: ノーツ -> スイング（重なり、窓、スライダー、チェーン）-> スイングの向き（ドットは流れから決める）->
角度の負担、リセット、爆弾の上の動的計画法でフォアハンドかバックハンドか -> スイングごとの手首のひねり、手の
位置、動き、BeatLeader 風の pass/tech の難しさ（壁による上乗せ付き。壁と爆弾は hazards.py）。
"""
import bisect
import math
from dataclasses import dataclass

from .chart import Chart, Note


@dataclass
class Swing:
    hand: int                 # 0 左（赤）、1 右（青）
    notes: list
    t: float                  # 最初のノーツ、秒
    t_end: float              # 最後のノーツ（チェーンの尾の時刻は分からないので頭の時刻）
    b: float                  # 最初のノーツ、拍
    kind: str                 # single | stack | window | slider | chain
    dot: bool                 # どのノーツもドット
    dir: float = 270.0        # スイングの向き、度（0 右、90 上）
    entry: tuple = (0.0, 0.0)
    exit: tuple = (0.0, 0.0)
    forehand: bool = True
    reset: bool = False       # この手の前のスイングと同じパリティ
    bomb_reset: bool = False
    reset_bombs: tuple = ()   # 爆弾のリセットを強いた爆弾の、chart.bombs への添字
    roll: float = 0.0         # このパリティの中立の向きからの、符号付きの手首のひねり、度
    dt: float = 0.0           # この手の前のスイングが始まってからの秒
    gap: float = 0.0          # 前のスイングの最後のノーツから、このスイングの最初のノーツまでの秒
    dbeat: float = 0.0        # この手の前のスイングが始まってからの拍
    strain: float = 0.0
    rotation: float = 0.0
    reposition: float = 0.0
    hit_distance: float = 0.0
    linear: bool = False
    diff: float = 0.0
    tech: float = 0.0


def angdiff(a, b):
    """符号付きの b - a。(-180, 180] の範囲。"""
    return (b - a + 540.0) % 360.0 - 180.0


def same_dir(a, b, tol):
    return abs(angdiff(a, b)) < tol


def unit(deg):
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


class Grid:
    def __init__(self, P):
        self.dx, self.rows = P["lane_spacing_m"], P["row_y_m"]

    def x(self, lane):
        return (lane - 1.5) * self.dx

    def y(self, row):
        r = self.rows
        if row <= 0:
            return r[0] + row * (r[1] - r[0])
        if row >= 2:
            return r[2] + (row - 2) * (r[2] - r[1])
        i = int(row)
        return r[i] + (row - i) * (r[i + 1] - r[i])

    def pos(self, n):
        return self.x(n.x), self.y(n.y)


# ---------------------------------------------------------------- まとめ方
def _joinable(prev: Note, n: Note, P):
    dt = n.t - prev.t
    if dt < P["simultaneous_s"] or dt < P["swing_merge_s"]:  # 誰もこんなに速く 2 回は振らない: 1 回のスイング
        return True
    if prev.tail is not None:
        return False
    close = dt <= P["slider_max_gap_s"]
    # 1/8 拍以内のスライダーの欠片（遅い曲では秒で上限を付ける）。矢印がそう示すとき
    beat_close = n.b - prev.b <= P["slider_max_beats"] + 1e-6 and dt <= P["slider_max_gap_long_s"]
    if not (close or beat_close):
        return False
    if (prev.ang is None or n.ang is None) and dt > P["dot_max_gap_s"]:
        return False
    if prev.ang is None and n.ang is None:
        return close  # ドットの連打は決まった窓の中だけ: ドットの vibro は別々のスイングのまま
    step = math.degrees(math.atan2(n.y - prev.y, n.x - prev.x)) % 360 if (n.x, n.y) != (prev.x, prev.y) else None
    if prev.ang is not None and n.ang is not None:
        return abs(angdiff(prev.ang, n.ang)) <= P["slider_max_angle"]
    arrow = prev.ang if prev.ang is not None else n.ang
    return step is None or same_dir(arrow, step, P["slider_geom_angle"])


def group(notes, hand, P):
    """1 色のノーツ（時刻順）-> [Swing]。"""
    swings, cur, turn = [], [], 0.0
    for n in notes:
        if cur and _joinable(cur[-1], n, P):
            arrows = [m.ang for m in cur if m.ang is not None]
            step = abs(angdiff(arrows[-1], n.ang)) if arrows and n.ang is not None else 0.0
            # 1 回のスイングは長くは曲がり続けられない: 長い、またはくねるストリームは何回かのスイング
            if n.t - cur[0].t <= P["slider_max_span_s"] and turn + step <= P["slider_max_turn"]:
                cur.append(n); turn += step
                continue
        if cur:
            swings.append(_swing(cur, hand, P))
        cur, turn = [n], 0.0
    if cur:
        swings.append(_swing(cur, hand, P))
    return swings


def _swing(ns, hand, P):
    if any(n.tail is not None for n in ns):
        kind = "chain"
    elif len(ns) == 1:
        kind = "single"
    elif ns[-1].t - ns[0].t < P["simultaneous_s"]:
        far = any(abs(a.x - b.x) > 1 or abs(a.y - b.y) > 1 for a, b in zip(ns, ns[1:]))
        kind = "window" if far else "stack"
    else:
        kind = "slider"
    return Swing(hand, list(ns), ns[0].t, ns[-1].t, ns[0].b, kind, all(n.ang is None for n in ns))


# ---------------------------------------------------------------- 向きと位置
def _circ_mean(angles):
    sx = sum(math.cos(math.radians(a)) for a in angles)
    sy = sum(math.sin(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sy, sx)) % 360


def resolve_directions(swings, grid, P):
    """矢印のスイングは（平均の）矢印の向き。ドットのスイングは流れを続ける: 前のスイングの逆で、形のほうへ向ける
    （複数のドットのスイングは、頭から尾への軸の向きのうち、それに近いほうを選ぶ）。"""
    prev = None
    for s in swings:
        arrows = [n.ang for n in s.notes if n.ang is not None]
        if arrows:
            s.dir = _circ_mean(arrows)
        else:
            want = (prev.dir + 180) % 360 if prev else 270.0
            a, b = s.notes[0], s.notes[-1]
            if (a.x, a.y) != (b.x, b.y):
                ax = math.degrees(math.atan2(grid.y(b.y) - grid.y(a.y), grid.x(b.x) - grid.x(a.x))) % 360
                s.dir = min((ax, (ax + 180) % 360), key=lambda c: abs(angdiff(want, c)))
            elif prev is not None:
                px, py = grid.pos(prev.notes[-1])
                cx, cy = grid.pos(a)
                if (px, py) != (cx, cy):
                    step = math.degrees(math.atan2(cy - py, cx - px)) % 360
                    d = angdiff(want, step)
                    if abs(d) < 90:
                        want = (want + d * P["dot_geom_blend"]) % 360
                s.dir = want
            else:
                s.dir = want
        ux, uy = unit(s.dir)
        s.notes.sort(key=lambda n: grid.x(n.x) * ux + grid.y(n.y) * uy if s.kind != "slider" else n.t)
        r = P["note_radius_m"]
        hx, hy = grid.pos(s.notes[0])
        last = s.notes[-1]
        tx, ty = (grid.x(last.tail[0]), grid.y(last.tail[1])) if last.tail else grid.pos(last)
        s.entry = (hx - ux * r, hy - uy * r)
        s.exit = (tx + ux * r, ty + uy * r)
        prev = s


# ---------------------------------------------------------------- パリティ
def neutral(hand, forehand, P):
    side = "right" if hand == 1 else "left"
    return P[f"neutral_{side}_{'forehand' if forehand else 'backhand'}"]


def falloff(gap, P):
    s = P["strain_falloff_start_s"]
    return 1.0 if gap < s else math.exp((s - gap) * math.log(P["strain_falloff_base"]))


def angle_strain(s, forehand, P):
    return (abs(angdiff(neutral(s.hand, forehand, P), s.dir)) / 180.0) ** 2


def _bombs_at_rest(prev, cur, bombs, times, grid, P, inert=None):
    """2 つのスイングの間の爆弾のうち、`prev` の後にセーバーが止まるところ（出口の先、近く）にあるものの添字。"""
    ux, uy = unit(prev.dir)
    lx, ly = grid.pos(prev.notes[-1])
    ex, ey = prev.exit
    lo, hi = bisect.bisect_right(times, prev.t_end + 1e-3), bisect.bisect_left(times, cur.t - 1e-3)
    out = []
    for k in range(lo, hi):
        if inert and inert[k]:  # 偽の爆弾と爆弾の絵は何も強いない（hazards.inert_bombs）
            continue
        bx, by = grid.x(bombs[k][1]), grid.y(bombs[k][2])
        if (bx - lx) * ux + (by - ly) * uy >= 0 and math.hypot(bx - ex, by - ey) <= P["bomb_rest_dist_m"]:
            out.append(k)
    return out


def assign_parity(swings, bombs, grid, P, inert=None):
    """スイングごとの（フォアハンド、バックハンド）の上の Viterbi。移る費用 = 新しいパリティの角度の負担 + 見込まれる
    向きからの回転（交互なら逆、リセットなら同じ）+ パリティを保つときのリセットの費用（爆弾が強いるなら安い）+
    そういう爆弾を交互で通り抜けるときの塞がれる費用。間が長ければどれも小さくする。"""
    n = len(swings)
    if n == 0:
        return
    INF = float("inf")
    times = [b[0] for b in bombs]
    sw = P["parity_strain_weight"]
    cost = [[sw * angle_strain(swings[0], False, P), sw * angle_strain(swings[0], True, P)]]
    back, bombed = [[0, 0]], [[]]
    for i in range(1, n):
        s, p = swings[i], swings[i - 1]
        gap = max(s.t - p.t_end, 0.0)
        f = falloff(gap, P)
        long = gap >= P["parity_long_gap_s"]
        swi = P["parity_strain_weight_long"] if long else sw
        rw = P["parity_rotation_weight_long"] if long else P["parity_rotation_weight"]
        rc = P["reset_cost_long"] if long else P["reset_cost"]
        rest = _bombs_at_rest(p, s, bombs, times, grid, P, inert) if bombs else []
        br = bool(rest)
        bombed.append(rest)
        row, brow = [INF, INF], [0, 0]
        for cp in (0, 1):
            base = swi * angle_strain(s, bool(cp), P)
            for pp in (0, 1):
                # 交互は動きを続ける（逆の向きを見込む）。リセットは同じ向きを見込む
                expect = p.dir if pp == cp else (p.dir + 180) % 360
                extra = rw * (abs(angdiff(expect, s.dir)) / 180.0) ** 2
                if pp == cp:
                    extra += P["bomb_reset_cost"] if br else rc
                elif br:
                    extra += P["bomb_block_cost"]
                c = cost[i - 1][pp] + (base + extra) * f
                if c < row[cp]:
                    row[cp], brow[cp] = c, pp
        cost.append(row); back.append(brow)
    par = [0] * n
    par[-1] = 0 if cost[-1][0] < cost[-1][1] else 1
    for i in range(n - 1, 0, -1):
        par[i - 1] = back[i][par[i]]
    for i, s in enumerate(swings):
        s.forehand = bool(par[i])
        s.reset = i > 0 and par[i] == par[i - 1]
        s.bomb_reset = s.reset and bool(bombed[i])
        s.reset_bombs = tuple(bombed[i]) if s.bomb_reset else ()
        s.roll = angdiff(neutral(s.hand, s.forehand, P), s.dir)


# ---------------------------------------------------------------- 動きと難しさ
def movement(swings, grid, P):
    for i, s in enumerate(swings):
        s.strain = angle_strain(s, s.forehand, P) * P["strain_scale"]
        if i == 0:
            s.strain = 0.0
            continue
        p = swings[i - 1]
        s.dt = max(s.t - p.t, P["min_swing_dt_s"])
        s.gap = max(s.t - p.t_end, 0.0)
        s.dbeat = max(s.b - p.b, 1e-3)
        s.strain *= falloff(s.gap, P)
        # BeatLeader の SwingMovement: 直角の位置の取り直し + 平均したスイングの中心のずれ
        dx, dy = s.entry[0] - p.exit[0], s.entry[1] - p.exit[1]
        px, py = unit(s.dir + 90)
        rep = abs(dx * px + dy * py) * P["reposition_perp_scale"]
        if i > 1:
            c = [((q.entry[0] + q.exit[0]) / 2, (q.entry[1] + q.exit[1]) / 2) for q in (s, p, swings[i - 2])]
            ab = ((c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2)
            bc = ((c[1][0] + c[2][0]) / 2, (c[1][1] + c[2][1]) / 2)
            rep += math.hypot(ab[0] - bc[0], ab[1] - bc[1]) * P["reposition_avg_scale"]
        rot = abs(angdiff(p.dir, s.dir))
        if not s.reset:
            rot = abs(rot - 180)
        rot /= 180
        if s.reset:
            rep /= 2; rot /= 2
        s.reposition, s.rotation = rep, rot
        s.hit_distance = math.hypot(s.entry[0] - p.entry[0], s.entry[1] - p.entry[1])
        back = (p.dir + 180) % 360
        step = math.degrees(math.atan2(grid.y(s.notes[0].y) - grid.y(p.notes[-1].y),
                                       grid.x(s.notes[0].x) - grid.x(p.notes[-1].x))) % 360
        moved = (s.notes[0].x, s.notes[0].y) != (p.notes[-1].x, p.notes[-1].y)
        s.linear = same_dir(back, s.dir, P["linear_tolerance_deg"]) and (not moved or same_dir(back, step, P["linear_tolerance_deg"]))


def njs_buff(njs, P):
    return 1.0 + P["njs_buff_per"] * (njs - P["njs_buff_from"]) if njs > P["njs_buff_from"] else 1.0


def swing_difficulty(both, njs, P, buffs=None):
    """両手の時刻順のスイングに BeatLeader の Difficulty.CalcSwingDiff をかける。`buffs`（スイングごとに 1 つ）は、
    避けたりしゃがんだりしながらのスイングへの壁による上乗せ（hazards.wall_buffs）。"""
    nb = njs_buff(njs, P)
    prev_hand = None
    for k, s in enumerate(both):
        dd = s.hit_distance / (s.hit_distance + P["distance_falloff"]) + 1
        speed = (1.0 / s.dt if s.dt else 0.0) * dd
        if s.reset:
            speed *= P["reset_speed_mult"]
        stress = (s.strain * P["stress_w_strain"] + s.reposition * P["stress_w_reposition"]
                  + s.rotation * P["stress_w_rotation"]) * dd
        low = 1.0 - P["speed_falloff_base"] ** (-speed)
        sm = P["stress_falloff"] * stress / (stress + P["stress_falloff"]) + 1.0
        s.diff = speed * low * sm * nb
        if prev_hand is not None and prev_hand != s.hand:
            s.diff *= P["stream_bonus"]
        if buffs is not None:
            s.diff *= buffs[k]
        prev_hand = s.hand
        s.tech = (s.strain + s.reposition + s.rotation) * nb


def _peak_mean(vals, w):
    """幅 w の後ろ向きの移動平均の最大（BeatLeader の CalcAverage: 窓が埋まるまでは 0）。"""
    if len(vals) < 2 or w < 1 or len(vals) <= w:
        return 0.0
    acc = sum(vals[:w])
    best = 0.0
    for i in range(w, len(vals)):
        acc += vals[i] - vals[i - w]
        best = max(best, acc / w)
    return best


def ratings(left, right, both, P):
    """-> dict(pass, tech, pass_short, pass_long, hands_ratio)。"""
    out = {"pass": 0.0, "tech": 0.0, "pass_short": 0.0, "pass_long": 0.0, "hands_ratio": 1.0}
    if not both:
        return out
    W = P["pass_windows"]
    d = lambda ss: [s.diff for s in ss]  # noqa: E731
    pl = sum(_peak_mean(d(left), w // 2) for w in W) / len(W)
    pr = sum(_peak_mean(d(right), w // 2) for w in W) / len(W)
    pc = sum(_peak_mean(d(both), w) for w in W) / len(W)
    hi, lo = max(pl, pr), min(pl, pr)
    ratio = min(lo / max(min(hi, pc), 1e-4), 1.0) if (hi > 0 or pc > 0) else 1.0
    passr = pc * (1.0 - (1.0 - ratio) * P["one_saber_nerf"]) * P["pass_calibration"]
    tech = 0.0
    if len(both) > 2:
        st = sorted(s.tech for s in both)[int(len(both) * P["tech_skip_frac"]):]
        tech = sum(st) / len(st) * (1.0 - P["tech_pass_base"] ** (-passr)) * P["tech_scale"]
    n = len(both)
    out.update({"pass": passr, "tech": tech, "pass_short": _peak_mean(d(both), min(W[0], n - 1)),
                "pass_long": _peak_mean(d(both), min(W[-1], n - 1)), "hands_ratio": ratio})
    return out


def simulate_all(chart: Chart, P):
    """-> (両手のスイングを時刻順に並べたもの, 評価の dict, hazards の dict(wall_roles, bomb_roles))。"""
    from . import hazards
    grid = Grid(P)
    inert = hazards.inert_bombs(chart.bombs, P)
    hands = []
    for h in (0, 1):
        ss = group([n for n in chart.notes if n.c == h], h, P)
        resolve_directions(ss, grid, P)
        assign_parity(ss, chart.bombs, grid, P, inert)
        movement(ss, grid, P)
        hands.append(ss)
    both = sorted(hands[0] + hands[1], key=lambda s: (s.t, s.hand))
    hz = hazards.analyse(chart, both, grid, P, inert)
    buffs = hazards.wall_buffs([s.t for s in both], hazards.wall_spans(hz["walls"], hz["wall_roles"], P))
    swing_difficulty(both, chart.njs, P, buffs)
    return both, ratings(hands[0], hands[1], both, P), hz
