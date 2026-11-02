"""すべての譜面（各 characteristic の各難易度）のパターンの統計。読み込みは swing/chart.py（譜面の形式 v2、v3、v4:
時刻は譜面の BPM の変化をたどり、Noodle と Mapping Extensions の位置を使い、プレイヤーが触れられない物は外す）。
回転のイベントは無視する。統計の意味はプレイスタイル（片手、矢印なし、回転）ごとに違う。検索は同じプレイスタイルの
譜面どうしだけを比べる（charts.py）。"""
import numpy as np

from ...common import charts
from ..swing.chart import beatmaps, bpm_of, parse
from .base import Encoder, register

GAPS = (0.126, 0.26, 0.51, 1.01)  # 同じ手の間隔のしきい値（拍。約 1/8、1/4、1/2、1）
NAMES = (["nps", "njs", "bomb_ps", "wall_ps", "wall_time_frac", "arc_ps", "chain_ps", "red_frac", "double_frac",
          "dot_frac", "up_down_frac", "horiz_frac", "diag_frac", "same_dir_repeat", "angle_change_mean", "flip180_frac",
          "x_spread", "y_mean", "outer_lane_frac", "top_row_frac", "bottom_row_frac", "crossover_frac",
          "density_cv", "density_p90_over_mean", "length_s"]
         + [f"gap_le_{g}" for g in GAPS] + ["gap_gt_1"])


def pattern_features(chart):
    """読み込んだ 1 つの難易度（swing/chart.py の Chart）の特徴ベクトル (len(NAMES),)。色ノーツが 16 個未満なら None。"""
    if len(chart.notes) < 16:
        return None
    notes = sorted(chart.notes, key=lambda n: n.b)
    b = np.array([n.b for n in notes], float)
    sec = np.array([n.t for n in notes], float)
    span = max(sec[-1] - sec[0], 1.0)
    x = np.array([n.x for n in notes], float); y = np.array([n.y for n in notes], float)
    c = np.array([n.c for n in notes]); d = np.array([n.cut for n in notes])
    ang = [n.ang for n in notes]
    bombs = [o for o in chart.bombs if not o[4]]
    walls = [o for o in chart.walls if not o[7]]
    double = np.mean(np.concatenate([[False], np.diff(b) < 1e-3]))
    gaps, angs, repeats, flips, cross = [], [], [], [], []
    for col in (0, 1):
        m = c == col
        bb, dd, xx = b[m], d[m], x[m]
        if len(bb) > 1:
            gaps += list(np.diff(bb))
            a = [v for v, k in zip(ang, m) if k]
            for a0, a1 in zip(a[:-1], a[1:]):
                if a0 is not None and a1 is not None:
                    ch = abs((a1 - a0 + 180) % 360 - 180)
                    angs.append(ch); flips.append(ch >= 135)
            repeats += list((dd[1:] == dd[:-1]) & (dd[1:] != 8))
        cross += list((xx >= 2) if col == 0 else (xx <= 1))  # 赤が右半分 / 青が左半分
    gaps = np.array(gaps) if gaps else np.array([1.0])
    hist = np.histogram(sec, bins=max(int(span // 4), 1))[0] / 4.0  # 4 秒の窓でのノーツ数/秒
    v = [len(notes) / span, chart.njs, len(bombs) / span, len(walls) / span, sum(w[1] for w in walls) / span,
         chart.arcs / span, chart.chains / span, float(np.mean(c == 0)), double,
         np.mean(d == 8), np.mean(np.isin(d, (0, 1))), np.mean(np.isin(d, (2, 3))), np.mean(np.isin(d, (4, 5, 6, 7))),
         np.mean(repeats) if repeats else 0, np.mean(angs) if angs else 0, np.mean(flips) if flips else 0,
         x.std(), y.mean(), np.mean((x == 0) | (x == 3)), np.mean(y == 2), np.mean(y == 0), np.mean(cross) if cross else 0,
         hist.std() / (hist.mean() + 1e-9), np.percentile(hist, 90) / (hist.mean() + 1e-9), span]
    prev = 0
    for g in GAPS:
        v.append(np.mean((gaps > prev) & (gaps <= g))); prev = g
    v.append(np.mean(gaps > GAPS[-1]))
    return np.nan_to_num(np.array(v, np.float32))


@register
class ChartPatterns(Encoder):
    kind = "zip"
    id = "chart_patterns"
    version = "3"  # 3: swing/chart.py で読む（BPM の変化、Noodle / Mapping Extensions、触れられない物を外す）
    rows = "charts"
    inputs = ("chart",)
    names = NAMES
    dim = len(NAMES)

    def params(self):
        return {"charts": "every chart", "features": list(self.names)}

    def embed_zip(self, mz):
        """-> ノーツが足りるすべての譜面の {譜面のキー: 特徴}。"""
        out, first_error = {}, None
        for c, d, fn, njs in beatmaps(mz.info):
            if self.play_styles and charts.play_style(c) not in self.play_styles:
                continue
            try:
                v = pattern_features(parse(mz.json(fn), bpm_of(mz.info), njs))
            except Exception as ex:  # 壊れた譜面が 1 つあっても、ほかは使える
                first_error = first_error or ex
                continue
            if v is not None:
                out[charts.key(mz.hash, c, d)] = v
        if not out and first_error is not None:
            raise first_error
        return out
