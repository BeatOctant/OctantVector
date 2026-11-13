"""譜面のファイルから測る、遊ぶときの速さと密度（すべての譜面）。DB の charts.parquet の列を埋めるためのもので
（dbrelease.METRICS）、ビューとして DB に入れるものではない。

BeatSaver のカタログの njs は譜面に書かれたまま: NJS 0 以下は、ゲームが遊ぶときに難易度の既定値へ置き換える。
カタログの nps は BeatSaver の式で、BPM の区間のファイルの外の拍を 0 秒とするので負になる譜面がある（2026-11-07、
OctantSpecs open.md F16）。DB の nps はここで測った値で、カタログの値は使わない。ここでは
譜面を swing/chart.py で読み（BPM の変化をたどり、偽のノーツを外す）、ゲームが遊ぶときの値を測る:
  njs_play    遊ぶときの NJS: 書かれた NJS、0 以下なら難易度の既定値（DEFAULT_NJS）
  jd          ジャンプ距離（m）: ノーツが現れてから打つ位置までの距離の 2 倍
  reaction_s  反応時間（秒）: ノーツが現れてから打つ位置に来るまで（ジャンプの時間の半分）
  nps         色ノーツの数 / 最初から最後の色ノーツまでの秒（1 秒未満は 1 秒）
  nps_peak    各色ノーツから 2 秒の窓のノーツ数/秒の、ノーツで数えた 95 百分位（短い山の密度）
ジャンプの時間は、ゲームの式として知られている形（推測。版による違いは確かめていない）: 半ジャンプは 4 拍から
始め、NJS x 拍の秒 x 拍数が 17.999 を超えるあいだ半分にし、開始オフセットを足し、0.25 拍を下限にする。拍の秒は
Info.dat の BPM（譜面の途中の BPM の変化はジャンプに効かない）。
色ノーツのない譜面は nps と nps_peak が NaN（ジャンプの値はある）。ストアは float32 なので、それに入らない値
（NJS 3.4e38 のような、誰もノーツを見られない譜面のジャンプ距離）は inf になる。推薦側は有限でない値を値なしと
同じに扱う（難易度合わせのある推薦では候補にならない）。
"""
import numpy as np

from ...common import charts
from ..swing.chart import beatmaps, bpm_of, bpm_regions, jump_offsets, parse
from .base import Encoder, register

DEFAULT_NJS = {"Easy": 10.0, "Normal": 10.0, "Hard": 10.0, "Expert": 12.0, "ExpertPlus": 16.0}  # 書かれた NJS が 0 以下のとき
PEAK_WINDOW_S = 2.0
PEAK_PERCENTILE = 95
NAMES = ("njs_play", "jd", "reaction_s", "nps", "nps_peak")


def play_njs(njs, difficulty):
    njs = float(njs or 0)
    return njs if njs > 0 else DEFAULT_NJS.get(charts.difficulty(difficulty) or difficulty, 10.0)


def half_jump_beats(njs, bpm, offset):
    beat = 60.0 / bpm
    half = 4.0
    while njs * beat * half > 17.999 and half > 1e-6:
        half /= 2
    return max(half + float(offset or 0), 0.25)


def timing(chart, difficulty, offset):
    """読んだ 1 つの譜面（swing/chart.py の Chart）の (len(NAMES),)。"""
    njs = play_njs(chart.njs, difficulty)
    bpm = chart.bpm if chart.bpm and chart.bpm > 0 else 120.0
    half = half_jump_beats(njs, bpm, offset) * 60.0 / bpm  # 秒
    t = np.sort(np.array([n.t for n in chart.notes], float))
    if len(t):
        nps = len(t) / max(t[-1] - t[0], 1.0)
        # 窓 [t_i, t_i + PEAK_WINDOW_S) のノーツの数。始まりも searchsorted で数える（同じ時刻のノーツは同じ窓）
        peak = float(np.percentile((np.searchsorted(t, t + PEAK_WINDOW_S) - np.searchsorted(t, t)) / PEAK_WINDOW_S,
                                   PEAK_PERCENTILE))
    else:
        nps = peak = np.nan
    return np.array([njs, 2 * njs * half, half, nps, peak], np.float64)


@register
class ChartTiming(Encoder):
    kind = "zip"
    id = "chart_timing"
    version = "3"  # 2: v4 の音声のファイルの BPM の区間をたどる。3: 同じ時刻のノーツを NPS の山の窓に数える
    rows = "charts"
    norm = "none"
    inputs = ("chart",)
    names = NAMES
    dim = len(NAMES)

    def params(self):
        return {"charts": "every chart", "features": list(self.names), "defaultNjs": DEFAULT_NJS,
                "peakWindowS": PEAK_WINDOW_S, "peakPercentile": PEAK_PERCENTILE}

    def embed_zip(self, mz):
        """-> 読めるすべての譜面の {譜面のキー: 値}。"""
        out, first_error = {}, None
        offsets, regions = jump_offsets(mz.info), bpm_regions(mz)
        for c, d, fn, njs in beatmaps(mz.info):
            try:
                v = timing(parse(mz.json(fn), bpm_of(mz.info), njs, regions), d, offsets.get((c, d), 0))
            except Exception as ex:  # 壊れた譜面が 1 つあっても、ほかは使える
                first_error = first_error or ex
                continue
            out[charts.key(mz.hash, c, d)] = v
        if not out and first_error is not None:
            raise first_error
        return out
