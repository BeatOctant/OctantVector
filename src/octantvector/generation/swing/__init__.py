"""Beat Saber の譜面のスイングシミュレーション: 想定されるパリティ、手首のひねり、リズムと動き、BeatLeader 風の
pass/tech の評価、スタイルタグ（学習した BeatSaver のタグと細かいタグ）。方法と出所は OctantSpecs
octantvector/swing.md。"""
from . import params  # noqa: F401
from .chart import bpm_of, bpm_regions, parse, standard_beatmaps
from .features import NAMES, chart_features  # noqa: F401


def standard_charts(mz):
    """開いた MapZip の Standard の譜面の [(difficulty, Chart)]。易しい順。"""
    regions = bpm_regions(mz)
    return [(d, parse(mz.json(fn), bpm_of(mz.info), njs, regions)) for d, fn, njs in standard_beatmaps(mz.info)]
