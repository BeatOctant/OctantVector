"""The measured chart columns (generation/encoders/chart_timing.py) and the robust ruler of the skill (runtime/skill.py,
OctantSpecs open.md F11)."""
import io

import numpy as np

from octantvector.generation.encoders.chart_timing import half_jump_beats, play_njs, timing
from octantvector.generation.swing.chart import bpm_regions, parse
from octantvector.runtime.skill import FAR, spread, standardise


def _v3(times):
    return {"version": "3.3.0", "colorNotes": [{"b": b, "x": 1, "y": 0, "c": n % 2, "d": 1} for n, b in enumerate(times)]}


def test_a_written_njs_of_zero_or_less_plays_at_the_difficulty_default():
    assert play_njs(0, "ExpertPlus") == 16 and play_njs(-3, "Expert") == 12 and play_njs(None, "Hard") == 10
    assert play_njs(0, "Expert+") == 16  # old zips spell the difficulty their own way
    assert play_njs(22.5, "Easy") == 22.5


def test_the_half_jump_halves_while_the_half_jump_distance_passes_18_m():
    assert half_jump_beats(8, 120, 0) == 4  # 8 x 0.5 s x 4 = 16 m
    assert half_jump_beats(10, 120, 0) == 2  # 10 x 0.5 s x 4 = 20 m > 18
    assert half_jump_beats(16, 120, 0) == 2  # 16 x 0.5 x 4 = 32 > 18, 16 x 0.5 x 2 = 16
    assert half_jump_beats(16, 120, -1) == 1
    assert half_jump_beats(16, 120, -5) == 0.25  # at least a quarter beat
    assert half_jump_beats(3.4e38, 120, 0) == 0.25


def test_timing_follows_the_bpm_changes_and_the_jump():
    dat = _v3([0, 1, 2, 3, 4, 5, 6, 7, 8])
    dat["bpmEvents"] = [{"b": 4, "m": 240}]  # 4 beats at 120 (2 s), then 4 at 240 (1 s)
    njs, jd, rt, nps, peak = timing(parse(dat, 120, 0), "ExpertPlus", 0)
    assert njs == 16 and rt == 1.0 and jd == 32.0
    assert nps == 9 / 3.0
    assert peak == 3.0  # the densest 2 s windows (from 1 s and 1.5 s) hold 6 notes


def test_a_chart_without_notes_has_a_jump_but_no_density():
    njs, jd, rt, nps, peak = timing(parse(_v3([]), 150, 18), "Expert", 0.5)
    assert njs == 18 and np.isnan(nps) and np.isnan(peak) and jd > 0


def test_a_few_extreme_values_do_not_set_the_scale():
    nps = np.r_[np.linspace(2, 12, 999), 99999.0]
    njs = np.r_[np.linspace(10, 22, 999), 3.4e38]
    X = standardise(np.c_[nps, njs])
    normal = X[:999]
    assert 0.5 < np.ptp(normal[:, 0]) < 5 and 0.5 < np.ptp(normal[:, 1]) < 5  # the usual charts stay apart
    assert X[999, 1] == FAR and np.isfinite((X ** 2).sum())


def test_a_column_whose_quartiles_meet_falls_back_to_its_standard_deviation():
    """6 in 10 charts at NJS 16: the quartiles are both 16, and the column must not stay in its raw unit."""
    njs = np.r_[np.full(600, 16.0), np.linspace(10, 25, 400)]
    nps = np.linspace(2, 12, 1000)
    X = standardise(np.c_[nps, njs])
    assert np.ptp(X[:, 1]) < 6  # 15 in its raw unit (NJS 10 to 25), next to the NPS column's 2
    assert np.allclose(X[:, 1], (njs - 16) / njs.std())
    assert (standardise(np.c_[np.full(5, 3.0)]) == 0).all()  # one value only: divided by 1


def test_one_far_strong_chart_does_not_open_the_tolerance():
    d = np.r_[np.full(50, 1.0) + np.linspace(-0.2, 0.2, 50), 1e6]
    assert spread(d) < 0.3 < d.std()
    assert spread(np.full(5, 2.0)) == 1.0 and spread(np.array([])) == 1.0


def test_the_default_distance_uses_measured_columns_every_play_style_has():
    from octantvector.common.params import resolve
    from octantvector.generation.dbrelease import METRICS
    assert set(resolve(None)["features"]) <= {m for m, v in METRICS.items() if v == "chart_timing"}


class _Zip:
    def __init__(self, audio, regions, freq=44100):
        self.info = {"version": "4.0.0", "audio": {"bpm": 120, **audio}}
        self.files = {"audiodata.dat": {"version": "4.0.0", "songFrequency": freq,
                                        "bpmData": [dict(zip(("si", "ei", "sb", "eb"), r)) for r in regions]}}
        self.names = {k: k for k in self.files}

    def json(self, fn):
        return self.files[fn.lower()]


def _v4(times):
    return {"version": "4.0.0", "colorNotes": [{"b": b, "i": 0} for b in times], "colorNotesData": [{"x": 1, "y": 0, "c": 0, "d": 1}]}


def test_a_v4_chart_follows_the_bpm_regions_of_its_audio_file():
    """v4 keeps its BPM changes in the audio file. A note at the end of the last region, or past it, still has a
    time (BeatSaver counts it as 0 s there, which made its NPS negative)."""
    f = 44100
    mz = _Zip({"audioDataFilename": "AudioData.dat"}, [(0, 4 * f, 0, 8), (4 * f, 6 * f, 8, 16)])  # 120, then 240
    regions = bpm_regions(mz)
    ch = parse(_v4([0, 8, 16, 20]), 120, 16, regions)
    assert [round(n.t, 6) for n in ch.notes] == [0, 4, 6, 7]
    assert timing(ch, "ExpertPlus", 0)[3] == 4 / 7
    paused = _Zip({"audioDataFilename": "AudioData.dat"}, [(0, 2 * f, 0, 4), (2 * f, 3 * f, 4, 4), (3 * f, 5 * f, 4, 8)])
    assert [n.t for n in parse(_v4([6]), 120, 16, bpm_regions(paused)).notes] == [4.0]  # time passes in the 0 BPM part
    assert bpm_regions(_Zip({}, [])) is None and parse(_v4([8]), 120, 16, None).notes[0].t == 4.0


def test_a_v3_file_that_also_holds_an_empty_v2_note_list_is_read_as_v3():
    dat = {**_v3([0, 1, 2, 3]), "_notes": [], "_events": []}
    assert len(parse(dat, 120, 16).notes) == 4
    assert len(parse({"_version": "2.0.0", "_notes": [{"_time": 1, "_type": 0, "_lineIndex": 1, "_lineLayer": 0,
                                                       "_cutDirection": 1}]}, 120, 16).notes) == 1


def test_chart_names_follow_the_catalogue():
    """BeatSaver names a v2 difficulty by its rank ("Expert" with rank 9 is ExpertPlus there) and spells
    characteristics without spaces ("One Saber" is OneSaber)."""
    from octantvector.generation.swing.chart import beatmaps, jump_offsets
    info = {"_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_difficultyRank": 9, "_beatmapFilename": "a.dat", "_noteJumpStartBeatOffset": 1},
            {"_difficulty": "Hard", "_beatmapFilename": "b.dat"}]},
        {"_beatmapCharacteristicName": "One Saber", "_difficultyBeatmaps": [
            {"_difficulty": "Normal", "_difficultyRank": 3, "_beatmapFilename": "c.dat"}]}]}
    assert [(c, d) for c, d, _, _ in beatmaps(info)] == [("Standard", "ExpertPlus"), ("Standard", "Hard"),
                                                         ("OneSaber", "Normal")]
    assert jump_offsets(info)[("Standard", "ExpertPlus")] == 1


def test_charts_whose_ranks_meet_keep_their_own_names():
    """Two charts of one set with the same rank would get the same name, and one chart's values would go silently
    under the other's key: those keep the name they are written with."""
    from octantvector.generation.swing.chart import beatmaps, jump_offsets
    info = {"_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "a.dat", "_noteJumpStartBeatOffset": 1},
        {"_difficulty": "ExpertPlus", "_difficultyRank": 7, "_beatmapFilename": "b.dat", "_noteJumpStartBeatOffset": 2},
        {"_difficulty": "Normal", "_difficultyRank": 5, "_beatmapFilename": "c.dat"}]}]}
    assert [(d, fn) for _, d, fn, _ in beatmaps(info)] == [("Expert", "a.dat"), ("ExpertPlus", "b.dat"), ("Hard", "c.dat")]
    assert jump_offsets(info)[("Standard", "ExpertPlus")] == 2 and jump_offsets(info)[("Standard", "Expert")] == 1


def test_charts_whose_names_still_meet_are_left_out_and_the_rest_kept():
    """Expert and ExpertPlus both at rank 7 keep their written names, and a Hard at rank 9 would become ExpertPlus
    too: which of the two ExpertPlus the catalogue means cannot be told, so both are left out instead of one chart
    overwriting another. The other charts of the set are kept."""
    from octantvector.generation.swing.chart import beatmaps, jump_offsets
    info = {"_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "a.dat"},
        {"_difficulty": "ExpertPlus", "_difficultyRank": 7, "_beatmapFilename": "b.dat"},
        {"_difficulty": "Hard", "_difficultyRank": 9, "_beatmapFilename": "c.dat", "_noteJumpStartBeatOffset": 1}]}]}
    assert [(d, fn) for _, d, fn, _ in beatmaps(info)] == [("Expert", "a.dat")]
    assert jump_offsets(info) == {("Standard", "Expert"): 0}


def test_a_chart_listed_twice_in_a_set_is_one_chart():
    """Info.dat lists the same chart (name, rank and file) twice: the catalogue has it once, so it is kept once."""
    from octantvector.generation.swing.chart import beatmaps, jump_offsets
    info = {"_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": "Hard", "_difficultyRank": 5, "_beatmapFilename": "Hard.dat", "_noteJumpStartBeatOffset": 1},
        {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "Expert.dat"},
        {"_difficulty": "Hard", "_difficultyRank": 5, "_beatmapFilename": "Hard.dat", "_noteJumpStartBeatOffset": 2}]}]}
    assert [(d, fn) for _, d, fn, _ in beatmaps(info)] == [("Hard", "Hard.dat"), ("Expert", "Expert.dat")]
    assert jump_offsets(info) == {("Standard", "Hard"): 1, ("Standard", "Expert"): 0}


def test_two_files_under_one_name_are_both_left_out():
    from octantvector.generation.swing.chart import beatmaps
    info = {"_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
        {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "Expert.dat"},
        {"_difficulty": "Expert", "_difficultyRank": 7, "_beatmapFilename": "ExpertStandard.dat"},
        {"_difficulty": "Hard", "_difficultyRank": 5, "_beatmapFilename": "Hard.dat"}]}]}
    assert [(d, fn) for _, d, fn, _ in beatmaps(info)] == [("Hard", "Hard.dat")]


def test_notes_at_the_same_time_count_in_each_others_peak_window():
    """Two notes at one time are 2 notes in the 2 s window of each: the peak is 1 NPS for both."""
    peak = timing(parse(_v3([0, 0]), 120, 0), "ExpertPlus", 0)[4]
    assert peak == 1.0


def test_numbers_too_big_for_a_float_in_the_audio_file_are_left_out():
    audio = {"audioDataFilename": "AudioData.dat"}
    assert bpm_regions(_Zip(audio, [(0, 10**400, 0, 8), (0, 88200, 0, 4)])) == [(0.0, 0.0, 120.0)]
    assert bpm_regions(_Zip(audio, [(0, 88200, 0, 4)], freq=10**400)) is None


def test_an_unreadable_audio_file_falls_back_to_the_info_bpm():
    """A broken AudioData.dat must not fail every chart of the map: the charts are timed by the Info.dat BPM."""
    class Broken(_Zip):
        def json(self, fn):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
    mz = Broken({"audioDataFilename": "AudioData.dat"}, [])
    assert bpm_regions(mz) is None
    assert parse(_v4([8]), 120, 16, bpm_regions(mz)).notes[0].t == 4.0


def test_a_joke_chart_billions_of_years_long_still_has_patterns():
    """Notes at beats near 1e18 once made the density windows too many for memory (MemoryError)."""
    from octantvector.generation.encoders.chart_patterns import pattern_features
    from octantvector.generation.swing.chart import MAX_WINDOWS, windows
    assert windows(0.5, 2) == 1 and windows(61, 2) == 30 and windows(1e18, 2) == MAX_WINDOWS
    v = pattern_features(parse(_v3([b / 2 for b in range(200)] + [3.9e18]), 120, 16))
    assert v is not None and np.isfinite(np.asarray(v, float)).all()


def test_evaluation_reads_the_measured_nps_not_the_catalogue(tmp_path):
    """eval（personal の難易度合わせ、neighbours の最大 NPS）は譜面のファイルから測った nps を使う。色ノーツのない
    譜面（NaN）は値がない。"""
    import pytest
    from types import SimpleNamespace
    from octantvector.cli import _with_max_nps
    from octantvector.common import charts
    from octantvector.generation import GenerationError
    from octantvector.generation.dbrelease import chart_metric
    from octantvector.generation.encoders.chart_timing import NAMES, ChartTiming
    from octantvector.generation.store import Store
    with pytest.raises(GenerationError):
        chart_metric(tmp_path, "nps")  # not computed: say so instead of evaluating without NPS
    def row(nps):
        v = np.zeros(len(NAMES), np.float32)
        v[NAMES.index("nps")] = nps
        return v
    s = Store(ChartTiming(), tmp_path)
    s.add(charts.key("a" * 40, "Standard", "Expert"), row(4.0))
    s.add(charts.key("a" * 40, "Standard", "Expert+"), row(7.0))  # an old zip's spelling: the catalogue's ExpertPlus
    s.add(charts.key("a" * 40, "OneSaber", "Expert"), row(9.0))   # not Standard: not in the maximum
    s.add(charts.key("b" * 40, "Standard", "Easy"), row(np.nan))  # no colour notes
    s.flush()
    view, nps = chart_metric(tmp_path, "nps")
    assert view == f"chart_timing@{ChartTiming.version}"
    assert nps == {charts.key("a" * 40, "Standard", "Expert"): 4.0, charts.key("a" * 40, "Standard", "ExpertPlus"): 7.0,
                   charts.key("a" * 40, "OneSaber", "Expert"): 9.0}
    log = io.StringIO()
    view2, md = _with_max_nps(SimpleNamespace(store=tmp_path), {"a" * 40: {"key": "1"}, "b" * 40: {"key": "2"}}, log)
    assert view2 == view and md == {"a" * 40: {"key": "1", "maxnps": 7.0}, "b" * 40: {"key": "2", "maxnps": None}}
    assert "1 of 2 maps" in log.getvalue()


def test_the_difficulty_fit_keeps_difficulties_without_a_measured_nps():
    """eval personal: 測った NPS のない難易度は NaN で残し（-2 の罰にしない）、測った NPS のないプレイは幅に入れない。
    何件に値があったかを出す。"""
    import pytest
    from octantvector.cli import _difficulty_points
    from octantvector.common import charts
    from octantvector.generation import GenerationError
    a, b, c = "a" * 40, "b" * 40, "c" * 40
    def rec(*diffs):
        return ({}, {"diffs": [{"characteristic": "Standard", "difficulty": d, "njs": 16.0} for d in diffs]})
    recs = {a: rec("Hard", "Expert"), b: rec("Expert"), c: rec()}
    nps = {charts.key(a, "Standard", "Expert"): 5.0}
    play = {"outcome": "finished", "mode": "SoloStandard", "speed": 1.0}
    plays = [{**play, "hash": a, "diff": "Expert"}, {**play, "hash": b, "diff": "Expert"}]
    md = {a: {"key": "1"}, b: {"key": "2"}}
    log = io.StringIO()
    view, diffs, pts = _difficulty_points(("chart_timing@2", nps), recs, [a, b, c], plays, md, log)
    assert view == "chart_timing@2" and pts == [("1", 5.0, 16.0)]
    assert np.isnan(diffs[a][0, 0]) and diffs[a][1].tolist() == [5.0, 16.0]
    assert diffs[b].shape == (1, 2) and np.isnan(diffs[b][0, 0]) and diffs[c].shape == (0, 2)
    assert "1 of 3 Standard difficulties" in log.getvalue() and "1 of 2 finished plays" in log.getvalue()
    with pytest.raises(GenerationError):  # no measured play: no band to fit to
        _difficulty_points(("chart_timing@2", {}), recs, [a, b, c], plays, md, io.StringIO())
