"""Song decoding in the heavy pass (archive.decode_audio, embed._work)."""
import io
import json
import shutil
import subprocess
import wave
import zipfile

import numpy as np
import pytest

from octantvector.generation import archive, embed
from octantvector.generation.encoders import get

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("songs are read with ffmpeg, which is not installed", allow_module_level=True)


def _tones(seconds, sr):
    """Two channels: 0.5 sin 220 Hz and 0.5 cos 330 Hz."""
    t = np.arange(int(seconds * sr)) / sr
    return np.c_[np.sin(2 * np.pi * 220 * t), np.cos(2 * np.pi * 330 * t)] * 0.5


def _wav(seconds, sr=8000):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((_tones(seconds, sr) * 32767).round().astype("<i2").tobytes())
    return b.getvalue()


def test_a_song_longer_than_the_limit_is_refused_before_it_is_decoded():
    y, sr = archive.decode_audio(_wav(3))
    assert sr == 8000 and y.shape == (24000,) and y.dtype == np.float32
    with pytest.raises(ValueError, match="longer than"):
        archive.decode_audio(_wav(3), max_s=2)


def test_the_song_is_the_mean_of_its_channels_whatever_the_block_size(monkeypatch):
    """ffmpeg's samples are taken a block at a time and made mono as they come: the result is the mean of the
    channels of the whole song, also when the first guess of its length is too short."""
    data = _wav(6)
    with wave.open(io.BytesIO(data)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)
    expected = (pcm.astype(np.float32) / 32768).mean(1, dtype=np.float32)
    monkeypatch.setattr(archive, "BLOCK_FRAMES", 1000)  # 48 blocks
    y, sr = archive.decode_audio(data)
    assert sr == 8000 and np.array_equal(y, expected)
    monkeypatch.setattr(archive, "available_memory", lambda: None)
    real = subprocess.run
    def no_duration(cmd, **kw):  # ffprobe does not know the length: the samples outgrow the first guess
        out = real(cmd, **kw)
        info = json.loads(out.stdout); info["format"].pop("duration", None); info["streams"][0].pop("duration", None)
        out.stdout = json.dumps(info)
        return out
    monkeypatch.setattr(subprocess, "run", no_duration)
    long = _wav(130)
    y, _ = archive.decode_audio(long)
    with wave.open(io.BytesIO(long)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2)
    assert np.array_equal(y, (pcm.astype(np.float32) / 32768).mean(1, dtype=np.float32))
    with pytest.raises(ValueError, match="longer than"):
        archive.decode_audio(long, max_s=100)


def test_a_file_that_is_not_a_song_is_an_error():
    with pytest.raises(ValueError, match="no audio"):
        archive.decode_audio(b"not a song at all" * 100)


class _Audio:
    kind = "audio"

    def prep(self, y, sr):
        return len(y)


def test_a_song_that_cannot_be_decoded_fails_only_the_audio_encoders(tmp_path, monkeypatch):
    info = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16}]}]}
    notes = {"_version": "2.2.0", "_notes": [{"_time": t / 2, "_lineIndex": t % 4, "_lineLayer": 0, "_type": t % 2,
                                              "_cutDirection": 1} for t in range(64)]}
    h = "cd" + "0" * 38
    (tmp_path / "maps" / "cd").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "cd" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("Expert.dat", json.dumps(notes))
        z.writestr("song.egg", _wav(3))
    monkeypatch.setattr(archive.decode_audio, "__defaults__", (2,))
    monkeypatch.setitem(embed._WORKER, "archive", tmp_path)
    monkeypatch.setitem(embed._WORKER, "encs", {"chart_patterns": get("chart_patterns"), "audio": _Audio()})
    _, _, sha, out, err = embed._work((h, ["chart_patterns", "audio"]))
    assert err is None and sha
    assert isinstance(out["audio"], ValueError)
    assert list(out["chart_patterns"]) == [f"{h}:Standard:Expert"]


class _AudioOutOfMemory(_Audio):
    def prep(self, y, sr):
        raise MemoryError("the features of a long song")


def test_an_audio_encoder_that_fails_on_a_decoded_song_fails_only_itself(tmp_path, monkeypatch):
    """A 4-hour song decodes, and then the encoder's features do not fit: the chart views of the map are kept."""
    info = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16}]}]}
    notes = {"_version": "2.2.0", "_notes": [{"_time": t / 2, "_lineIndex": t % 4, "_lineLayer": 0, "_type": t % 2,
                                              "_cutDirection": 1} for t in range(64)]}
    h = "ce" + "0" * 38
    (tmp_path / "maps" / "ce").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "ce" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("Expert.dat", json.dumps(notes))
        z.writestr("song.egg", _wav(3))
    monkeypatch.setitem(embed._WORKER, "archive", tmp_path)
    monkeypatch.setitem(embed._WORKER, "encs", {"chart_patterns": get("chart_patterns"), "big": _AudioOutOfMemory(),
                                                "audio": _Audio()})
    _, _, sha, out, err = embed._work((h, ["chart_patterns", "big", "audio"]))
    assert err is None and sha
    assert isinstance(out["big"], MemoryError)
    assert out["audio"] == 24000
    assert list(out["chart_patterns"]) == [f"{h}:Standard:Expert"]


def test_an_audio_encoder_that_finds_nothing_records_a_skip_not_an_error():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Quiet(_Audio):
        def embed(self, x):
            return None if x == 0 else np.ones(2, np.float32)

    led, store = Led(), Store()
    results = [("a" * 40, ["q"], "s", {"q": 0}, None), ("b" * 40, ["q"], "s", {"q": 1}, None)]
    embed._consume(led, {"q": Quiet()}, {"q": store}, iter(results), 2, 0, io.StringIO())
    assert led.rows == [("a" * 40, "skip", None, None)] and store.added == ["b" * 40]


def test_an_audio_vector_that_is_not_finite_is_an_error_and_is_not_stored():
    """A vocal view once made a NaN vector for a 16 s song; normalising turns it into zeros, a meaningless DB row."""
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Broken(_Audio):
        id = "q"

        def embed(self, x):
            return np.array([np.nan, 1.0], np.float32) if x == 0 else np.ones(2, np.float32)

    led, store = Led(), Store()
    results = [("a" * 40, ["q"], "s", {"q": 0}, None), ("b" * 40, ["q"], "s", {"q": 1}, None)]
    embed._consume(led, {"q": Broken()}, {"q": store}, iter(results), 2, 0, io.StringIO())
    assert [(h, r) for h, r, _, _ in led.rows] == [("a" * 40, "error")] and "not finite" in led.rows[0][3]
    assert store.added == ["b" * 40]


def test_an_encoder_with_a_batch_gets_its_maps_in_groups_and_the_rest_at_the_end():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Grouped(_Audio):
        batch = 2
        calls = []

        def embed_many(self, xs):
            self.calls.append(len(xs))
            return [None if x == 0 else np.ones(2, np.float32) for x in xs]

    enc, led, store = Grouped(), Led(), Store()
    results = [(str(n) * 40, ["g"], "s", {"g": n % 2}, None) for n in range(5)]
    embed._consume(led, {"g": enc}, {"g": store}, iter(results), 5, 0, io.StringIO())
    assert enc.calls == [2, 2, 1]
    assert store.added == ["1" * 40, "3" * 40] and [r[1] for r in led.rows] == ["skip"] * 3


def test_the_model_works_while_the_main_process_takes_in_further_results():
    import threading

    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    taken = threading.Event()

    class Slow(_Audio):
        overlapped = []

        def embed(self, x):  # the first map waits until the main process has taken in the third result
            if x == 0:
                self.overlapped.append(taken.wait(5))
            return np.ones(2, np.float32)

    def results():
        for n in range(4):
            if n == 2:  # asked for while the model holds the first map
                taken.set()
            yield str(n) * 40, ["s"], "s", {"s": n}, None

    enc, store = Slow(), Store()
    embed._consume(Led(), {"s": enc}, {"s": store}, results(), 4, 0, io.StringIO())
    assert enc.overlapped == [True] and store.added == [str(n) * 40 for n in range(4)]


def test_a_group_the_model_fails_on_is_recorded_as_errors_and_the_others_go_on():
    class Led:
        rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"
        added = []

        def add(self, h, x):
            self.added.append(h)

    class Flaky(_Audio):
        batch = 2

        def embed_many(self, xs):
            if 0 in xs:
                raise RuntimeError("the model broke")
            return [np.ones(2, np.float32) for _ in xs]

    led, store = Led(), Store()
    results = [(str(n) * 40, ["f"], "s", {"f": n}, None) for n in range(4)]
    embed._consume(led, {"f": Flaky()}, {"f": store}, iter(results), 4, 0, io.StringIO())
    assert [(r[0], r[1]) for r in led.rows] == [("0" * 40, "error"), ("1" * 40, "error")]
    assert "the model broke" in led.rows[0][3] and store.added == ["2" * 40, "3" * 40]


class OutOfMemoryError(RuntimeError):
    """Named like torch.cuda.OutOfMemoryError, which embed tells by its name."""


def _ledger_and_store():
    class Led:
        def __init__(self):
            self.rows = []

        def zip_sha(self, h, sha):
            return True

        def record(self, enc, fp, rows):
            self.rows += rows

    class Store:
        fingerprint = "f"

        def __init__(self):
            self.added = []

        def add(self, h, x):
            self.added.append(h)

    return Led(), Store()


def test_a_group_the_gpu_has_no_room_for_is_split_and_the_encoder_keeps_the_smaller_size():
    class Tight(_Audio):
        id = "t"
        batch = 4
        calls = []

        def embed_many(self, xs):
            self.calls.append(len(xs))
            if len(xs) > 1:
                raise OutOfMemoryError("CUDA out of memory")
            return [np.ones(2, np.float32) for _ in xs]

    (led, store), enc, log = _ledger_and_store(), Tight(), io.StringIO()
    results = [(str(n) * 40, ["t"], "s", {"t": n}, None) for n in range(8)]
    assert embed._consume(led, {"t": enc}, {"t": store}, iter(results), 8, 0, log) == {}
    assert enc.calls == [4, 2, 1, 1, 1, 1, 1, 1, 1, 1]  # the second group goes one map at a time from the start
    assert store.added == [str(n) * 40 for n in range(8)] and led.rows == []
    assert "t: the GPU had no room, 2 maps at a time from now" in log.getvalue()


def test_a_map_the_gpu_has_no_room_for_even_alone_is_not_recorded_so_a_later_run_computes_it():
    class Huge(_Audio):
        id = "h"
        tries = []

        def embed(self, x):
            if x == 1:
                self.tries.append(x)
                raise OutOfMemoryError("CUDA out of memory")
            return np.ones(2, np.float32)

    (led, store), enc = _ledger_and_store(), Huge()
    results = [(str(n) * 40, ["h"], "s", {"h": n}, None) for n in range(3)]
    assert embed._consume(led, {"h": enc}, {"h": store}, iter(results), 3, 0, io.StringIO()) == {"h": 1}
    assert enc.tries == [1, 1]  # tried again once with the cache emptied
    assert store.added == ["0" * 40, "2" * 40] and led.rows == []


def test_a_song_that_does_not_fit_in_the_free_memory_is_left_for_a_later_run(monkeypatch):
    """With too little free memory for the song, decoding stops before the OS kills the worker (which stops the whole
    run), and the map is not recorded, so a later run with room computes it."""
    monkeypatch.setattr(archive, "available_memory", lambda: archive.MEMORY_MARGIN + 1000)
    with pytest.raises(archive.NotEnoughMemory, match="GiB is free"):
        archive.decode_audio(_wav(3))  # 24000 samples, 96 kB
    monkeypatch.setattr(archive, "available_memory", lambda: None)  # not known: decode
    assert len(archive.decode_audio(_wav(3))[0]) == 24000

    (led, store), log = _ledger_and_store(), io.StringIO()
    results = [("a" * 40, ["q"], "s", {"q": archive.NotEnoughMemory("the song needs 2.5 GiB")}, None)]
    assert embed._consume(led, {"q": _Audio()}, {"q": store}, iter(results), 1, 0, log) == {"q": 1}
    assert led.rows == [] and store.added == [] and "left for a later run" in log.getvalue()


def test_a_song_bigger_than_the_memory_of_the_pc_is_an_error_not_left_for_later(monkeypatch):
    """Waiting for free memory does not help: a later run would meet the same song again and again."""
    monkeypatch.setattr(archive, "available_memory", lambda: None)
    monkeypatch.setattr(archive, "total_memory", lambda: archive.MEMORY_MARGIN + 1000)
    with pytest.raises(ValueError, match="more than this PC has") as ex:
        archive.decode_audio(_wav(3))
    assert not isinstance(ex.value, archive.LeftForLater)


def _ffmpeg_song(tmp_path, args, suffix):
    (tmp_path / "in.wav").write_bytes(_wav(3, sr=44100))
    out = tmp_path / f"song{suffix}"
    subprocess.run(["ffmpeg", "-v", "error", *args[0], "-i", str(tmp_path / "in.wav"), *args[1], str(out)], check=True)
    return out.read_bytes()


@pytest.mark.parametrize("kind", ["ogg", "mp4", "ogg with a video", "mp3"])
def test_songs_of_every_kind_maps_hold_are_read(tmp_path, kind):
    """A map's .egg is mostly an Ogg Vorbis, and sometimes an MP4 (AAC), an MP3, or an Ogg that also holds a Theora
    video: each is read, mono, at the song's own rate."""
    args, suffix = {
        "ogg": (([], ["-c:a", "libvorbis", "-f", "ogg"]), ".egg"),
        "mp4": (([], ["-c:a", "aac", "-f", "mp4"]), ".m4a"),
        "ogg with a video": ((["-f", "lavfi", "-i", "color=c=black:s=32x32:d=3"],
                              ["-map", "0:v", "-map", "1:a", "-c:v", "libtheora", "-c:a", "libvorbis", "-f", "ogg"]),
                             ".ogv"),
        "mp3": (([], ["-c:a", "libmp3lame", "-f", "mp3"]), ".mp3"),
    }[kind]
    data = _ffmpeg_song(tmp_path, args, suffix)
    y, sr = archive.decode_audio(data)
    assert sr == 44100 and y.dtype == np.float32 and abs(len(y) - 3 * sr) < 0.1 * sr
    rms = float(np.sqrt(np.mean(y[sr // 2:-sr // 2] ** 2)))
    assert 0.2 < rms < 0.3  # the mono mix of two 0.5 tones: 0.25
    with pytest.raises(ValueError, match="longer than"):
        archive.decode_audio(data, 2)


def test_a_song_cut_short_is_an_error_not_half_a_song(tmp_path):
    """A VBR MP3 cut in the middle: ffprobe gives the whole length from its header, and ffmpeg reads half without a
    word. Taking half would put the clips (15-75 % of the song) somewhere else."""
    (tmp_path / "in.wav").write_bytes(_wav(20, sr=44100))
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp_path / "in.wav"), "-c:a", "libmp3lame", "-q:a", "2",
                    str(tmp_path / "v.mp3")], check=True)
    data = (tmp_path / "v.mp3").read_bytes()
    assert abs(len(archive.decode_audio(data)[0]) - 20 * 44100) < 4410
    with pytest.raises(ValueError, match="of a 20 s song"):
        archive.decode_audio(data[:len(data) // 2])


@pytest.mark.parametrize("kind", ["ogg with a longer video", "vbr mp3 without a length", "raw aac",
                                  "matroska with a longer video"])
def test_a_whole_song_is_not_taken_for_one_cut_short(tmp_path, kind):
    """The length a song is held against is the audio stream's, not the file's (a video may run longer), and not one
    ffprobe only estimated from the bitrate (a VBR MP3 without a Xing header, a raw AAC stream)."""
    (tmp_path / "in.wav").write_bytes(_wav(20, sr=44100))
    pre, post = {
        "ogg with a longer video": (["-f", "lavfi", "-i", "color=c=black:s=32x32:d=40"],
                                    ["-map", "0:v", "-map", "1:a", "-c:v", "libtheora", "-c:a", "libvorbis", "-f", "ogg"]),
        "vbr mp3 without a length": ([], ["-c:a", "libmp3lame", "-q:a", "6", "-write_xing", "0", "-f", "mp3"]),
        "raw aac": ([], ["-c:a", "aac", "-f", "adts"]),
        "matroska with a longer video": (["-f", "lavfi", "-i", "color=c=black:s=32x32:d=40"],
                                         ["-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "libvorbis",
                                          "-f", "matroska"]),
    }[kind]
    out = tmp_path / "song.egg"
    subprocess.run(["ffmpeg", "-v", "error", *pre, "-i", str(tmp_path / "in.wav"), *post, str(out)], check=True)
    y, sr = archive.decode_audio(out.read_bytes())
    assert sr == 44100 and abs(len(y) - 20 * sr) < sr // 5


def test_a_song_that_outgrows_the_limit_while_read_is_refused_without_holding_it_all(monkeypatch):
    """ffprobe may not know the length (or say too little): the samples stop at the limit, not at the end of the song."""
    real = subprocess.run
    def no_duration(cmd, **kw):
        out = real(cmd, **kw)
        info = json.loads(out.stdout); info["format"].pop("duration", None); info["streams"][0].pop("duration", None)
        out.stdout = json.dumps(info)
        return out
    monkeypatch.setattr(subprocess, "run", no_duration)
    sizes = []
    real_room = np.empty
    monkeypatch.setattr(archive.np, "empty", lambda n, *a, **k: sizes.append(n) or real_room(n, *a, **k))
    with pytest.raises(ValueError, match="longer than"):
        archive.decode_audio(_wav(130), max_s=70)
    assert max(sizes) <= 70 * 8000 + 1


def test_a_decoder_stopped_by_a_signal_leaves_the_map_for_a_later_run(monkeypatch, tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nkill -9 $$\n")
    fake.chmod(0o755)
    real_probe = shutil.which("ffprobe")
    monkeypatch.setattr(archive, "need_ffmpeg", lambda: (str(fake), real_probe))
    with pytest.raises(archive.DecoderStopped, match="signal 9"):
        archive.decode_audio(_wav(3))
    assert issubclass(archive.DecoderStopped, archive.LeftForLater)
    assert issubclass(archive.NotEnoughMemory, archive.LeftForLater)


def test_a_decoder_stopped_in_the_middle_of_a_sample_leaves_the_map_for_a_later_run(monkeypatch, tmp_path):
    """Its last bytes end inside a sample: they are dropped, and the signal decides, not a numpy error."""
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nprintf 'abcdefgh123'\nkill -9 $$\n")
    fake.chmod(0o755)
    real_probe = shutil.which("ffprobe")
    monkeypatch.setattr(archive, "need_ffmpeg", lambda: (str(fake), real_probe))
    with pytest.raises(archive.DecoderStopped, match="signal 9"):
        archive.decode_audio(_wav(3))


def test_a_decoder_that_crashes_on_the_song_is_an_error_not_left_for_later(monkeypatch, tmp_path):
    """It would crash on the same song in every later run."""
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\nkill -SEGV $$\n")
    fake.chmod(0o755)
    real_probe = shutil.which("ffprobe")
    monkeypatch.setattr(archive, "need_ffmpeg", lambda: (str(fake), real_probe))
    with pytest.raises(ValueError, match="crashed on the song"):
        archive.decode_audio(_wav(3))


def test_an_os_other_than_linux_cannot_read_songs(monkeypatch):
    """macOS has /dev/fd, but /dev/fd/N shares the offset of the written file and reads nothing."""
    monkeypatch.setattr(archive.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="only on Linux"):
        archive.need_ffmpeg()


def test_an_os_without_dev_fd_cannot_read_songs(monkeypatch):
    monkeypatch.setattr(archive.os.path, "isdir", lambda p: False)
    with pytest.raises(ValueError, match="/dev/fd"):
        archive.need_ffmpeg()


def test_without_ffmpeg_songs_are_an_error_and_the_reader_is_part_of_the_audio_views(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="not on the PATH"):
        archive.decode_audio(_wav(3))
    from octantvector.generation.encoders.base import AudioEncoder

    class A(AudioEncoder):
        id, dim = "a", 2
    assert A().definition()["song_reader"] == archive.SONG_READER


def test_a_map_without_a_song_file_fails_only_the_audio_encoders(tmp_path, monkeypatch):
    """Some maps have no song in their zip (map 2466): its chart views are still computed."""
    info = {"_beatsPerMinute": 120, "_songFilename": "song.egg", "_difficultyBeatmapSets": [
        {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
            {"_difficulty": "Expert", "_beatmapFilename": "Expert.dat", "_noteJumpMovementSpeed": 16}]}]}
    notes = {"_version": "2.2.0", "_notes": [{"_time": t / 2, "_lineIndex": t % 4, "_lineLayer": 0, "_type": t % 2,
                                              "_cutDirection": 1} for t in range(64)]}
    h = "ce" + "0" * 38
    (tmp_path / "maps" / "ce").mkdir(parents=True)
    with zipfile.ZipFile(tmp_path / "maps" / "ce" / f"{h}.zip", "w") as z:
        z.writestr("Info.dat", json.dumps(info))
        z.writestr("Expert.dat", json.dumps(notes))
    monkeypatch.setitem(embed._WORKER, "archive", tmp_path)
    monkeypatch.setitem(embed._WORKER, "encs", {"chart_patterns": get("chart_patterns"), "audio": _Audio()})
    _, _, sha, out, err = embed._work((h, ["chart_patterns", "audio"]))
    assert err is None and sha
    assert isinstance(out["audio"], ValueError) and "no song file" in str(out["audio"])
    assert list(out["chart_patterns"]) == [f"{h}:Standard:Expert"]
