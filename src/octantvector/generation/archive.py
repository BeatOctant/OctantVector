"""手元のマップのアーカイブを読むだけの入口: maps/<hash[:2]>/<hash>.zip と download.journal.jsonl。"""
import io
import json
import os
import signal
import sys
import zipfile
from pathlib import Path

import numpy as np


def zip_path(archive: Path, version_hash: str) -> Path:
    return Path(archive) / "maps" / version_hash[:2] / f"{version_hash}.zip"


def commit_order(archive: Path) -> dict[str, int]:
    """確かめたすべての zip の 版の hash -> commit_seq（ディスク上の順。zip はこの順で読む）。"""
    seq = {}
    with open(Path(archive) / "download.journal.jsonl") as f:
        for line in f:
            if '"committed"' in line:  # ほかの出来事の多くは解析せずに飛ばす。出来事の種類は下で確かめる
                try:
                    j = json.loads(line)
                except ValueError:  # 書き込みが途中で切れた行。その zip は確定していない
                    continue
                if j.get("event") == "committed":
                    seq[j["hash"]] = j["commit_seq"]
    return seq


def read_json(zf: zipfile.ZipFile, name: str):
    """マップの .dat は JSON で、たいてい UTF-8（BOM はあってもなくても）、まれに UTF-16。json.loads は BOM から、
    BOM がなければ先頭のバイトから文字コードを見分ける（RFC 8259 / 4627）。"""
    return json.loads(zf.read(name))


class MapZip:
    """1 つのマップの zip を読み取り専用で開くコンテキストマネージャー。`data` は呼ぶ側がもう読んだ zip のバイト列で、
    ファイルをディスクから二度読まずに済む。"""

    def __init__(self, archive: Path, version_hash: str, data: bytes | None = None):
        self.path = zip_path(archive, version_hash)
        self.hash = version_hash
        self.data = data

    def __enter__(self):
        self.zf = zipfile.ZipFile(self.path if self.data is None else io.BytesIO(self.data))
        self.names = {n.lower(): n for n in self.zf.namelist()}  # Info.dat と info.dat など
        self.info = read_json(self.zf, self.names["info.dat"])
        return self

    def __exit__(self, *exc):
        self.zf.close()

    def song_bytes(self) -> bytes:
        """Info.dat にある曲のファイル。zip の中の名前とは大文字小文字を区別せずに照らす。なければ zip の最初の .egg、
        .ogg、.wav。それもなければ ValueError（曲のファイルのないマップがある。マップ 2466）。"""
        fn = self.info.get("_songFilename") or self.info.get("audio", {}).get("songFilename") or "song.egg"
        name = self.names.get(fn.lower()) if isinstance(fn, str) else None
        if name is None:
            name = next((n for n in self.zf.namelist() if n.lower().endswith((".egg", ".ogg", ".wav"))), None)
        if name is None:
            raise ValueError(f"the map has no song file ({fn!r} is not in its zip)")
        return self.zf.read(name)

    def cover_bytes(self) -> bytes | None:
        """Info.dat にあるカバー画像（`_coverImageFilename`、v4 は `coverImageFilename`）。zip の中の名前とは大文字小文字を
        区別せずに照らす。なければ zip の最初の .jpg、.jpeg、.png。それもなければ None。"""
        fn = self.info.get("_coverImageFilename") or self.info.get("coverImageFilename")
        name = self.names.get(fn.lower()) if isinstance(fn, str) and fn else None
        if name is None:
            name = next((n for n in self.zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))), None)
        return None if name is None else self.zf.read(name)

    def json(self, filename: str):
        return read_json(self.zf, self.names[filename.lower()])


# 計算する曲の最大の長さ。これより長いものは、ヘッダーの長さが壊れているか、ジョークかテストのマップ。曲はモノラルの
# float32 で持つので、4 時間の 44.1 kHz で 2.5 GB
MAX_SONG_S = 4 * 3600
BLOCK_FRAMES = 1 << 16  # ffmpeg の出力を一度に受け取るフレーム数
MEMORY_MARGIN = 512 * 2**20  # 曲を展開したあとにも残しておく空きメモリ（エンコーダーの準備の分）


class LeftForLater(Exception):
    """この PC のいまの状態のせいで曲を読めなかった。そのマップは記録せず、後の実行で計算する（embed._consume）。"""


class NotEnoughMemory(LeftForLater, MemoryError):
    """曲を展開するだけの空きメモリが、いまの PC にない。"""


class DecoderStopped(LeftForLater):
    """ffmpeg が曲を読み終える前に、外からシグナルで止められた（OS のメモリ不足など）。"""


# ffmpeg が自分で落ちたときのシグナル。同じ曲でまた落ちるので、後の実行に残さずエラーとして記録する。ほかのシグナル
# （SIGKILL、SIGTERM など）は外から止められたもの
CRASH_SIGNALS = frozenset(getattr(signal, n) for n in ("SIGSEGV", "SIGBUS", "SIGILL", "SIGABRT", "SIGFPE")
                          if hasattr(signal, n))


def _stopped(program, code):
    """シグナルで終わった program（code < 0）の例外: 自分で落ちたなら ValueError、外から止められたなら DecoderStopped。"""
    if -code in CRASH_SIGNALS:
        return ValueError(f"{program} crashed on the song (signal {-code})")
    return DecoderStopped(f"{program} was stopped by signal {-code}")


# 曲の読み方。音声のビューの定義（encoders/base.py の AudioEncoder.definition）に入るので、変えれば音声のビューの
# fingerprint が変わり、前の読み方のベクトルと混ざらない
SONG_READER = "ffmpeg: f32le at the song's own rate, the mean of its channels"


def need_ffmpeg():
    """-> (ffmpeg, ffprobe) のパス。どちらかがなければ ValueError。曲は /dev/fd から読ませる（decode_audio）ので、
    /dev/fd のない OS でも ValueError。"""
    import shutil
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("ffmpeg and ffprobe are needed to read songs, and they are not on the PATH")
    # Linux の /dev/fd/N は開き直すたびに先頭から読める。macOS と BSD では元の記述子と読む位置を分け合い、書いた
    # あとの終わりから読むので、どの曲も「音がない」になる
    if not sys.platform.startswith("linux") or not os.path.isdir("/dev/fd"):
        raise ValueError("songs are handed to ffmpeg through /dev/fd, which works only on Linux")
    return ffmpeg, ffprobe


def _meminfo(field):
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def available_memory():
    """-> いま使えるメモリのバイト数（Linux の /proc/meminfo の MemAvailable）。分からなければ None。"""
    return _meminfo("MemAvailable")


def total_memory():
    """-> この PC のメモリの全体のバイト数（MemTotal）。分からなければ None。"""
    return _meminfo("MemTotal")


def decode_audio(data: bytes, max_s: float = MAX_SONG_S):
    """-> (モノラルの float32 のサンプル, サンプルレート)。曲は ffmpeg で読む（拡張子が .egg でも、中身は Ogg Vorbis、
    動画の入った Ogg、MP4（AAC）、MP3、WAV などがある）。サンプルレートは曲のまま、モノラルはチャンネルの平均
    （SONG_READER）。ffmpeg には ffprobe の見たチャンネル数とレートで出させるので、途中で形の変わる曲もその形にそろう。

    ffprobe で max_s 秒より長いと分かる曲は、デコードする前に ValueError を投げる。読んでいる途中で max_s 秒を超えても
    ValueError。展開する曲と MEMORY_MARGIN が空きメモリに入らなければ NotEnoughMemory、ffmpeg が外からシグナルで
    止められたら DecoderStopped（どちらも記録せずに後の実行へ）。ffmpeg が自分で落ちた曲（CRASH_SIGNALS）は ValueError。
    ffmpeg の出力は BLOCK_FRAMES ずつ受け取ってその場でモノラルにするので、チャンネルごとのサンプルを一度に持たない。
    ffmpeg が読めない曲、音声のストリームの長さの 9 割に届かない曲（途中で切れた曲）、空くのを待ってもこの PC の
    メモリに入らない曲は ValueError。

    長さは音声のストリームのもの（なければファイル全体のもの）: 動画のほうが長い曲では、ファイル全体の長さは音の長さ
    ではない。ffprobe がビットレートから見積もっただけの長さ（Xing ヘッダーのない VBR の MP3、生の AAC）は当てに
    ならないので、長さが分からないものとして読む。"""
    import subprocess
    import tempfile
    ffmpeg, ffprobe = need_ffmpeg()
    # 名前のない一時ファイル（ワーカーが殺されても残らない）を /dev/fd から読ませる。MP4 は目次が後ろにあることがあり、
    # パイプでは読めない
    with tempfile.TemporaryFile() as t, tempfile.TemporaryFile() as err:
        t.write(data); t.flush()
        src = f"/dev/fd/{t.fileno()}"
        try:
            probe = subprocess.run([ffprobe, "-v", "warning", "-select_streams", "a:0", "-show_entries",
                                    "stream=sample_rate,channels,duration:format=duration", "-of", "json", src],
                                   capture_output=True, text=True, errors="replace", pass_fds=(t.fileno(),))
        except OSError as ex:
            raise DecoderStopped(f"ffprobe could not be started: {ex}") from ex
        if probe.returncode < 0:
            raise _stopped("ffprobe", probe.returncode)
        try:
            info = json.loads(probe.stdout)
            sr, ch = int(info["streams"][0]["sample_rate"]), int(info["streams"][0]["channels"])
        except (ValueError, KeyError, IndexError, TypeError):
            raise ValueError(f"ffprobe finds no audio in the song: {probe.stderr.strip()[-200:]}") from None
        if sr <= 0 or ch <= 0:
            raise ValueError(f"ffprobe finds no audio in the song ({sr} Hz, {ch} channels)")
        seconds, of_stream = None, False
        if "Estimating duration from bitrate" not in probe.stderr:  # ffprobe の警告。見積もりの長さは使わない
            for where in (info["streams"][0], info.get("format") or {}):
                try:
                    seconds = float(where["duration"])
                except (KeyError, TypeError, ValueError):
                    continue
                if seconds > 0:
                    of_stream = where is info["streams"][0]
                    break
                seconds = None
        if seconds is not None and seconds > max_s:
            raise ValueError(f"song is {seconds / 60:.0f} min, longer than the {max_s / 60:.0f} min that is computed")
        limit = int(max_s * sr)

        def room(frames):  # frames 個のサンプルを新しく持てるか（ffprobe の長さは短すぎることもあるので、広げるたびに見る）
            need, free, total = frames * 4, available_memory(), total_memory()
            if total is not None and need + MEMORY_MARGIN > total:  # 空くのを待っても入らない: 後の実行に残さない
                raise ValueError(f"the song needs {need / 2**30:.1f} GiB, more than this PC has ({total / 2**30:.1f} GiB)")
            if free is not None and need + MEMORY_MARGIN > free:
                raise NotEnoughMemory(f"the song needs {need / 2**30:.1f} GiB more and {free / 2**30:.1f} GiB is free")
            try:
                return np.empty(frames, np.float32)
            except MemoryError as ex:  # /proc/meminfo のない OS
                raise NotEnoughMemory(f"the song needs {need / 2**30:.1f} GiB more") from ex

        # ffprobe の長さに 1 秒の余裕。長さが分からなければ 1 分から広げる
        y = room(min(int((seconds if seconds is not None else 60) * sr) + sr, limit + 1))
        try:
            p = subprocess.Popen([ffmpeg, "-nostdin", "-v", "error", "-i", src, "-map", "0:a:0", "-t", str(max_s + 1),
                                  "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"],
                                 stdout=subprocess.PIPE, stderr=err, pass_fds=(t.fileno(),))
        except OSError as ex:
            raise DecoderStopped(f"ffmpeg could not be started: {ex}") from ex
        n, step, too_long = 0, BLOCK_FRAMES * ch * 4, False
        try:
            while block := p.stdout.read(step):
                # 最後のまとまりは、ffmpeg が書く途中で止められると、サンプルやフレームの途中で終わる。半端は捨てる
                # （止められたことは下の終了コードで分かる。半端のまま np.frombuffer に渡すと ValueError になり、
                # 外から止められた曲が、後の実行に残らずエラーとして記録されていた）
                m = np.frombuffer(block, np.float32, count=len(block) // (4 * ch) * ch).reshape(-1, ch).mean(
                    1, dtype=np.float32)
                if n + len(m) > limit:
                    too_long = True
                    break
                if n + len(m) > len(y):  # 倍に広げる（limit + 1 まで）。古いものと新しいものを一時に持つ分も見る
                    bigger = room(min(max(2 * len(y), n + len(m)), limit + 1))
                    bigger[:n] = y[:n]
                    y = bigger
                y[n:n + len(m)] = m
                n += len(m)
        finally:
            p.stdout.close()
            code = p.wait()
        if too_long:
            raise ValueError(f"song is longer than the {max_s / 60:.0f} min that is computed")
        err.seek(0)
        msg = err.read().decode(errors="replace").strip()
        if code < 0:
            raise _stopped("ffmpeg", code)
        if code != 0 or n == 0:
            raise ValueError(f"ffmpeg cannot decode the song: {msg[-200:] or 'no samples'}")
        # 途中で切れたかは音声のストリームの長さでだけ見る。ファイル全体の長さは、動画のほうが長い曲では音の長さではない
        if of_stream and n < 0.9 * seconds * sr:  # ffmpeg は途中で切れた MP3 を黙って半分だけ読むことがある
            raise ValueError(f"ffmpeg read {n / sr:.0f} s of a {seconds:.0f} s song: {msg[-200:] or 'cut short?'}")
    return y[:n], sr
