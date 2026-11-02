"""手元のマップのアーカイブを読むだけの入口: maps/<hash[:2]>/<hash>.zip と download.journal.jsonl。"""
import io
import json
import zipfile
from pathlib import Path


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
        fn = self.info.get("_songFilename") or self.info.get("audio", {}).get("songFilename") or "song.egg"
        name = self.names.get(fn.lower()) or next(
            n for n in self.zf.namelist() if n.lower().endswith((".egg", ".ogg", ".wav")))
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


# デコードする曲の最大の長さ: 曲全体を float32 でメモリに展開するので、とても長い曲で libsndfile がプロセスごと
# 落ちる（9 時間の曲、マップ 5149c）。30 分を超える Beat Saber の曲は冗談かテストのマップ
MAX_SONG_S = 1800


def decode_audio(data: bytes, max_s: float = MAX_SONG_S):
    """-> (モノラルの float32 のサンプル, サンプルレート)。まず libsndfile、だめなら ffmpeg（librosa 経由）。
    ヘッダーで max_s 秒より長い曲は、何もデコードする前に ValueError を投げる。"""
    import soundfile as sf
    try:
        f = sf.SoundFile(io.BytesIO(data))
    except Exception:
        f = None
    if f is not None:
        with f:
            if f.frames > max_s * f.samplerate:
                raise ValueError(f"song is {f.frames / f.samplerate / 60:.0f} min, longer than the "
                                 f"{max_s / 60:.0f} min that is decoded")
            try:
                return f.read(dtype="float32", always_2d=True).mean(1), f.samplerate
            except Exception:
                pass
    import librosa, tempfile
    with tempfile.NamedTemporaryFile(suffix=".egg") as t:
        t.write(data); t.flush()
        y, sr = librosa.load(t.name, sr=None, mono=True, duration=max_s + 1)
    if len(y) > max_s * sr:
        raise ValueError(f"song is longer than the {max_s / 60:.0f} min that is decoded")
    return y, sr
