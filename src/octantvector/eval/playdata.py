"""プレイヤー自身のデータを読む関数。どの関数も渡されたパスを読むだけで、呼ぶ側が戻り値を保存しない限り、
何もコピー、アップロード、書き込みをしない。各 signal がラベルとして何を意味するかは OctantSpecs 法 4.1。
よくある出所（パスはユーザー自身のもの。ここでは場所を決めつけない）:

  votedSongs.json        BeatSaverVoting（UserData/）          read_votes
  *.bplist               プレイリスト（Playlists/）            read_playlists
  PlayerData.dat         ゲームのプロフィール（AppData/LocalLow/...） read_player_data
  beatsaber.db           DataRecorder（UserData/DataRecorder） read_datarecorder
  *.bsor / *.dat         BeatLeader / ScoreSaber のリプレイ    replay_hashes（ファイル名だけ）
  CustomLevels/          入れてあるマップ                      installed_keys
"""
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ..common.charts import difficulty

HASH_RE = re.compile(r"[0-9A-Fa-f]{40}")


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_votes(path) -> dict:
    """-> {版の hash: "Upvote" | "Downvote"}"""
    return {h.lower(): v["voteType"] for h, v in _load_json(path).items()}


def read_playlists(paths) -> Counter:
    """-> 渡した .bplist のファイルでの、版の hash の Counter。"""
    c = Counter()
    for p in paths:
        for s in _load_json(p).get("songs", []):
            if s.get("hash"):
                c[s["hash"].lower()] += 1
    return c


def read_player_data(path):
    """-> (お気に入りの hash, {hash: playCount の合計}, {hash: 有効なスコアのある難易度の数})。"""
    pd = _load_json(path)["localPlayers"][0]
    fav = {lid[13:53].lower() for lid in pd.get("favoritesLevelIds", []) if lid.startswith("custom_level_")}
    plays, cleared = Counter(), Counter()
    for st in pd.get("levelsStatsData", []):
        if st["levelId"].startswith("custom_level_") and st.get("playCount", 0) > 0:
            h = st["levelId"][13:53].lower()
            plays[h] += st["playCount"]
            cleared[h] += bool(st.get("validScore"))
    return fav, plays, cleared


def read_datarecorder(db_path) -> list:
    """DataRecorder の SQLite（読み取り専用、immutable で開く）-> カスタムのレベルのプレイごとに dict を 1 つ:
    hash、結果（"finished" | "failed" | "quit"）、difficulty、mode、speed。"""
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro&immutable=1", uri=True)
    out = []
    for h, cleared, diff, mode, speed in con.execute(
            "select songHash, cleared, difficulty, mode, songSpeedMultiplier from MovieCutRecord "
            "where songHash is not null and songHash != ''"):
        if not HASH_RE.fullmatch(h or ""):
            continue  # 組み込みのレベル
        outcome = {"finished": "finished", "failed": "failed", "softFail": "failed"}.get(cleared, "quit")
        out.append({"hash": h.lower(), "outcome": outcome, "diff": difficulty(diff), "mode": mode,
                    "speed": float(speed) if speed not in (None, "None") else 1.0})
    con.close()
    return out


def replay_hashes(folder) -> Counter:
    """リプレイのファイル名（BeatLeader の .bsor、ScoreSaber の .dat）にある版の hash。ファイルは開かない。"""
    c = Counter()
    for f in Path(folder).iterdir():
        m = HASH_RE.search(f.name)
        if m:
            c[m.group(0).lower()] += 1
    return c


def installed_keys(custom_levels) -> set:
    """入れてあるマップの BeatSaver の key（フォルダ名 "<key> (<title> - <mapper>)"）。"""
    return {f.name.split(" ")[0] for f in Path(custom_levels).iterdir() if f.is_dir()}


def build_signals(votes=None, playlists=None, player_data=None, plays=None, replays=(), installed=None, key_to_hash=None):
    """各関数の出力を {版の hash: {signal: 数}} にまとめる。signal: upvote、downvote、playlist、favorite、
    pd_playcount、pd_cleared、play、finished、failed、quit、replay、installed。"""
    sig = defaultdict(Counter)
    for h, t in (votes or {}).items():
        sig[h]["upvote" if t == "Upvote" else "downvote"] += 1
    for h, n in (playlists or {}).items():
        sig[h]["playlist"] += n
    if player_data:
        fav, pc, cl = player_data
        for h in fav:
            sig[h]["favorite"] += 1
        for h, n in pc.items():
            sig[h]["pd_playcount"] += n
        for h, n in cl.items():
            sig[h]["pd_cleared"] += n
    for p in plays or []:
        sig[p["hash"]]["play"] += 1
        sig[p["hash"]][p["outcome"]] += 1
    for c in replays:
        for h, n in c.items():
            sig[h]["replay"] += n
    for k in installed or ():
        if key_to_hash and k in key_to_hash:
            sig[key_to_hash[k]]["installed"] += 1
    return sig


INTERACTION = ("installed", "play", "replay", "pd_playcount", "favorite", "upvote")


def merge_by_key(sig, meta, pool_hash_of_key):
    """マップのすべての版の signal を、そのマップのプールの版の hash にまとめる -> {プールの hash: Counter}。
    meta: catalogue.load_meta の出力（hash -> {key, ...}）。pool_hash_of_key: {key: プールの hash}。"""
    out = defaultdict(Counter)
    for h, s in sig.items():
        ph = pool_hash_of_key.get(meta.get(h, {}).get("key"))
        if ph:
            out[ph].update(s)
    return out

