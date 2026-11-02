"""譜面の識別とプレイスタイル（OctantSpecs constitution.md、原則 1 と 2。contracts/、契約2（DB build to
recommendation core））。

譜面はプレイヤーが遊ぶもの: マップの 1 つの版の、1 つの characteristic の、1 つの難易度。譜面から計算したビューは
譜面ごとに "<版の hash>:<characteristic>:<difficulty>" のキーで持ち、曲やマップのメタデータから計算したビューは
版の hash ごとに持つ。key_hash() はどちらのキーも版の hash にするので、台帳、プール、較正はマップの版ごとに働く。

譜面にはそれぞれプレイスタイルがある。プレイスタイルの違う譜面どうしを同じものとして比べない: 譜面のビューは
同じプレイスタイルの譜面どうしの似ている度合いだけを測る。プレイスタイルは今は characteristic の名前。
characteristic だけでは見分けられないプレイスタイルが出てきたら、ここで決める。
"""
import numpy as np

SEP = ":"
DIFF_ORDER = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
STANDARD = "Standard"


def key(version_hash, characteristic, difficulty):
    return f"{version_hash}{SEP}{characteristic}{SEP}{difficulty}"


_DIFF_SPELLING = {**{d.lower(): d for d in DIFF_ORDER}, "expert+": "ExpertPlus"}


def difficulty(name):
    """難易度の名前を BeatSaver のカタログの書き方にする（"normal" -> "Normal"、"Expert+" -> "ExpertPlus"）。
    ほかの名前はそのまま返す。"""
    return _DIFF_SPELLING.get(name.lower(), name) if isinstance(name, str) else name


def difficulty_rank(name):
    """Easy（0）から ExpertPlus（4）までの難易度の位置。DIFF_ORDER にない名前は -1。"""
    return DIFF_ORDER.index(name) if name in DIFF_ORDER else -1


def catalogue_spelling(chart_key):
    """難易度を BeatSaver のカタログの書き方にした譜面のキー。古い zip には、カタログの "Normal" と "ExpertPlus" を
    Info.dat に "normal" や "Expert+" と書くものがある。ほかのキーはそのまま返す（書き方以上に違う難易度は別の譜面）。"""
    if not is_chart_key(chart_key):
        return chart_key
    h, c, d = split(chart_key)
    return key(h, c, difficulty(d))


def split(chart_key):
    """-> (版の hash, characteristic, difficulty)"""
    h, c, d = chart_key.split(SEP, 2)
    return h, c, d


def key_hash(k):
    """譜面のキーの版の hash。版の hash（マップの行）はそのまま返す。"""
    return k.split(SEP, 1)[0]


def is_chart_key(k):
    return SEP in k


def play_style(characteristic):
    return characteristic or STANDARD


def _rank(k):
    _, c, d = split(k)
    r = difficulty_rank(d)
    return r if play_style(c) == STANDARD and r >= 0 else None


def hardest_standard(keys):
    """譜面のキーのうち、一番難しい Standard の譜面（ExpertPlus まで）。なければ None。"""
    ranked = [(_rank(k), k) for k in keys if _rank(k) is not None]
    return max(ranked)[1] if ranked else None


def representative(keys, vecs):
    """マップ単位の実験（search、eval-personal）のための、マップの版ごとに 1 つの譜面: ExpertPlus までで一番難しい
    Standard の難易度。譜面のビューを譜面ごとに持つ前に表していたもの。-> (版の hash, ベクトル)。"""
    best = {}
    for n, k in enumerate(keys):
        rank, h = _rank(k), key_hash(k)
        if rank is not None and (h not in best or rank > best[h][0]):
            best[h] = (rank, n)
    hs = sorted(best)
    x = np.stack([vecs[best[h][1]] for h in hs]) if hs else np.zeros((0, np.asarray(vecs).shape[-1]), np.float32)
    return hs, x
