"""プレイヤー自身の好意のマップに対する 1 つ抜きの評価。

候補 = プレイヤーが関わったマップのうち索引にあるもの。好意のマップは OctantSpecs 法 4.1 で決める
（runtime/likes.py。推薦と同じ決まり）。それ以外はラベルなし。好意のマップ p を 1 つずつ抜き、プロファイル、
クラスター、難易度の幅を p なしで作り、ほかの好意のマップを除いた候補の中で p の順位を付ける。
出すもの: hit@10/20、順位の中央値、MRR。
"""
import numpy as np

from ..runtime import likes
from .profile import difficulty_band, fit_penalty, spherical_kmeans


def view_score(Xc, liked_local, kind="km3", r_hub=None):
    """1 つのビューの、候補の行に対する z スコアにしたプロファイルの点。liked_local: Xc への添字。"""
    X = Xc[liked_local]
    if kind == "mean":
        q = X.mean(0)
        s = Xc @ (q / np.linalg.norm(q))
    else:
        s = (Xc @ spherical_kmeans(X, int(kind[2:])).T).max(1)
    if r_hub is not None:  # CSLS: 2cos - r_q - r_c。r_q はクエリごとに一定なので、cos - r_c / 2 で順位を付ける
        s = s - 0.5 * r_hub
    return (s - s.mean()) / s.std()


def loo_ranks(score_fn, liked_local, n_cand):
    """score_fn(p, others) -> 候補に対する点。-> 抜いた p ごとの 1 始まりの順位。"""
    liked_local = np.asarray(liked_local)
    ranks = []
    for p in liked_local:
        others = liked_local[liked_local != p]
        s = np.array(score_fn(p, others), np.float64)
        s[others] = -np.inf
        ranks.append(1 + int((s > s[p]).sum()))
    return np.array(ranks)


def summarise(ranks, n_rankable):
    r = np.asarray(ranks)
    return {"n_liked": len(r), "rank_base": n_rankable, "hit@10": 100 * float(np.mean(r <= 10)),
            "hit@20": 100 * float(np.mean(r <= 20)), "median_rank": float(np.median(r)), "mrr": float(np.mean(1 / r)),
            "random_hit@10": 100 * 10 / n_rankable, "random_median_rank": n_rankable / 2}


def labels(merged):
    """merged: {プールの hash: signal の Counter} -> 関わったマップの {プールの hash: 1（好意）| 0（ラベルなし）}。
    signal に日付がないので、「後の日にまた遊んだ」はここでは数えない。"""
    from .playdata import INTERACTION
    return {h: int(likes.classify(s) == "liked") for h, s in merged.items() if any(s.get(k) for k in INTERACTION)}


def run(index, cand_hashes, liked_hashes, kind="km3", csls=True, beta=0.0,
        difficulties=None, finished_points=None, csls_density="view", key_of=None, keep_ranks=False):
    """ビューごとの z スコアを後で合わせたもの（索引の重みで重み付け）+ beta * 難易度合わせ。
    difficulties: {hash: (n_diff, 2) の nps/njs}。finished_points: 最後まで遊んだプレイの [(遊んだマップの BeatSaver
    の key, nps, njs)]（幅の元）。key_of: {候補の hash: key}。抜いたマップの幅は、どの版を遊んだかによらず、その key の
    プレイをすべて除いて作る。
    csls_density: "view" = ハブの密度を各ビューの中で測る（直す対象のビューごとの点に合う）。"combined" = 索引の
    すべてのビューを重み付きでつないだもので 1 つの密度を測り、すべてのビューにかける。
    keep_ranks: "ranks"（{抜いた hash: 順位}）も返す（個人のデータ: results.py は頼まれたところにだけ書く）。"""
    cand = [index.pos[h] for h in cand_hashes]
    pos = {h: j for j, h in enumerate(cand_hashes)}
    liked = np.array([pos[h] for h in liked_hashes if h in pos])
    weights = {v: w for v, (_, w) in index.views.items()}
    tot = sum(weights.values())
    Xs = {v: index.views[v][0][cand] for v in weights}
    r_hub = {v: None for v in weights}
    if csls:
        if csls_density == "combined":
            r_all = index.local_density(10)[cand]
            r_hub = {v: r_all for v in weights}
        elif csls_density == "view":
            r_hub = {v: index.view_density(v, 10)[cand] for v in weights}
        else:
            raise ValueError(f"unknown csls_density {csls_density!r}")
    pen_cache = {}

    def penalty(p):
        if beta == 0 or difficulties is None:
            return 0.0
        h = cand_hashes[p]
        if h not in pen_cache:
            key = key_of[h] if key_of else h
            pts = np.array([(n, j) for k, n, j in finished_points if k != key])
            lo, hi, sd = difficulty_band(pts)
            pen_cache[h] = fit_penalty([difficulties.get(c, np.zeros((0, 2))) for c in cand_hashes], lo, hi, sd)
        return pen_cache[h]

    def score(p, others):
        s = sum(weights[v] / tot * view_score(Xs[v], others, kind, r_hub[v]) for v in weights)
        return s + beta * penalty(p)

    ranks = loo_ranks(score, liked, len(cand))
    res = summarise(ranks, len(cand) - len(liked) + 1)
    if keep_ranks:
        res["ranks"] = {cand_hashes[p]: int(r) for p, r in zip(liked, ranks)}
    return res
