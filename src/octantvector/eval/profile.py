"""好意のマップの集まりからの好みのクラスター（personal.view_score がそれに対して点を付ける）と、難易度合わせの項。"""
import numpy as np


def spherical_kmeans(X, k=3, weights=None, seed=0, iters=25, restarts=4):
    """重み付きの球面 k-means（コサイン）。初期値は k-means++。X の各行は単位ベクトルであること。-> (k, d) の重心。"""
    X = np.asarray(X, np.float64)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, np.float64)
    k = min(k, len(X))
    best, best_obj = None, -np.inf
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        cent = [X[rng.choice(len(X), p=w / w.sum())]]
        for _ in range(k - 1):
            d = np.clip(1 - np.max(X @ np.array(cent).T, axis=1), 0, None) * w  # clip: 浮動小数の誤差で 0 未満になり得る
            cent.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
        cent = np.array(cent)
        for _ in range(iters):
            lab = np.argmax(X @ cent.T, axis=1)
            new = np.array([(w[lab == j, None] * X[lab == j]).sum(0) if np.any(lab == j) else cent[j] for j in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, cent):
                break
            cent = new
        obj = (w * np.max(X @ cent.T, axis=1)).sum()
        if obj > best_obj:
            best, best_obj = cent, obj
    return best


def difficulty_band(points, lo_pct=10, hi_pct=90):
    """points: 最後まで遊んだプレイで実際に選んだ難易度の (nps, njs) の (n, 2) 配列。
    -> 次元ごとの (lo, hi, sd)。遊びやすい範囲は [lo, hi]。"""
    a = np.asarray(points, np.float64)
    return np.percentile(a, lo_pct, axis=0), np.percentile(a, hi_pct, axis=0), a.std(0)


def fit_penalty(difficulties, lo, hi, sd):
    """difficulties: マップごとの (n_diff, 2) の (nps, njs) 配列。-> マップごとに、難易度のうち一番よい（一番負でない）
    罰 -0.5 * sum((幅の外の距離 / sd)^2)。0 ならどれかの難易度が範囲内。NaN の値（測った NPS のない難易度）はその次元を
    罰に数えない（ほかの次元だけで合わせる）。難易度が 1 つもないマップは -2。"""
    out = np.empty(len(difficulties))
    for i, D in enumerate(difficulties):
        if len(D) == 0:
            out[i] = -2.0
            continue
        z = np.nan_to_num(np.maximum(0, np.maximum(lo - D, D - hi)) / sd, nan=0.0)
        out[i] = (-0.5 * (z ** 2).sum(1)).max()
    return out
