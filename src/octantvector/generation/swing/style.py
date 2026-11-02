"""スイングの特徴からのプレイスタイルのタグ（OctantSpecs octantvector/swing.md、「スタイルタグ」）。

2 種類:
  learned  BeatSaver のスタイルタグ（accuracy、balanced など）ごとに 1 つのロジスティックのモデル。譜面の作者が
           選んだタグで `octantvector swing-train` が学習し、確率を出す。
  fine     特徴の上の決まりで定める、もっと細かいタグ（stream、vibro、inverted、bomb_reset など）。決まりは
           特徴を、全体の分位点（"q>=" / "q<="、0..1）か絶対値（">=" / "<="）と比べる。すべての条件を満たすと
           タグが付き、点（0..1）は一番弱い条件の点なので、タグが付かないマップも並べられる。分位点の条件は
           タグの側の分位点を点にする。絶対値の条件は、しきい値の下では届いた割合の半分、上では 0.5 + 分位点の
           半分を点にするので、しきい値を超えたマップも並んだまま。
分位点と学習した重みは、`swing-train` が書く 1 つのモデルのファイル（JSON）にある。決まりは設定ファイルの
`[swing_tags]` の表で変えたり足したりできる。例 `stream = [["alternation", "q>=", 0.7]]`。
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import config
from ..errors import GenerationError
from .features import HAZARD_NAMES, NAMES

# BeatSaver のスタイルタグ、"poodle" のタグ、"modded"（Noodle Extensions、Mapping Extensions、Vivify のどれかが要る譜面）
LEARNED_TAGS = ["accuracy", "balanced", "challenge", "dance-style", "fitness", "speed", "tech", "poodle", "modded"]

FINE_TAGS = {
    # 速さ
    "speed":        [["ebpm_p90", "q>=", 0.85]],
    "stream":       [["alternation", "q>=", 0.6], ["gap_lt_1", "q>=", 0.8]],
    "vibro":        [["gap_lt_0", "q>=", 0.95], ["gap_lt_0", ">=", 0.05]],
    "burst":        [["sustain_ratio", "q>=", 0.85], ["pass_short", "q>=", 0.6]],
    "stamina":      [["pass_long", "q>=", 0.8], ["length_s", "q>=", 0.6]],
    "midspeed":     [["ebpm_median", "q>=", 0.35], ["ebpm_median", "q<=", 0.75], ["strain_mean", "q<=", 0.6]],
    "jumps":        [["hit_distance_p90", "q>=", 0.8], ["wide_jump_frac", "q>=", 0.8]],
    # テク
    "angle_tech":   [["strain_p90", "q>=", 0.85]],
    "inverted":     [["inverted_frac", "q>=", 0.9], ["inverted_frac", ">=", 0.03]],
    "wrist_roll":   [["roll_change_mean", "q>=", 0.85], ["roll_speed_p90", "q>=", 0.6]],
    "rotation":     [["rotation_mean", "q>=", 0.85]],
    "parity_breaks": [["rebound_reset_frac", "q>=", 0.9], ["rebound_reset_frac", ">=", 0.03]],
    "bomb_resets":  [["bomb_reset_frac", "q>=", 0.9], ["bomb_reset_frac", ">=", 0.02]],
    "rhythm_tech":  [["rhythm_entropy", "q>=", 0.8], ["rhythm_change", "q>=", 0.8]],
    "syncopation":  [["offbeat_frac", "q>=", 0.9], ["offbeat_frac", ">=", 0.1]],
    "triplets":     [["triplet_frac", "q>=", 0.9], ["triplet_frac", ">=", 0.1]],
    "unsnapped":    [["unsnapped_frac", "q>=", 0.9], ["unsnapped_frac", ">=", 0.1]],
    "crossover":    [["crossover_frac", "q>=", 0.9]],
    "wide_reach":   [["outer_lane_frac", "q>=", 0.8], ["hand_x_std", "q>=", 0.8]],
    "dots":         [["dot_frac", "q>=", 0.9], ["dot_frac", ">=", 0.2]],
    "free_angles":  [["offgrid_angle_frac", ">=", 0.05]],
    # フロー / 精度
    "flowy":        [["linear_frac", "q>=", 0.75], ["rotation_mean", "q<=", 0.4], ["ebpm_median", "q<=", 0.7]],
    "acc":          [["full_swing_frac", "q>=", 0.65], ["ebpm_median", "q<=", 0.4], ["strain_mean", "q<=", 0.5]],
    # ノーツのパターン
    "windows":      [["window_frac", "q>=", 0.9], ["window_frac", ">=", 0.02]],
    "stacks":       [["stack_frac", "q>=", 0.9], ["stack_frac", ">=", 0.03]],
    "sliders":      [["slider_frac", "q>=", 0.9], ["slider_frac", ">=", 0.03]],
    "chains":       [["chain_frac", ">=", 0.02]],
    "arcs":         [["arc_ps", ">=", 0.05]],
    "doubles":      [["double_frac", "q>=", 0.85]],
    # 壁
    "art_walls":    [["art_wall_ps", ">=", 0.5], ["art_wall_frac", ">=", 0.8]],
    "workout_walls": [["head_move_mps", "q>=", 0.9], ["head_move_mps", ">=", 0.05]],
    "dodge_walls":  [["dodge_wall_ps", ">=", 0.05]],
    "crouch_walls": [["crouch_wall_ps", ">=", 0.02]],
    "rhythm_walls": [["wall_rhythm_frac", ">=", 0.5], ["dodge_wall_ps", ">=", 0.03]],
    "dance_walls":  [["wall_rhythm_frac", ">=", 0.5], ["head_move_mps", "q>=", 0.8]],
    "vision_walls": [["vision_wall_frac", ">=", 0.05]],
    # 爆弾
    "bombs":        [["bomb_ps", "q>=", 0.9], ["bomb_ps", ">=", 0.1]],
    "art_bombs":    [["bomb_art_frac", ">=", 0.3], ["bomb_ps", ">=", 0.2]],
    "dodge_bombs":  [["bomb_dodge_ps", "q>=", 0.9], ["bomb_dodge_ps", ">=", 0.1]],
    "rhythm_bombs": [["bomb_rhythm_frac", ">=", 0.5], ["bomb_dodge_ps", ">=", 0.05]],
    "bomb_walls":   [["bomb_cluster_frac", ">=", 0.3], ["bomb_dodge_ps", ">=", 0.05]],
    # 体の動き
    "one_hand":     [["hands_ratio", "<=", 0.15]],
    # 読み
    "dense_low_njs": [["nps", "q>=", 0.7], ["njs", "q<=", 0.3]],
    "high_njs":     [["njs", "q>=", 0.9]],
}
OPS = ("q>=", "q<=", ">=", "<=")
_IDX = {n: i for i, n in enumerate(NAMES)}
N_QUANTILES = 101


def is_hazard(conds):
    """壁と爆弾のタグ（どの条件も壁か爆弾の特徴のもの）は `swing_hazards` のビューに、残りは `swing_style` に入れる。"""
    return all(f in HAZARD_NAMES for f, _, _ in conds)


def load_rules(group=None):
    """設定ファイルの `[swing_tags]` の表でタグを置き換えたり足したりした FINE_TAGS。group が "style" か "hazards" なら、
    そのビューのタグだけを残す。"""
    rules = {k: [list(c) for c in v] for k, v in FINE_TAGS.items()}
    for name, conds in config.section("swing_tags").items():
        for c in conds:
            if len(c) != 3 or c[0] not in _IDX or c[1] not in OPS:
                raise GenerationError(f"[swing_tags] {name}: each condition is [feature, one of {OPS}, number]; "
                                      f"got {c}")
        rules[name] = [[f, op, float(v)] for f, op, v in conds]
    if group:
        rules = {k: v for k, v in rules.items() if is_hazard(v) == (group == "hazards")}
    return rules


# ---------------------------------------------------------------- モデルのファイル
def signed_log(X):
    """sign(x) * log(1 + |x|): 裾の重い特徴を、線形のモデルが使える尺度にする。"""
    return np.sign(X) * np.log1p(np.abs(X))


def fit(X_all, X_tagged, tags, lam=10.0):
    """X_all: すべての行の特徴（分位点用）。X_tagged + tags: マップごとに 1 行（一番難しい難易度）と、その BeatSaver の
    スタイルタグ。-> モデルの dict（JSON にできる）。"""
    from .dataset import logistic_weights
    q = np.quantile(X_all, np.linspace(0, 1, N_QUANTILES), axis=0).T  # (features, N_QUANTILES)
    L = signed_log(np.asarray(X_tagged, float))
    mu, sd = L.mean(0), L.std(0) + 1e-6
    learned = {}
    for t in LEARNED_TAGS:
        y = np.array([t in ts for ts in tags], float)
        if y.sum() >= 20:
            w = logistic_weights(np.clip((L - mu) / sd, -6, 6), y, lam)
            learned[t] = {"w": w[:-1].round(6).tolist(), "b": float(w[-1]), "n_pos": int(y.sum())}
    return {"features": NAMES, "quantiles": q.round(6).tolist(), "mu": mu.round(6).tolist(), "sd": sd.round(6).tolist(),
            "learned": learned, "n_rows": int(len(X_all)), "n_maps": int(len(X_tagged))}


def save(model, path):
    Path(path).write_text(json.dumps(model))


def load_model(path):
    m = json.loads(Path(path).read_text())
    if m["features"] != NAMES:
        raise GenerationError(f"{path} was trained on a different feature list; run `octantvector swing-train` again")
    return m


def model_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 当てはめ
def quantile_of(model, X):
    """各特徴の値の、全体での分位点（0..1）。"""
    q = np.asarray(model["quantiles"])
    X = np.atleast_2d(X)
    # 中間の順位: 多くのマップが同じ値（例 「爆弾のリセットなし」の 0）のとき、その並びの上ではなく真ん中に置く
    return np.stack([(np.searchsorted(q[j], X[:, j], "left") + np.searchsorted(q[j], X[:, j], "right")) / (2 * q.shape[1])
                     for j in range(X.shape[1])], 1)


def learned_probs(model, X):
    """-> (n, len(LEARNED_TAGS))。モデルに重みのないタグは NaN。"""
    Z = np.clip((signed_log(np.atleast_2d(X)) - np.asarray(model["mu"])) / np.asarray(model["sd"]), -6, 6)
    out = np.full((Z.shape[0], len(LEARNED_TAGS)), np.nan)
    for j, t in enumerate(LEARNED_TAGS):
        if t in model["learned"]:
            m = model["learned"][t]
            out[:, j] = 1 / (1 + np.exp(-np.clip(Z @ np.asarray(m["w"]) + m["b"], -30, 30)))
    return out


def fine_tags(model, X, rules=None):
    """-> (点 (n, len(rules))、0..1, 付いたか (n, len(rules)) の bool)。列は rules の順。"""
    rules = rules or FINE_TAGS
    X = np.atleast_2d(X)
    Q = quantile_of(model, X)
    S = np.zeros((X.shape[0], len(rules))); on = np.zeros_like(S, bool)
    for j, conds in enumerate(rules.values()):
        ok = np.ones(X.shape[0], bool); sc = np.ones(X.shape[0])
        for f, op, v in conds:
            i = _IDX[f]
            if op == "q>=":
                ok &= Q[:, i] >= v; sc = np.minimum(sc, Q[:, i])
            elif op == "q<=":
                ok &= Q[:, i] <= v; sc = np.minimum(sc, 1 - Q[:, i])
            elif op == ">=":
                met = X[:, i] >= v; ok &= met
                below = 0.5 * np.clip(X[:, i] / v, 0, 1) if v > 0 else 0.0
                sc = np.minimum(sc, np.where(met, 0.5 + 0.5 * Q[:, i], below))
            else:
                met = X[:, i] <= v; ok &= met
                above = 0.5 * np.clip(v / np.maximum(X[:, i], 1e-9), 0, 1) if v > 0 else 0.0
                sc = np.minimum(sc, np.where(met, 1 - 0.5 * Q[:, i], above))
        S[:, j], on[:, j] = sc, ok
    return S, on


def describe(model, x, rules=None):
    """1 つの譜面の特徴 -> (learned {tag: 確率}, 付いた細かいタグ。よいものから)。"""
    rules = rules or load_rules()
    p = learned_probs(model, x)[0]
    S, on = fine_tags(model, x, rules)
    names = list(rules)
    fine = sorted((names[j] for j in np.where(on[0])[0]), key=lambda n: -S[0, names.index(n)])
    return {t: round(float(v), 3) for t, v in zip(LEARNED_TAGS, p) if not np.isnan(v)}, fine
