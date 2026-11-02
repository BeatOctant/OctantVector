"""複数ビューの重み付きコサインの k-NN。CSLS でハブを抑えることもできる。

各ビューはそれぞれ正規化する（normalisation.py。encoder.norm: "center" = 中心を引いて L2、"zscore" = z スコアに
して L2）。統計は、較正のファイルを渡せばそこから取る（calibration.py）ので、行を足しても座標は動かない。渡さ
なければ読み込んだ行から計算する。合わせた点はビューごとのコサインの重み付き平均。missing="drop" では、すべての
ビューにある hash だけを索引に入れ、各ビューに sqrt(w / sum(w)) を掛けてつなぐ。そうすると 1 回の積で同じ点になる。

missing="reweight" では、ビューの重みが全体の `min_weight` 以上ある hash を残す。対のどちらかにないビューは、
重みをそのままにコサイン 0 と数える（点は共通の重みではなく全体の重みで割る）ので、共通のビューが少ない対が、
軽いビュー 1 つで一致しただけで、全部を共通に持つ対を上回ることはない。search_views() は各ビュー単独の上位 k を
返す（土台の出力）。重み付きの組み合わせはその上の層（query_rows）で、重みは要求ごとに変えられる。似ている度合いと
上位 k は、使えれば torch で CUDA の上で行の塊ごとに計算し（device "auto"）、なければ numpy。結果はどちらでも numpy。

索引はマップ単位（実験用の道具: search、eval-neighbours、eval-personal）。譜面の行を持つビューは、マップの版ごとに
1 つの譜面（一番難しい Standard の難易度。charts.representative）で表す。配布 DB はすべての譜面を持つ（runtime/db.py）。
"""
import numpy as np

from ..common import charts
from ..common.params import DEFAULTS
from .encoders import get
from .normalisation import calibrated, normalise
from .store import Store

CHUNK = 2048


def parse_views(spec: str) -> dict:
    """'clap_htsat_unfused:0.5,meta_tags:0.5' -> {id: 重み}（重みの既定は 1）。"""
    out = {}
    for part in spec.split(","):
        i, _, w = part.partition(":")
        out[i.strip()] = float(w or 1)
    return out


def resolve_device(device="auto"):
    """設定 `device`（"auto" か "cpu"）に対する CUDA のデバイス。"cpu" か、デバイスがなければ None。"""
    if device == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def _topk_np(s, k):
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)


class Index:
    def __init__(self, weights: dict, store_root, restrict=None, device="auto", vectors=None, calibration=None,
                 missing="drop", min_weight=DEFAULTS["minViewWeight"]):
        """weights: {encoder id: 重み}。vectors: ストアを読む代わりに渡せる {encoder id: (hashes, array)}。
        calibration: すべてのビューの固定した統計を持つ較正（calibration.load）。
        missing: "drop"（すべてのビューにある hash だけ）か "reweight"（モジュールの docstring を参照）。"""
        if missing not in ("drop", "reweight"):
            raise ValueError(f"missing must be 'drop' or 'reweight', got {missing!r}")
        raw = {}
        for i in weights:
            enc = get(i)
            hs, vs = vectors[i] if vectors is not None else Store(enc, store_root).load()
            if enc.row_kind() == "charts" and any(charts.is_chart_key(h) for h in hs):
                hs, vs = charts.representative(hs, vs)
            raw[i] = (enc, dict(zip(hs, vs)))
        sets = [set(d) for _, d in raw.values()]
        keep = set.intersection(*sets) if missing == "drop" else set.union(*sets)
        if restrict is not None:
            keep &= set(restrict)
        total = sum(weights.values())
        if missing == "reweight":  # hash を検索に入れるには、重みが十分にそろっていること
            keep = {h for h in keep if sum(weights[i] for i, (_, d) in raw.items() if h in d) >= min_weight * total}
        self.hashes = sorted(keep)
        self.pos = {h: n for n, h in enumerate(self.hashes)}
        self.missing = missing
        self.encoders = {i: enc for i, (enc, _) in raw.items()}
        self.calibration_id = calibration["id"] if calibration else None
        self.views, self.present, parts = {}, {}, []
        for i, (enc, d) in raw.items():
            mask = np.array([h in d for h in self.hashes], bool)
            st = calibrated(calibration, enc) if calibration else None
            dim = len(next(iter(d.values()))) if d else enc.dim
            x = np.zeros((len(self.hashes), dim), np.float32)
            if mask.any():
                x[mask] = normalise(enc, np.stack([d[h] for h, m in zip(self.hashes, mask) if m]), st)
            self.views[i] = (x, weights[i] / total)
            self.present[i] = mask
            parts.append(np.sqrt(weights[i] / total) * x)
        self.emb = np.hstack(parts).astype(np.float32) if missing == "drop" else None
        self.device = resolve_device(device)
        self._t, self._r, self._vt = None, {}, {}

    def __len__(self):
        return len(self.hashes)

    def _view_t(self, view):
        if view not in self._vt:
            import torch
            x, _ = self.views[view]
            self._vt[view] = (torch.from_numpy(x).to(self.device),
                              torch.from_numpy(self.present[view]).to(self.device))
        return self._vt[view]

    def _weighted(self, rows, weights):
        """`rows` からすべての項目への、ビューごとのコサインの重み付き平均。どちらかにないビューは 0 と数える。
        共通のビューのない対は -inf。"""
        tot = sum(weights.values())
        if self.device is not None:
            import torch
            rt = torch.as_tensor(rows, device=self.device)
            num = den = 0
            for v, w in weights.items():
                x, m = self._view_t(v)
                pair = (m[rt][:, None] & m[None, :]).float() * (w / tot)
                num = num + pair * (x[rt] @ x.T)
                den = den + pair
            return torch.where(den > 0, num, torch.tensor(-torch.inf, device=self.device))
        num = den = 0
        for v, w in weights.items():
            x, m = self.views[v][0], self.present[v]
            pair = (m[rows][:, None] & m[None, :]) * (w / tot)
            num = num + pair * (x[rows] @ x.T)
            den = den + pair
        return np.where(den > 0, num, -np.inf).astype(np.float32)

    def _scores(self, rows, r=None, weights=None):
        """自分を除いた、`rows` からすべての項目への似ている度合い。r（局所の密度）を渡せば CSLS で補正する。
        weights: 要求ごとの {view: 重み}（既定は索引の重み）。"""
        if weights is not None or self.emb is None:
            s = self._weighted(rows, weights or {v: w for v, (_, w) in self.views.items()})
            if self.device is not None:
                import torch
                rt = torch.as_tensor(rows, device=self.device)
                s[torch.arange(len(rows), device=self.device), rt] = -torch.inf
                if r is not None:
                    r_t = torch.as_tensor(r, device=self.device)
                    s = 2 * s - r_t[rt][:, None] - r_t[None, :]
                return s
            s[np.arange(len(rows)), rows] = -np.inf
            if r is not None:
                s = 2 * s - r[rows][:, None] - r[None, :]
            return s
        if self.device is not None:
            import torch
            if self._t is None:
                self._t = torch.from_numpy(self.emb).to(self.device)
            rt = torch.as_tensor(rows, device=self.device)
            s = self._t[rt] @ self._t.T
            s[torch.arange(len(rows), device=self.device), rt] = -torch.inf
            if r is not None:
                r_t = torch.as_tensor(r, device=self.device)
                s = 2 * s - r_t[rt][:, None] - r_t[None, :]
            return s
        s = self.emb[rows] @ self.emb.T
        s[np.arange(len(rows)), rows] = -np.inf
        if r is not None:
            s = 2 * s - r[rows][:, None] - r[None, :]
        return s

    def _topk(self, s, k):
        if self.device is not None:
            import torch
            v, i = torch.topk(s, k, dim=1)
            return i.cpu().numpy(), v.cpu().numpy()
        i = _topk_np(s, k)
        return i, np.take_along_axis(s, i, 1)

    def local_density(self, k=10):
        """CSLS の r_i: 各項目から k 近傍へのコサインの平均。"""
        if k not in self._r:
            r = np.empty(len(self), np.float32)
            for a in range(0, len(self), CHUNK):
                rows = np.arange(a, min(a + CHUNK, len(self)))
                r[rows] = self._topk(self._scores(rows), k)[1].mean(1)
            self._r[k] = r
        return self._r[k]

    def view_density(self, view, k=10):
        """1 つのビューの中だけでの CSLS の r_i: そのビューでの、各項目から k 近傍へのコサインの平均。"""
        key = (view, k)
        if key not in self._r and self.emb is None:  # ないビュー: 点を伏せる。そのビューのない行は 0
            r = np.zeros(len(self), np.float32)
            have = np.flatnonzero(self.present[view])
            for a in range(0, len(have), CHUNK):
                rows = have[a:a + CHUNK]
                v = self._topk(self._scores(rows, weights={view: 1.0}), k)[1]
                r[rows] = _finite_mean(v)
            self._r[key] = r
        if key not in self._r:
            x = self.views[view][0]
            r = np.empty(len(self), np.float32)
            for a in range(0, len(self), CHUNK):
                rows = np.arange(a, min(a + CHUNK, len(self)))
                if self.device is not None:
                    import torch
                    xt = torch.from_numpy(x).to(self.device)
                    s = xt[torch.as_tensor(rows, device=self.device)] @ xt.T
                    s[torch.arange(len(rows), device=self.device), torch.as_tensor(rows, device=self.device)] = -torch.inf
                    r[rows] = torch.topk(s, k, dim=1).values.mean(1).cpu().numpy()
                else:
                    s = x[rows] @ x.T
                    s[np.arange(len(rows)), rows] = -np.inf
                    r[rows] = np.take_along_axis(s, _topk_np(s, k), 1).mean(1)
            self._r[key] = r
        return self._r[key]

    def query_rows(self, rows, k=10, csls=True, csls_k=10, weights=None):
        """渡した行の添字の、重み付きの組み合わせでの上位 k の近傍 -> (添字, 点)。
        weights: 要求ごとの {view: 重み}。CSLS の密度は索引の重みでのもの。"""
        rows = np.asarray(rows)
        return self._topk(self._scores(rows, self.local_density(csls_k) if csls else None, weights), k)

    def search_views(self, rows, k=10):
        """各ビュー単独の上位 k -> {view: (添字, 点)}。そのビューのない項目は返さず、ない行の結果は空。"""
        rows = np.asarray(rows)
        out = {}
        for v in self.views:
            have = self.present[v][rows]
            idx = np.full((len(rows), k), -1, np.int64)
            sc = np.full((len(rows), k), -np.inf, np.float32)
            if have.any():
                i, s = self._topk(self._scores(rows[have], weights={v: 1.0}), k)
                idx[have], sc[have] = i, s
            out[v] = (idx, sc)
        return out

    def knn(self, k=10, csls=False):
        parts = [self.query_rows(np.arange(a, min(a + CHUNK, len(self))), k, csls, k) for a in range(0, len(self), CHUNK)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def view_cosines(self, a, b):
        """項目 a と b のビューごとのコサイン。どちらかにビューがなければ None。"""
        return {i: float(x[a] @ x[b]) if self.present[i][a] and self.present[i][b] else None
                for i, (x, _) in self.views.items()}


def _finite_mean(v):
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v)
    return np.where(ok.any(1), np.where(ok, v, 0).sum(1) / np.maximum(ok.sum(1), 1), 0.0)
