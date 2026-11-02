"""得意な譜面からの距離としてのプレイヤーの実力（OctantSpecs 法 3.3）。

プレイスタイルごとに（プレイスタイルの違う譜面どうしは比べない、憲法の原則 2）:
  features   charts.parquet の譜面の特徴。DB のそのプレイスタイルの譜面で標準化する
  strong set 得意な譜面の枠: 直近の素のクリア（速度 1.0、修飾なし。途中終了は数えない）、続いて空きがあるあいだ、
             OctantPath の前にクリアしたと PlayerData が言う譜面（`cleared` のある imported_play）をプレイの多い順に:
             PlayerData には日付がなく、そのクリアには修飾があったかもしれない。枠が埋まると、自分の新しいクリアが
             その中で一番遊んでいないものと入れ替わる。minClears に数えるので、実力は最初のプレイからあり、
             実力の位置だけを決める: 曲線と苦手な方向はプレイヤー自身のプレイから取る（実力が安定したプレイヤー
             でも、PlayerData が今の実力より易しいプレイヤーでも、全部の履歴に一番近かった。早く外しても
             よくならなかった）
  distance   譜面の距離: k 個の一番近い得意な譜面への距離の平均（その譜面自体は除く）
  weak directions  苦手な方向: 得意な譜面より特徴が高い（か低い）方向のうち、プレイヤーが設定の重みの言う以上に
             失敗するもの。直近の素のクリアと失敗（得意な譜面の一番古いクリア以降）から、その失敗が十分に
             あるときに学び、距離で重く数える（Style.learn）
  curve      距離に対するクリアの見込み。失敗が十分にあるとき、素のクリアと失敗にロジスティックを当てはめる
狙いはここで目標の距離になる（契約1（OctantCompass to OctantVector））: normal = 得意な譜面どうしの距離の中央値、
warmup = その一部、challenge = 曲線が挑戦のクリアの見込みを与える距離。
"""
import numpy as np

RIDGE = 1.0  # 苦手な方向の当てはめの L2 の罰: 1 つの方向だけでクリアと失敗が分かれるときも有限に保つ。どこまで信じるかは、
             # その標準誤差と directionShrinkPlays が決める（Style.learn）


class Style:
    """1 つのプレイスタイルでの実力。rows: そのスタイルの譜面の行。X: その標準化した特徴。w: 特徴の重み。
    strong: 得意な譜面の枠の譜面の行。譜面と得意な譜面との差は、譜面が高い特徴では `up`、低い特徴では `down` で
    重みを付ける。苦手な方向を学ぶまでは、どちらも `w`。"""

    def __init__(self, style, rows, X, w, strong, p):
        self.style, self.X, self.w, self.p = style, X, w, p
        self.up, self.down = w, w
        self.weak = []  # 学んだ苦手な方向: (特徴の添字, "high" か "low", 設定の重みに対する重み)
        self.rows = np.asarray(rows)
        self.pos = {int(r): n for n, r in enumerate(rows)}
        self.strong = [int(r) for r in strong]
        self.curve = None
        self._measure()

    def _measure(self):
        """得意な譜面どうしの距離とその広がり（目標の単位）。一から計算する。"""
        self._all = None
        self.self_d = np.array([self._dist(self.X[self.pos[r]], exclude=r) for r in self.strong])
        self.sd = float(self.self_d.std()) or 1.0

    def _to_strong(self, x, exclude=None):
        """-> (`exclude` を除いた得意な譜面の行, 特徴 x からそれぞれへの距離)。"""
        S = [s for s in self.strong if s != exclude]
        e = x - self.X[[self.pos[s] for s in S]]
        return S, np.sqrt(((e ** 2) * np.where(e > 0, self.up, self.down)).sum(1))

    def _dist(self, x, exclude=None):
        S, d = self._to_strong(x, exclude)
        if not S:
            return np.nan
        k = min(self.p["k"], len(d))
        return float(np.sort(d)[:k].mean())

    def distance(self, row):
        n = self.pos.get(row)
        if n is None or not np.isfinite(self.X[n]).all():
            return None
        return self._dist(self.X[n], exclude=row)

    def distances(self):
        """-> スタイルのすべての譜面の (距離, 一番近い得意な譜面の行)。`rows` の順で、`distance` と `nearest_strong` が
        1 つずつ返すのと同じ値（`distance` が None なら距離は NaN）。すべての譜面と得意な譜面の枠の行列として一度だけ
        計算する: 推薦ではすべての候補にこれが要る。"""
        if self._all is None:
            S = np.array([self.pos[s] for s in self.strong])
            Xs, k = self.X[S], min(self.p["k"], len(S))
            dist, nearest = np.empty(len(self.X)), np.empty(len(self.X), np.int64)
            for a in range(0, len(self.X), 8192):  # 塊ごとにして、行列を小さく保つ
                X = self.X[a:a + 8192]
                E = X[:, :1] - Xs[:, 0]
                D = (E ** 2) * self._weights(E, 0)  # _dist と同じ順で、特徴ごとに足す
                for f in range(1, X.shape[1]):
                    E = X[:, f:f + 1] - Xs[:, f]
                    D += (E ** 2) * self._weights(E, f)
                D = np.sqrt(D)
                dist[a:a + 8192] = np.sort(np.partition(D, k - 1, 1)[:, :k], 1).mean(1)
                own = np.flatnonzero((S >= a) & (S < a + 8192))
                D[S[own] - a, own] = np.inf  # 得意な譜面は、自分の一番近い得意な譜面にはならない
                nearest[a:a + 8192] = np.array(self.strong)[np.argmin(D, 1)]
            dist[~np.isfinite(self.X).all(1)] = np.nan
            for r in self.strong:  # その距離は、自分を得意な譜面の枠から外して測る
                dist[self.pos[r]] = self._dist(self.X[self.pos[r]], exclude=r)
            self._all = dist, nearest
        return self._all

    def _weights(self, E, f):
        """特徴 f での差 E（譜面 - 得意な譜面）の重み。符号で重みが違うときは符号ごと。"""
        return self.up[f] if self.up[f] == self.down[f] else np.where(E > 0, self.up[f], self.down[f])

    def nearest_strong(self, row):
        return int(self.distances()[1][self.pos[row]])

    def learn(self, rows, cleared):
        """プレイ（その譜面の行と、それぞれクリアだったか）からの苦手な方向。プレイの譜面から一番近い得意な譜面を引くと、
        特徴ごとに、どれだけ高くどれだけ低いかが分かる（距離と同じく、設定の重みの平方根で尺度を合わせる）。この
        2 x 特徴数 の列でクリアをロジスティック回帰すると、各方向の落ち方が出る: その方向に沿ってクリアの見込みが
        どれだけ速く下がるか（上がる方向はなし）。設定の重みはどの方向も同じに数え、2 x 特徴数 のすべての方向の平均の
        落ち方で落ちるかのように扱う。落ち方から標準誤差（当てはめのぶれ）を引いても平均より速い方向は苦手: その重みに
        （その落ち方 / 平均）の 2 乗を掛けるので、その方向への一歩はそれだけ遠い距離に数える。ほかは設定の重みのまま:
        得意な譜面より易しい譜面は、やはり合わない。n 回のプレイから学んだ倍率 m は (n m + n0) / (n + n0) で 1 へ縮める。
        n0 = directionShrinkPlays。そのあと新しい重みで距離を測り直す。ここで測るのは遊んだ譜面だけで、スタイル全体では
        ない（`distances`）。"""
        at = [self.pos[r] for r in rows]
        near = (self._to_strong(self.X[n], exclude=r) for n, r in zip(at, rows))
        E = self.X[at] - self.X[[self.pos[S[int(np.argmin(d))]] for S, d in near]]
        U = np.sqrt(np.r_[self.w, self.w]) * np.concatenate([np.maximum(E, 0), np.maximum(-E, 0)], 1)
        y = np.asarray(cleared, float)
        out = _logistic(U, y, RIDGE, noise=True)
        if out is None:
            return
        fit, se = out
        drop = np.maximum(0.0, -np.array(fit[1:]))
        if not (drop - se > drop.mean()).any():
            return  # ぶれを超えてほかより速く落ちる方向がない: 学ぶものはない
        m = np.maximum(1.0, (drop - se) / drop.mean()) ** 2
        n0 = float(self.p["directionShrinkPlays"])
        m = (len(y) * m + n0) / (len(y) + n0)
        F = len(self.w)
        self.up, self.down = self.w * m[:F], self.w * m[F:]
        weak = [(f, side, float(m[s * F + f])) for s, side in enumerate(("high", "low")) for f in range(F)]
        self.weak = sorted((x for x in weak if x[2] > 1), key=lambda x: -x[2])
        self._measure()

    def chance(self, d):
        if self.curve is None or d is None:
            return None
        a, b = self.curve
        return float(1 / (1 + np.exp(-(a + b * d))))

    def target(self, difficulty):
        """-> (実際に使った狙い, 目標の距離)。"""
        normal = float(np.median(self.self_d))
        if isinstance(difficulty, (int, float)) and not isinstance(difficulty, bool):
            if self.curve is not None and 0 < difficulty < 1 and self.curve[1] < 0:
                a, b = self.curve
                return "clearChance", max(0.0, (np.log(difficulty / (1 - difficulty)) - a) / b)
            return "normal", normal
        if difficulty == "warmup":
            return "warmup", self.p["warmupScale"] * normal
        if difficulty == "challenge":
            c = self.p["challengeClearChance"]
            if self.curve is not None and self.curve[1] < 0:
                a, b = self.curve
                return "challenge", max(normal, (np.log(c / (1 - c)) - a) / b)
            return "challenge", float(np.percentile(self.self_d, self.p["normalPercentile"])) + self.sd
        return "normal", normal


def _logistic(x, y, ridge=0.0, iters=50, noise=False):
    """ニュートン法によるロジスティック回帰。P(y) = 1 / (1 + exp(-(a + x b)))。列 x は 1 つでも複数でもよく、b に L2 の
    罰 `ridge` をかける。-> (a, *b)。収束しなければ None。`noise` があれば ((a, *b), 当てはめた点での罰付きのヘッセ
    行列からの b の標準誤差)。"""
    X = np.column_stack([np.ones(len(x)), x])
    pen = np.r_[0.0, np.full(X.shape[1] - 1, float(ridge))]
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen) + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(H, X.T @ (y - p) - pen * beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    if not np.isfinite(beta).all():
        return None
    fit = tuple(float(v) for v in beta)
    return (fit, np.sqrt(np.diag(np.linalg.inv(H)))[1:]) if noise else fit


def estimate(db, hist, p):
    """-> 素のクリアが十分にあるプレイスタイルの {プレイスタイル: Style}。"""
    feats = [f for f, w in p["features"].items() if w and f in db.chart_metrics]
    if not feats:
        return {}
    w = np.array([float(p["features"][f]) for f in feats])
    raw = np.stack([db.chart_metrics[f] for f in feats], 1)
    styles = np.array(db.play_styles, dtype=object)
    out = {}
    plain = [pl for pl in hist.plays if pl.plain and pl.chart is not None and pl.result in ("clear", "fail")]
    imported = [db.chart_pos[k] for k, _ in sorted(hist.imported_clears.items(), key=lambda kv: (-kv[1], kv[0]))]
    for style in sorted(set(db.play_styles)):
        rows = np.flatnonzero(styles == style)
        X = raw[rows]
        ok = np.isfinite(X).all(1)
        if ok.sum() < 2:
            continue
        mu, sd = X[ok].mean(0), X[ok].std(0)
        X = (X - mu) / np.where(sd > 0, sd, 1)
        mine = [pl for pl in plain if db.play_styles[db.chart_pos[pl.chart]] == style]
        # 枠が埋まったら、直近のプレイは得意な譜面の枠の一番古いクリアから始まる。それまではどのプレイも直近
        clears, strong, since = [pl for pl in mine if pl.result == "clear"], [], ""
        for pl in reversed(clears):  # 新しい順、譜面ごとに 1 つ
            r = db.chart_pos[pl.chart]
            if r not in strong and np.isfinite(raw[r]).all():
                strong.append(r)
            if len(strong) >= p["strongRecentClears"]:
                since = pl.at
                break
        for r in imported:  # 残りは PlayerData のクリアで埋める
            if len(strong) >= p["strongRecentClears"]:
                break
            if db.play_styles[r] == style and r not in strong and np.isfinite(raw[r]).all():
                strong.append(r)
        if len(strong) < max(2, p["minClears"]):
            continue
        st = Style(style, rows, X, w, strong, p)
        fails = [pl for pl in mine if pl.result == "fail"]
        if len(fails) >= p["minFailsForCurve"]:
            pts = [(db.chart_pos[pl.chart], pl.result == "clear", pl.at >= since) for pl in mine]
            pts = [(r, c, new) for r, c, new in pts if st.distance(r) is not None]
            if pts:
                if sum(pl.at >= since for pl in fails) >= p["minFailsForCurve"]:  # 十分に新しい失敗
                    st.learn([r for r, _, new in pts if new], [c for _, c, new in pts if new])
                charts, cleared, _ = zip(*pts)
                st.curve = _logistic(np.array([st.distance(r) for r in charts]), np.array(cleared, float))
        out[style] = st
    return out


def ceilings(db, hist, p):
    """-> {プレイスタイル: {指標: (下限, 上限)}}: 狙いとは別に、プレイヤーが入れた制限（`ceiling`）。各プレイスタイルで、
    プレイヤーの直近の素のクリア（譜面ごとに 1 つ、多くても `strongRecentClears`）から決めるので、一番よかったときでは
    なく今の実力に沿う。足りなければ PlayerData のクリア（プレイの多い順）でその数まで埋める。上限は `ceilingFeatures`
    のそれぞれの `ceilingPercentile`（下限は -inf）、下限は `floorFeatures` のそれぞれの中央値から `floorBelowMedian` を
    引いたもの（上限は inf）。両方にある指標は両方を持つ。クリアが `ceilingMinClears` より少ないプレイスタイルと、DB に
    ないかクリアに値のない指標は、どちらも持たない。"""
    tops = [f for f in p["ceilingFeatures"] if f in db.chart_metrics]
    bottoms = [f for f in p["floorFeatures"] if f in db.chart_metrics]
    if not tops and not bottoms:
        return {}
    imported = [db.chart_pos[k] for k, _ in sorted(hist.imported_clears.items(), key=lambda kv: (-kv[1], kv[0]))]
    clears = [db.chart_pos[pl.chart] for pl in hist.plays
              if pl.plain and pl.chart is not None and pl.result == "clear"]
    out = {}
    for style in sorted(set(db.play_styles)):
        rows = []
        for r in reversed(clears):  # 新しい順、譜面ごとに 1 つ
            if db.play_styles[r] == style and r not in rows:
                rows.append(r)
            if len(rows) >= p["strongRecentClears"]:
                break
        for r in imported:
            if len(rows) >= p["strongRecentClears"]:
                break
            if db.play_styles[r] == style and r not in rows:
                rows.append(r)
        if len(rows) < p["ceilingMinClears"]:
            continue
        cap = {}
        for f in dict.fromkeys(tops + bottoms):
            v = db.chart_metrics[f][rows]
            v = v[np.isfinite(v)]
            if len(v):
                cap[f] = (float(np.median(v)) - p["floorBelowMedian"] if f in bottoms else -np.inf,
                          float(np.percentile(v, p["ceilingPercentile"])) if f in tops else np.inf)
        if cap:
            out[style] = cap
    return out
