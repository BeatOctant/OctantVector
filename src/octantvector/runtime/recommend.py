"""recommend と describe（契約1）。

recommend(db, history, request, params) は 1 人のプレイヤーのために譜面に順位を付ける:
  1. 履歴をそのまま読む（history.py）: likes.classify による好意のマップ、譜面ごとのプレイ、旧い版は現行版に移す
  2. 入口で基準を選ぶ: 起点があれば起点の譜面。そのとき候補は、ビューによらず起点のプレイスタイルの譜面だけ（起点の
     譜面を選んだことがプレイスタイルを選んだことになる）。ジャンルの語があれば、プレイヤーが最近遊んだ好意のマップ
     （音楽のビューだけ。語に一番近いマップの中で）。どちらでもなければ、すべての好意のマップ。
     譜面のビューは、そのマップでプレイヤーが遊んだ譜面を取る（プレイのないマップは imported_play の譜面）。
     起点とジャンルの語の両方があれば、起点が基準のままで、語は候補を語に一番近いマップに絞るだけ。`genreFrom`
     （ジャンルの語の list）があり、起点も語もなければ、語をプレイヤーのために選ぶ: 好意のマップでの値の平均が一番
     高い一覧の語。基準はすべての好意のマップで、候補はその語のマップ（「あなたのジャンル」）。どの項目の理由も、
     一覧を絞った語（`genre`）を示すので、プレイヤーはどのジャンルが選ばれたかが分かる。
  3. どの候補の譜面にも、ビューごとの基準へのコサインの重み付き平均を付ける。譜面（かそのマップ）がベクトルを
     持たないビューは 0 と数え、重みは正規化し直さない。ビューの重みが `minViewWeight` に満たない譜面は外す
     （OctantSpecs 法 3.1）。どの基準もベクトルを持たないビューは使わない。
     譜面のビューは 1 つのプレイスタイルの譜面どうしだけを比べる。`all`（`weights` の代わりの、`weights` と同じ形の
     重み付きの 2 つ以上のビューの list）があれば、候補はどれでも近いこと: 各項目の重み付き平均を、すべての項目を
     持つ譜面の中での百分位にし、一番低い百分位で譜面を並べる（「これでもあれでも近い」。すべての項目で同じしきい値を
     十分に残るまで下げるのと同じ順）。順位は、すべての項目のビューでの重み付き平均の尺度に戻すので、難易度の罰は
     `weights` のときと同じ重さになる。項目の欠けた譜面（その重みの `minViewWeight` に満たない）は外す。どの基準も
     出せない項目は無視して重みと同じく返し、項目が 1 つだけ残れば `weights` と同じく並べる。
     関係のビュー（アーティスト、契約2）は、メンバーのある各基準のマップへの近さの平均をマップに与える。メンバーの
     ないマップは持たない。
  4. 難易度（skill.py）: プレイヤーのクリアが十分にあるプレイスタイルでは、狙いの目標の距離から許容幅より遠い譜面を
     外し（残りが少なすぎれば最大まで広げる）、残りに目標からの距離につれて大きくなる罰を付ける。クリアの足りない
     プレイスタイルは絞らない。
     `revisitDays` が 0 より大きければ、候補を好意のマップのうち最後のプレイがその日数より前のもの（history.revisit）
     だけにし、既知の除外は当てない（「久しぶり」）。
     プレイヤーがもう知っているマップは、`includeKnown` でなければ外す。挑戦の狙いでは、目標（法 4.1）を実力の
     あるプレイスタイルでだけ残すので、実力が追いつけば目標がまた出てくる。
     `ceiling`（プレイヤーが入れる）があれば、`ceilingFeatures` のどれかでプレイヤーの上限を超える譜面と、
     `floorFeatures` のどれかで下限を下回る譜面（skill.ceilings。直近のクリアから）を、どの手札でも外し、広げても
     戻さない。挑戦の狙いは、目標の近くが少なすぎれば、その下に残る一番難しい譜面を取る。
  5. 各マップの上位 `chartsPerMap` の譜面、各曲の上位 `mapsPerSong` のマップ、各アーティストの上位 `mapsPerArtist`
     のマップ、そして全体の上位 `count`。通るものが少なければ少なく返す。2 つのマップが同じ曲なのは、音楽のビューの
     平均コサインが `sameSongCosine` 以上（1 曲のほかの作者のマップは音声がほぼ同じで、そうしないと一覧を埋める）で、
     両方にアーティストがあれば一方のアーティストの組がもう一方を含むとき（ほかのアーティストのカバーは原曲と別のまま、
     "A" と "A feat. B" は 1 曲になり得る）。共通の音楽のビューのないマップは同じ曲にならない。起点のマップは、一覧に
     もう入ったその曲のマップと数えるので、既定の `mapsPerSong` では起点の曲のほかの作者のマップは出ない。2 つの
     マップが同じアーティストなのは、アーティストの組が同じとき。アーティストのないマップには上限をかけない。名前は
     使わない（OctantSpecs 法 3.2 と 3.4）。
`narrow`（`weights` と同じ形の重み付きのビュー）があれば、候補をまず、そのビューで同じ基準に一番近い
`narrowTopMaps` のマップに絞り（もう外したマップは枠を取らない）、残りを上と同じく `weights` で並べる。難易度の
許容幅を広げても `count` より少なければ、十分に残るかすべてのマップが入るまで、絞るマップの数に `narrowWiden` を
掛ける（まれなプレイスタイルは、一番近いマップの中に譜面が少ない）:「音で近く、譜面で並べる」（扱い方
narrow_then_rank）。DB か基準が出せない名前は無視して重みと同じく返す。
`mix`（{"hand", "request"} の list）があれば、一覧をいくつもの手札から一度に引く: 各項目をそれぞれの要求で単独に
推薦し（`count` と `chartsPerMap` は外の要求から。項目の `difficulty` を省くか "auto" なら外のものを取る）、`count`
の枠を重みで項目に分ける。どれも少なくとも 1 つ（最大剰余。項目が枠より多ければ重い項目が取る）。手札の重みは、
その手札のプレイヤーのそれまでの下見での残した率（`recommendation` の出来事、契約3: プレイヤーがダウンロードか
プレイで先へ進んだ下見。残した = ダウンロードか選択、外した = 削除か入れ替え。下見の反応はまとめて来るので、下見
ごとに 1 つの率）を、ミックスの手札の平均の率へ `mixPriorDrafts` 回分の重みで縮めたもの。下見が 1 つもなければ
どの手札も同じ重み。手札の譜面は重い手札から順に、各手札のよいものから交互に並べ、前の手札が置いた譜面やマップは
後の手札に譲り、手札が埋められない枠はほかが取る。各項目の理由はその `hand` を示す。反応が変えるのは手札が取る枠の
数だけで、手札の中の譜面の順位は変えず、マップを好意にもしない。
起点があり `chainEvery` が 0 より大きければ、一覧を進めながら起点を動かす: `chainEvery` マップごとに、上の段が
次に選ぶはずのマップを見せずに起点にし、残りをそれに対して同じく点を付ける（その譜面が起点。もう選んだマップは
外す）。`chainFirstWeight` は、その割合の最初の起点の点を後のどの段にも残す（0: 残さない）。絞り込み、難易度合わせ、
曲とアーティストの上限は一覧全体にかかる。項目の理由は、それを選んだ起点からのもの。起点がなければ何もしない。
同じ DB、履歴、要求、params なら結果は同じ。点は入れない: 理由は表示のためだけ（憲法の原則 4 と 6）。
"""
import numpy as np

from ..common import charts as chartids
from ..common.hands import view_keys
from . import history as hist_mod
from . import skill as skill_mod
from ..common.params import AIMS, resolve, why_invalid

TASTE_ROLES = ("music", "chart", "meta")


class RequestError(ValueError):
    """要求が DB に合わない（知らないジャンルの語か狙い）。"""


class SeedNotInDBError(RequestError):
    """起点のマップか譜面が DB にない（DB にあるマップの旧い版としてもない）。"""


class SeedNotEmbeddedError(SeedNotInDBError):
    """起点の譜面は DB のカタログにあるが、それもそのマップも、要求が使えるどのビューにもベクトルを持たないので、候補と
    比べるものがない。SeedNotInDBError の一種なので、そのエラーしか知らない呼ぶ側も、起点を使えないことをプレイヤーに
    伝えられる。どの起点がなぜかはメッセージにある。"""


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _keys(db, name):
    return view_keys(name, db.roles())


def _split(db, wanted):
    """({view: 重み}, DB にない名前): 各名前の重みを、それが指すビューに等しく分ける（common.hands.view_keys: 版のない
    id は DB にある版に分ける）。"""
    out, missing = {}, []
    for name, w in wanted.items():
        views = _keys(db, name)
        if not views:
            missing.append(name)
        for v in views:
            out[v] = out.get(v, 0.0) + float(w) / len(views)
    return {v: w for v, w in out.items() if w > 0}, missing


def _even(db, roles):
    """DB にある `roles` のすべてのビュー。役割ごとに同じ重みを、そのビューに分ける。"""
    out = {}
    for r in roles:
        views = db.views_of_role(r)
        out.update((v, 1.0 / len(views)) for v in views)
    return out


def _weights(db, wanted):
    """({view: 重み}, 無視した名前)。DB にない名前は無視して返すので、DB を替えても推薦は止まらない（契約1）。既定か、
    どの名前も無視されたときは、ある好みの役割ごとに同じ重みを、そのビューに分ける。"""
    out, ignored = _split(db, wanted or {})
    if not wanted or len(ignored) == len(wanted):
        out = _even(db, TASTE_ROLES)
    return out, ignored


def _unserved(db, wanted, weights, refs, ignored):
    """頼まれた名前のうち、どのビューにも基準がつかなかった（履歴か起点がそこにベクトルを持たない）ものを `ignored` に
    足す: DB にない名前と同じく返す（契約1）。"""
    for name in wanted:
        if name not in ignored and not any(v in refs and v in weights for v in _keys(db, name)):
            ignored.append(name)


def _settle(db, wanted, weights, refs, ignored, default, make_refs):
    """基準が出せる重み（_unserved）。重みを付けたどのビューにも基準がなく、重みが頼まれていたら、頼まれなかったかの
    ように `default` を使う。"""
    if not wanted:
        return weights, refs
    _unserved(db, wanted, weights, refs, ignored)
    if any(v in refs for v in weights):
        return weights, refs
    return default, make_refs(default)


def _map_view_ref(db, view, hashes):
    x, _ = db.views[view]
    vecs = [x[n] for n in (db.row_in_view(view, db.pos[h]) for h in hashes) if n is not None]
    return _unit(np.mean(vecs, 0)) if vecs else None


def _chart_view_refs(db, view, chart_rows):
    """譜面の行からの {プレイスタイル: 基準のベクトル}。"""
    x, _ = db.views[view]
    by_style = {}
    for r in chart_rows:
        n = db.row_in_view(view, r)
        if n is not None:
            by_style.setdefault(db.play_styles[r], []).append(x[n])
    return {s: u for s, v in by_style.items() if (u := _unit(np.mean(v, 0))) is not None}


def _view_scores(db, view, ref):
    """-> 譜面の行ごとの (コサイン, has): マップのビューは譜面のマップのコサイン、譜面のビューは譜面そのもののコサイン、
    関係のビューは譜面のマップの近さ。行が持たないところでは `has` は False。"""
    n = len(db.chart_keys)
    if db.rows[view] == "charts":
        x, rows = db.views[view]
        out, has = np.zeros(n), np.zeros(n, bool)
        styles = db.styles_in_view(view)
        for style, q in ref.items():
            m = styles == style
            out[rows[m]] = x[m] @ q
            has[rows[m]] = True
        return out, has
    per_map, has_map = np.zeros(len(db.hashes)), np.zeros(len(db.hashes), bool)
    if view in db.relations:
        rows, s = db.relations[view].rows, ref
    else:
        x, rows = db.views[view]
        s = x @ ref
    per_map[rows], has_map[rows] = s, True
    m = db.chart_map_array()
    ok = m >= 0
    return np.where(ok, per_map[m], 0.0), ok & has_map[m]


def _references(db, views, ref_maps, ref_charts):
    refs = {}
    for v in views:
        if v in db.relations:
            r = db.relations[v].reference([db.pos[h] for h in ref_maps])
        elif db.rows[v] == "charts":
            r = _chart_view_refs(db, v, ref_charts)
        else:
            r = _map_view_ref(db, v, ref_maps)
        if r is not None and (not isinstance(r, dict) or r):
            refs[v] = r
    return refs


class _Taste:
    """すべての譜面の行の点: `score`（重み付き平均）、`ok`（重みが十分にあり、譜面のマップが DB にある）、ビューごとの
    寄与とそれがあるところ。"""

    def __init__(self, n):
        self.score, self.ok, self.parts = np.zeros(n), np.zeros(n, bool), {}

    def parts_of(self, c):
        return {v: float(s[c]) for v, (s, has) in self.parts.items() if has[c]}


def _taste(db, weights, refs, min_share):
    """-> _Taste。基準のないビュー（基準のマップや譜面のどれもそこにベクトルを持たない）は重みから外す: 欠けているのは
    候補ではなくプレイヤーの履歴のほうだから。"""
    weights = {v: w for v, w in weights.items() if v in refs}
    total = sum(weights.values())
    out = _Taste(len(db.chart_keys))
    if not total:
        return out
    present = 0.0
    for v, w in weights.items():
        s, has = _view_scores(db, v, refs[v])
        part = np.where(has, w * s / total, 0.0)
        out.parts[v] = part, has
        out.score = out.score + part
        present = present + np.where(has, w, 0.0)
    out.ok = (db.chart_map_array() >= 0) & (present / total >= min_share)
    return out


def _taste_all(db, weights, groups, refs, min_share):
    """-> `all` のための _Taste: `groups`（それぞれ {view: 重み}。`refs` が出す）を一番低い百分位で並べ、`weights`（すべての
    グループのビュー）での重み付き平均の尺度にする。内訳はグループごとのもの。"""
    base = _taste(db, weights, refs, 0.0)
    each = [_taste(db, g, refs, min_share) for g in groups]
    out = _Taste(len(db.chart_keys))
    out.ok = base.ok & np.logical_and.reduce([t.ok for t in each])
    for t in each:
        out.parts.update(t.parts)
    n = int(out.ok.sum())
    if not n:
        return out
    pct = [np.searchsorted(np.sort(t.score[out.ok]), t.score, "left") / max(n - 1, 1) for t in each]
    low = np.minimum(np.minimum.reduce(pct), 1.0)
    scale = np.sort(base.score[out.ok])
    # 一番低い百分位が同じなら、平均の百分位の高いほうを上に（ほかの順位を越えないほど小さい量）
    out.score = scale[np.rint(low * (n - 1)).astype(int)] + 1e-9 * np.mean(pct, 0)
    return out


def _all(db, request):
    """`all` のある要求の (基準のための `weights` と同じ形の重み付きの名前, 項目ごとの [{view: 重み}])。なければ
    (request["weights"], None)。"""
    groups = request.get("all")
    if groups is None:
        return request.get("weights"), None
    if request.get("weights") is not None:
        raise RequestError("a request has weights or all, not both")
    if not isinstance(groups, list) or len(groups) < 2 or not all(isinstance(g, dict) and g for g in groups):
        raise RequestError("all is not a list of two or more views with weights")
    wanted = {}
    for g in groups:
        for name, w in g.items():
            wanted[name] = wanted.get(name, 0.0) + float(w)
    return wanted, [_split(db, g)[0] for g in groups]


def _served(weights, groups, refs):
    """1 組の基準の (weights, groups): 各グループは、重みが付いていて基準の出せるビューを残す（ジャンルの項目は音楽の
    ビューだけを重くし、動いた起点はビューを欠くことがある）。グループが 2 つ未満しか残らなければ、残った 1 つ（か
    `weights`）を `weights` と同じく並べる。"""
    if groups is None:
        return weights, None
    groups = [g for g in ({v: w for v, w in g.items() if v in weights and v in refs} for g in groups) if g]
    if len(groups) >= 2:
        return weights, groups
    return (groups[0] if groups else weights), None


def _seed_chart(db, seed, replaced):
    if not isinstance(seed, dict):
        raise RequestError(f"seed must be an object with hash, characteristic and difficulty, got {seed!r}")
    h, c, d = seed.get("hash"), seed.get("characteristic"), seed.get("difficulty")
    cur = db.current(h)
    if cur is None:
        raise SeedNotInDBError(f"seed {h} is not in DB {db.version}")
    key = chartids.key(cur, c, d)
    if key not in db.chart_pos:
        raise SeedNotInDBError(f"seed chart {chartids.key(h, c, d)} is not in DB {db.version}")
    if cur != h:
        replaced.append({"from": chartids.key(h, c, d), "to": key})
    return cur, db.chart_pos[key]


def _exclusions(db, exclude):
    maps, chart_rows = set(), set()
    for e in exclude or ():
        if not isinstance(e, dict):
            raise RequestError(f"an exclude entry must be an object with a hash, got {e!r}")
        h = e.get("hash")
        h = db.current(h) or h
        if e.get("characteristic") and e.get("difficulty"):
            r = db.chart_pos.get(chartids.key(h, e["characteristic"], e["difficulty"]))
            if r is not None:
                chart_rows.add(r)
        else:
            maps.add(h)
    return maps, chart_rows


def _taste_word(db, words, liked):
    """`words` の語のうち、その値を持つ好意のマップでの平均が一番高いもの。なければ None。"""
    g = db.genres()
    missing = [w for w in words if w not in g]
    if missing:
        raise RequestError(f"genre words {missing!r} are not in DB {db.version}; words: {sorted(g)}")
    best = None
    for w in words:
        view, dim = g[w]
        x, rows = db.views[view]
        on = np.isin(rows, [db.pos[h] for h in liked])
        if on.any():
            mean = float(x[on, dim].mean())
            if best is None or mean > best[0]:
                best = (mean, w)
    return None if best is None else best[1]


def _genre_maps(db, word, share):
    g = db.genres()
    if word not in g:
        raise RequestError(f"genre word {word!r} is not in DB {db.version}; words: {sorted(g)}")
    view, dim = g[word]
    x, rows = db.views[view]
    s = x[:, dim]
    keep = max(1, int(round(len(s) * share)))
    top = np.argsort(-s, kind="stable")[:keep]
    return {db.hashes[int(rows[n])] for n in top}


def _today():
    """「久しぶり」の日数を数える今日（UTC の日付。プレイの `at` は UTC、契約3）。試験で差し替える。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date()


def recommend(db, history, request, params=None):
    why = why_invalid(params)
    if why:  # 暴走し得る値（際限のない許容幅の広げ）や numpy を壊す値はここで止める
        raise RequestError(f"params: {why}")
    p = resolve(params)
    request = request or {}
    if request.get("mix") is not None:
        return _mixed(db, history, request, params, p)
    hist = hist_mod.parse(db, history)
    replaced = list(hist.replaced)
    liked = hist.liked()
    wanted, groups = _all(db, request)
    weights, ignored = _weights(db, wanted)
    default = _weights(db, None)[0]
    ex_maps, ex_charts = _exclusions(db, request.get("exclude"))
    # 「久しぶり」（revisitDays）: 候補を、好意のマップのうち最近遊んでいないものだけにする。どれも既知のマップなので、
    # 既知の除外は当てない（削除したマップは hist.revisit が外す）
    revisit = hist.revisit(int(p["revisitDays"]), _today()) if int(p["revisitDays"]) > 0 else None
    # 目標（法 4.1）は挑戦の狙いの候補に残す。実力が追いついたら戻ってくるように
    goals = hist.goals() if request.get("difficulty") == "challenge" and not p["includeKnown"] else set()
    if not p["includeKnown"] and revisit is None:
        ex_maps |= hist.known() - goals
    allowed_maps = None

    def played_charts(maps):
        """`maps` でプレイヤーが遊んだ譜面の行: プレイから、プレイのないマップは imported_play から（好意のマップと同じ
        決まり）。OctantPath がまだプレイを見ていなくても、譜面のビューに基準があるように。"""
        rows = {db.chart_pos[pl.chart] for pl in hist.plays if pl.chart and pl.hash in maps}
        with_plays = {pl.hash for pl in hist.plays}
        rows |= {db.chart_pos[k] for h in maps if h not in with_plays for k in hist.imported.get(h, ())}
        return sorted(rows)

    def taste(refs, min_share):
        w, g = _served(weights, groups, refs)
        return _taste(db, w, refs, min_share) if g is None else _taste_all(db, w, g, refs, min_share)

    word = request.get("genre")  # 一覧を絞るジャンルの語。あれば（各項目の理由がそれを示す）
    if request.get("seed") is not None:
        if request.get("genre") is not None:  # 起点があれば、語は絞るだけ:「これに近く、そのジャンルで」
            allowed_maps = _genre_maps(db, request["genre"], float(p["genreTopShare"]))
        seed_map, seed_row = _seed_chart(db, request["seed"], replaced)
        ex_maps.add(seed_map)
        ref_maps, ref_charts = [seed_map], [seed_row]
        weights, refs = _settle(db, wanted, weights, _references(db, weights, [seed_map], [seed_row]), ignored,
                                default, lambda w: _references(db, w, [seed_map], [seed_row]))
        if not refs:
            raise SeedNotEmbeddedError(f"seed chart {db.chart_keys[seed_row]} has no vector in DB {db.version}")
        mix = float(p["seedTasteWeight"])
        if mix > 0 and liked:
            taste_refs = _references(db, weights, sorted(liked), played_charts(liked))
            scored = taste(refs, p["minViewWeight"])
            b = taste(taste_refs, 0.0)
            scored.score = (1 - mix) * scored.score + mix * np.where(b.ok, b.score, 0.0)
        else:
            scored = taste(refs, p["minViewWeight"])
    else:
        if request.get("genre") is not None:
            allowed_maps = _genre_maps(db, request["genre"], float(p["genreTopShare"]))
            recent = [pl.hash for pl in hist.plays[-int(p["recentPlays"]):]]
            ref_maps = sorted(set(recent) & liked)
            default = _even(db, ("music",))
            weights = {v: w for v, w in weights.items() if db.manifest["views"][v].get("role") == "music"} or default
        elif request.get("genreFrom") is not None:
            word = _taste_word(db, request["genreFrom"], liked)
            if word is not None:
                allowed_maps = _genre_maps(db, word, float(p["genreTopShare"]))
            ref_maps = sorted(liked)
        else:
            ref_maps = sorted(liked)
        if not ref_maps or not weights:
            return _result(db, [], replaced, {}, hist, ignored, word)
        ref_charts = played_charts(set(ref_maps))
        weights, refs = _settle(db, wanted, weights, _references(db, weights, ref_maps, ref_charts), ignored, default,
                                lambda w: _references(db, w, ref_maps, ref_charts))
        scored = taste(refs, p["minViewWeight"])

    chain = None
    if request.get("seed") is not None and int(p["chainEvery"]) > 0:
        def chain(row):
            refs = _references(db, weights, [db.hashes[db.chart_map_array()[row]]], [row])
            return taste(refs, p["minViewWeight"]) if refs else None

    narrowed = _narrow(db, request["narrow"], ref_maps, ref_charts, ex_maps, ignored, p) if request.get("narrow") else None

    difficulty = request.get("difficulty", "normal")
    if difficulty in (None, "auto"):
        difficulty = "normal"
    if not (difficulty in AIMS or isinstance(difficulty, (int, float))):
        raise RequestError(f"difficulty must be one of {AIMS} or a clear chance, got {difficulty!r}")
    skills = skill_mod.estimate(db, hist, p)
    targets = {s: st.target(difficulty) for s, st in skills.items()}

    chart_map = db.chart_map_array()
    cand = scored.ok & ~_map_mask(db, ex_maps)[chart_map]
    cand[sorted(ex_charts)] = False
    if allowed_maps is not None:
        cand &= _map_mask(db, allowed_maps)[chart_map]
    if revisit is not None:
        cand &= _map_mask(db, revisit)[chart_map]
    if request.get("seed") is not None:  # 起点のプレイスタイルだけ: マップのビューはマップのどの譜面にも同じ値を与える
        cand &= db.play_style_array() == db.play_styles[seed_row]
    in_narrow = None
    if narrowed is not None:
        rank = np.full(len(db.hashes), len(narrowed))
        rank[[db.pos[h] for h in narrowed]] = np.arange(len(narrowed))
        in_narrow = rank[chart_map]
    skilled, dist, z = np.zeros(len(cand), bool), np.full(len(cand), np.nan), np.zeros(len(cand))
    for style, st in skills.items():
        d = st.distances()[0]
        skilled[st.rows], dist[st.rows] = True, d
        z[st.rows] = np.abs(d - targets[style][1]) / st.sd
    cand &= ~skilled | np.isfinite(dist)
    cand &= skilled | ~_map_mask(db, goals)[chart_map]  # 目標は、実力で見分けられるプレイスタイルでだけ
    caps = skill_mod.ceilings(db, hist, p) if p["ceiling"] else {}
    if caps:
        styles = db.play_style_array()
        for style, cap in caps.items():
            rows = np.flatnonzero(styles == style)
            for f, (low, top) in cap.items():
                v = db.chart_metrics[f][rows]
                cand[rows[(v < low) | (v > top)]] = False  # 値のない譜面は残す

    def diff(c):
        if not skilled[c]:
            return None
        st, d = skills[db.play_styles[c]], float(dist[c])
        aim, target = targets[db.play_styles[c]]
        return {"aim": aim, "distance": d, "target": target, "z": float(z[c]), "clearChance": st.chance(d),
                "nearestStrong": db.chart_keys[st.nearest_strong(c)]}

    count = int(request.get("count", 10))
    per_map = int(request.get("chartsPerMap", 1))
    tol, tol_max = float(p["toleranceSd"]), float(p["toleranceMaxSd"])
    songs = _Songs(db, float(p["sameSongCosine"]), int(p["mapsPerSong"]), int(p["mapsPerArtist"]),
                   [seed_map] if request.get("seed") is not None else [])
    pen = np.where(skilled, p["penaltyWeight"] * 0.05 * z ** 2, 0.0)
    top = int(p["narrowTopMaps"])
    parts = {}

    def pick(within, tol):
        if chain is None:
            out = _pick(db, within, skilled, z, scored.score - pen, tol, count, per_map, songs)
            parts.update((c, scored.parts_of(c)) for c in out)
            return out
        return _chain_pick(db, within, skilled, z, scored, pen, tol, count, per_map, songs, chain,
                           int(p["chainEvery"]), float(p["chainFirstWeight"]), parts)

    while True:
        within = cand if in_narrow is None else cand & (in_narrow < top)
        tol = float(p["toleranceSd"])
        picked = pick(within, tol)
        while len(picked) < count and tol < tol_max:
            tol = min(tol_max, tol + 0.5)
            picked = pick(within, tol)
        if len(picked) < count and caps and difficulty == "challenge":
            # 上限の下では、挑戦の目標が上限より上にあり得る: 残りを点で取る。点の罰は目標に一番近い譜面（上限の下に残る一番
            # 難しいもの）を先に置く
            picked = pick(within, np.inf)
        if len(picked) >= count or in_narrow is None or top >= len(narrowed):
            break
        top *= max(2, int(p["narrowWiden"]))  # 絞ったマップの中に少なすぎる（まれなプレイスタイル）: もっと取る
    picked = [(c, db.hashes[chart_map[c]], parts[c], diff(c)) for c in picked]
    return _result(db, picked, replaced, {s: t[0] for s, t in targets.items()}, hist, ignored, word)


def _mix_weights(history, hands, prior):
    """各手札の下見へのプレイヤーの反応（契約3 の `recommendation` の出来事。段階 shown / skipped / downloaded /
    selected。requestId でつなぐ）からの {手札: 重み}: プレイヤーが先へ進んだ下見ごとに、反応した譜面のうち残した
    ものの割合。手札の率は、その下見での平均を、どのミックスの手札の下見での平均へも `prior` 回分の重みで縮めたもの。
    下見のない手札と、どの手札にも下見がないときは、共通の率を取る。下見の手札は、出来事に `hand` があればそれ
    （ミックスの下見の譜面）、なければその `preset`。"""
    drafts = {}
    for e in history or ():
        if not isinstance(e, dict) or e.get("type") != "recommendation":
            continue
        hand, rid, stage, key = e.get("hand") or e.get("preset"), e.get("requestId"), e.get("stage"), (
            e.get("hash"), e.get("characteristic"), e.get("difficulty"))
        if hand not in hands or not rid or stage not in ("shown", "skipped", "downloaded", "selected"):
            continue
        charts = drafts.setdefault((hand, rid), {})
        charts.setdefault(key, set()).add(stage)
    rates = {h: [] for h in hands}
    for (hand, _), charts in sorted(drafts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        kept = sum(1 for st in charts.values() if st & {"downloaded", "selected"})
        skipped = sum(1 for st in charts.values() if "skipped" in st and not st & {"downloaded", "selected"})
        if kept:  # プレイヤーが下見から先へ進んだので、反応が付いている
            rates[hand].append(kept / (kept + skipped))
    everything = [r for rs in rates.values() for r in rs]
    common = sum(everything) / len(everything) if everything else 1.0
    return {h: (sum(rs) + prior * common) / (len(rs) + prior) if rs and len(rs) + prior > 0 else common
            for h, rs in rates.items()}


def _mix_places(weights, count):
    """`count` を重みで分けた {手札: 枠}（最大剰余。同じなら前の手札）。続いて枠が続く限り、一番多い手札から取って
    どれにも少なくとも 1 つ。枠が手札より少なければ、重い手札が取る。"""
    order = sorted(weights, key=lambda h: (-weights[h], list(weights).index(h)))
    chosen = order[:max(count, 0)]
    total = sum(weights[h] for h in chosen) or 1.0
    exact = {h: count * weights[h] / total for h in chosen}
    places = {h: int(exact[h]) for h in chosen}
    rest = count - sum(places.values())
    for h in sorted(chosen, key=lambda h: (-(exact[h] - places[h]), order.index(h)))[:rest]:
        places[h] += 1
    for h in reversed(chosen):  # 軽いものから: 一番多い手札から 1 枠
        if places[h] == 0:
            most = max(chosen, key=lambda o: (places[o], -order.index(o)))
            places[most] -= 1
            places[h] += 1
    return {h: places.get(h, 0) for h in weights}


def _mixed(db, history, request, params, p):
    """`mix` の要求のための recommend: 項目ごとに 1 つの推薦。枠を _mix_weights と _mix_places で分け、一覧を交互に並べる
    （recommend の docstring）。"""
    mix = request["mix"]
    if not isinstance(mix, list) or not mix or not all(
            isinstance(m, dict) and isinstance(m.get("hand"), str) and isinstance(m.get("request"), dict) for m in mix):
        raise RequestError("mix must be a list of {hand, request} entries")
    hands = [m["hand"] for m in mix]
    if len(set(hands)) != len(hands):
        raise RequestError("mix names a hand twice")
    count = int(request.get("count", 10))
    per_map = int(request.get("chartsPerMap", 1))
    weights = _mix_weights(history, hands, float(p["mixPriorDrafts"]))
    places = _mix_places(weights, count)
    results = {}
    for m in mix:
        sub = {k: v for k, v in m["request"].items() if k != "mix"}
        sub["count"], sub["chartsPerMap"] = count, per_map  # count 全体: 手札の余った譜面がほかの枠を埋める
        if sub.get("difficulty") in (None, "auto"):
            sub["difficulty"] = request.get("difficulty", "normal")
        results[m["hand"]] = recommend(db, history, sub, {**(params or {}), **(sub.get("params") or {})})
    order = sorted(hands, key=lambda h: (-places[h], -weights[h], hands.index(h)))
    queues = {h: list(results[h]["items"]) for h in hands}
    picked, seen_charts, per = [], set(), {}

    def take(h):
        while queues[h]:
            it = queues[h].pop(0)
            key = (it["hash"], it["characteristic"], it["difficulty"])
            if key in seen_charts or per.get(it["hash"], 0) >= per_map:
                continue
            seen_charts.add(key)
            per[it["hash"]] = per.get(it["hash"], 0) + 1
            picked.append({**it, "reason": {**it["reason"], "hand": h}})
            return True
        return False

    filled, open_ = {h: 0 for h in hands}, list(order)
    while open_:  # 1 巡: 枠の残る各手札から 1 譜面ずつ、重いものから。各手札は自分の順で
        for h in list(open_):
            if filled[h] >= places[h] or not take(h):
                open_.remove(h)
            else:
                filled[h] += 1
    for h in order:  # 枠が残った（手札に少なすぎた）: ほかの余った譜面を、重い手札から
        while len(picked) < count and take(h):
            pass
    first = results[hands[0]]
    replaced, seen = [], set()
    for h in hands:
        for r in results[h]["replaced"]:
            k = (r["from"], r["to"])
            if k not in seen:
                seen.add(k)
                replaced.append(r)
    ignored = []
    for h in hands:
        ignored.extend(n for n in results[h]["ignoredWeights"] if n not in ignored)
    aims = {}
    for h in hands:  # プレイスタイルの狙い: 手札の狙いが違えば最初の手札のもの（各項目は自分のものを言う）
        for style, aim in results[h]["aims"].items():
            aims.setdefault(style, aim)
    return {"dbVersion": db.version, "items": [{**it, "rank": n} for n, it in enumerate(picked, 1)], "aims": aims,
            "replaced": replaced, "ignoredWeights": ignored, "unknownEvents": first["unknownEvents"],
            "mix": {h: {"weight": round(weights[h], 4), "places": places[h], "given": filled[h]} for h in hands}}


def _map_mask(db, hashes):
    """`hashes` のマップの行（DB にあるもの）をマスクにしたもの。"""
    out = np.zeros(len(db.hashes), bool)
    out[[db.pos[h] for h in hashes if h in db.pos]] = True
    return out


def _narrow(db, wanted, ref_maps, ref_charts, ex_maps, ignored, p):
    """外していないすべてのマップを、`narrow` のビューで基準に近い順に（recommend は最初の `narrowTopMaps` を残し、譜面が
    少なすぎればもっと残す）。そこのどの名前も出せなければ None（その名前は `ignored` に入り、何も絞らない）。"""
    weights = _split(db, wanted)[0]
    refs = _references(db, weights, ref_maps, ref_charts)
    _unserved(db, wanted, weights, refs, ignored)
    weights = {v: w for v, w in weights.items() if v in refs}
    if not weights:
        return None
    taste = _taste(db, weights, refs, p["minViewWeight"])
    chart_map = db.chart_map_array()
    keep = taste.ok & ~_map_mask(db, ex_maps)[chart_map]
    best = np.full(len(db.hashes), -np.inf)
    np.maximum.at(best, chart_map[keep], taste.score[keep])
    maps = np.flatnonzero(np.isfinite(best))
    ranked = maps[np.lexsort((db.hash_array()[maps], -best[maps]))]
    return [db.hashes[m] for m in ranked]


class _Songs:
    """選んだマップのうち、候補のマップと同じ曲か同じアーティストのもの（OctantSpecs 法 3.4）: 音楽のビュー（マップの行）
    と、DB にあれば役割 `artist` の関係のビューのアーティストの組。"""

    def __init__(self, db, threshold, per_song, per_artist=0, seeds=()):
        self.db, self.threshold, self.per_song, self.per_artist = db, threshold, per_song, per_artist
        self.views = db.views_of_role("music")
        self.rel = db.artists()
        self.seeds = list(seeds)  # 曲が一覧にもうあると数えるマップ（アーティストは数えない）

    def artists(self, h):
        return frozenset(self.rel.members_of(self.db.pos[h])) if self.rel is not None else frozenset()

    def vectors(self, h):
        out = {}
        for v in self.views:
            n = self.db.row_in_view(v, self.db.pos[h])
            if n is not None:
                out[v] = self.db.views[v][0][n]
        return out

    def same(self, a, b, artists_a=frozenset(), artists_b=frozenset()):
        if artists_a and artists_b and not (artists_a <= artists_b or artists_b <= artists_a):
            return False  # 片方だけにいる客演のアーティストなら、まだ 1 曲
        shared = [v for v in a if v in b]
        return bool(shared) and float(np.mean([a[v] @ b[v] for v in shared])) >= self.threshold

    def full(self, h, picked):
        """h の曲のマップが `mapsPerSong`、または h のアーティストのマップが `mapsPerArtist`、もう `picked`（マップの hash、
        順に）にあれば True。"""
        artists = self.artists(h)
        if self.per_artist > 0 and artists and sum(self.artists(o) == artists for o in picked) >= self.per_artist:
            return True
        if self.per_song <= 0 or not self.views:
            return False
        a = self.vectors(h)
        if not a:
            return False
        others = self.seeds + list(picked)
        return sum(self.same(a, self.vectors(o), artists, self.artists(o)) for o in others) >= self.per_song


def _chain_pick(db, cand, skilled, z, first, pen, tol, count, per_map, songs, make, every, first_weight, parts):
    """-> 譜面の行: _pick で一度に `every` マップずつ選び、_pick が次に取るマップを起点にする（その譜面が基準で、マップは
    見せない）。`count` 選ぶか、起点を作れなくなるまで。`parts` は、選んだ各行の理由を、それを選んだ起点から受け取る。"""
    chart_map = db.chart_map_array()
    cand = cand.copy()
    taste, score = first, first.score
    out, maps = [], []
    while len(out) < count:
        n = min(every, count - len(out))
        step = _pick(db, cand, skilled, z, score - pen, tol, n + 1, per_map, songs, maps)
        taken = step[:n]
        for c in taken:
            parts[c] = taste.parts_of(c)
            if db.hashes[chart_map[c]] not in maps:
                maps.append(db.hashes[chart_map[c]])
        out += taken
        if len(step) <= n or len(out) >= count:
            break
        seed = step[n]
        cand &= ~_map_mask(db, maps + [db.hashes[chart_map[seed]]])[chart_map]
        nxt = make(seed)
        if nxt is None:
            break
        taste = nxt
        score = np.where(nxt.ok, nxt.score, 0.0)
        if first_weight > 0:
            score = (1 - first_weight) * score + first_weight * first.score
    return out


def _pick(db, cand, skilled, z, score, tol, count, per_map, songs, prior=()):
    """-> 譜面の行。点、続いて行の順でよいものから: 目標から `tol` 以内、マップごとに `per_map`、曲とアーティストの上限
    （_Songs）。`prior`（一覧にもうあるマップの hash）のマップを選んだものと数える。"""
    rows = np.flatnonzero(cand & ~(skilled & (z > tol)))
    rows = rows[np.lexsort((rows, -score[rows]))]
    chart_map = db.chart_map_array()
    out, per, maps = [], {}, list(prior)
    for c in rows.tolist():
        h = db.hashes[chart_map[c]]
        if per.get(h, 0) >= per_map:
            continue
        if h not in per:
            if songs.full(h, maps):
                continue
            maps.append(h)
        per[h] = per.get(h, 0) + 1
        out.append(c)
        if len(out) >= count:
            break
    return out


def _result(db, picked, replaced, aims, hist, ignored, word=None):
    """aims: {プレイスタイル: 使った狙い}。そこにないプレイスタイルは難易度で絞らなかった（クリアが少なすぎる）。
    word: 一覧を絞ったジャンルの語（頼まれたか、`genreFrom` が選んだ）。どの項目の理由にも示す。"""
    items, nps = [], db.chart_metrics.get("nps")
    for rank, (c, h, parts, diff) in enumerate(picked, 1):
        _, ch, d = chartids.split(db.chart_keys[c])
        reason = {"views": {v: round(s, 6) for v, s in parts.items()}, "difficulty": None}
        if word is not None:
            reason["genre"] = word
        if diff is not None:
            nh, nc, nd = chartids.split(diff["nearestStrong"])
            reason["difficulty"] = {"aim": diff["aim"], "distance": round(diff["distance"], 6),
                                    "target": round(diff["target"], 6),
                                    "clearChance": None if diff["clearChance"] is None else round(diff["clearChance"], 4),
                                    "nearestStrong": {"hash": nh, "characteristic": nc, "difficulty": nd}}
        v = None if nps is None or not np.isfinite(nps[c]) else round(float(nps[c]), 4)
        items.append({"hash": h, "characteristic": ch, "difficulty": d, "rank": rank, "nps": v, "reason": reason})
    return {"dbVersion": db.version, "items": items, "aims": aims, "replaced": replaced, "ignoredWeights": ignored,
            "unknownEvents": hist.unknown}


def describe(db, history, params=None):
    """プレイヤーの表示用の要約（契約1 の `describe`）。"""
    why = why_invalid(params)
    if why:
        raise RequestError(f"params: {why}")
    p = resolve(params)
    hist = hist_mod.parse(db, history)
    skills = skill_mod.estimate(db, hist, p)
    feats = [f for f, w in p["features"].items() if w and f in db.chart_metrics]
    styles = {}
    for style, st in skills.items():
        strong = st.strong
        styles[style] = {
            "strongCharts": len(strong),
            "features": {f: float(np.nanmedian(db.chart_metrics[f][strong])) for f in feats},
            "typicalDistance": float(np.median(st.self_d)),
            "clearCurve": st.curve is not None,
            # 重いものから。weight: その向きの差を、設定の重みの何倍に数えるか（skill.py）
            "weakDirections": [{"feature": feats[f], "direction": side, "weight": m} for f, side, m in st.weak],
        }
    return {"dbVersion": db.version, "likedMaps": len(hist.liked()), "plays": len(hist.plays),
            "playStyles": styles, "replaced": hist.replaced, "unknownEvents": hist.unknown}
