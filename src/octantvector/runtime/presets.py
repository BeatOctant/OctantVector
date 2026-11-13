"""手札（presets、hands）: 埋め込みの使い方に名前を付けたもの（契約1）。

手札は 1 つの扱い方（埋め込みをどう使うか。扱い方そのものはここのコードで、DB には入らない）とそのつまみに、
id、ラベル、要る入力を付けたもの。DB はマニフェスト（`presets`）に自分の手札を持てて、そのときは下の既定を
置き換える。なければ既定を使う。

手札に要るものは、作り手が宣言するのではなく、つまみから読む: 手札を一覧に出すのは次のときだけ
  - この OctantVector がその扱い方を知り、どのつまみもその扱い方が取るもので、受け付ける値であること
  - つまみが名指すどのビューも、どのジャンルの語も DB にあること（重みはビューの id（どの版でも）か id@version を
    名指す: common.hands.view_keys。役割はラベルにすぎない）
  - 要るどの入力も、この OctantVector が埋められること: 起点か、`name` が扱い方のつまみで選択肢の残る選択
    （選択肢のない `genre` の選択は DB の語を取る）
どれかに反する手札は、それだけを外す。一覧の残りと DB はそのまま使う。

  presets(db)                 -> {"revision", "presets": [{id, group?, label, hint?, inputs, genreLabels?}]}
表示する文字（label、group、hint、入力や選択肢の label）は文字列か {言語: 文字列}。例
{"en": "Similar songs", "ja": "似た曲"}。そのまま渡す。
手札は `genreLabels`（{ジャンルの語: 表示する文字}）も持てる: 一覧を絞ったジャンルの語に出す名前（各項目の理由が
それを示す、recommend.py）。DB にない語と、文字でない名前は一覧から外し、どれもない手札は語そのものを出す。
  expand_preset(db, request)  -> `preset` と `choices` を契約1（OctantCompass to OctantVector）の条件にした要求。
                                 要求が自分で送った条件が勝つ。`preset` のない要求はそのまま返す
ミックスの手札（扱い方 `mixed`、つまみ `hands`: 一覧のほかの手札の id）は `mix` に広がる: 名指した手札ごとに 1 つの
項目で、それぞれその手札が単独で受け取る要求を持つ（要求の条件、ミックスの手札のつまみ、名指した手札のつまみの順）。
確かめる処理は common/hands.py にある。
広げる処理は条件を組み合わせるだけで、候補に点を付けたり、並べたり、削ったりはしない。
"""
import hashlib
import json

from ..common.hands import EITHER_KNOBS, HANDLINGS, inputs, is_text, why_not, why_not_mixing
from .recommend import RequestError


# 自分の手札を持たない DB の一覧: "all" だけ。ほかの手札はどれもビューを名指し、DB がどのビューを持つかは作り手が
# 決めるため。英語の言い回しは推測。
DEFAULT_PRESETS = [
    {"id": "all", "group": {"en": "Song & map", "ja": "曲と譜面"}, "label": {"en": "All", "ja": "すべて"},
     "hint": {"en": "Similar to the selected map overall", "ja": "選んでいる譜面に近いものを、どの軸でも"},
     "inputs": [{"kind": "seed"}], "handling": "near", "knobs": {}},
]


class UnknownPresetError(RequestError):
    """要求が今の一覧にない手札を名指した。"""


def _db_facts(db):
    return db.roles(), set(db.genres())


def _hands(db):
    """この DB とこの OctantVector が出せる手札の {id: (手札, 一覧に出す入力)}。一覧の順。"""
    hands = db.manifest.get("presets")
    hands = hands if isinstance(hands, list) and hands else DEFAULT_PRESETS
    views, genres = _db_facts(db)
    out = {}
    for hand in hands:
        if why_not(hand, views, genres) is None and hand["id"] not in out:
            out[hand["id"]] = (hand, inputs(hand, views, genres)[0])
    served = {k: h for k, (h, _) in out.items()}
    return {k: v for k, v in out.items() if why_not_mixing(v[0], served, views, genres) is None}


def presets(db):
    listed = []
    genres = _db_facts(db)[1]
    for hand, shown in _hands(db).values():
        out = {"id": hand["id"], "label": hand["label"]}
        for key in ("group", "hint"):
            if is_text(hand.get(key)):
                out[key] = hand[key]
        names = hand.get("genreLabels")
        names = {w: t for w, t in names.items() if w in genres and is_text(t)} if isinstance(names, dict) else {}
        if names:
            out["genreLabels"] = names
        out["inputs"] = shown
        listed.append(out)
    digest = hashlib.sha256(json.dumps([db.version, listed], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"revision": digest[:12], "presets": listed}


def expand_preset(db, request):
    request = dict(request or {})
    if request.get("preset") is None:
        return request  # `choices` は手札に付くもの。手札がなければ、ここで捨ててよいものではない（契約1）
    preset_id = request.pop("preset")
    choices = request.pop("choices", None) or {}
    hands = _hands(db)
    if preset_id not in hands:
        raise UnknownPresetError(f"preset {preset_id!r} is not in the current list")
    hand, inputs = hands[preset_id]
    request = _expand(hand, inputs, request, choices, preset_id)
    if hand["handling"] == "mixed":
        # 名指したどの手札も、要求をそのまま（要求の条件、続いてミックスの手札のつまみ）受け取り、残りを自分のつまみで
        # 埋める。入力は、ミックスの手札を一覧に出したときに決まっている（why_not_mixing）
        base = {k: v for k, v in request.items() if k != "mix"}
        request["mix"] = [{"hand": name, "request": _expand(hands[name][0], [], dict(base), {}, name)}
                          for name in request["mix"]]
    request.setdefault("difficulty", "auto")
    return request


def _expand(hand, inputs, request, choices, preset_id):
    """要求に何もないところに、手札の入力（`choices` から）とつまみを入れた `request`。手札の `difficulty` のつまみは
    残すので、recommend は要求自身のものと見分けられる（ミックスの手札の項目では、そうしないと要求の "auto" が
    手札自身の狙いを覆ってしまう）。"""
    knobs = HANDLINGS[hand["handling"]]
    for item in inputs:
        if item["kind"] == "seed":
            if request.get("seed") is None and not item.get("optional"):
                raise RequestError(f"preset {preset_id} needs a seed")
        elif item["kind"] == "choice":
            value = choices.get(item["name"])
            if value is None and item.get("optional"):
                continue
            if value not in {o.get("value") for o in item["options"]}:
                raise RequestError(f"preset {preset_id} needs a choice for {item['name']}")
            request.setdefault(knobs[item["name"]][1], value)
    for name, value in hand.get("knobs", {}).items():
        condition = knobs[name][1]
        if any(condition in pair and any(request.get(c) is not None for c in pair) for pair in EITHER_KNOBS):
            continue  # 要求自身の重み（か all）が手札のものの代わりになる
        if condition == "params":
            request["params"] = {**value, **(request.get("params") or {})}
        else:
            request.setdefault(condition, value)
    return request
