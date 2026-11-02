"""手札が何を言ってよいか。推薦側なしで確かめる（契約1）ので、DB を作る側と推薦側が同じ手札を受け入れる。
`views` は {ビューのキー: 役割}、`genres` は DB のジャンルの語。"""
import numbers

from .params import AIMS, DEFAULTS, why_invalid


def is_text(value):
    """表示する文字: 空でない文字列か、言語が 1 つ以上ある {言語: 空でない文字列}。そのまま渡す。"""
    if isinstance(value, str):
        return bool(value)
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items()))


def view_keys(name, views):
    """重みの名前が指すビューのキー: ビューそのもの（id@version）か、DB にあるビューの id のすべての版（版のない id。
    1 つの埋め込みを名指す手札が、その新しい版でも生き残るように）。役割はラベルで、手札が使える名前ではない: DB が
    どのビューを持つかは作り手が決めるので、手札はどのビューのことかを自分で言う。
    `views` は DB のビューのキー。どれもなければ []。"""
    if name in views:
        return [name]
    return [k for k in views if k.split("@")[0] == name]


def _views(value, views, genres):
    if not isinstance(value, dict) or not value:
        return "needs views with weights"
    for name, w in value.items():
        if not isinstance(w, numbers.Real) or isinstance(w, bool) or w <= 0:
            return f"the weight of {name!r} is not a positive number"
        if not view_keys(name, views):
            return f"the DB has no view {name!r}"
    return None


def _view_groups(value, views, genres):
    if not isinstance(value, list) or len(value) < 2:
        return "all is not a list of two or more views with weights"
    for group in value:
        why = _views(group, views, genres)
        if why:
            return why
    return None


def _difficulty(value, views, genres):
    if value in AIMS or value == "auto":
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and 0 <= value <= 1:
        return None
    return f"difficulty {value!r} is not an aim or a clear chance"


def _genre(value, views, genres):
    return None if value in genres else f"the DB has no genre word {value!r}"


def _genre_from(value, views, genres):
    if not isinstance(value, list) or not value:
        return "genreFrom is not a list of genre words"
    missing = [w for w in value if w not in genres]
    return f"the DB has no genre words {missing!r}" if missing else None


def _params(value, views, genres):
    if not isinstance(value, dict):
        return "params is not an object"
    unknown = sorted(set(value) - set(DEFAULTS))
    if unknown:
        return f"unknown params {unknown}"
    # DB は手札を運ぶので、その値も要求と同じく確かめる（暴走する値が DB から来てはならない）
    return why_invalid(value)


def _hand_ids(value, views, genres):
    """ミックスの手札が引く手札: 2 つ以上の異なる id。それぞれが一覧にあって出せるかは、ここではなく一覧全体で
    確かめる（why_not_mixing）。"""
    if not isinstance(value, list) or len(value) < 2:
        return "hands is not a list of two or more hand ids"
    if not all(isinstance(h, str) and h for h in value):
        return "hands has an entry that is not a hand id"
    if len(set(value)) != len(value):
        return "hands names a hand twice"
    return None


# この OctantVector が知る扱い方: {扱い方: {つまみ: (確かめる関数, それがなる契約1 の条件)}}。
#   near              起点（なければ履歴）のまわりを重み付きのビューで並べる。起点がなければ、`genreFrom` は一覧の
#                     語のうち好意のマップが一番近い語を選び、その語のマップを残す。
#                     `weights` の代わりの `all`: 重み付きの 2 つ以上のビューで、マップはどれでも近いこと（recommend
#                     は一番低い百分位で並べる:「これでもあれでも近い」）
#   narrow_then_rank  `narrow` で一番近いマップを残し、`rank` で並べる（recommend の `narrow` の条件）
#   mixed             `hands` で名指した手札（同じ一覧のほかの手札の id。ミックスの手札は不可）から 1 つの一覧を
#                     引く。各手札の枠の割合は、それまでの下見へのプレイヤーの反応で決まる（recommend の `mix` の
#                     条件）。要求の起点、難易度、params はどの手札にも届き、要求が空けたところを手札のつまみが埋める
HANDLINGS = {
    "near": {"weights": (_views, "weights"), "all": (_view_groups, "all"), "difficulty": (_difficulty, "difficulty"), "genre": (_genre, "genre"),
             "genreFrom": (_genre_from, "genreFrom"), "params": (_params, "params")},
    "narrow_then_rank": {"narrow": (_views, "narrow"), "rank": (_views, "weights"),
                         "difficulty": (_difficulty, "difficulty"), "params": (_params, "params")},
    "mixed": {"hands": (_hand_ids, "mix"), "difficulty": (_difficulty, "difficulty"), "params": (_params, "params")},
}
REQUIRED_KNOBS = {"narrow_then_rank": ("narrow", "rank"), "mixed": ("hands",)}
EITHER_KNOBS = (("weights", "all"),)  # 手札が多くても 1 つしか持たないつまみ（条件の枠も 1 つ: presets._expand）
CHOICE_KNOBS = ("genre", "difficulty")  # 選択の入力が埋められるつまみ（どれも値は 1 つ）

def inputs(hand, views, genres):
    """(一覧に出す入力。選択肢は埋めて絞る。または None) と、None のときの理由。"""
    knobs = HANDLINGS[hand["handling"]]
    out = []
    for item in hand.get("inputs", []):
        if not isinstance(item, dict):
            return None, "an input is not an object"
        item = dict(item)
        kind = item.get("kind")
        if "label" in item and not is_text(item["label"]):
            return None, "an input's label is not a text"
        if kind == "choice":
            name = item.get("name")
            if name not in knobs or name not in CHOICE_KNOBS:
                return None, f"choice {name!r} fills no knob of {hand['handling']}"
            if "options" in item:
                if not isinstance(item["options"], list):
                    return None, f"the options of choice {name!r} are not a list"
                item["options"] = [o for o in item["options"]
                                   if isinstance(o, dict) and knobs[name][0](o.get("value"), views, genres) is None
                                   and ("label" not in o or is_text(o["label"]))]
            elif name == "genre":
                # DB はまだジャンルの語の表示名を宣言しない。語そのものを出す。
                item["options"] = [{"value": g, "label": g} for g in sorted(genres)]
            if not item.get("options"):
                if item.get("optional"):
                    continue
                return None, f"choice {name!r} has no option this DB can serve"
        elif kind != "seed":
            if item.get("optional"):
                continue  # この OctantVector が埋められない入力: 省ける枠からは外す
            return None, f"input kind {kind!r} is unknown"
        out.append(item)
    return out, None


def why_not(hand, views, genres):
    """手札をこの DB で出せるなら None（`views`: {ビューのキー: 役割}、`genres`: 語）。出せなければその理由。"""
    if not isinstance(hand, dict):
        return "not an object"
    if not isinstance(hand.get("id"), str) or not hand["id"]:
        return "no id"
    if not is_text(hand.get("label")):
        return "no label"
    handling = hand.get("handling")
    if handling not in HANDLINGS:
        return f"handling {handling!r} is unknown to this OctantVector"
    knobs = hand.get("knobs", {})
    if not isinstance(knobs, dict):
        return "knobs is not an object"
    for name in REQUIRED_KNOBS.get(handling, ()):
        if name not in knobs:
            return f"{handling} needs the knob {name!r}"
    for pair in EITHER_KNOBS:
        if all(name in knobs for name in pair):
            return f"knobs {' and '.join(pair)} are one or the other"
    for name, value in knobs.items():
        if name not in HANDLINGS[handling]:
            return f"knob {name!r} is unknown to {handling}"
        why = HANDLINGS[handling][name][0](value, views, genres)
        if why:
            return why
    if not isinstance(hand.get("inputs", []), list):
        return "inputs is not a list"
    return inputs(hand, views, genres)[1]


def needs_seed(hand, views, genres):
    """起点なしでは頼めない手札（省けない起点の入力がある）なら True。"""
    listed = inputs(hand, views, genres)[0] or []
    return any(item["kind"] == "seed" and not item.get("optional") for item in listed)


def why_not_mixing(hand, served, views, genres):
    """ミックスの手札が名指すどの手札からも引けるなら None。引けなければその理由。`served`: 同じ一覧のうち why_not を
    通った手札の {id: 手札}。名指す手札はそこにあり、それ自体がミックスでないこと（1 段だけ: 反応は手札ごとに記録
    するので、ミックスのミックスには学べる反応がない）。ミックスの手札が起点なしで頼めるなら、名指す手札もどれも
    そうであること（でないとその手札を頼めない）。"""
    if hand.get("handling") != "mixed":
        return None
    for name in hand["knobs"]["hands"]:
        other = served.get(name)
        if other is None or name == hand["id"]:
            return f"mixed names the hand {name!r}, which is not in the list"
        if other.get("handling") == "mixed":
            return f"mixed names the hand {name!r}, which is mixed itself"
        if not needs_seed(hand, views, genres) and needs_seed(other, views, genres):
            return f"mixed may be asked without a seed but the hand {name!r} needs one"
        if any(item["kind"] == "choice" and not item.get("optional") for item in inputs(other, views, genres)[0] or []):
            return f"the hand {name!r} needs a choice, which a mixed hand does not gather"
    return None


def check_hands(hands, views, genres):
    """外れる手札（と重なる id）ごとの {添字か id: 理由}。"""
    out, seen, served = {}, set(), {}
    for n, hand in enumerate(hands):
        why = why_not(hand, views, genres)
        key = hand.get("id") if isinstance(hand, dict) and isinstance(hand.get("id"), str) else n
        if why is None and key in seen:
            why = "the id is used by an earlier hand"
        if why:
            out[key] = why
        else:
            served[key] = hand
        seen.add(key)
    for hand in hands:  # ミックスの手札は、一覧のどこにあっても通った手札から引く
        if isinstance(hand, dict) and hand.get("handling") == "mixed" and hand.get("id") not in out:
            why = why_not_mixing(hand, served, views, genres)
            if why:
                out[hand["id"]] = why
    return out
