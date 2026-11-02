"""名前の付いたプール: 計算がどの現行版を対象にするか。プールの名前は呼ぶ側が渡す（api.compute、`octantvector
update --pool`）。較正とリリースはいつも `distribution`。既定は下のとおりで、設定ファイルの [pools.<name>] の表
（契約1: params["pools"]）で変えたり足したりできる。キー:

  min_upvotes  int、既定 0: Upvote の下限
  standard     bool、既定 false: Standard の譜面がある版だけ
  sample       int か null: マップをこの数だけ残す。同じ seed と同じカタログなら毎回同じマップ
               （sha256("<seed>:<map id>") の順。map id は版が変わってもマップについて回る）
  seed         str か int、既定 "octantvector"
  include      パスか null: 1 行に BeatSaver の map id か版の hash を 1 つ書いたテキストファイル（'#' から後は
               コメント）。そのマップは、その版が現行である限り、ほかに加えてメンバーになる

  distribution  すべての現行版（配布 DB が持つもの）
  trial         すべての現行版から抜いた 20,000 マップ（全部を計算せずにビューを試す）
"""
import hashlib
from pathlib import Path

from . import config
from .catalogue import standard_diffs
from .errors import GenerationError

KEYS = {"min_upvotes": 0, "standard": False, "sample": None, "seed": "octantvector", "include": None}
DEFAULTS = {"distribution": {}, "trial": {"sample": 20000}}


def _checked(name, d):
    bad = [k for k, ok in (("min_upvotes", isinstance(d["min_upvotes"], int) and not isinstance(d["min_upvotes"], bool)),
                           ("standard", isinstance(d["standard"], bool)),
                           ("sample", d["sample"] is None or (isinstance(d["sample"], int)
                                                              and not isinstance(d["sample"], bool) and d["sample"] >= 0)),
                           ("seed", isinstance(d["seed"], (str, int)) and not isinstance(d["seed"], bool)),
                           ("include", d["include"] is None or isinstance(d["include"], (str, Path))))
           if not ok]
    if bad:
        raise GenerationError(f"[pools.{name}]: wrong type of {', '.join(bad)} (pools.py)")
    return d


def load():
    """すべてのキーを埋めた {name: 定義}。"""
    given = {name: {} for name in DEFAULTS}
    for name, d in config.section("pools").items():
        if not isinstance(d, dict) or set(d) - set(KEYS):
            raise GenerationError(f"[pools.{name}] may only set {', '.join(KEYS)}")
        given.setdefault(name, {}).update(d)
    return {name: _checked(name, {**KEYS, **DEFAULTS.get(name, {}), **d}) for name, d in given.items()}


def matches(d, m, v):
    """定義のうち版ごとの絞り込み（min_upvotes、standard）。sample と include はプール全体にかかる。"""
    return m.get("stats", {}).get("upvotes", 0) >= d["min_upvotes"] and (not d["standard"] or bool(standard_diffs(v)))


def _included(path):
    try:
        lines = Path(path).expanduser().read_text().splitlines()
    except OSError as ex:
        raise GenerationError(f"pool include file {path}: {ex}") from None
    return {s.lower() for s in (line.split("#", 1)[0].strip() for line in lines) if s}


def members(d, eligible, current):
    """プールの hash。eligible: `matches` を通る現行版の [(map id, hash)]。current: すべての現行版の {map id: hash}。"""
    if d["sample"] is None:
        out = {h for _, h in eligible}
    else:
        rank = sorted(eligible, key=lambda t: (hashlib.sha256(f"{d['seed']}:{t[0]}".encode()).hexdigest(), t[1]))
        out = {h for _, h in rank[:d["sample"]]}
    if d["include"] is not None:
        hashes = {h.lower(): h for h in current.values()}
        keys = {str(k).lower(): h for k, h in current.items()}
        out |= {keys.get(x) or hashes[x] for x in _included(d["include"]) if x in keys or x in hashes}
    return out
