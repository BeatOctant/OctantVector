"""プレイヤーの履歴: 契約3（OctantPath to OctantCompass）の `events` を選り分けずに受け取る（OctantSpecs contracts/、
契約1（OctantCompass to OctantVector））。

各出来事は `type`（upvote / downvote / like / play / user_delete / recommendation / imported_play）、`hash`、
`at` を持つ。プレイはさらに `characteristic`、`difficulty`、`result`（clear / fail / quit / restart）を持ち、
`modifiers` とスコアを持つことがある。
`imported_play`（譜面ごとの `count`。OctantPath を入れたときに、ゲームの PlayerData から一度だけ作られる）だけでは、
好意にも目標にもならない（likes.py）: 回数は途中終了を含み、結果がない（OctantSpecs 法 4.1）。それでも、その譜面は
好意のマップのどの譜面を遊んだかを示すので、好みの基準としてプレイの代わりになれる（recommend.py）。`cleared` の
あるもの（PlayerData にその譜面のスコアがある: 一度はクリアした。修飾は分からない）は、自分のクリアの後ろで得意な
譜面の枠を埋める（skill.py）ので、実力が最初のプレイからある。ほかのどこでもクリアにも失敗にも数えない。
粒度は保つ: プレイは譜面ごと、ほかの signal はマップごと（OctantSpecs constitution.md、原則 2）。

旧い版の出来事は、同じマップの現行版に移す。プレイは characteristic と difficulty が同じ譜面に移す（作り直された
譜面のこともある）。移したものはすべて報告する。DB が知らないマップの出来事は入れずに数える。
"""
from dataclasses import dataclass, field
from datetime import timedelta

from ..common import charts as chartids
from . import likes


@dataclass
class Play:
    hash: str
    chart: str | None  # DB での譜面のキー。現行版にその譜面がなければ None
    result: str        # 送られたままの結果（clear / fail / quit / restart）。数では restart を quit にまとめる
    at: str
    plain: bool        # 速度 1.0 で修飾なし: 実力に使える


@dataclass
class History:
    counts: dict = field(default_factory=dict)    # hash -> likes.py の形でまとめた数
    plays: list = field(default_factory=list)     # Play。古い順
    replaced: list = field(default_factory=list)  # 現行版に移した旧い版ごとの {"from", "to"}
    imported: dict = field(default_factory=dict)  # hash -> その imported_play の出来事が指す、DB での譜面のキー
    imported_clears: dict = field(default_factory=dict)  # `cleared` のある imported_play の出来事の、DB での譜面のキー -> PlayerData のプレイ回数
    unknown: int = 0                              # DB にないマップの出来事

    def liked(self):
        return {h for h, s in self.counts.items() if likes.classify(s) == "liked"}

    def known(self):
        """プレイヤーがもう関わったマップ（遊んだ、お気に入りにした、投票した、削除した）。"""
        return {h for h, s in self.counts.items() if any(s.values())}

    def goals(self):
        """目標（まだクリアしていないマップのお気に入りか Upvote、法 4.1）。ただし、プレイヤーが削除したマップは、ほかの
        既知のマップと同じく外したまま。推測: 法 4.1 は削除したマップを出にくくするとだけ書き、どれだけかは書いていない。"""
        return {h for h, s in self.counts.items() if likes.classify(s) == "goal" and not s["deleted"]}

    def revisit(self, days, today):
        """「久しぶり」の候補: 好意のマップ（法 4.1）のうち、最後のプレイ（途中終了も数える）の日が `today`（UTC の日付）の
        `days` 日前か、それより前のもの。プレイの記録がない好意のマップ（OctantPath を入れる前のプレイだけ）は前に遊んだ
        ものとして入れる。削除したマップは入れない。"""
        last = {}
        for pl in self.plays:
            last[pl.hash] = max(last.get(pl.hash, ""), pl.at[:10])
        cutoff = (today - timedelta(days=days)).isoformat()
        return {h for h in self.liked() if not self.counts[h]["deleted"] and last.get(h, "") <= cutoff}


def _plain(e):
    """速度 1.0 で、ほかの修飾なし（`modifiers`: {"speed": 1.0, "noFail": false}、契約3）。"""
    mods = e.get("modifiers") or {}
    return mods.get("speed") in (None, 1, 1.0) and not any(v for k, v in mods.items() if k != "speed")


def _current(db, h, replaced, seen):
    cur = db.current(h)
    if cur is not None and cur != h and h not in seen:
        seen.add(h)
        replaced.append({"from": h, "to": cur})
    return cur


def parse(db, events):
    hist, seen = History(), set()
    votes, first_clear = {}, {}
    for e in sorted(events or (), key=lambda e: str(e.get("at", ""))):
        t, h = e.get("type"), e.get("hash")
        if not h or t not in ("upvote", "downvote", "like", "like_removed", "play", "user_delete", "recommendation", "imported_play"):
            continue
        if t == "recommendation":  # 好みの signal には使わない（法 4.1）
            continue
        cur = _current(db, h, hist.replaced, seen)
        if cur is None:
            hist.unknown += 1
            continue
        s = hist.counts.setdefault(cur, {"favorite": 0, "upvote": 0, "play": 0, "finished": 0, "failed": 0,
                                         "quit": 0, "deleted": 0, "pd_playcount": 0, "returned": 0})
        if t in ("upvote", "downvote"):
            votes[cur] = t  # 最後の投票が勝つ（votedSongs.json も上書きする）
        elif t == "like":
            s["favorite"] = 1
        elif t == "like_removed":  # お気に入りでなくなっただけで、嫌いではない。最後の like/like_removed が勝つ
            s["favorite"] = 0
        elif t == "imported_play":
            s["pd_playcount"] += int(e.get("count") or 0)
            key = chartids.key(cur, e.get("characteristic"), e.get("difficulty"))
            if key in db.chart_pos:
                hist.imported.setdefault(cur, set()).add(key)
                if e.get("cleared") is True:
                    hist.imported_clears[key] = hist.imported_clears.get(key, 0) + int(e.get("count") or 0)
        elif t == "user_delete":
            s["deleted"] += 1
        else:
            res = e.get("result")
            s["play"] += 1
            s[{"clear": "finished", "fail": "failed"}.get(res, "quit")] += 1
            day = str(e.get("at", ""))[:10]  # `at` は UTC（契約3）
            if res in ("clear", "fail") and cur in first_clear and day > first_clear[cur]:
                s["returned"] = 1  # 最初のクリアより後の日にまた遊んだ（法 4.1）
            if res == "clear":
                first_clear.setdefault(cur, day)
            key = chartids.key(cur, e.get("characteristic"), e.get("difficulty"))
            hist.plays.append(Play(cur, key if key in db.chart_pos else None, res, str(e.get("at", "")), _plain(e)))
    for h, v in votes.items():
        hist.counts[h]["upvote"] = int(v == "upvote")
    return hist
