"""プレイヤー自身のデータで、どのマップを好意的（正例）とみなすか（OctantSpecs 法 4.1）。

決まり（`classify`）:
  liked        一度でもクリアし、お気に入りか Upvote（どちらか一方でよい）があるか、最初のクリアより
               後の日にまた遊んだ
  goal         まだクリアしていないマップのお気に入りか Upvote: 好みには入れず、挑戦用に取っておく
  challenging  クリアしないまま MIN_CHALLENGE_PLAYS 回以上失敗した: 難しさの手がかりにだけ使う
  None         それ以外（1 回きりのクリア、途中終了、失敗、未プレイ）: ラベルなし。負例にはしない
数えるプレイはクリアか失敗で終わったもの。途中終了（曲の途中でやめる）とやり直しは数えない。
`imported_play`（OctantPath を入れる前のゲームの PlayerData のプレイ回数。結果も日付もない）だけでは、
好意にも目標にもならない。ただし、そういうマップにお気に入りか Upvote があれば好意とみなす。クリアしたかは
分からないが、そういう印の多くはクリアしたマップに付いていた（1 人の記録で 165 のうち 142）。「日」は
`at`（UTC）の日付。
"""
MIN_CHALLENGE_PLAYS = 2  # OctantSpecs 法 4.1 の「繰り返し」


def classify(s: dict):
    """s: history.py が持つ 1 マップ分の数（`returned`: 最初のクリアより後の日のクリアか失敗）。
    -> "liked" | "goal" | "challenging" | None（OctantSpecs 法 4.1）。"""
    marked = bool(s.get("favorite") or s.get("upvote"))
    if s.get("finished", 0):
        return "liked" if marked or s.get("returned") else None
    if s.get("play", 0) == 0 and s.get("pd_playcount", 0):  # OctantPath の前にクリアしたかは分からない
        return "liked" if marked else None
    if marked:
        return "goal"
    if s.get("failed", 0) >= MIN_CHALLENGE_PLAYS:
        return "challenging"
    return None

