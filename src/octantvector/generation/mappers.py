"""関係のビュー beatsaver_mappers@1、役割 `mapper`（OctantSpecs contracts/、契約2（DB build to recommendation
core）ビューの種類）: 各マップの版を作った BeatSaver のアカウント。カタログ（契約4（OctantHarvest crawl to
OctantVector））から作る。メンバーは、アップロードした人（`uploader.id`）とカタログにある共同作者
（`collaborators[].id`）の "beatsaver:<account id>"。マップのどの版も、そのマップのメンバーを持つ。対は持たない:
2 つのマップが近い（1）のは、アカウントを共有するときだけ。使うのはアカウント ID だけで、名前は使わない
（OctantSpecs 法 3.2）。

リリースのたびにカタログから作るので、そのリリースのマップに沿う。DB の作り手がビューに挙げれば（NAME）
DB に入る。DB 自体の BeatSaver の表記のほかにクレジットは要らない。
"""
NAME = "beatsaver_mappers"
KEY = f"{NAME}@1"


def _accounts(m):
    people = [m.get("uploader")] + list(m.get("collaborators") or ())
    return tuple(sorted({f"beatsaver:{p['id']}" for p in people if isinstance(p, dict) and p.get("id") is not None}))


def relation(recs, hashes):
    """マップ `hashes` の (ビューのキー, common/dbformat.py の形の関係の中身)。recs: カタログの {hash: (map, version)}。"""
    members = {h: ids for h in hashes if h in recs and (ids := _accounts(recs[h][0]))}
    meta = {"role": "mapper", "rows": "maps", "source": "beatsaver:uploader+collaborators",
            "fingerprint": "beatsaver:uploader+collaborators/1", "sources": []}
    return KEY, {"meta": meta, "members": members, "pairs": []}
