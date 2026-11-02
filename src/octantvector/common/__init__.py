"""生成側と推薦側が共有するもの: 譜面の識別とプレイスタイル（charts.py）と、配布 DB のファイル（契約2（DB build
to recommendation core）、dbformat.py）。両側が同じ関数で全量を書くので、「全量」と「前の全量 + 差分」は
同じ中身になる。

このパッケージは octantvector のほかの部分を import せず、設定を読まず、生成の入力もユーザーのデータも使わない。
"""
