"""スイングのシミュレーションのビュー（方法は OctantSpecs octantvector/swing.md。シミュレーションは
octantvector.generation.swing）。"""
import numpy as np

from ...common import charts
from .. import config
from ..errors import GenerationError
from ..swing import NAMES, chart_features, params, standard_charts, style
from .base import DerivedEncoder, Encoder, register


@register
class SwingFlow(Encoder):
    """すべての Standard の譜面の、読み取れるスイングのシミュレーションの特徴。シミュレーションは 2 本のセーバーと
    Standard の遊び方を前提にしているので、ほかのプレイスタイルは、そのためのシミュレーションを確かめるまで
    ベクトルを持たない。ノーツが `max_notes` を超える譜面も持たない: 50 万ノーツの譜面のシミュレーションは終わらない。"""
    kind = "zip"
    id = "swing_flow"
    version = "12"
    max_notes = 50000
    rows = "charts"
    play_styles = ("Standard",)
    inputs = ("chart",)
    credits = ("BeatLeader analyzer, https://github.com/BeatLeader/beatleader-analyzer, formulas ported into "
               "OctantVector (THIRD_PARTY_NOTICES.md), MIT",
               "JoshaParity (Joshabi), https://github.com/Joshabi/JoshaParity, formulas ported into OctantVector "
               "(THIRD_PARTY_NOTICES.md), MIT")

    def __init__(self):
        self.names, self.P = NAMES, params.load()
        self.dim = len(NAMES)

    def params(self):
        return {"charts": "every Standard chart", "max_notes": self.max_notes, "features": list(self.names),
                "swing": self.P}

    def embed_zip(self, mz):
        """-> ノーツが足りて max_notes 以下のすべての Standard の譜面の {譜面のキー: 特徴}。"""
        out = {}
        for d, ch in standard_charts(mz):
            if len(ch.notes) > self.max_notes:
                continue
            v = chart_features(ch, self.P)[0]
            if v is not None:
                out[charts.key(mz.hash, "Standard", d)] = v
        return out


class _SwingTags(DerivedEncoder):
    """`swing_flow` から派生するタグのビューで共通。`group` で細かいタグを選ぶ（style.load_rules）。"""
    inputs = ("view:swing_flow", "file:swing_model")
    group = None
    learned = False

    def __init__(self):
        self.path = config.load().swing_model
        self.rules = style.load_rules(self.group)
        self.names = ([f"p:{t}" for t in style.LEARNED_TAGS] if self.learned else []) + list(self.rules)
        self.dim = len(self.names)

    def params(self):
        return {"source": self.source, "names": self.names, "rules": self.rules,
                "model": style.model_hash(self.path) if self.path and self.path.exists() else None}

    def derive(self, store_root, device="cpu"):
        # ストアを開く前に確かめる: モデルがないと、params() はそこに "model": None を記録してしまう
        if not self.path or not self.path.exists():
            raise GenerationError(f"{self.id} needs a style model: run `octantvector swing-train` and set swing_model "
                                  "(https://github.com/BeatOctant/OctantSpecs/blob/main/octantvector/config.md)")
        self.model = style.load_model(self.path)
        return super().derive(store_root, device)

    def values(self, X, device="cpu"):  # 行ごと: 分位点と重みはモデルから取り、X からは取らない
        S, _ = style.fine_tags(self.model, X, self.rules)
        if self.learned:
            S = np.c_[np.nan_to_num(style.learned_probs(self.model, X), nan=0.0), S]
        return S


@register
class SwingStyle(_SwingTags):
    """`swing_flow` からのプレイスタイルのタグ: BeatSaver のスタイルタグごとの確率（学習）と、それに続いて、壁や爆弾に
    関わらない細かいタグごとの点（0..1）。`octantvector swing-train` のモデルのファイルが要る（設定 `swing_model`）。"""
    id = "swing_style"
    version = "13"
    group = "style"
    learned = True


@register
class SwingHazards(_SwingTags):
    """`swing_flow` からの壁と爆弾のタグ（アートの壁 / ワークアウトの壁 / リズムの壁、アートの爆弾 / 避ける爆弾 /
    リズムの爆弾 など）: それぞれの点（0..1）。推薦で別々に重みを付けられるよう、`swing_style` とは分けている。"""
    id = "swing_hazards"
    version = "10"
    group = "hazards"
