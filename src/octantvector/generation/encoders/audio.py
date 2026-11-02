"""音声のビューが取る形。組み込みのビューとプラグイン（octantvector.plugin）で共通。

ClipAudioEncoder  clip_s 秒の箇所を n_clips 個（base.clips）取り、`sr` にリサンプルし、CPU のワーカーで
                  `features` がモデルの入力にする。モデルは箇所ごとにベクトルを 1 つ出し（`clip_vectors`）、
                  `l2_clips` なら箇所ごとに L2 正規化し、曲のベクトルはその平均。`stem = "vocal"` なら、ミックスの
                  代わりに箇所のボーカルのステムを埋め込む（stem.py）。
HiddenMean        箇所のベクトルを、エンコーダーの層 `layers`（1 始まり）の出力の時間平均を層で平均したものにする
                  ClipAudioEncoder。

`models` の最初のものが、model_dir() と読み込みのフックが読むモデル。サブクラスは load_processor、load_model と
（ClipAudioEncoder では）clip_vectors を用意する。重い import はその中に置く。
"""
import numpy as np

from . import stem as stems
from .base import AudioEncoder, clips, resample


class ClipAudioEncoder(AudioEncoder):
    clip_s = 10
    n_clips = 3
    l2_clips = False
    stem = None  # か "vocal"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if cls.stem not in (None, "vocal"):
            raise ValueError(f"{cls.__name__}: stem must be None or 'vocal', got {cls.stem!r}")
        if cls.stem == "vocal" and "stem" in cls.__dict__:  # ステムを有効にしたクラスがその設定を受け取る
            for k, v in stems.VOCAL.items():
                if k not in cls.__dict__:
                    setattr(cls, k, v)

    def own_models(self):
        return tuple(self.models) + ((stems.HTDEMUCS,) if self.stem else ())

    def own_credits(self):
        return tuple(self.credits) + (stems.CREDITS if self.stem else ())

    def params(self):
        p = {"clip_s": self.clip_s, "n_clips": self.n_clips, "l2_clips": self.l2_clips,
             "pool": "mean over clips" + (" of L2-normed clip vectors" if self.l2_clips else "")}
        if self.stem:
            p["stem"] = stems.params()
        return p

    # フック
    def load_processor(self, folder):
        """-> モデルの手元のフォルダから読んだ特徴抽出器。"""
        raise NotImplementedError

    def load_model(self, folder, device):
        """-> モデルの手元のフォルダから読み、`device` に載せ、eval モードにしたモデル。"""
        raise NotImplementedError

    def features(self, clips):
        """self.sr の箇所 -> モデルの入力（pickle できること: ワーカーのプロセスから渡るため）。"""
        return self.proc(clips, sampling_rate=self.sr, return_tensors="np")["input_values"].astype(np.float32)

    def clip_vectors(self, x):
        """n 個の箇所のモデルの入力 -> (n, dim) の float32 の numpy。"""
        raise NotImplementedError

    # 重い処理
    def setup_worker(self):
        if not self.stem:  # ステムがあると、箇所は分離した後にメインのプロセスでしかできない
            self.proc = self.load_processor(self.model_dir())

    def setup_model(self, device):
        self.device = device
        self.model = self.load_model(self.model_dir(), device)
        if self.stem:
            self.proc = self.load_processor(self.model_dir())
            self._sep = stems.Separator(self.model_dir(stems.HTDEMUCS), device)

    def prep(self, y, sr):
        if self.stem:
            return stems.prep(y, sr, self.clip_s, self.n_clips)
        return self.features([resample(c, sr, self.sr) for c in clips(y, sr, self.clip_s, self.n_clips)])

    def _rows(self, v):
        v = np.asarray(v, np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) if self.l2_clips else v

    def embed(self, x):
        if self.stem:
            return self.embed_many([x])[0]
        return self._rows(self.clip_vectors(x)).mean(0)

    def embed_many(self, prepped):
        if self.stem:
            return stems.embed_many(self, prepped)
        return super().embed_many(prepped)

    def vocals(self, mix):
        return self._sep.vocals(mix)

    def stem_vectors(self, voc):
        """SEP_SR の箇所のボーカルのステム -> (clips, dim)。"""
        return self._rows(self.clip_vectors(self.features([resample(v, stems.SEP_SR, self.sr) for v in voc])))


class HiddenMean(ClipAudioEncoder):
    layers = ()  # エンコーダーの層の番号（1 始まり）
    half = False  # CUDA ではモデルを fp16 で動かす

    def params(self):
        return {**super().params(), "layers": list(self.layers),
                "pool": "time-mean per layer, mean over layers and clips"}

    def encoder_layers(self):
        """モデルのエンコーダーの層、順に（隠れ状態 i は層 i の出力）。"""
        return self.model.encoder.layers

    def setup_model(self, device):
        super().setup_model(device)
        if self.half and str(device).startswith("cuda"):
            self.model = self.model.half()
        # 最後に使う層より後は誰も読まないので外す: 出力は同じで、手間が減る
        del self.encoder_layers()[max(self.layers):]
        self._cap = {}
        for i in self.layers:
            self.encoder_layers()[i - 1].register_forward_hook(
                lambda mod, inp, out, i=i: self._cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))

    def clip_vectors(self, x):
        """(clips, samples) の特徴抽出した入力 -> (clips, dim): 層ごとの時間平均を、層で平均する。"""
        import torch
        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            self._cap.clear()
            self.model(torch.from_numpy(x).to(self.device, dtype))
            return torch.stack([self._cap[i].float() for i in self.layers]).mean(2).mean(0).cpu().numpy()
