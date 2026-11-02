"""ボーカルのステム。箇所の音声のエンコーダー（`stem = "vocal"` の ClipAudioEncoder）の前段: 箇所を Demucs（htdemucs）で
分離し、ボーカルのステムだけを埋め込む。役割は `vocal` で、手札がミックス全体とは別に、歌が似ているマップを頼める。
箇所の声が少なすぎる曲（インスト）はベクトルを持たないので、その曲にはボーカルのビューがないだけになる。

771 マップで測った: ミックスの CLAP に足すと、どのボーカルのビューでも「近い 10 件が同じアーティスト」が 10.6 % から
12.2-13.2 % に上がり、ボーカルのビューが見つける同じアーティストのマップの 40-56 % はミックスの CLAP の上位 10 件に
なかった。7 つの窓から声の多い 3 つを取っても、決まった 3 か所（13.2 と 12.9 %）とほとんど変わらず、決まった 3 か所なら
分離は 3 か所で済む。箇所は音楽のビューと同じ場所だが長さは 7.8 秒（測ったときは 10 秒）: htdemucs は一度に 7.8 秒を
処理するので、10 秒の箇所は重なる 2 回の forward になった。ワーカーは曲をモノラルでデコードするので、分離器には
モノラルのミックスが両方のチャンネルに入る（測ったときはステレオのファイル。分離の質が少し下がると推測）。

費用: 分離が GPU の時間の大半で、アーカイブを読むのがもう 1 つの大きな部分。ボーカルのビューはどちらも分け合う:
1 回の重い処理が、挙げたすべてのビューのために各曲を一度だけ読み、マップのまとまりのステムは一度だけ分離して、
同じ処理のほかのボーカルのビューで使い回す（下の分離のキャッシュ）。重い処理は `batch` マップをまとめて渡す
（embed_many）。その箇所は `sep_clips` ずつ分離器を通り、埋め込みのモデルはまとめて通る。Demucs の無作為な時間の
ずらしは切ってある（shifts=0）ので、同じ曲はいつも同じベクトルになる。

分離器の重みはほかと同じく宣言したモデル（HTDEMUCS）で、models_dir からだけ読む。"""
import hashlib
from collections import OrderedDict

import numpy as np

from ..errors import GenerationError
from .base import Model, clips, resample

SEP_SR = 44100
HTDEMUCS = Model("adefossez/HTDemucs", "cbc8a9b1a87023b7fd74e7b3412e6321c0eab003",
                 ("htdemucs.yaml", "955717e8.safetensors"))
CREDITS = ("Demucs (Alexandre Défossez), https://github.com/adefossez/demucs, code under MIT; "
           "HTDemucs weights, https://huggingface.co/adefossez/HTDemucs, no license stated (the author: "
           "\"Weights are provided for research purpose!\", https://github.com/facebookresearch/demucs/issues/508); "
           "used only to compute the vectors, not redistributed",)
# ボーカルのステムを有効にしたクラスが、自分で決めなければ受け取る値（ClipAudioEncoder.__init_subclass__）
VOCAL = {"role": "vocal",
         "batch": 8,  # embed_many 1 回のマップ数
         "clip_s": 7.8}  # htdemucs の学習の区切り: 箇所ごとに forward 1 回で、重なる塊はない
SEP_CLIPS = 12  # 分離器 1 回の箇所数（GPU のメモリ）
SEP_HALF = True  # CUDA では分離器を fp16 の autocast で動かす
MIN_VOCAL_SHARE = 0.3  # 仮の値（インストのタグの付いた 18 マップのうち 3 が上に残り、771 マップのうち 608 が上に残る）


def params():
    """ステムがステムのエンコーダーの定義に足すもの。"""
    return {"separator": "demucs htdemucs", "stem": "vocals", "shifts": 0,
            "sep_precision": "fp16 autocast on cuda" if SEP_HALF else "fp32",
            "input": "mono decode on both channels", "min_vocal_share": MIN_VOCAL_SHARE,
            "vocal_share": "mean over clips of 100 ms frames with vocal energy >= 10% of the mix's"}


def vocal_share(voc, mix, sr, frame=0.1, ratio=0.1):
    """箇所ごと: `frame` 秒の区切りのうち、ボーカルのエネルギーがミックスの `ratio` 以上あるものの割合。"""
    n = int(frame * sr)
    k = voc.shape[-1] // n
    ev = (voc[..., :k * n].reshape(len(voc), k, n) ** 2).mean(-1)
    em = (mix[..., :k * n].reshape(len(mix), k, n) ** 2).mean(-1) + 1e-10
    return ((ev / em) >= ratio).mean(-1)


def prep(y, sr, clip_s, n_clips):
    """SEP_SR での曲の箇所。どれもちょうど clip_s の長さ（リサンプルで 1 サンプルずれることがある）。"""
    n = int(clip_s * SEP_SR)
    cl = [resample(c, sr, SEP_SR)[:n] for c in clips(y, sr, clip_s, n_clips)]
    return np.stack([np.pad(c, (0, n - len(c))) for c in cl]).astype(np.float32)


def load_htdemucs(folder):
    """手元のファイルからの htdemucs のモデルの束（demucs.hf.get_hf_model と同じ作り方で、ハブは使わない）。"""
    import yaml
    from demucs.apply import BagOfModels
    from demucs.hf import load_safetensors_model
    bag = yaml.safe_load((folder / "htdemucs.yaml").read_text())
    missing = [s for s in bag["models"] if f"{s}.safetensors" not in HTDEMUCS.files]
    if missing:
        raise GenerationError(f"{HTDEMUCS.key}: htdemucs.yaml names models not declared: {', '.join(missing)}")
    models = [load_safetensors_model(folder / f"{s}.safetensors") for s in bag["models"]]
    return BagOfModels(models, bag.get("weights"), bag.get("segment"))


class Separator:
    """プロセスの中でデバイスごとに 1 つの Demucs のモデル。直近のいくつかのまとまりのステムを（精度ごとに）持つので、
    1 回の処理のボーカルのビューは各まとまりを一度だけ分離する。"""
    _models, _caches = {}, {}

    def __init__(self, folder, device, keep=4):
        import torch
        from demucs.apply import apply_model
        self.torch, self.device = torch, device
        key = str(device)
        if key not in Separator._models:
            Separator._models[key] = load_htdemucs(folder).to(device).eval()
        self.model, self._apply = Separator._models[key], apply_model
        self._stem = list(self.model.sources).index("vocals")
        self.segment = int(self.model.models[0].segment * SEP_SR) if hasattr(self.model, "models") else \
            int(self.model.segment * SEP_SR)
        self._cache, self._keep = Separator._caches.setdefault((key, SEP_HALF), OrderedDict()), keep

    def vocals(self, mix):
        """SEP_SR の (clips, n) のモノラル -> (clips, n) のモノラルのボーカルのステム。"""
        key = hashlib.blake2b(mix.tobytes(), digest_size=16).digest() + str(mix.shape).encode()
        if key in self._cache:
            return self._cache[key]
        torch = self.torch
        out = []
        cuda = str(self.device).startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=cuda and SEP_HALF):
            for s in range(0, len(mix), SEP_CLIPS):
                x = torch.from_numpy(mix[s:s + SEP_CLIPS]).to(self.device)[:, None].expand(-1, 2, -1)
                mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-8  # demucs.separate と同じ正規化
                split = x.shape[-1] > self.segment  # 1 つの区切りの箇所は 1 回の forward で通す
                y = self._apply(self.model, (x - mu) / sd, shifts=0, split=split, overlap=0.25, device=self.device)
                out.append((y[:, self._stem].float() * sd + mu).mean(1).cpu().numpy())
        voc = np.concatenate(out)
        self._cache[key] = voc
        while len(self._cache) > self._keep:
            self._cache.popitem(last=False)
        return voc


def embed_many(enc, prepped):
    """曲のまとまりの箇所を一度だけ分離し、声のある曲のステムを `enc` で埋め込む（enc.stem_vectors: (SEP_SR の箇所) ->
    (clips, dim)）。曲のベクトルはその箇所の平均で、声のない曲は None。"""
    mix = np.concatenate(prepped)
    voc = enc.vocals(mix)
    bounds = np.cumsum([0] + [len(p) for p in prepped])
    voiced = [float(vocal_share(voc[a:b], mix[a:b], SEP_SR).mean()) >= MIN_VOCAL_SHARE
              for a, b in zip(bounds[:-1], bounds[1:])]
    keep = [j for c, a, b in zip(voiced, bounds[:-1], bounds[1:]) if c for j in range(a, b)]  # 声のある箇所
    if not keep:
        return [None] * len(prepped)
    vecs = dict(zip(keep, enc.stem_vectors(voc[keep])))
    return [np.mean([vecs[i] for i in range(a, b)], 0) if c else None
            for c, a, b in zip(voiced, bounds[:-1], bounds[1:])]
