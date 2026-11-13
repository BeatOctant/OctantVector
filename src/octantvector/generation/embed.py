"""`octantvector update`: カタログとアーカイブを台帳に取り込み、プールを見直し、台帳が足りないと言うものだけを計算する
（ledger.py）。直後にもう一度実行しても何も計算しない。

順番: マップのエンコーダー（カタログだけ）、続いて音声、画像、zip のエンコーダーのために zip を一通り、最後に派生の
エンコーダー。zip はジャーナルの commit_seq の順（ディスク上で順に並ぶ）で読む。各曲は一度だけデコードしてすべての
音声のエンコーダーで分け合い、カバー画像もすべての画像のエンコーダーのために一度だけ読む。ビューと、それを計算する
プール（pools.py）は呼ぶ側が名指す。
"""
import hashlib
import multiprocessing as mp
import os
import random
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import closing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from . import archive, catalogue, config, pools
from ..common.charts import key_hash
from .encoders import get, registry
from .errors import GenerationError
from .ledger import Ledger, meta_digest, meta_fields
from .search import resolve_device
from .store import Store

_WORKER = {}
MAIN_KINDS = ("audio", "image")  # ワーカーで用意し、メインのプロセス（GPU）で埋め込む


def _init(ids, archive_root, given=None):
    if given is not None:  # 契約1（OctantHarvest to OctantVector）の呼び出し: ワーカーはファイルではなく呼ぶ側の設定を見る
        config._GIVEN.set(given)
    try:  # zip だけのエンコーダー（chart_patterns、swing_flow）は torch なしで動く
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    _WORKER["archive"] = Path(archive_root)
    _WORKER["encs"] = {}
    for i in ids:
        e = get(i)
        if hasattr(e, "setup_worker"):
            e.setup_worker()
        _WORKER["encs"][i] = e


def _work(task):
    """-> (hash, エンコーダーの id, zip の sha256, {encoder id: 出力 | None | Exception}, マップ単位のエラー)"""
    h, ids = task
    out, sha = {}, None
    try:
        raw = archive.zip_path(_WORKER["archive"], h).read_bytes()  # 一度だけ読む: hash を取り、メモリから開く
        sha = hashlib.sha256(raw).hexdigest()
        with archive.MapZip(_WORKER["archive"], h, raw) as mz:
            for i in ids:
                e = _WORKER["encs"][i]
                if e.kind == "zip":
                    try:
                        out[i] = e.embed_zip(mz)
                    except Exception as ex:  # 壊れた譜面は、そのエンコーダーだけを失敗させる
                        out[i] = ex
            audio = [i for i in ids if _WORKER["encs"][i].kind == "audio"]
            try:
                data = mz.song_bytes() if audio else None
            except Exception as ex:  # 曲のファイルのないマップは、音声のエンコーダーだけを失敗させる
                data = ex
            image = [i for i in ids if _WORKER["encs"][i].kind == "image"]
            try:
                cover = mz.cover_bytes() if image else None
            except Exception as ex:  # zip から読めないカバーは、画像のエンコーダーだけを失敗させる
                cover = ex
        for i in image:  # カバーのないマップは skip。読めないかデコードできないものは、そのエンコーダーを失敗させる
            if isinstance(cover, Exception):
                out[i] = cover
                continue
            try:
                out[i] = None if cover is None else _WORKER["encs"][i].prep(cover)
            except Exception as ex:
                out[i] = ex
        if audio:
            try:
                if isinstance(data, Exception):
                    raise data
                y, sr = archive.decode_audio(data)
            except Exception as ex:  # デコードできない曲は、音声のエンコーダーだけを失敗させる
                out.update((i, ex) for i in audio)
            else:
                for i in audio:
                    try:
                        out[i] = _WORKER["encs"][i].prep(y, sr)
                    except Exception as ex:  # 長い曲で特徴抽出がメモリに入らないなど: その音声のエンコーダーだけを失敗させる
                        out[i] = ex
        return h, ids, sha, out, None
    except Exception as ex:
        return h, ids, sha, out, repr(ex)


def run(cfg, encoder_ids, pool="distribution", n=None, seed=2, retry_errors=False, log=sys.stderr):
    """プール `pool` で `encoder_ids`（と派生のものの元）を計算する。n: プールの無作為な部分（テストと手早い実験）。
    -> {encoder id: 記録せず後の実行に残したマップ}（残したエンコーダーだけ。GPU に空きがなかった、PC のいまの状態の
    せいで曲を読めなかった: _consume）。"""
    arch, cat = cfg.require("archive"), cfg.require("catalogue")
    if not encoder_ids:
        raise GenerationError("name the views to compute")
    ids = list(dict.fromkeys(_with_sources(encoder_ids)))
    encs = {i: get(i) for i in ids}
    defs = pools.load()
    if pool not in defs:
        raise GenerationError(f"unknown pool {pool!r}; known: {', '.join(sorted(defs))}")
    led = Ledger(Path(cfg.store) / "ledger.sqlite")
    try:
        all_map = {i: c() for i, c in registry().items() if c.kind == "map"}  # どのマップのエンコーダーの要約も今のものに保つ
        counts = led.ingest(catalogue.read_jsonl(cat), archive.commit_order(arch), all_map, defs)
        print("ingest: " + ", ".join(f"{k} {v}" for k, v in counts.items()), file=log)

        pending = {}
        stores = {i: Store(e, cfg.store, on_flush=lambda hs, i=i: _flushed(led, encs[i], stores[i], pending, hs))
                  for i, e in encs.items()}
        members = led.pool(pool)
        if n is not None:
            members = random.Random(seed).sample(members, min(n, len(members)))
            members.sort(key=lambda r: (r[1] is None, r[1] or 0, r[0]))
        plan = {}
        for i, e in encs.items():
            adopted = led.adopt(e, stores[i].fingerprint, stores[i].done())
            if adopted:
                print(f"{i}: {adopted} vectors already in the store recorded in the ledger", file=log)
            if e.kind == "derived":
                continue
            plan[i] = led.todo(e, stores[i].fingerprint, members, retry_errors)
            print(f"{i} (pool {pool}): {len(plan[i])} to compute", file=log)

        mapped = [i for i in plan if encs[i].kind == "map"]
        if any(plan[i] for i in mapped):
            recs = catalogue.records_for(cat, {h for i in mapped for h in plan[i]})
            for i in mapped:
                _run_map(led, encs[i], stores[i], pending, plan[i], recs)
        heavy = [i for i in plan if encs[i].kind in MAIN_KINDS + ("zip",) and plan[i]]
        if any(encs[i].kind == "audio" for i in heavy):  # なければ全曲がエラーとして記録されてしまう
            try:
                archive.need_ffmpeg()
            except ValueError as ex:
                raise GenerationError(str(ex)) from None
        no_room = _run_heavy(cfg, arch, led, encs, stores, pending, {i: plan[i] for i in heavy}, log) if heavy else {}
        for i in [i for i in encs if encs[i].kind == "derived"]:
            # 派生のビューは元のストアに従う。足したものはほかの結果と同じく記録する
            try:
                print(f"{i}: {encs[i].derive(cfg.store, _device(cfg))} vectors", file=log)
            except GenerationError as ex:  # 例 スタイルのモデルが設定されていない: ほかのビューはそれでも済ませる
                print(f"{i}: skipped ({ex})", file=log)
                continue
            led.adopt(encs[i], stores[i].fingerprint, Store(encs[i], cfg.store).done())
        print("ledger: " + str(led.summary()), file=log)
        if no_room:
            print("left for a later run (the GPU had no room, or the PC could not read the song for now): "
                  + ", ".join(f"{i} {k}" for i, k in no_room.items()), file=log)
        return no_room
    finally:
        led.close()


def _with_sources(ids):
    """元を先に。派生のビューが、元を計算済みで見つけられるように。"""
    for i in ids:
        yield from _with_sources(get(i).sources())
        yield i


def _flushed(led, enc, store, pending, keys):
    hashes = dict.fromkeys(key_hash(k) for k in keys)  # 譜面の行: マップの版ごとに 1 つの結果
    led.record(enc, store.fingerprint, [(h, "ok", pending.pop((enc.id, h), None), None) for h in hashes])


def _add(store, h, x):
    """1 マップの出力: ベクトル（マップの行）か {譜面のキー: ベクトル}（譜面の行）。-> 何もなければ False。"""
    if isinstance(x, dict):
        if not x:
            return False
        store.add_many(x)
    else:
        store.add(h, x)
    return True


def _run_map(led, enc, store, pending, todo, recs):
    fields, rows = meta_fields(enc), []
    for h in todo:
        if h not in recs:
            continue
        m, v = recs[h]
        d = meta_digest(m, v, fields)
        vec = enc.embed_map(m, v)
        if vec is None or (isinstance(vec, dict) and not vec):
            rows.append((h, "skip", d, None))
        else:
            pending[(enc.id, h)] = d
            _add(store, h, vec)
    store.flush()
    led.record(enc, store.fingerprint, rows)


def _run_heavy(cfg, arch, led, encs, stores, pending, plan, log):
    """-> {encoder id: 記録せず後の実行に残したマップ}（_consume）。"""
    heavy = list(plan)
    want = {}
    for i, hs in plan.items():
        for h in hs:
            want.setdefault(h, []).append(i)
    seq = dict(led.db.execute("SELECT hash, seq FROM versions WHERE seq IS NOT NULL"))
    tasks = sorted(want.items(), key=lambda t: seq.get(t[0], 0))
    print(f"{len(tasks)} maps to read for {heavy}", file=log)
    device = _device(cfg)
    if device == "cuda" and any(encs[i].kind in MAIN_KINDS for i in heavy):  # zip だけなら GPU は使わない
        _reserve_gpu(cfg.gpu_reserve_gb, log)
    for i in heavy:
        if encs[i].kind in MAIN_KINDS:
            encs[i].setup_model(device)
    t0 = time.time()
    try:
        if cfg.workers <= 1:  # プロセスの中で（テスト、デバッグ）
            _init(heavy, str(arch))
            results = map(_work, tasks)
            return _consume(led, encs, stores, results, len(tasks), t0, log)
        else:
            limit = cfg.worker_memory_gb * 2**30 if cfg.worker_memory_gb > 0 else None
            with closing(parallel(_work, tasks, cfg.workers, _init, (heavy, str(arch), config.current()),
                                  memory_limit=limit)) as results:
                return _consume(led, encs, stores, (_unpack(t, r, log) for t, r in results), len(tasks), t0, log)
    finally:
        for i in heavy:
            stores[i].flush()


# 数値計算のライブラリは、どのワーカーでも CPU ごとに 1 つのスレッドを始める。ワーカーが多いと、そのスレッドは
# 互いに取り合うだけ
ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ワーカーはこの数のマップを扱ったら入れ替える。長く生きたワーカーにたまるメモリを返すため
TASKS_PER_WORKER = 200


class MemoryLimitExceeded(Exception):
    """ワーカーがこのタスクでメモリの上限を超え、止められた（失うのはそのタスクだけ）。"""


def _tracked(fn, notes, i, task):
    """ワーカーで走る: このプロセスがどのタスクを持つかを書き残し、親がそれを名指して止められるようにする。"""
    note = Path(notes) / str(os.getpid())
    note.write_text(str(i))
    try:
        return fn(task)
    finally:  # 入れ替えたワーカー（TASKS_PER_WORKER）は、pid を使い回され得る書き置きを残さない
        note.unlink()


def _anon(pid):
    """プロセス自身の（匿名の）メモリのバイト数。常駐の大きさ全体ではない: ワーカーが割り当てる共有ライブラリ（torch
    だけで数 GB）もそこに数えられるが、自分のメモリは使わない。"""
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) * 1024
    return 0


def _watch(notes, limit, killed, stop):
    """親のスレッド: 自分のメモリが `limit` バイトを超えたワーカーを止め、持っていたタスクを書き残す。"""
    while not stop.wait(0.2):
        for note in Path(notes).iterdir():
            try:
                pid = int(note.name)
                if _anon(pid) > limit:
                    killed.add(int(note.read_text()))
                    os.kill(pid, signal.SIGKILL)
            except (OSError, ValueError):  # ワーカーが終わったか、ちょうど書き置きを書いている
                continue


def parallel(fn, tasks, workers, initializer, initargs, in_flight=4, memory_limit=None):
    """-> すべてのタスクの (task, fn(task))。ワーカーのプロセスで走らせ、終わった順に返す（順は決まらない）。ワーカーごとに
    受け取り待ちは in_flight まで。受け取りが遅くても、結果がメモリにたまらないようにするため。

    memory_limit（バイト、Linux）: 自分の（匿名の）メモリがこれを超えたワーカーは止める。そのタスクは
    (task, MemoryLimitExceeded) で返り、ほかのタスクは新しいワーカーで続く。ほかの形で死んだワーカー（ネイティブの
    コードで落ちた、外から殺された）は、持っていたかもしれないタスクとともに実行を止める: やり直せば台帳から続く。"""
    saved = {k: os.environ.get(k) for k in ONE_THREAD}
    for k in ONE_THREAD:  # 各ワーカーが始まるときに読む（ワーカーはタスクを渡すときに始まる）。呼ぶ側の値は
        os.environ.setdefault(k, "1")  # 残し、実行が終われば全部を元に戻す
    watch = bool(memory_limit) and os.path.exists("/proc/self/status")
    todo = deque(enumerate(tasks))
    try:
        while todo:
            with tempfile.TemporaryDirectory() as notes:
                killed, stop = set(), threading.Event()
                watcher = threading.Thread(target=_watch, args=(notes, memory_limit, killed, stop), daemon=True)
                # fork ではなく spawn: メインのプロセスがもう CUDA やダウンロードのスレッドを持っているかもしれない（fork した
                # ワーカーは止まってしまう）
                ex = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=initializer,
                                         initargs=initargs, max_tasks_per_child=TASKS_PER_WORKER)
                if watch:
                    watcher.start()
                running = {}
                try:
                    while todo or running:
                        broken = False
                        while todo and len(running) < workers * in_flight:
                            i, task = todo.popleft()
                            try:
                                running[ex.submit(_tracked, fn, notes, i, task)] = (i, task)
                            except BrokenProcessPool:  # 結果を集める前にワーカーが死んだ
                                todo.appendleft((i, task))
                                broken = True
                                break
                        if broken:
                            break
                        done, _ = wait(running, return_when=FIRST_COMPLETED)
                        for f in done:
                            i, task = running.pop(f)
                            try:
                                r = f.result()
                            except BrokenProcessPool:
                                running[f] = (i, task)
                                broken = True
                                continue
                            yield task, r
                        if broken:
                            break
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
                    stop.set()
                    if watcher.is_alive():
                        watcher.join()
                if not running and not todo:
                    return
                if not killed:
                    held = [t[0] if isinstance(t, tuple) else t for _, t in running.values()]
                    raise GenerationError(f"a worker process died (crashed in native code, or killed from outside); "
                                          f"maps it may have held: {', '.join(map(str, held))}. Run again to continue.")
                for i, task in sorted(running.values(), key=lambda t: t[0], reverse=True):
                    if i in killed:
                        yield task, MemoryLimitExceeded(f"used more than {memory_limit / 2**30:.1f} GiB of memory")
                    else:  # プールと一緒に止められただけのワーカーが持っていた: もう一度計算する
                        todo.appendleft((i, task))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _unpack(task, r, log):
    """メモリのために止めたタスクは、そのマップを求めたどのエンコーダーも失敗にする（読めない zip と同じ）。"""
    if isinstance(r, MemoryLimitExceeded):
        h, ids = task
        print(f"{h}: worker stopped, it {r}", file=log, flush=True)
        return h, ids, None, {}, f"worker {r}"
    return r


# メインのプロセスが一番古いものを待つ前にモデルのスレッドへ渡すまとまりの数: メインのプロセスがワーカーの結果を
# 受け取るあいだも GPU がいつも次のまとまりを持てるほど多く、用意したマップがたまらないほど少なく
GPU_AHEAD = 2


def _consume(led, encs, stores, results, total, t0, log):
    """`batch` が 1 より大きい音声と画像のエンコーダーは、マップをまとまりで受け取る（embed_many）ので、GPU は一度に
    いくつものマップを扱う。まとまりの結果は、そのまとまりを埋め込んだときに記録する（残りは最後に）。

    モデルは専用の 1 つのスレッドで動くので、GPU が働くあいだもメインのプロセスはワーカーの結果を受け取り続け（ワーカー
    はデコードを続ける）、どちらも相手を待たない。台帳とストアに書くのはメインのプロセスだけで、結果はまとまりを渡した
    順に記録する。

    GPU に空きがなければ、モデルはもっと小さいまとまりを受け取る（_run_model）。1 つでも入らないマップと、ワーカーで
    PC のいまの状態のせいで曲を読めなかったマップ（archive.LeftForLater: 曲を展開する空きメモリがない、ffmpeg が外から止められた）
    は記録しないので、次の実行で計算する。-> {encoder id: そういうマップ}（残したエンコーダーだけ）。"""
    groups, ahead, no_room = {}, deque(), {}
    sizes = {i: [getattr(e, "batch", 1)] for i, e in encs.items()}  # モデル 1 回の呼び出しのマップ数。_run_model が縮める
    model = ThreadPoolExecutor(1, thread_name_prefix="octantvector-model")

    def hand_over(i, group):
        ahead.append((i, group, model.submit(_run_model, encs[i], [x for _, x in group], sizes[i], log)))
        while ahead and (len(ahead) > GPU_AHEAD or ahead[0][2].done()):
            _record_group(led, encs, stores, no_room, *ahead.popleft())

    try:
        for k, (h, ids, sha, out, err) in enumerate(results, 1):
            if sha and not led.zip_sha(h, sha):
                print(f"{h}: zip differs from the one first read for this hash; not used", file=log)
                err = f"zip sha256 {sha} differs from the recorded one"
            fails = {}
            for i, x in out.items():
                if err:
                    fails[i] = ("error", err)
                elif x is None or (isinstance(x, dict) and not x):
                    fails[i] = ("skip", None)
                elif isinstance(x, archive.LeftForLater):  # 記録しない: 後の実行で計算する
                    no_room[i] = no_room.get(i, 0) + 1
                    print(f"{h}: {i}: {type(x).__name__} ({x}); left for a later run", file=log)
                elif isinstance(x, Exception):
                    fails[i] = ("error", repr(x))
                elif encs[i].kind in MAIN_KINDS:
                    g = groups.setdefault(i, [])
                    g.append((h, x))
                    if len(g) >= getattr(encs[i], "batch", 1):
                        hand_over(i, groups.pop(i))
                else:
                    try:
                        _embedded(led, encs[i], stores[i], h, x)
                    except Exception as ex:
                        fails[i] = ("error", repr(ex))
            if err:  # zip がまったく読めなかった: それを求めたどのエンコーダーも失敗した
                for i in ids:
                    fails.setdefault(i, ("error", err))
            for i, (r, e) in fails.items():
                led.record(encs[i], stores[i].fingerprint, [(h, r, None, e)])
            if k % 100 == 0:
                rate = k / (time.time() - t0)
                print(f"{k}/{total}  {rate:.1f} maps/s  eta {(total - k) / rate / 60:.1f} min", file=log, flush=True)
    finally:  # 実行が止まるときも: 読んだものは計算し直さない
        try:
            for i, g in groups.items():
                hand_over(i, g)
            while ahead:
                _record_group(led, encs, stores, no_room, *ahead.popleft())
        finally:
            model.shutdown(wait=True)
    return no_room


def _embedded(led, enc, store, h, v):
    """1 マップの出力。音声か画像のエンコーダーからの None は、表すものが見つからなかったこと（例 曲に声がない）: skip と
    して記録する。有限でない値を持つ音声か画像のベクトルは書かずにエラーにする（ボーカルのビューで 16 秒の曲が NaN の
    ベクトルになったことがある）: 正規化がそれを 0 に変え、DB に意味のない行が入るため。譜面のビューの NaN と inf は
    エンコーダーが決めた値（chart_timing）なので、ここでは見ない。"""
    if v is None:
        led.record(enc, store.fingerprint, [(h, "skip", None, None)])
        return
    if enc.kind in MAIN_KINDS and not np.isfinite(v).all():
        raise ValueError(f"{enc.id}: the vector has values that are not finite")
    _add(store, h, v)


# 1 つでも GPU に入らなかったマップの出力: 何も記録しないので、後の実行で計算する
NO_ROOM = object()


def _out_of_memory(ex):
    """`ex` が torch の CUDA のメモリ不足か（名前で見分ける: ここでは torch を import しない）。"""
    return type(ex).__name__ == "OutOfMemoryError"


def _empty_cache():
    """torch がほかのモデルのためにキャッシュしている GPU のメモリを返し、次の試しがその空きを使えるようにする。"""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_model(enc, xs, size, log):
    """モデルのスレッドで: 用意したマップごとに 1 つの出力（batch のあるエンコーダーは embed_many、なければ embed）。
    1 回の呼び出しで `size[0]` マップ。GPU に空きがなければ（ほかのモデルやほかのプログラムのメモリ、例 VR のサーバー）、
    キャッシュを空けて、マップを半分にしてもう一度試し、1 つまで下げる。エンコーダーは実行の残りでも小さい数を使う
    （size はそのまとまりで共有する）。2 回試しても 1 つで入らないマップは NO_ROOM。"""
    out, again = [], False
    while len(out) < len(xs):
        part = xs[len(out):len(out) + size[0]]
        try:
            out += enc.embed_many(part) if getattr(enc, "batch", 1) > 1 else [enc.embed(x) for x in part]
            again = False
            continue
        except Exception as ex:
            if not _out_of_memory(ex):
                raise
        _empty_cache()  # except の外: 失敗した呼び出しの、traceback が持つテンソルはもう解放されている
        if size[0] > 1:
            size[0] //= 2
            print(f"{enc.id}: the GPU had no room, {size[0]} maps at a time from now", file=log, flush=True)
        elif not again:
            again = True
        else:
            out.append(NO_ROOM)
            again = False
    return out


def _record_group(led, encs, stores, no_room, i, group, future):
    """モデルのスレッドに渡したまとまりを記録する。モデルがまだ働いていれば待つ。GPU に空きのなかったマップは no_room に
    数えるだけ。"""
    enc, store = encs[i], stores[i]
    try:
        vs = future.result()
    except Exception as ex:
        led.record(enc, store.fingerprint, [(h, "error", None, repr(ex)) for h, _ in group])
        return
    for (h, _), v in zip(group, vs):
        if v is NO_ROOM:
            no_room[i] = no_room.get(i, 0) + 1
            continue
        try:
            _embedded(led, enc, store, h, v)
        except Exception as ex:
            led.record(enc, store.fingerprint, [(h, "error", None, repr(ex))])


def _reserve_gpu(reserve_gb, log):
    """torch がこのプロセスで確保する GPU のメモリを、全体から `reserve_gb` を除いた量までにする。キャッシュも含めて
    この量を超える確保はメモリ不足になり、_run_model がマップを減らして試し直す。CUDA の文脈（数百 MB）はこの外。
    0 なら上限なし（このプロセスで前に上限を付けていれば外す）。"""
    if reserve_gb <= 0:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.set_per_process_memory_fraction(1.0, torch.cuda.current_device())
        return
    import torch
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory
    cap = total - reserve_gb * 2**30
    if cap <= 0:
        raise GenerationError(f"gpu_reserve_gb ({reserve_gb:g}) is not less than the GPU memory "
                              f"({total / 2**30:.1f} GB)")
    torch.cuda.set_per_process_memory_fraction(cap / total, dev)
    print(f"GPU: models use at most {cap / 2**30:.1f} of {total / 2**30:.1f} GB "
          f"({reserve_gb:g} GB left for other programs)", file=log, flush=True)


def _device(cfg):
    """モデルを動かすデバイス（"cuda" か "cpu"）。設定 `device` で決める（search.resolve_device）。"""
    return "cpu" if resolve_device(cfg.device) is None else "cuda"
