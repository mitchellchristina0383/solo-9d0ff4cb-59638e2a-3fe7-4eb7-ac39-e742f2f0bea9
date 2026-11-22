"""暂停到磁盘 / 恢复测试：共享块只存一份、写盘期间续写作废、恢复块不足可重试、
恢复后续写、文件损坏说明原因。"""
import json
import os
import threading

import numpy as np
import pytest

from app.blocks import CapacityError
from app.engine import ConflictError, Engine
from app.spill import SpillCorruptError, SpillWriteError

VOCAB = 64


def make_engine(tmp_path, num_blocks=8, block_size=4, d_model=16, seed=7):
    return Engine.from_config(vocab_size=VOCAB, d_model=d_model,
                              block_size=block_size, num_blocks=num_blocks,
                              seed=seed, spill_dir=tmp_path / "spill")


def pause(eng, sid):
    job = eng.begin_spill(sid)
    stored = eng.write_snapshot(job)
    return eng.commit_spill(sid, job.id, stored)


def cached_logits(eng, sid, bid):
    with eng.lock:
        br = eng.sessions[sid].branches[bid]
        _, _, q = eng.model.kvq(br.tokens[-1], br.length - 1)
        K, V = eng._gather_kv(br)
        return eng.model.logits_from_kv(q, K, V)


def assert_matches_scratch(eng, sid, bid, tokens):
    np.testing.assert_allclose(cached_logits(eng, sid, bid),
                               eng.model.scratch_logits(tokens),
                               rtol=1e-10, atol=1e-10)


def snapshot_state(eng):
    return {sid: (s.state, {bid: (b.name, list(b.tokens), list(b.block_ids))
                            for bid, b in s.branches.items()})
            for sid, s in eng.sessions.items()}


# ---------------------------------------------------------------- 基本闭环

def test_pause_frees_blocks_and_stores_shared_blocks_once(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([5, 6, 7, 8, 9, 10])           # 块0满、块1占2槽
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 5)["branch"]["id"]         # 块内分叉，共享块0/1
    eng.append(sid, child, [21, 22])                       # 子分支写时复制出块2
    eng.append(sid, main, [11, 12, 13])                    # 父分支补块3

    before = eng.get_session(sid)["branches"]
    preds = {b["id"]: b["prediction"]["argmax"] for b in before}
    used = eng.stats()["used_blocks"]
    assert used == 4                                       # 块0(共享), 块1, 块2, 块3

    out = pause(eng, sid)
    assert out["state"] == "paused"
    assert out["spill"]["unique_blocks"] == 4
    assert out["spill"]["total_block_refs"] == 5           # 3 + 2 个逻辑块引用
    assert out["freed_blocks"] == 4
    assert eng.stats()["used_blocks"] == 0
    assert eng.stats()["free_blocks"] == 8
    assert os.path.getsize(out["spill"]["path"]) > 0

    paused = eng.get_session(sid)
    assert paused["state"] == "paused"
    assert all(b["state"] == "paused" and b["block_ids"] == []
               for b in paused["branches"])

    # 暂停期间不能追加 / 分叉 / 预测 / 关闭分支
    with pytest.raises(ConflictError):
        eng.append(sid, main, [1])
    with pytest.raises(ConflictError):
        eng.fork(sid, main, 3)
    with pytest.raises(ConflictError):
        eng.predict(sid, main)
    with pytest.raises(ConflictError):
        eng.close_branch(sid, main)

    # 恢复：块用量回到暂停前，分支 id / 名称 / 词元 / 共享关系 / 预测全部保留
    restored = eng.restore_session(paused["spill"]["path"])
    assert restored["session_id"] == sid
    assert restored["restored_blocks"] == used
    after = eng.get_session(sid)["branches"]
    assert {b["id"] for b in after} == {b["id"] for b in before}
    for b in after:
        assert b["state"] == "resident"
        assert b["tokens"] == next(x["tokens"] for x in before if x["id"] == b["id"])
        assert b["prediction"]["argmax"] == preds[b["id"]]
        # K/V 按位原样装回，top-5 概率也应逐位一致（float64 精确相等）
        old_top = next(x["prediction"]["top"] for x in before if x["id"] == b["id"])
        assert [t["token"] for t in b["prediction"]["top"]] == \
            [t["token"] for t in old_top]
        assert [t["prob"] for t in b["prediction"]["top"]] == \
            [t["prob"] for t in old_top]
    # 共享关系保留：两条分支共享第 0 个逻辑块对应的同一物理块
    with eng.lock:
        session = eng.sessions[sid]
        m = session.branches[main].block_ids[0]
        c = session.branches[child].block_ids[0]
        assert m == c
        assert eng.pool.refcount[m] == 2
        # 父分支未满块与子分支私有未满块已分家
        assert session.branches[main].block_ids[-1] != session.branches[child].block_ids[-1]
    assert eng.stats()["used_blocks"] == 4

    # 原分支可以继续续写，且与从头计算一致
    eng.append(sid, main, [31, 32])
    assert_matches_scratch(eng, sid, main, [5, 6, 7, 8, 9, 10, 11, 12, 13, 31, 32])
    eng.append(sid, child, [41])
    assert_matches_scratch(eng, sid, child, [5, 6, 7, 8, 9, 21, 22, 41])


def test_spill_file_deduplicates_shared_blocks(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session(list(range(1, 9)))             # 2 块
    sid, main = r["session_id"], r["branch"]["id"]
    for pos in (8, 4, 0):
        eng.fork(sid, main, pos)
    out = pause(eng, sid)
    # 4 条分支共享同一对物理块 -> 文件里只有 2 个唯一块，引用 6 次
    assert out["spill"]["unique_blocks"] == 2
    assert out["spill"]["total_block_refs"] == 5   # 2 + 1 + 1 + 1（位置 0 的分支没有块）
    with np.load(out["spill"]["path"], allow_pickle=False) as z:
        assert z["K"].shape == (2, 4, 16)
        assert z["V"].shape == (2, 4, 16)
    assert out["spill"]["file_size"] > 0


# ---------------------------------------------------------------- 写盘期间续写

def test_append_during_write_invalidates_pause(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5])
    sid, main = r["session_id"], r["branch"]["id"]
    used_before = eng.stats()["used_blocks"]
    tokens_before = eng.get_session(sid)["branches"][0]["tokens"]
    observed = {}

    def during_write(snapshot, path):
        # 写盘钩子运行在引擎锁外：同一进程内其他线程此时应能操作会话
        done = []

        def worker():
            eng.append(sid, main, [9, 9])           # 写盘期间续写
            done.append(True)

        th = threading.Thread(target=worker)
        th.start()
        th.join(timeout=5)
        observed["append_ok"] = bool(done)
        observed["blocks_during_write"] = eng.stats()["used_blocks"]

    eng.spill_store.before_write = during_write
    job = eng.begin_spill(sid)
    assert eng.stats()["used_blocks"] == used_before   # 拍快照阶段不释放块
    stored = eng.write_snapshot(job)
    with pytest.raises(ConflictError, match="作废"):
        eng.commit_spill(sid, job.id, stored)

    assert observed["append_ok"] is True
    assert not os.path.exists(stored["path"])          # 作废后文件已删除
    detail = eng.get_session(sid)
    assert detail["state"] == "resident"
    assert detail["branches"][0]["tokens"] == tokens_before + [9, 9]
    assert all(b["block_ids"] for b in detail["branches"])
    # 追加 5->7 词元正好补满块1的槽位（4..7），仍用原 2 块，未被释放
    assert eng.stats()["used_blocks"] == used_before

    # 作废后可立即重试暂停，这次成功
    eng.spill_store.before_write = None
    out = pause(eng, sid)
    assert out["state"] == "paused"
    assert eng.stats()["used_blocks"] == 0


def test_close_branch_during_write_invalidates_pause(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4])
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 2)["branch"]["id"]
    used_before = eng.stats()["used_blocks"]

    def during_write(snapshot, path):
        eng.close_branch(sid, child)

    eng.spill_store.before_write = during_write
    job = eng.begin_spill(sid)
    stored = eng.write_snapshot(job)
    with pytest.raises(ConflictError):
        eng.commit_spill(sid, job.id, stored)
    assert not os.path.exists(stored["path"])
    detail = eng.get_session(sid)
    assert detail["state"] == "resident"
    assert [b["id"] for b in detail["branches"]] == [main]
    assert eng.stats()["used_blocks"] == used_before     # 会话未暂停，块未释放


def test_write_failure_leaves_session_resident(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4])
    sid = r["session_id"]
    used_before = eng.stats()["used_blocks"]

    def broken(snapshot, path):
        raise OSError("磁盘满（模拟）")

    eng.spill_store.before_write = broken
    job = eng.begin_spill(sid)
    with pytest.raises(SpillWriteError, match="磁盘满"):
        eng.write_snapshot(job)
    eng.abort_spill(sid, job.id)
    detail = eng.get_session(sid)
    assert detail["state"] == "resident"
    assert eng.stats()["used_blocks"] == used_before
    eng.append(sid, r["branch"]["id"], [7])              # 会话完全可用


# ---------------------------------------------------------------- 恢复时争用块

def test_restore_fails_cleanly_when_blocks_short_then_retry(tmp_path):
    eng = make_engine(tmp_path, num_blocks=6, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])     # 2 块
    sid = r["session_id"]
    out = pause(eng, sid)
    assert eng.stats()["used_blocks"] == 0
    path = out["spill"]["path"]

    # 别的会话占满 6 块
    other = eng.create_session([9] * 24)["session_id"]
    assert eng.stats()["free_blocks"] == 0

    with pytest.raises(CapacityError, match="恢复需要 2 个空闲块"):
        eng.restore_session(path)
    detail = eng.get_session(sid)
    assert detail["state"] == "paused"                   # 仍暂停
    assert all(b["block_ids"] == [] for b in detail["branches"])  # 仍不占块
    assert os.path.exists(path)                          # 文件保留，可重试

    # 释放 1 块仍不够 -> 仍失败且状态不变
    eng.close_session(other)
    eng.create_session([9] * 20)                         # 占 5 块
    assert eng.stats()["free_blocks"] == 1
    with pytest.raises(CapacityError):
        eng.restore_session(path)
    assert eng.get_session(sid)["state"] == "paused"

    # 再释放足够块 -> 重试成功，内容与预测保持
    eng.close_session(eng.list_sessions()["sessions"][-1]["session_id"])
    assert eng.stats()["free_blocks"] == 6
    res = eng.restore_session(path)
    assert res["restored_blocks"] == 2
    detail = eng.get_session(sid)
    assert detail["state"] == "resident"
    assert detail["branches"][0]["tokens"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert detail["branches"][0]["prediction"]["argmax"] == int(
        np.argmax(eng.model.scratch_logits([1, 2, 3, 4, 5, 6, 7, 8])))
    assert eng.stats()["used_blocks"] == 2


def test_restore_races_with_concurrent_allocation(tmp_path):
    """多个会话同时抢块：恢复要么完整成功，要么保持暂停且不占块。"""
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])
    sid = r["session_id"]
    path = pause(eng, sid)["spill"]["path"]
    results = []
    barrier = threading.Barrier(4)

    def worker(i):
        barrier.wait()
        try:
            if i % 2 == 0:
                eng.restore_session(path)
                results.append(("restore", True))
            else:
                try:
                    eng.create_session([10 + i] * 16)     # 每次抢 4 块
                    results.append(("create", True))
                except CapacityError:
                    results.append(("create", False))
        except CapacityError:
            results.append(("restore", False))
        except ConflictError:
            results.append(("restore", "conflict"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    detail = eng.get_session(sid)
    restores = [ok for kind, ok in results if kind == "restore"]
    assert restores, "至少有一个恢复请求"
    if detail["state"] == "paused":
        assert all(b["block_ids"] == [] for b in detail["branches"])
        assert restores.count(True) == 0
    else:
        assert restores.count(True) == 1
        assert detail["branches"][0]["length"] == 8
    st = eng.stats()
    assert st["used_blocks"] <= st["num_blocks"]
    # 占用块数 = 恢复的 2 块（若成功） + 各成功建会话的 4 块
    created = sum(1 for kind, ok in results if kind == "create" and ok)
    expect = (2 if detail["state"] == "resident" else 0) + 4 * created
    assert st["used_blocks"] == expect


# ---------------------------------------------------------------- 文件损坏

def _rewrite_npz(path, mutate):
    with np.load(path, allow_pickle=False) as z:
        meta = z["meta"].tobytes()
        K = np.array(z["K"])
        V = np.array(z["V"])
        digest = str(z["digest"].item())
    meta2, K2, V2, digest2 = mutate(meta, K, V, digest)
    with open(path, "wb") as f:
        np.savez_compressed(
            f, meta=np.frombuffer(meta2, dtype=np.uint8), K=K2, V=V2,
            digest=np.array(digest2))


@pytest.mark.parametrize("how", ["garbage", "digest", "structure", "truncated"])
def test_corrupt_file_reports_reason_and_keeps_paused(tmp_path, how):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5])
    sid = r["session_id"]
    path = pause(eng, sid)["spill"]["path"]

    if how == "garbage":
        with open(path, "wb") as f:
            f.write(b"not a zip file at all")
    elif how == "truncated":
        data = open(path, "rb").read()
        with open(path, "wb") as f:
            f.write(data[: len(data) // 2])
    elif how == "digest":
        def mutate(meta, K, V, digest):
            K2 = K.copy()
            K2[0, 0, 0] += 1.0
            return meta, K2, V, digest
        _rewrite_npz(path, mutate)
    else:  # structure: 分支块表长度与词元数不符
        def mutate(meta, K, V, digest):
            obj = json.loads(meta.decode())
            obj["branches"][0]["block_refs"] = [0]
            m = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode()
            return m, K, V, digest
        _rewrite_npz(path, mutate)

    with pytest.raises(SpillCorruptError) as ei:
        eng.restore_session(path)
    assert ei.value.args[0]
    detail = eng.get_session(sid)
    assert detail["state"] == "paused"
    assert all(b["block_ids"] == [] for b in detail["branches"])
    assert eng.stats()["used_blocks"] == 0

    # 文件损坏无法修复，但删除暂停会话仍可用
    eng.close_session(sid)
    assert eng.list_sessions()["sessions"] == []
    assert not os.path.exists(path)


def test_config_mismatch_is_reported(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4, d_model=16)
    r = eng.create_session([1, 2, 3, 4])
    sid = r["session_id"]
    path = pause(eng, sid)["spill"]["path"]
    other = Engine.from_config(vocab_size=VOCAB, d_model=32, block_size=4,
                               num_blocks=8, seed=7, spill_dir=tmp_path / "x")
    with pytest.raises(SpillCorruptError, match="配置不一致"):
        other.load_spill(path)


def test_retry_restore_after_file_repaired(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5])
    sid = r["session_id"]
    path = pause(eng, sid)["spill"]["path"]
    good = open(path, "rb").read()
    with open(path, "wb") as f:
        f.write(good[:20])
    with pytest.raises(SpillCorruptError):
        eng.restore_session(path)
    assert eng.get_session(sid)["state"] == "paused"

    with open(path, "wb") as f:
        f.write(good)                      # 修复文件后重试
    out = eng.restore_session(path)
    assert out["restored_blocks"] == 2
    assert eng.get_session(sid)["state"] == "resident"
    assert not os.path.exists(path)


def test_session_with_only_empty_branch_roundtrip(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3])
    sid, main = r["session_id"], r["branch"]["id"]
    empty = eng.fork(sid, main, 0)["branch"]["id"]
    eng.close_branch(sid, main)            # 只剩空分支，不占任何块
    assert eng.stats()["used_blocks"] == 0
    out = pause(eng, sid)
    assert out["spill"]["unique_blocks"] == 0
    assert out["spill"]["total_block_refs"] == 0
    res = eng.restore_session(out["spill"]["path"])
    assert res["restored_blocks"] == 0
    detail = eng.get_session(sid)
    assert detail["state"] == "resident"
    assert [b["id"] for b in detail["branches"]] == [empty]
    assert detail["branches"][0]["prediction"]["argmax"] is None
    eng.append(sid, empty, [4, 5])
    assert_matches_scratch(eng, sid, empty, [4, 5])


def test_other_sessions_stay_responsive_during_write_and_read(tmp_path):
    eng = make_engine(tmp_path, num_blocks=8, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5])
    sid = r["session_id"]
    other = eng.create_session([6, 7, 8, 9])["session_id"]
    other_main = eng.list_sessions()["sessions"][1]["branches"][0]["id"]

    def probe():
        """在 I/O 钩子里从另一线程访问引擎；若锁被持有会超时。"""
        def work():
            eng.append(other, other_main, [2])
            eng.get_session(other)
            eng.stats()
        th = threading.Thread(target=work)
        th.start()
        th.join(timeout=5)
        return not th.is_alive()

    assert probe() is True  # 基线

    seen = []
    eng.spill_store.before_write = lambda snap, p: seen.append(probe())
    job = eng.begin_spill(sid)
    stored = eng.write_snapshot(job)
    eng.commit_spill(sid, job.id, stored)
    assert seen == [True]

    seen.clear()
    eng.spill_store.before_load = lambda p: seen.append(probe())
    eng.restore_session(stored["path"])
    assert seen == [True]
    assert eng.get_session(sid)["state"] == "resident"
