"""引擎层测试：KV 缓存正确性、块内分叉、写时复制、满池分叉、原子拒绝、回收、并发。"""
import threading

import numpy as np
import pytest

from app.blocks import CapacityError
from app.engine import Engine, ValidationError

VOCAB = 64


def make_engine(num_blocks=8, block_size=4, d_model=16, seed=7):
    return Engine.from_config(vocab_size=VOCAB, d_model=d_model,
                              block_size=block_size, num_blocks=num_blocks,
                              seed=seed)


def cached_logits(eng, session_id, branch_id):
    """从块缓存路径算出的 logits（即预测端实际走的路径）。"""
    with eng.lock:
        br = eng.sessions[session_id].branches[branch_id]
        _, _, q = eng.model.kvq(br.tokens[-1], br.length - 1)
        K, V = eng._gather_kv(br)
        return eng.model.logits_from_kv(q, K, V)


def assert_matches_scratch(eng, session_id, branch_id, tokens):
    """缓存路径的预测必须与按各自词元从头计算一致。"""
    got = cached_logits(eng, session_id, branch_id)
    want = eng.model.scratch_logits(tokens)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
    assert int(np.argmax(got)) == int(np.argmax(want))


def test_cached_prediction_matches_scratch_across_appends():
    eng = make_engine()
    r = eng.create_session([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])  # 10 词元，跨 3 块
    sid, bid = r["session_id"], r["branch"]["id"]
    eng.append(sid, bid, [5, 8, 9])
    eng.append(sid, bid, [7, 9])
    tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]
    assert_matches_scratch(eng, sid, bid, tokens)
    pred = eng.predict(sid, bid)["prediction"]
    assert pred["argmax"] == int(np.argmax(eng.model.scratch_logits(tokens)))


def test_in_block_fork_shares_blocks_then_cow():
    eng = make_engine(num_blocks=8, block_size=4)
    r = eng.create_session([5, 6, 7, 8, 9, 10])  # 块0满，块1占2槽
    sid = r["session_id"]
    main = r["branch"]["id"]

    f = eng.fork(sid, main, 5)  # 分叉点 5 落在块1内部（块1只有槽0有效）
    child = f["branch"]["id"]
    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert c.block_ids == p.block_ids            # 前缀块共享
        assert eng.pool.refcount[p.block_ids[0]] == 2
        assert eng.pool.refcount[p.block_ids[1]] == 2
        shared_last = p.block_ids[1]                 # 被共享的未满块

    eng.append(sid, main, [11, 12, 13])             # 父分支先写 -> 父分支写时复制
    pred_parent_before = eng.predict(sid, main)["prediction"]["argmax"]
    eng.append(sid, child, [21, 22])                # 子分支续写

    with eng.lock:
        p = eng.sessions[sid].branches[main]
        c = eng.sessions[sid].branches[child]
        assert c.block_ids[0] == p.block_ids[0]      # 满块仍共享
        assert c.block_ids[1] != p.block_ids[1]      # 未满共享块已因写时复制而分家
        assert c.block_ids[1] == shared_last         # 原块归子分支（父分支先触发复制）
        assert eng.pool.refcount[shared_last] == 1
        assert p.tokens == [5, 6, 7, 8, 9, 10, 11, 12, 13]

    # 两条分支的预测都与各自从头计算一致
    assert_matches_scratch(eng, sid, main, [5, 6, 7, 8, 9, 10, 11, 12, 13])
    assert_matches_scratch(eng, sid, child, [5, 6, 7, 8, 9, 21, 22])
    # 子分支的写时复制不影响父分支预测
    assert eng.predict(sid, main)["prediction"]["argmax"] == pred_parent_before


def test_fork_works_when_pool_full():
    eng = make_engine(num_blocks=3, block_size=4)   # 容量 12 词元
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])
    sid, main = r["session_id"], r["branch"]["id"]
    eng.create_session([9, 10, 11, 12])             # 占满最后一块
    assert eng.stats()["free_blocks"] == 0

    f1 = eng.fork(sid, main, 8)                     # 块边界分叉：不分配块
    f2 = eng.fork(sid, main, 3)                     # 块内分叉：同样不分配块
    assert eng.stats()["free_blocks"] == 0
    assert f2["branch"]["block_ids"] == f1["branch"]["block_ids"][:1]

    with pytest.raises(CapacityError):              # 但任何追加都放不下
        eng.append(sid, f1["branch"]["id"], [9])
    # 拒绝后状态不变
    s = eng.get_session(sid)
    lens = {b["name"]: b["length"] for b in s["branches"]}
    assert lens == {"main": 8, "main@8": 8, "main@3": 3}
    assert eng.stats()["used_blocks"] == 3


def test_append_rejected_atomically_when_too_large():
    eng = make_engine(num_blocks=4, block_size=4)   # 容量 16 词元
    r = eng.create_session([1, 2, 3, 4])
    sid, bid = r["session_id"], r["branch"]["id"]
    eng.create_session([5, 6, 7, 8, 9, 10, 11, 12])  # 再占 2 块，只剩 1 块空闲
    before = eng.get_session(sid)["branches"][0]
    used_before = eng.stats()["used_blocks"]

    with pytest.raises(CapacityError):
        eng.append(sid, bid, list(range(20, 28)))   # 8 个词元需要 2 个新块，只剩 1 个

    after = eng.get_session(sid)["branches"][0]
    assert after == before                          # 分支内容、块表完全没变
    assert eng.stats()["used_blocks"] == used_before


def test_close_branch_recycles_only_unreferenced_blocks():
    eng = make_engine(num_blocks=6, block_size=4)
    r = eng.create_session([1, 2, 3, 4, 5, 6, 7, 8])  # 块0、块1
    sid, main = r["session_id"], r["branch"]["id"]
    child = eng.fork(sid, main, 8)["branch"]["id"]    # 共享块0、块1
    grand = eng.fork(sid, main, 4)["branch"]["id"]    # 只共享块0
    assert eng.stats()["used_blocks"] == 2

    eng.close_branch(sid, main)                       # 块0:3->2，块1:2->1，都不回收
    assert eng.stats()["used_blocks"] == 2
    eng.close_branch(sid, child)                      # 块1 归零回收，块0 仍被引用
    assert eng.stats()["used_blocks"] == 1
    eng.close_branch(sid, grand)                      # 全部回收
    assert eng.stats()["used_blocks"] == 0
    assert eng.stats()["free_blocks"] == 6


def test_fork_at_zero_and_empty_branch():
    eng = make_engine()
    r = eng.create_session([1, 2, 3])
    sid, main = r["session_id"], r["branch"]["id"]
    empty = eng.fork(sid, main, 0)["branch"]
    assert empty["block_ids"] == [] and empty["length"] == 0
    with pytest.raises(ValidationError):
        eng.predict(sid, empty["id"])
    eng.append(sid, empty["id"], [7, 7, 7, 7, 7])
    assert_matches_scratch(eng, sid, empty["id"], [7, 7, 7, 7, 7])


def test_concurrent_appends_respect_total_capacity():
    eng = make_engine(num_blocks=8, block_size=4)     # 容量 32 词元
    r = eng.create_session([1, 2, 3, 4])              # 占 1 块，空闲 7
    sid, main = r["session_id"], r["branch"]["id"]
    other = eng.fork(sid, main, 4)["branch"]["id"]
    branches = [main, other]
    results, errors = [], []

    def worker(i):
        try:
            eng.append(sid, branches[i % 2], [10 + i] * 8)  # 每次需 2 个新块
            results.append(i)
        except CapacityError:
            errors.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 空闲 7 块、每次追加需 2 块：成功次数至多 3 次，且必有被拒
    assert 1 <= len(results) <= 3
    assert len(results) + len(errors) == 10
    st = eng.stats()
    assert st["used_blocks"] <= st["num_blocks"]
    assert st["used_blocks"] == 1 + 2 * len(results)

    # 每条分支的长度 = 4 + 8 * 其成功次数，且预测与从头计算一致
    sess = eng.get_session(sid)
    for b in sess["branches"]:
        n_ok = sum(1 for i in results if branches[i % 2] == b["id"])
        assert b["length"] == 4 + 8 * n_ok
        assert_matches_scratch(eng, sid, b["id"], b["tokens"])
