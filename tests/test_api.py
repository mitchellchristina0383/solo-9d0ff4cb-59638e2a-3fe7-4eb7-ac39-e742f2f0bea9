"""HTTP 端到端测试：真实 uvicorn 服务 + requests，覆盖块内分叉、并发满额、回收。"""
import concurrent.futures as cf
import threading
import time

import numpy as np
import pytest
import requests
import uvicorn

from app.engine import Engine
from app.main import create_app

BASE = "http://127.0.0.1:8471"


@pytest.fixture(scope="module")
def server():
    eng = Engine.from_config(vocab_size=64, d_model=16, block_size=4,
                             num_blocks=32, seed=7)
    app = create_app(eng)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8471,
                                        log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        try:
            requests.get(BASE + "/api/stats", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        raise RuntimeError("server 未能启动")
    yield eng
    srv.should_exit = True
    th.join(timeout=10)


def stats():
    return requests.get(BASE + "/api/stats", timeout=5).json()


def test_full_flow_in_block_fork_and_recycle(server):
    eng = server
    free0 = stats()["free_blocks"]

    # 建会话：6 词元 -> 块0满、块1占2槽
    r = requests.post(BASE + "/api/sessions", json={"tokens": [5, 6, 7, 8, 9, 10]})
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    main = r.json()["branch"]["id"]
    assert stats()["used_blocks"] == 2

    # 分叉点 5 落在块1内部
    r = requests.post(BASE + f"/api/sessions/{sid}/fork",
                      json={"branch_id": main, "position": 5})
    assert r.status_code == 201, r.text
    child = r.json()["branch"]["id"]
    assert r.json()["branch"]["tokens"] == [5, 6, 7, 8, 9]
    assert stats()["used_blocks"] == 2  # 分叉不新占块

    # 两条分支各自续写不同词元
    r = requests.post(BASE + f"/api/sessions/{sid}/append",
                      json={"branch_id": main, "tokens": [11, 12, 13]})
    assert r.status_code == 200, r.text
    pred_main = r.json()["branch"]["prediction"]["argmax"]
    r = requests.post(BASE + f"/api/sessions/{sid}/append",
                      json={"branch_id": child, "tokens": [21, 22]})
    assert r.status_code == 200, r.text
    pred_child = r.json()["branch"]["prediction"]["argmax"]

    # 预测与按各自词元从头计算一致
    assert pred_main == int(np.argmax(eng.model.scratch_logits([5, 6, 7, 8, 9, 10, 11, 12, 13])))
    assert pred_child == int(np.argmax(eng.model.scratch_logits([5, 6, 7, 8, 9, 21, 22])))

    # 关闭子分支：共享块仍被父分支引用，只有写时复制出的私有块被回收
    used_before = stats()["used_blocks"]
    r = requests.delete(BASE + f"/api/sessions/{sid}/branches/{child}")
    assert r.status_code == 200
    assert stats()["used_blocks"] == used_before - 1  # 子分支私有块 1 个

    # 删除会话：全部回收
    r = requests.delete(BASE + f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert stats()["free_blocks"] == free0


def test_validation_errors(server):
    r = requests.post(BASE + "/api/sessions", json={"tokens": [0, 999]})
    assert r.status_code == 400
    r = requests.post(BASE + "/api/sessions", json={"tokens": []})
    assert r.status_code == 400
    r = requests.post(BASE + "/api/sessions", json={"tokens": [1, 2]})
    sid = r.json()["session_id"]
    bid = r.json()["branch"]["id"]
    r = requests.post(BASE + f"/api/sessions/{sid}/fork",
                      json={"branch_id": bid, "position": 99})
    assert r.status_code == 400
    r = requests.get(BASE + "/api/sessions/nope")
    assert r.status_code == 404
    requests.delete(BASE + f"/api/sessions/{sid}")
    assert stats()["used_blocks"] == 0


def test_concurrent_appends_over_http_respect_capacity(server):
    eng = server
    assert stats()["free_blocks"] == 32  # 前序测试已清理干净

    r = requests.post(BASE + "/api/sessions", json={"tokens": [1, 2, 3, 4]})
    sid = r.json()["session_id"]
    main = r.json()["branch"]["id"]
    child = requests.post(BASE + f"/api/sessions/{sid}/fork",
                          json={"branch_id": main, "position": 4}).json()["branch"]["id"]
    branches = [main, child]
    free = stats()["free_blocks"]  # 31

    def one_append(i):
        return requests.post(BASE + f"/api/sessions/{sid}/append",
                             json={"branch_id": branches[i % 2],
                                   "tokens": [10 + i] * 8}, timeout=10)

    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        rs = list(ex.map(one_append, range(40)))  # 每次需 2 块，共要 80 块，远超 31

    ok = [r for r in rs if r.status_code == 200]
    rejected = [r for r in rs if r.status_code == 409]
    assert len(ok) + len(rejected) == 40
    assert len(rejected) > 0                    # 必然有整体拒绝
    assert len(ok) <= free // 2                 # 成功数受总块数约束
    st = stats()
    assert st["used_blocks"] == 1 + 2 * len(ok)  # 占用与实际成功完全一致

    # 分支内容与成功追加一致，预测与从头计算一致
    sess = requests.get(BASE + f"/api/sessions/{sid}").json()
    for b in sess["branches"]:
        assert b["length"] == len(b["tokens"])
        assert (b["length"] - 4) % 8 == 0
        want = int(np.argmax(eng.model.scratch_logits(b["tokens"])))
        assert b["prediction"]["argmax"] == want

    requests.delete(BASE + f"/api/sessions/{sid}")
    assert stats()["used_blocks"] == 0
