"""HTTP 端到端测试：暂停到磁盘 / 恢复的接口与页面所依赖的错误信息。"""
import os
import threading
import time

import numpy as np
import pytest
import requests
import uvicorn

from app.engine import Engine
from app.main import create_app

BASE = "http://127.0.0.1:8472"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    spill_dir = tmp_path_factory.mktemp("spill-api")
    eng = Engine.from_config(vocab_size=64, d_model=16, block_size=4,
                             num_blocks=12, seed=11, spill_dir=spill_dir)
    app = create_app(eng)
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8472,
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


def st():
    return requests.get(BASE + "/api/stats", timeout=5).json()


def make_session(tokens):
    r = requests.post(BASE + "/api/sessions", json={"tokens": tokens}, timeout=5)
    assert r.status_code == 201, r.text
    return r.json()


def test_pause_resume_http_preserves_branches_and_predictions(ctx):
    eng = ctx
    s = make_session([5, 6, 7, 8, 9, 10])
    sid, main = s["session_id"], s["branch"]["id"]
    child = requests.post(BASE + f"/api/sessions/{sid}/fork",
                          json={"branch_id": main, "position": 5},
                          timeout=5).json()["branch"]["id"]
    requests.post(BASE + f"/api/sessions/{sid}/append",
                  json={"branch_id": main, "tokens": [11, 12, 13]}, timeout=5)
    requests.post(BASE + f"/api/sessions/{sid}/append",
                  json={"branch_id": child, "tokens": [21]}, timeout=5)

    detail = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    preds = {b["id"]: b["prediction"]["argmax"] for b in detail["branches"]}
    used = st()["used_blocks"]
    assert used == 4    # 共享块0 + 子分支块1 + 父分支写时复制/新增的块2、块3

    r = requests.post(BASE + f"/api/sessions/{sid}/pause", timeout=10)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "paused"
    assert body["spill"]["unique_blocks"] == 4
    assert body["spill"]["total_block_refs"] == 5   # 父 3 块 + 子 2 块
    assert os.path.exists(body["spill"]["path"])
    assert st()["used_blocks"] == 0

    # 暂停期间追加返回 409 且说明原因
    r = requests.post(BASE + f"/api/sessions/{sid}/append",
                      json={"branch_id": main, "tokens": [1]}, timeout=5)
    assert r.status_code == 409
    assert "暂停" in r.json()["detail"]

    # 页面读取的会话详情包含暂停状态与文件信息
    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    assert d["state"] == "paused" and d["spill"]["path"]
    assert all(b["state"] == "paused" and b["block_ids"] == []
               for b in d["branches"])

    r = requests.post(BASE + f"/api/sessions/{sid}/resume", timeout=10)
    assert r.status_code == 200, r.text
    assert r.json()["restored_blocks"] == 4
    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    assert d["state"] == "resident"
    assert {b["id"]: b["prediction"]["argmax"] for b in d["branches"]} == preds
    assert st()["used_blocks"] == 4

    # 恢复后原分支可继续续写，预测与从头计算一致
    r = requests.post(BASE + f"/api/sessions/{sid}/append",
                      json={"branch_id": main, "tokens": [31, 32]}, timeout=5)
    assert r.status_code == 200, r.text
    tokens = [5, 6, 7, 8, 9, 10, 11, 12, 13, 31, 32]
    want = int(np.argmax(eng.model.scratch_logits(tokens)))
    assert r.json()["branch"]["prediction"]["argmax"] == want

    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)


def test_pause_invalidated_by_append_during_write_over_http(ctx):
    eng = ctx
    s = make_session([1, 2, 3, 4, 5])
    sid, main = s["session_id"], s["branch"]["id"]
    used_before = st()["used_blocks"]

    def during_write(snapshot, path):
        r = requests.post(BASE + f"/api/sessions/{sid}/append",
                          json={"branch_id": main, "tokens": [9]}, timeout=10)
        assert r.status_code == 200, r.text

    eng.spill_store.before_write = during_write
    try:
        r = requests.post(BASE + f"/api/sessions/{sid}/pause", timeout=15)
    finally:
        eng.spill_store.before_write = None
    assert r.status_code == 409
    assert "作废" in r.json()["detail"]

    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    assert d["state"] == "resident"
    assert d["branches"][0]["tokens"] == [1, 2, 3, 4, 5, 9]
    assert st()["used_blocks"] == used_before   # 块未被释放

    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)


def test_resume_capacity_shortage_reports_and_can_retry(ctx):
    s = make_session([1, 2, 3, 4, 5, 6])   # 2 块
    sid = s["session_id"]
    r = requests.post(BASE + f"/api/sessions/{sid}/pause", timeout=10)
    assert r.status_code == 200
    path = r.json()["spill"]["path"]

    # 占满 12 块（注意此时该会话已暂停、不占块）
    hog = make_session([7] * 48)["session_id"]
    assert st()["free_blocks"] == 0

    r = requests.post(BASE + f"/api/sessions/{sid}/resume", timeout=10)
    assert r.status_code == 409
    assert "恢复需要" in r.json()["detail"]
    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    assert d["state"] == "paused"
    assert all(b["block_ids"] == [] for b in d["branches"])
    assert st()["used_blocks"] == 12
    assert os.path.exists(path)

    # 释放后重试成功
    requests.delete(BASE + f"/api/sessions/{hog}", timeout=5)
    r = requests.post(BASE + f"/api/sessions/{sid}/resume", timeout=10)
    assert r.status_code == 200, r.text
    assert r.json()["restored_blocks"] == 2
    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)
    assert st()["used_blocks"] == 0


def test_resume_corrupt_file_reports_reason(ctx):
    s = make_session([1, 2, 3, 4])
    sid = s["session_id"]
    r = requests.post(BASE + f"/api/sessions/{sid}/pause", timeout=10)
    path = r.json()["spill"]["path"]
    with open(path, "wb") as f:
        f.write(b"corrupted!")

    r = requests.post(BASE + f"/api/sessions/{sid}/resume", timeout=10)
    assert r.status_code == 409
    assert "损坏" in r.json()["detail"]
    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    assert d["state"] == "paused"
    assert all(b["block_ids"] == [] for b in d["branches"])
    assert st()["used_blocks"] == 0

    # 会话仍可删除，文件一并清理
    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)
    assert not os.path.exists(path)


def test_empty_branch_session_roundtrip(ctx):
    s = make_session([1, 2, 3])
    sid, main = s["session_id"], s["branch"]["id"]
    empty = requests.post(BASE + f"/api/sessions/{sid}/fork",
                          json={"branch_id": main, "position": 0},
                          timeout=5).json()["branch"]["id"]
    r = requests.post(BASE + f"/api/sessions/{sid}/pause", timeout=10)
    assert r.status_code == 200, r.text
    r = requests.post(BASE + f"/api/sessions/{sid}/resume", timeout=10)
    assert r.status_code == 200, r.text
    d = requests.get(BASE + f"/api/sessions/{sid}", timeout=5).json()
    by = {b["id"]: b for b in d["branches"]}
    assert by[empty]["length"] == 0 and by[empty]["block_ids"] == []
    assert by[main]["length"] == 3
    requests.delete(BASE + f"/api/sessions/{sid}", timeout=5)
