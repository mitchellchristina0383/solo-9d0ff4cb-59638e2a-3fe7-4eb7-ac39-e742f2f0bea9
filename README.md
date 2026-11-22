# KV-Cache 分叉对比

给推理工程师的小工具：从一次推理的中途分叉出多条分支，各自续写不同词元，
对比下一词元预测。内置固定权重的单层因果注意力小模型，续写复用实际算出的
KV 缓存；KV 按固定大小的物理块管理，分叉共享前缀块（写时复制），块数有上限，
网页实时显示块占用。

## 启动

```bash
docker compose up --build
# 打开 http://localhost:8000
```

不用 Docker 也可以：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

可调参数（环境变量，compose.yaml 里有默认值）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `VOCAB_SIZE` | 64 | 词表大小，词元取值 `0..63` |
| `D_MODEL` | 32 | 注意力维度 |
| `BLOCK_SIZE` | 4 | 每个物理块容纳的词元数 |
| `NUM_BLOCKS` | 32 | 物理块总数（全部会话共享的容量上限） |
| `SEED` | 1234 | 固定权重的伪随机种子 |

## 用法

1. 输入一串整数词元建会话（自动算出 KV 并写入块）；
2. 选中一条分支，在任意位置（包括块中间）分叉 —— 只共享前缀块、不新占块，池满也能分叉；
3. 给各分支追加一串词元，页面展示各自的下一词元预测（top-1 与 top-5）；
4. 关闭分支只回收引用计数归零的块；页头与块网格实时显示占用。

## 设计要点

- **模型**（`app/model.py`）：单层单头因果注意力，`x = E[tok] + P[pos]`，
  `k/v/q = x @ W_{k,v,q}`，`logits = softmax(Kq/√d)·V @ Wo`。权重由固定种子生成，
  float64。K/V 只取决于 `(token, position)`，因此缓存值与从头逐位计算完全等价。
- **块池**（`app/blocks.py`）：固定数量物理块，每块 `BLOCK_SIZE` 个槽位，
  引用计数管理共享；归零才回到空闲池。
- **引擎**（`app/engine.py`）：
  - 分叉：子分支引用前缀块并增加引用计数，不分配新块；
  - 追加：先算清所需块数（末尾共享未满块需写时复制 1 块 + 新增整块数），
    空闲不足则整体拒绝，分支内容与块占用不变；检查与提交在同一把锁内完成，
    并发追加安全；
  - 分叉点落在块内时，子分支只承认该块前 `position` 个槽位，首次写入触发写时复制；
  - 预测只读缓存的 K/V（加上末词元的 q），不重算历史。
- **API**（`app/main.py`）：`POST /api/sessions`、`POST /api/sessions/{id}/fork`、
  `POST /api/sessions/{id}/append`、`GET /api/sessions/{id}/branches/{bid}/predict`、
  `DELETE /api/sessions/{id}/branches/{bid}`、`DELETE /api/sessions/{id}`、
  `GET /api/stats`。容量不足返回 409，参数非法 400，不存在 404。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -v
```

覆盖：缓存路径 vs 从头计算的 logits 一致（1e-10）、块内分叉的共享与写时复制、
满池分叉、追加整体拒绝的原子性、关闭分支的精确回收、引擎内与 HTTP 层的并发容量约束。
