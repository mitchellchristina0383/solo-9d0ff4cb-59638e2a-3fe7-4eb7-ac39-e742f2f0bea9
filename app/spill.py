"""会话的 KV 快照落盘 / 读回。

文件是单个 ``.npz``（zip 容器），内容：
- ``meta``：UTF-8 JSON，记录会话、分支、逻辑块表、模型配置等；
- ``K`` / ``V``：所有被引用物理块的实际 K/V，``(唯一块数, block_size, d_model)``；
- ``digest``：对魔数、meta 字节和 K/V 原始字节的 SHA-256。

共享块按物理块去重，只存一份；分支的 ``block_refs`` 指向唯一块下标，
因此恢复时可以按引用次数重建共享关系。写入用临时文件 + ``os.replace``
保证要么读到完整文件、要么旧文件保持不变；读取时先校验结构，再校验摘要。
本模块只做纯文件 I/O，不持有引擎锁，调用方负责把读写放在锁外。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MAGIC = "kv-spill"
FORMAT_VERSION = 1


class SpillError(Exception):
    """暂存文件相关错误的基类。"""


class SpillCorruptError(SpillError):
    """文件缺失、结构非法、配置不符或摘要不匹配。"""


class SpillWriteError(SpillError):
    """快照无法写入本地文件。"""


@dataclass
class SpillSnapshot:
    """引擎在锁内拍下的不可变快照，供锁外写盘。"""

    session_id: str
    revision: int
    block_size: int
    d_model: int
    vocab_size: int
    max_positions: int
    branches: list[dict]      # {id, name, tokens, block_refs}
    source_block_ids: list[int]  # 唯一物理块 id（仅供排查，恢复时不依赖）
    K: np.ndarray             # (unique_blocks, block_size, d_model)
    V: np.ndarray


@dataclass
class LoadedSpill:
    """校验通过、已从文件读回的快照。"""

    session_id: str
    revision: int
    block_size: int
    d_model: int
    vocab_size: int
    max_positions: int
    branches: list[dict]
    K: np.ndarray
    V: np.ndarray
    path: str
    file_size: int
    digest: str


def _canonical_meta(meta: dict) -> bytes:
    return json.dumps(meta, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _digest(meta_bytes: bytes, K: np.ndarray, V: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(MAGIC.encode("utf-8"))
    h.update(b"\0")
    h.update(meta_bytes)
    h.update(b"\0")
    h.update(np.ascontiguousarray(K, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(V, dtype=np.float64).tobytes())
    return h.hexdigest()


class SpillStore:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)
        # 测试注入点：让测试能在“写盘/读盘进行中”暂停 I/O，验证锁外行为
        self.before_write: Callable[[SpillSnapshot, Path], None] | None = None
        self.before_load: Callable[[Path], None] | None = None

    # ---------- 写 ----------

    def save(self, snapshot: SpillSnapshot) -> dict:
        U = int(snapshot.K.shape[0])
        branches = [
            {
                "id": b["id"],
                "name": b["name"],
                "tokens": [int(t) for t in b["tokens"]],
                "block_refs": [int(u) for u in b["block_refs"]],
            }
            for b in snapshot.branches
        ]
        total_refs = sum(len(b["block_refs"]) for b in branches)
        meta = {
            "magic": MAGIC,
            "version": FORMAT_VERSION,
            "session_id": snapshot.session_id,
            "revision": int(snapshot.revision),
            "block_size": int(snapshot.block_size),
            "d_model": int(snapshot.d_model),
            "vocab_size": int(snapshot.vocab_size),
            "max_positions": int(snapshot.max_positions),
            "branches": branches,
            "unique_blocks": U,
            "total_block_refs": total_refs,
            "source_block_ids": [int(b) for b in snapshot.source_block_ids],
            "created_at": time.time(),
        }
        meta_bytes = _canonical_meta(meta)
        digest = _digest(meta_bytes, snapshot.K, snapshot.V)

        self.directory.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", snapshot.session_id)[:80] or "session"
        path = self.directory / (
            f"{safe}-rev{snapshot.revision}-{uuid.uuid4().hex[:8]}.npz")
        tmp = self.directory / f".{path.name}.tmp-{uuid.uuid4().hex[:8]}"

        if self.before_write is not None:
            try:
                self.before_write(snapshot, path)
            except Exception as exc:  # noqa: BLE001
                raise SpillWriteError(f"写入本地暂存文件失败：{exc}") from exc

        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    meta=np.frombuffer(meta_bytes, dtype=np.uint8),
                    K=np.ascontiguousarray(snapshot.K, dtype=np.float64),
                    V=np.ascontiguousarray(snapshot.V, dtype=np.float64),
                    digest=np.array(digest),
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            # 目录项也尽量落盘；部分文件系统不支持对目录 fsync，忽略即可
            try:
                dfd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001 - 统一转成带原因的业务异常
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise SpillWriteError(f"写入本地暂存文件失败：{exc}") from exc

        return {
            "path": str(path),
            "digest": digest,
            "file_size": int(path.stat().st_size),
            "unique_blocks": U,
            "total_block_refs": total_refs,
            "created_at": float(meta["created_at"]),
        }

    # ---------- 读 ----------

    def load(self, path: str | os.PathLike[str], *, block_size: int,
             d_model: int, vocab_size: int, max_positions: int) -> LoadedSpill:
        p = Path(path)
        if self.before_load is not None:
            self.before_load(p)
        try:
            with open(p, "rb") as f:
                with np.load(f, allow_pickle=False) as z:
                    meta_raw = np.asarray(z["meta"], dtype=np.uint8)
                    if meta_raw.size > 4 * 1024 * 1024:
                        raise SpillCorruptError("暂存文件的元数据异常过大")
                    meta_bytes = meta_raw.tobytes()
                    digest = str(z["digest"].item())
                    K = np.array(z["K"], dtype=np.float64, copy=True)
                    V = np.array(z["V"], dtype=np.float64, copy=True)
            file_size = int(p.stat().st_size)
        except SpillCorruptError:
            raise
        except Exception as exc:  # noqa: BLE001 - zip/数组损坏都归为文件损坏
            raise SpillCorruptError(
                f"暂存文件无法读取（可能已损坏或被删除）：{exc}") from exc

        try:
            meta = json.loads(meta_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SpillCorruptError(f"暂存文件元数据不是合法 JSON：{exc}") from exc

        self._validate(meta, K, V, block_size=block_size, d_model=d_model,
                       vocab_size=vocab_size, max_positions=max_positions)
        want = _digest(meta_bytes, K, V)
        if not hmac.compare_digest(want, digest):
            raise SpillCorruptError("暂存文件校验和不匹配，内容已损坏")

        return LoadedSpill(
            session_id=meta["session_id"],
            revision=int(meta["revision"]),
            block_size=int(meta["block_size"]),
            d_model=int(meta["d_model"]),
            vocab_size=int(meta["vocab_size"]),
            max_positions=int(meta["max_positions"]),
            branches=meta["branches"],
            K=K,
            V=V,
            path=str(p),
            file_size=file_size,
            digest=digest,
        )

    @staticmethod
    def _validate(meta: object, K: np.ndarray, V: np.ndarray, *, block_size: int,
                  d_model: int, vocab_size: int, max_positions: int) -> None:
        def bad(msg: str) -> SpillCorruptError:
            return SpillCorruptError(f"暂存文件结构非法：{msg}")

        if not isinstance(meta, dict):
            raise bad("元数据必须是对象")
        if meta.get("magic") != MAGIC or meta.get("version") != FORMAT_VERSION:
            raise bad("魔数或格式版本不匹配")
        expected = {
            "block_size": block_size,
            "d_model": d_model,
            "vocab_size": vocab_size,
            "max_positions": max_positions,
        }
        for key, val in expected.items():
            if meta.get(key) != val:
                raise SpillCorruptError(
                    f"暂存文件与当前模型配置不一致：{key}={meta.get(key)!r}，"
                    f"当前为 {val}")
        if not isinstance(meta.get("session_id"), str) or not meta["session_id"]:
            raise bad("缺少会话 id")
        if not isinstance(meta.get("revision"), int) or isinstance(
                meta.get("revision"), bool):
            raise bad("revision 必须是整数")

        if K.ndim != 3 or V.ndim != 3 or K.shape != V.shape:
            raise bad("K/V 必须是形状相同的三维数组")
        U, B, D = K.shape
        if B != block_size or D != d_model:
            raise bad(f"K/V 形状 {(U, B, D)} 与配置不一致")
        if int(meta.get("unique_blocks", -1)) != U:
            raise bad("唯一块数与此 K/V 数组不一致")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(V)):
            raise bad("K/V 含非有限数值")

        branches = meta.get("branches")
        if not isinstance(branches, list):
            raise bad("branches 必须是数组")
        seen_ids: set[str] = set()
        used: set[int] = set()
        total_refs = 0
        for i, b in enumerate(branches):
            if not isinstance(b, dict):
                raise bad(f"第 {i} 个分支不是对象")
            bid = b.get("id")
            if not isinstance(bid, str) or not bid:
                raise bad(f"第 {i} 个分支 id 非法")
            if bid in seen_ids:
                raise bad(f"分支 id 重复：{bid}")
            seen_ids.add(bid)
            if not isinstance(b.get("name"), str):
                raise bad(f"分支 {bid} 缺少名称")
            tokens = b.get("tokens")
            refs = b.get("block_refs")
            if not isinstance(tokens, list) or not isinstance(refs, list):
                raise bad(f"分支 {bid} 的 tokens/block_refs 必须是数组")
            if len(tokens) > max_positions:
                raise bad(f"分支 {bid} 长度超出位置表")
            for t in tokens:
                if not isinstance(t, int) or isinstance(t, bool) or not 0 <= t < vocab_size:
                    raise bad(f"分支 {bid} 含非法词元 {t!r}")
            need = (len(tokens) + block_size - 1) // block_size
            if len(refs) != need:
                raise bad(f"分支 {bid} 的块表长度 {len(refs)} 与词元数不符（应为 {need}）")
            local: set[int] = set()
            for u in refs:
                if not isinstance(u, int) or isinstance(u, bool) or not 0 <= u < U:
                    raise bad(f"分支 {bid} 引用了不存在的唯一块 {u!r}")
                if u in local:
                    raise bad(f"分支 {bid} 重复引用同一块 {u}")
                local.add(u)
                used.add(u)
            total_refs += len(refs)
        if used != set(range(U)):
            raise bad("存在未被子句引用的唯一块")
        if meta.get("total_block_refs") != total_refs:
            raise bad("总块引用次数不一致")

    def delete(self, path: str | os.PathLike[str]) -> bool:
        """尽力删除文件；不抛异常，便于清理路径不掩盖原始错误。"""
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except OSError:
            return False
