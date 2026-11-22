"""KV 缓存块池：固定数量的物理块，每块存固定数量词元的 K/V，引用计数管理共享。"""
from __future__ import annotations

import numpy as np


class CapacityError(Exception):
    """池内空闲块不足，本次追加被整体拒绝。"""


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int, d_model: int) -> None:
        self.num_blocks = int(num_blocks)
        self.block_size = int(block_size)
        self.d_model = int(d_model)
        # 每个物理块：block_size 个槽位，每槽一份 K 和 V
        self.K = np.zeros((self.num_blocks, self.block_size, self.d_model))
        self.V = np.zeros((self.num_blocks, self.block_size, self.d_model))
        self.refcount = [0] * self.num_blocks
        self._free = list(range(self.num_blocks))  # 空闲块栈

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def alloc(self) -> int:
        if not self._free:
            raise CapacityError("块池已满")
        b = self._free.pop()
        assert self.refcount[b] == 0
        self.refcount[b] = 1
        return b

    def incref(self, b: int) -> None:
        assert self.refcount[b] > 0, f"块 {b} 不在使用中"
        self.refcount[b] += 1

    def decref(self, b: int) -> None:
        assert self.refcount[b] > 0, f"块 {b} 引用计数异常"
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self._free.append(b)

    def copy_prefix(self, src: int, dst: int, n: int) -> None:
        """把 src 块前 n 个槽位的 K/V 复制到 dst 块（写时复制用）。"""
        self.K[dst, :n] = self.K[src, :n]
        self.V[dst, :n] = self.V[src, :n]

    def write(self, b: int, slot: int, k: np.ndarray, v: np.ndarray) -> None:
        self.K[b, slot] = k
        self.V[b, slot] = v

    def snapshot(self) -> list[dict]:
        return [{"id": i, "refcount": r} for i, r in enumerate(self.refcount)]
