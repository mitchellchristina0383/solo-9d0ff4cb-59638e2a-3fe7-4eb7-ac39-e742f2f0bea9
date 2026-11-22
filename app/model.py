"""固定权重的单层因果注意力小模型。

权重由固定种子的伪随机数生成，进程内只构造一次，所有会话共享。
K/V 只取决于 (token, position)，因此按块缓存的 K/V 与从头逐位计算
在数学上完全等价（float64 下误差 ~1e-15）。
"""
from __future__ import annotations

import math

import numpy as np


class TinyCausalAttention:
    """单层、单头因果注意力：x = E[tok] + P[pos]；k/v/q = x @ W；logits = attn @ Wo。"""

    def __init__(self, vocab_size: int = 64, d_model: int = 32,
                 max_positions: int = 128, seed: int = 1234) -> None:
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.max_positions = int(max_positions)
        rng = np.random.default_rng(seed)
        s = 1.0 / math.sqrt(self.d_model)
        self.E = rng.normal(0.0, 1.0, (self.vocab_size, self.d_model))
        self.P = rng.normal(0.0, 0.5, (self.max_positions, self.d_model))
        self.Wq = rng.normal(0.0, s, (self.d_model, self.d_model))
        self.Wk = rng.normal(0.0, s, (self.d_model, self.d_model))
        self.Wv = rng.normal(0.0, s, (self.d_model, self.d_model))
        self.Wo = rng.normal(0.0, s, (self.d_model, self.vocab_size))

    def kvq(self, token: int, position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回该 token 在指定位置上的 (k, v, q)，各为 (d,) 向量。"""
        if not 0 <= position < self.max_positions:
            raise ValueError(f"position {position} 超出位置表范围 {self.max_positions}")
        x = self.E[token] + self.P[position]
        return x @ self.Wk, x @ self.Wv, x @ self.Wq

    def logits_from_kv(self, q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """用查询向量 q 对缓存的 K/V 做因果注意力（调用方保证只传入可见前缀），返回 logits。"""
        scores = (K @ q) / math.sqrt(self.d_model)
        scores = scores - scores.max()
        w = np.exp(scores)
        w /= w.sum()
        out = w @ V
        return out @ self.Wo

    def scratch_logits(self, tokens: list[int]) -> np.ndarray:
        """参考实现：对完整词元序列从头计算最后一个位置的 logits（不走任何缓存）。"""
        n = len(tokens)
        if n == 0:
            raise ValueError("空序列无法预测")
        K = np.empty((n, self.d_model))
        V = np.empty((n, self.d_model))
        q = None
        for i, t in enumerate(tokens):
            k, v, q = self.kvq(t, i)
            K[i] = k
            V[i] = v
        return self.logits_from_kv(q, K, V)

    @staticmethod
    def top_k(logits: np.ndarray, k: int = 5) -> list[dict]:
        """softmax 后取概率最高的 k 个词元。"""
        z = logits - logits.max()
        p = np.exp(z)
        p /= p.sum()
        k = min(k, len(p))
        idx = np.argpartition(-p, k - 1)[:k]
        idx = idx[np.argsort(-p[idx])]
        return [{"token": int(i), "prob": float(p[i])} for i in idx]
