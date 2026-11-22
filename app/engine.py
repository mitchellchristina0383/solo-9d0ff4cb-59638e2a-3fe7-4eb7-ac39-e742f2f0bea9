"""会话 / 分支管理与 KV 块调度。

核心性质：
- 分叉只增加引用计数，不分配新块 —— 池满也能分叉；
- 追加前先算清所需块数（含写时复制），空闲不足则整体拒绝，
  分支内容与块占用保持不变（原子性由全局锁保证）；
- 分叉点落在块内时，子分支引用该块但只承认前 position 个槽位，
  追加时对该共享块写时复制，父分支内容不受影响；
- 关闭分支只回收引用计数归零的块。

暂停到磁盘（会话级，按请求分三步，只有第一、三步持有锁）：
1. ``begin_spill`` 在锁内拍下会话全部分支与实际 K/V 的不可变快照
   （共享物理块按 id 去重，只取一份），并把会话标为 pausing；
2. ``write_snapshot`` 锁外写文件（临时文件 + rename，原子替换）；
3. ``commit_spill`` 重新加锁，若写盘期间会话 revision 变了（追加/分叉/关闭）
   或会话已删除，则作废本次暂停、删除文件、**不释放任何块**；
   只有文件写好且会话未变，才把会话标记为 paused 并释放其全部块。
写盘和读盘期间锁不持有，其他会话（以及本会话的其他请求）照常操作。
恢复时先锁外读文件并校验，再在锁内一次性检查空闲块、分配、重建分支表，
因此恢复与别的分配并发时也只会成功或整体保持暂停。
"""
from __future__ import annotations

import itertools
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .blocks import BlockPool, CapacityError
from .model import TinyCausalAttention
from .spill import (LoadedSpill, SpillCorruptError, SpillSnapshot,
                    SpillStore)


class NotFoundError(Exception):
    pass


class ValidationError(Exception):
    pass


class ConflictError(Exception):
    """并发状态冲突：暂停作废、会话已删除或不在可恢复状态。"""


@dataclass
class Branch:
    id: str
    name: str
    tokens: list[int]
    block_ids: list[int]  # 逻辑块表：第 i 个逻辑块 -> 物理块 id
    state: str = "resident"  # resident | paused

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class Session:
    id: str
    branches: dict[str, Branch] = field(default_factory=dict)
    revision: int = 0       # 任何分支/内容变化都会自增，用于作废进行中的暂停
    state: str = "resident"  # resident | pausing | paused
    pending_pause: str | None = None
    spill: dict | None = None


@dataclass
class SpillJob:
    """一次进行中的暂停写盘。snapshot 为锁内拍下的不可变副本。"""

    id: str
    session_id: str
    revision: int
    branch_ids: list[str]
    snapshot: SpillSnapshot
    stored: dict | None = None  # 锁外写盘成功后回填，供失败清理


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class Engine:
    def __init__(self, model: TinyCausalAttention, pool: BlockPool,
                 spill_store: SpillStore | None = None) -> None:
        self.model = model
        self.pool = pool
        self.lock = threading.RLock()  # 所有状态变更都在锁内完成，保证并发追加的原子性
        self.sessions: dict[str, Session] = {}
        self._branch_seq = itertools.count(1)
        self.spill_store = spill_store or SpillStore(
            os.getenv("SPILL_DIR", str(Path(__file__).resolve().parent.parent
                                     / "spill")))
        # 进行中的暂停任务；任务本身不持有块引用，块在提交前仍归会话所有
        self.spill_jobs: dict[str, SpillJob] = {}
        # 恢复中的文件路径，避免同一文件被并发恢复两次
        self._restoring: set[str] = set()

    # ---------- 构造 ----------

    @classmethod
    def from_config(cls, vocab_size: int = 64, d_model: int = 32,
                    block_size: int = 4, num_blocks: int = 32,
                    seed: int = 1234,
                    spill_dir: str | os.PathLike[str] | None = None) -> "Engine":
        max_positions = block_size * num_blocks  # 单条分支最长不超过池容量
        model = TinyCausalAttention(vocab_size=vocab_size, d_model=d_model,
                                    max_positions=max_positions, seed=seed)
        pool = BlockPool(num_blocks=num_blocks, block_size=block_size,
                         d_model=model.d_model)
        store = SpillStore(spill_dir) if spill_dir is not None else None
        return cls(model, pool, spill_store=store)

    # ---------- 校验 ----------

    def _check_tokens(self, tokens: list[int], allow_empty: bool = False) -> None:
        if not tokens and not allow_empty:
            raise ValidationError("词元列表不能为空")
        for t in tokens:
            if not isinstance(t, int) or isinstance(t, bool):
                raise ValidationError(f"词元必须是整数，收到 {t!r}")
            if not 0 <= t < self.model.vocab_size:
                raise ValidationError(
                    f"词元 {t} 超出词表范围 [0, {self.model.vocab_size})")

    def _get_session(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise NotFoundError(f"会话不存在: {session_id}")

    def _get_branch(self, session: Session, branch_id: str) -> Branch:
        try:
            return session.branches[branch_id]
        except KeyError:
            raise NotFoundError(f"分支不存在: {branch_id}")

    # ---------- 内部工具（须在锁内调用） ----------

    def _write_token_kv(self, branch: Branch, token: int, pos: int) -> None:
        k, v, _ = self.model.kvq(token, pos)
        b = branch.block_ids[pos // self.pool.block_size]
        self.pool.write(b, pos % self.pool.block_size, k, v)

    def _gather_kv(self, branch: Branch) -> tuple[np.ndarray, np.ndarray]:
        """从块中按逻辑顺序收集该分支全部缓存的 K/V —— 预测只依赖缓存，不重算历史。"""
        n = branch.length
        B = self.pool.block_size
        K = np.empty((n, self.model.d_model))
        V = np.empty((n, self.model.d_model))
        for i, b in enumerate(branch.block_ids):
            lo, hi = i * B, min((i + 1) * B, n)
            K[lo:hi] = self.pool.K[b, :hi - lo]
            V[lo:hi] = self.pool.V[b, :hi - lo]
        return K, V

    def _predict_locked(self, branch: Branch) -> dict:
        if branch.length == 0:
            return {"argmax": None, "top": []}
        _, _, q = self.model.kvq(branch.tokens[-1], branch.length - 1)
        K, V = self._gather_kv(branch)
        logits = self.model.logits_from_kv(q, K, V)
        top = self.model.top_k(logits, k=5)
        return {"argmax": top[0]["token"], "top": top}

    def _branch_dict(self, branch: Branch, with_prediction: bool = True) -> dict:
        d = {
            "id": branch.id,
            "name": branch.name,
            "length": branch.length,
            "tokens": list(branch.tokens),
            "block_ids": list(branch.block_ids),
            "state": branch.state,
        }
        if with_prediction:
            d["prediction"] = (self._predict_locked(branch)
                               if branch.state == "resident"
                               else {"argmax": None, "top": []})
        return d

    def _session_dict(self, session: Session) -> dict:
        d = {
            "session_id": session.id,
            "state": session.state,
            "revision": session.revision,
            "branches": [self._branch_dict(b) for b in session.branches.values()],
        }
        if session.state == "paused" and session.spill:
            d["spill"] = dict(session.spill)
        return d

    def _touch(self, session: Session) -> None:
        """会话内容/结构变化时自增版本号，进行中的暂停会因此作废。"""
        session.revision += 1

    def _ensure_resident(self, session: Session, branch: Branch) -> None:
        if session.state == "paused" or branch.state == "paused":
            raise ConflictError("该分支已暂停到磁盘，请先恢复会话再操作")

    # ---------- 会话 ----------

    def create_session(self, tokens: list[int]) -> dict:
        self._check_tokens(tokens)
        with self.lock:
            need = _ceil_div(len(tokens), self.pool.block_size)
            if self.pool.num_free < need:
                raise CapacityError(
                    f"建会话需要 {need} 个块，空闲仅 {self.pool.num_free} 个")
            branch = Branch(id="b" + str(next(self._branch_seq)),
                            name="main", tokens=list(tokens),
                            block_ids=[self.pool.alloc() for _ in range(need)])
            for pos, tok in enumerate(tokens):
                self._write_token_kv(branch, tok, pos)
            session = Session(id=uuid.uuid4().hex[:8])
            session.branches[branch.id] = branch
            self.sessions[session.id] = session
            return {"session_id": session.id,
                    "branch": self._branch_dict(branch)}

    def close_session(self, session_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            self._cancel_pause_locked(session)  # 进行中的写盘一并作废
            jobs = [j for j in self.spill_jobs.values()
                    if j.session_id == session_id]
            for job in jobs:
                self.spill_jobs.pop(job.id, None)
            spill = session.spill
            self._release_session_blocks(session)
            del self.sessions[session_id]
        # 删除会话时一并清理暂存/写了一半的文件（尽力而为，不影响关闭）
        if spill:
            self.spill_store.delete(spill["path"])
        for job in jobs:
            if job.stored is not None:
                self.spill_store.delete(job.stored["path"])
        return {"closed": session_id}

    def _release_branch_blocks(self, branch: Branch) -> None:
        for b in branch.block_ids:
            self.pool.decref(b)
        branch.block_ids = []

    def _release_session_blocks(self, session: Session) -> None:
        for branch in session.branches.values():
            self._release_branch_blocks(branch)

    # ---------- 分叉 ----------

    def fork(self, session_id: str, branch_id: str, position: int,
             name: str | None = None) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            parent = self._get_branch(session, branch_id)
            self._ensure_resident(session, parent)
            if not isinstance(position, int) or not 0 <= position <= parent.length:
                raise ValidationError(
                    f"分叉点须为 [0, {parent.length}] 内的整数，收到 {position!r}")
            # 分叉只共享前缀块，不分配新块 —— 池满也能分叉
            n_blocks = _ceil_div(position, self.pool.block_size)
            shared = parent.block_ids[:n_blocks]
            for b in shared:
                self.pool.incref(b)
            child = Branch(
                id="b" + str(next(self._branch_seq)),
                name=name or f"{parent.name}@{position}",
                tokens=parent.tokens[:position],
                block_ids=list(shared),
            )
            session.branches[child.id] = child
            self._touch(session)
            return {"session_id": session_id, "branch": self._branch_dict(child)}

    # ---------- 追加（原子，受总块数限制） ----------

    def append(self, session_id: str, branch_id: str, tokens: list[int]) -> dict:
        self._check_tokens(tokens)
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            self._ensure_resident(session, branch)
            B = self.pool.block_size
            L = branch.length
            new_len = L + len(tokens)
            if new_len > self.model.max_positions:
                raise ValidationError(f"追加后长度 {new_len} 超出位置表上限 {self.model.max_positions}")

            cur_blocks = len(branch.block_ids)
            need_blocks = _ceil_div(new_len, B)
            # 末尾块未满且被共享 -> 追加前必须写时复制一块
            cow = (L % B != 0 and cur_blocks > 0
                   and self.pool.refcount[branch.block_ids[-1]] > 1)
            extra = (need_blocks - cur_blocks) + (1 if cow else 0)
            if self.pool.num_free < extra:
                # 整体拒绝：分支内容与块占用保持原样
                raise CapacityError(
                    f"追加 {len(tokens)} 个词元需要 {extra} 个空闲块，"
                    f"当前仅 {self.pool.num_free} 个，本次追加已整体拒绝")

            # 提交阶段：先写时复制，再补齐新块，最后写入 K/V
            if cow:
                old = branch.block_ids[-1]
                nb = self.pool.alloc()
                self.pool.copy_prefix(old, nb, L % B)
                self.pool.decref(old)
                branch.block_ids[-1] = nb
            while len(branch.block_ids) < need_blocks:
                branch.block_ids.append(self.pool.alloc())
            for i, tok in enumerate(tokens):
                self._write_token_kv(branch, tok, L + i)
                branch.tokens.append(tok)
            self._touch(session)
            return {"session_id": session_id,
                    "branch": self._branch_dict(branch)}

    # ---------- 查询 ----------

    def predict(self, session_id: str, branch_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            self._ensure_resident(session, branch)
            if branch.length == 0:
                raise ValidationError("空分支没有可预测的上一词元")
            return {"session_id": session_id, "branch_id": branch_id,
                    "prediction": self._predict_locked(branch)}

    def get_session(self, session_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            return self._session_dict(session)

    def list_sessions(self) -> dict:
        with self.lock:
            return {"sessions": [
                {"session_id": s.id,
                 "state": s.state,
                 "spilled": s.state == "paused" and bool(s.spill),
                 "branches": [{"id": b.id, "name": b.name, "length": b.length,
                               "state": b.state}
                              for b in s.branches.values()]}
                for s in self.sessions.values()
            ]}

    # ---------- 关闭 ----------

    def close_branch(self, session_id: str, branch_id: str) -> dict:
        with self.lock:
            session = self._get_session(session_id)
            branch = self._get_branch(session, branch_id)
            # 写盘期间允许关闭分支：分支被移除、块按引用计数回收，
            # revision 自增会让进行中的暂停在提交时作废。已暂停则须先恢复。
            if session.state == "paused":
                raise ConflictError("会话已暂停到磁盘，请先恢复再关闭分支")
            del session.branches[branch_id]
            self._release_branch_blocks(branch)  # 只有引用归零的块才真正回到空闲池
            self._touch(session)
            return {"closed": branch_id}

    # ---------- 暂停到磁盘（锁内拍快照 -> 锁外写盘 -> 锁内提交） ----------

    def _freeze_snapshot_locked(self, session: Session) -> SpillSnapshot:
        """在锁内复制会话面临的状态与 K/V；共享物理块只取一份。"""
        if any(b.state != "resident" for b in session.branches.values()):
            raise ConflictError("会话中有分支不在内存中，无法暂停")
        unique = sorted({b for br in session.branches.values()
                         for b in br.block_ids})
        index = {b: i for i, b in enumerate(unique)}
        B = self.pool.block_size
        K = np.zeros((len(unique), B, self.model.d_model))
        V = np.zeros((len(unique), B, self.model.d_model))
        for b in unique:
            K[index[b]] = self.pool.K[b]
            V[index[b]] = self.pool.V[b]
        branches = [
            {
                "id": br.id,
                "name": br.name,
                "tokens": list(br.tokens),
                "block_refs": [index[b] for b in br.block_ids],
            }
            for br in session.branches.values()
        ]
        return SpillSnapshot(
            session_id=session.id,
            revision=session.revision,
            block_size=B,
            d_model=self.model.d_model,
            vocab_size=self.model.vocab_size,
            max_positions=self.model.max_positions,
            branches=branches,
            source_block_ids=unique,
            K=K,
            V=V,
        )

    def begin_spill(self, session_id: str) -> dict:
        """第一阶段：锁内拍快照并登记任务，随后锁外写文件。

        会话在写盘期间保持原状（仍可追加/分叉），任何改动都会使 revision 自增，
        导致提交时本次暂停作废。返回的 job id 用于提交。
        """
        with self.lock:
            session = self._get_session(session_id)
            if session.state != "resident":
                raise ConflictError("该会话已在暂停或恢复流程中")
            snapshot = self._freeze_snapshot_locked(session)
            job = SpillJob(
                id="sp" + uuid.uuid4().hex[:10],
                session_id=session_id,
                revision=session.revision,
                branch_ids=[b.id for b in session.branches.values()],
                snapshot=snapshot,
            )
            self.spill_jobs[job.id] = job
            session.state = "pausing"
            session.pending_pause = job.id
            return job

    def write_snapshot(self, job: SpillJob) -> dict:
        """第二阶段（锁外）：把快照写成本地文件。"""
        stored = self.spill_store.save(job.snapshot)
        job.stored = stored
        return stored

    def commit_spill(self, session_id: str, job_id: str,
                     stored: dict) -> dict:
        """第三阶段：锁内校验会话未变，然后标记暂停并释放全部块。

        ``stored`` 是锁外 ``write_snapshot`` 的返回值。写盘期间会话若有变化
        （追加/分叉/关闭），本次暂停作废：删除刚落盘的文件，块一个都不释放。
        文件删除放在锁外，避免把 I/O 带进临界面。
        """
        with self.lock:
            job = self.spill_jobs.get(job_id)
            session = self.sessions.get(session_id)
            invalid_reason = None
            result = None
            if (job is None or job.session_id != session_id or session is None
                    or session.pending_pause != job_id):
                self.spill_jobs.pop(job_id, None)
                invalid_reason = "暂停任务不存在或已失效，本次暂停已作废"
            else:
                self.spill_jobs.pop(job_id, None)
                session.pending_pause = None
                if session.state == "pausing":
                    session.state = "resident"
                changed = (session.revision != job.revision
                           or set(session.branches) != set(job.branch_ids)
                           or any(b.state != "resident"
                                  for b in session.branches.values()))
                if changed:
                    # 写盘期间分支有变：本次暂停作废，块一个都不动
                    invalid_reason = (
                        "写盘期间会话发生了变化（追加/分叉/关闭），本次暂停已作废，"
                        "块占用未变，可重试暂停")
                else:
                    # 文件已写好且状态未变：标记暂停并释放该会话全部块
                    for branch in session.branches.values():
                        branch.state = "paused"
                    self._release_session_blocks(session)
                    session.state = "paused"
                    session.spill = {
                        "path": stored["path"],
                        "digest": stored["digest"],
                        "file_size": stored["file_size"],
                        "created_at": stored["created_at"],
                        "revision": job.revision,
                        "unique_blocks": stored["unique_blocks"],
                        "total_block_refs": stored["total_block_refs"],
                    }
                    result = {
                        "session_id": session_id,
                        "state": session.state,
                        "spill": dict(session.spill),
                        "freed_blocks": stored["unique_blocks"],
                        "branches": [self._branch_dict(b)
                                     for b in session.branches.values()],
                    }
        if invalid_reason is not None:
            self.spill_store.delete(stored["path"])
            raise ConflictError(invalid_reason)
        return result

    def _cancel_pause_locked(self, session: Session) -> None:
        """作废进行中的暂停（文件清理由调用方用取出的 job.stored 完成）。"""
        if session.pending_pause is not None:
            self.spill_jobs.pop(session.pending_pause, None)
            session.pending_pause = None
        if session.state == "pausing":
            session.state = "resident"

    def abort_spill(self, session_id: str, job_id: str) -> None:
        """写盘或提交失败时调用：清理任务与文件，会话保持原状与块占用。"""
        with self.lock:
            session = self.sessions.get(session_id)
            job = self.spill_jobs.pop(job_id, None)
            if session is not None and session.pending_pause == job_id:
                session.pending_pause = None
                if session.state == "pausing":
                    session.state = "resident"
        if job is not None and job.stored is not None:
            self.spill_store.delete(job.stored["path"])

    # ---------- 恢复（锁外读文件校验 -> 锁内一次性分配/重建） ----------

    def load_spill(self, path: str) -> LoadedSpill:
        """锁外读文件：不阻塞其他会话。结构/配置/摘要任一不符都抛 SpillCorruptError。"""
        return self.spill_store.load(
            path,
            block_size=self.pool.block_size,
            d_model=self.model.d_model,
            vocab_size=self.model.vocab_size,
            max_positions=self.model.max_positions,
        )

    def restore_session(self, path: str,
                        loaded: LoadedSpill | None = None,
                        session_id: str | None = None) -> dict:
        """从文件恢复会话：保留分支 id、共享关系与预测。

        ``loaded`` 为锁外预读的结果；未提供时在本方法内读（仍不持有锁）。
        失败时（块不足、文件损坏）会话保持暂停、不占块，可重试。
        成功后文件已读入内存，删除暂存文件（失败不影响恢复结果）。
        """
        key = os.path.abspath(path)
        with self.lock:
            if key in self._restoring:
                raise ConflictError("该暂存文件正在被另一个恢复请求使用")
            self._restoring.add(key)
        ok = False
        try:
            if loaded is None:
                loaded = self.load_spill(path)
            if os.path.abspath(loaded.path) != key:
                raise ConflictError("预读的文件内容与恢复目标不一致")
            with self.lock:
                session = self.sessions.get(loaded.session_id)
                if session_id is not None and session_id != loaded.session_id:
                    raise ConflictError("恢复目标与会话 id 不一致")
                if session is None:
                    raise SpillCorruptError(
                        "该文件对应的会话已不存在（可能已被删除），无法恢复")
                if session.state != "paused":
                    raise ConflictError("该会话不在暂停状态，无需恢复")
                if session.spill is None or os.path.abspath(
                        session.spill["path"]) != key:
                    raise ConflictError("文件与当前会话的暂停记录不匹配")
                result = self._restore_locked(session, loaded)
            ok = True
            return result
        finally:
            with self.lock:
                self._restoring.discard(key)
            if ok:
                self.spill_store.delete(key)

    def _restore_locked(self, session: Session, loaded: LoadedSpill) -> dict:
        U = int(loaded.K.shape[0])
        if self.pool.num_free < U:
            raise CapacityError(
                f"恢复需要 {U} 个空闲块，当前仅 {self.pool.num_free} 个；"
                f"会话保持暂停且不占块，可释放其他会话后重试")
        try:
            mapping = [self.pool.alloc() for _ in range(U)]
        except CapacityError as exc:  # 理论上已被上面的检查拦住
            raise CapacityError(f"恢复分配块失败：{exc}") from exc

        # 写入实际 K/V：每个唯一块只写一份，共享关系由分支块表引用同一物理块
        B = self.pool.block_size
        for u, b in enumerate(mapping):
            self.pool.K[b, :B] = loaded.K[u]
            self.pool.V[b, :B] = loaded.V[u]

        branches: dict[str, Branch] = {}
        for rec in loaded.branches:
            refs = [mapping[u] for u in rec["block_refs"]]
            branches[rec["id"]] = Branch(
                id=rec["id"], name=rec["name"], tokens=list(rec["tokens"]),
                block_ids=refs, state="resident")
        # 共享块：同一物理块被多个逻辑块/分支引用时，把引用计数补到实际次数
        counts: dict[int, int] = {}
        for br in branches.values():
            for b in br.block_ids:
                counts[b] = counts.get(b, 0) + 1
        for b, count in counts.items():
            for _ in range(count - 1):
                self.pool.incref(b)

        # 校验重建后的缓存与快照预测一致（防止块映射或数值出错）
        old_branches = session.branches
        session.branches = branches
        try:
            for rec in loaded.branches:
                want = self._prediction_from_loaded(loaded, rec)
                got = self._predict_locked(branches[rec["id"]])
                if got["argmax"] != want["argmax"]:
                    raise SpillCorruptError(
                        f"恢复后分支 {rec['name']} 的预测与暂存文件不一致")
        except Exception:
            # 回滚：释放刚分配的块（含共享引用计数）、还原分支表，会话保持暂停
            for br in branches.values():
                for b in br.block_ids:
                    self.pool.decref(b)
            session.branches = old_branches
            raise

        session.state = "resident"
        session.spill = None
        session.revision += 1
        return {
            "session_id": session.id,
            "restored_blocks": U,
            "branches": [self._branch_dict(b) for b in session.branches.values()],
        }

    def _prediction_from_loaded(self, loaded: LoadedSpill, rec: dict) -> dict:
        """用暂存文件里的 K/V 直接算该分支预测，作为恢复结果的独立参照。"""
        n = len(rec["tokens"])
        if n == 0:
            return {"argmax": None, "top": []}
        B = self.pool.block_size
        K = np.empty((n, self.model.d_model))
        V = np.empty((n, self.model.d_model))
        for i, u in enumerate(rec["block_refs"]):
            lo, hi = i * B, min((i + 1) * B, n)
            K[lo:hi] = loaded.K[u, :hi - lo]
            V[lo:hi] = loaded.V[u, :hi - lo]
        _, _, q = self.model.kvq(rec["tokens"][-1], n - 1)
        top = self.model.top_k(self.model.logits_from_kv(q, K, V), k=5)
        return {"argmax": top[0]["token"], "top": top}

    # ---------- 统计 ----------

    def stats(self) -> dict:
        with self.lock:
            return {
                "block_size": self.pool.block_size,
                "num_blocks": self.pool.num_blocks,
                "used_blocks": self.pool.num_used,
                "free_blocks": self.pool.num_free,
                "vocab_size": self.model.vocab_size,
                "d_model": self.model.d_model,
                "blocks": self.pool.snapshot(),
            }
