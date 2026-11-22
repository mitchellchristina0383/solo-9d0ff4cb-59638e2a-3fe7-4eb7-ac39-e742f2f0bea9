"""FastAPI 应用：会话/分叉/追加/预测/关闭/暂停到磁盘/恢复/占用统计 + 单页网页。

追加、分叉、关闭都是同步视图函数（FastAPI 在线程池中执行），
引擎内部用一把锁串行化状态变更，保证并发追加的原子性。

暂停分三步：``begin_spill``（锁内拍快照）-> ``write_snapshot``（锁外写盘）
-> ``commit_spill``（锁内校验版本并释放块）。写盘期间不持有引擎锁，
其他会话照常读写；若本会话在写盘期间被改动，提交时整次暂停作废。
恢复同样先锁外读文件校验，再锁内一次性分配块与重建分支表。
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .blocks import CapacityError
from .engine import (ConflictError, Engine, NotFoundError, ValidationError)
from .spill import SpillError

STATIC = Path(__file__).parent / "static" / "index.html"


class CreateSessionReq(BaseModel):
    tokens: list[int]


class ForkReq(BaseModel):
    branch_id: str
    position: int
    name: str | None = None


class AppendReq(BaseModel):
    branch_id: str
    tokens: list[int]


def _engine_from_env() -> Engine:
    return Engine.from_config(
        vocab_size=int(os.getenv("VOCAB_SIZE", "64")),
        d_model=int(os.getenv("D_MODEL", "32")),
        block_size=int(os.getenv("BLOCK_SIZE", "4")),
        num_blocks=int(os.getenv("NUM_BLOCKS", "32")),
        seed=int(os.getenv("SEED", "1234")),
        spill_dir=os.getenv("SPILL_DIR") or None,
    )


def create_app(engine: Engine | None = None) -> FastAPI:
    eng = engine or _engine_from_env()
    app = FastAPI(title="KV-Cache 分叉对比")
    app.state.engine = eng

    @app.exception_handler(NotFoundError)
    async def _404(_: Request, exc: NotFoundError):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValidationError)
    async def _400(_: Request, exc: ValidationError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(CapacityError)
    async def _409_capacity(_: Request, exc: CapacityError):
        return JSONResponse({"detail": str(exc), "reason": "capacity"},
                            status_code=409)

    @app.exception_handler(ConflictError)
    async def _409_conflict(_: Request, exc: ConflictError):
        return JSONResponse({"detail": str(exc), "reason": "conflict"},
                            status_code=409)

    @app.exception_handler(SpillError)
    async def _409_spill(_: Request, exc: SpillError):
        # 写盘/读盘失败的详细信息原样返回，页面据此说明原因并可重试
        return JSONResponse({"detail": str(exc), "reason": "spill"},
                            status_code=409)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC)

    @app.post("/api/sessions", status_code=201)
    def create_session(req: CreateSessionReq):
        return eng.create_session(req.tokens)

    @app.get("/api/sessions")
    def list_sessions():
        return eng.list_sessions()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return eng.get_session(session_id)

    @app.delete("/api/sessions/{session_id}")
    def close_session(session_id: str):
        return eng.close_session(session_id)

    @app.post("/api/sessions/{session_id}/fork", status_code=201)
    def fork(session_id: str, req: ForkReq):
        return eng.fork(session_id, req.branch_id, req.position, req.name)

    @app.post("/api/sessions/{session_id}/append")
    def append(session_id: str, req: AppendReq):
        return eng.append(session_id, req.branch_id, req.tokens)

    @app.get("/api/sessions/{session_id}/branches/{branch_id}/predict")
    def predict(session_id: str, branch_id: str):
        return eng.predict(session_id, branch_id)

    @app.delete("/api/sessions/{session_id}/branches/{branch_id}")
    def close_branch(session_id: str, branch_id: str):
        return eng.close_branch(session_id, branch_id)

    @app.post("/api/sessions/{session_id}/pause")
    def pause(session_id: str):
        """锁内拍快照，锁外写盘，锁内提交；中途会话有变则作废。"""
        job = eng.begin_spill(session_id)
        try:
            stored = eng.write_snapshot(job)   # 不持有引擎锁
            return eng.commit_spill(session_id, job.id, stored)
        except BaseException:
            eng.abort_spill(session_id, job.id)
            raise

    @app.post("/api/sessions/{session_id}/resume")
    def resume(session_id: str):
        """先锁外读文件并校验，再锁内分配块与重建会话。"""
        detail = eng.get_session(session_id)
        if not detail.get("spill"):
            raise ConflictError("该会话没有可恢复的暂存文件")
        loaded = eng.load_spill(detail["spill"]["path"])  # 不持有引擎锁
        return eng.restore_session(detail["spill"]["path"], loaded,
                                   session_id=session_id)

    @app.get("/api/stats")
    def stats():
        return eng.stats()

    return app


app = create_app()

