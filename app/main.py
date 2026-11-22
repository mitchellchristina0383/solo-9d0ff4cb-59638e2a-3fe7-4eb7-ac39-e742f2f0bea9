"""FastAPI 应用：会话/分叉/追加/预测/关闭/占用统计的 HTTP 接口 + 单页网页。

追加、分叉、关闭都是同步视图函数（FastAPI 在线程池中执行），
引擎内部用一把锁串行化状态变更，保证并发追加的原子性。
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .blocks import CapacityError
from .engine import Engine, NotFoundError, ValidationError

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
    async def _409(_: Request, exc: CapacityError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

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

    @app.get("/api/stats")
    def stats():
        return eng.stats()

    return app


app = create_app()
