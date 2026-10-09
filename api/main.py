"""交通运维知识库问答系统 FastAPI 服务（Phase 4 工程化）。

提供：
- GET  /healthz   健康检查
- POST /query     检索问答（复用 kotaemon 检索+生成链路）
- POST /ingest    文档入库（复用 kotaemon IndexPipeline）

运行（PowerShell）：
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

说明：本服务只做「进程内复用」，不重复实现检索逻辑；索引固定使用
traffic_ops_kb（collection = index_4）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Swagger UI 静态资源（本地化，不依赖 cdn.jsdelivr.net——国内网络拉不到会导致 /docs 空白）
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="TrafficAgent",
    description="TrafficAgent: 城市轨道交通运维知识库问答系统（BM25+向量混合检索 + bge 重排 + ReAct Agent）",
    version="1.0.0",
    docs_url=None,   # 禁用默认 CDN 版 /docs，下方用本地资源重写
    redoc_url=None,  # redoc 同样依赖 CDN，一并禁用
)

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/docs", include_in_schema=False)
def swagger_ui():
    """本地资源版 Swagger UI（js/css 均由本服务提供，离线可用）。"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
    )


@app.get("/", include_in_schema=False)
def root():
    """服务首页：TrafficAgent 交互界面（深色运维风）。"""
    return FileResponse(_STATIC_DIR / "agent.html")


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数")
    use_rerank: bool = Field(True, description="是否启用 bge 重排")


class Citation(BaseModel):
    doc: str = Field(..., description="文档名")
    page: Optional[str] = Field(None, description="页码")
    score: Optional[float] = Field(None, description="相关性得分")
    text: str = Field("", description="片段摘要")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="答案文本")
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float = Field(..., description="总延迟（毫秒）")
    tokens: dict = Field(default_factory=dict)
    trace_id: str = Field("", description="链路追踪 id")


class IngestRequest(BaseModel):
    file_path: str = Field(..., description="待入库文件绝对路径（暂支持本地文件）")
    reindex: bool = Field(False, description="是否强制重建")


class AgentRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: str = Field("default", description="多轮会话 id")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "traffic_ops_qa"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    from .service import get_service

    svc = get_service(use_rerank=req.use_rerank, top_k=req.top_k)
    result = svc.query(req.question, top_k=req.top_k)

    citations = [
        Citation(
            doc=c["doc"],
            page=str(c["page"]) if c.get("page") is not None else None,
            score=c.get("score"),
            text=c.get("text", ""),
        )
        for c in result["citations"]
    ]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        latency_ms=result["latency_ms"],
        tokens=result["tokens"],
        trace_id=result["trace_id"],
    )


@app.post("/ingest")
def ingest(req: IngestRequest):
    """文档入库：复用 kotaemon 的 IndexDocumentPipeline（与 scripts/ingest_traffic.py 同源）。

    接收本地文件路径，调用 FileIndex 的 indexing pipeline 完成入库，
    索引固定为 traffic_ops_kb（与 Phase 1 建库一致）。
    """
    import time
    from pathlib import Path

    path = Path(req.file_path)
    if not path.exists():
        return {"status": "error", "message": f"文件不存在: {req.file_path}"}

    INDEX_NAME = "traffic_ops_kb"

    try:
        from ktem.app import BaseApp

        app = BaseApp()
        mgr = app.index_manager

        idx = None
        for each in mgr.indices:
            if each.name == INDEX_NAME:
                idx = each
                break

        if idx is None:
            from ktem.index.models import Index
            from sqlmodel import Session, select
            from ktem.db.models import engine

            with Session(engine) as sess:
                row = sess.exec(select(Index).where(Index.name == INDEX_NAME)).one_or_none()
            if row is not None:
                idx = mgr.start_index(row.id, row.name, row.config, row.index_type)
            else:
                idx = mgr.build_index(
                    name=INDEX_NAME,
                    config={"supported_file_types": ".pdf, .txt, .md", "private": True},
                    index_type="ktem.index.file.FileIndex",
                )

        settings = {f"index.options.{idx.id}.reader_mode": "default"}
        pipeline = idx.get_indexing_pipeline(settings, user_id="default")

        s = time.time()
        n_chunks = 0
        error = None
        gen = pipeline.stream([str(path)], reindex=req.reindex)
        try:
            while True:
                next(gen)
        except StopIteration as e:
            _file_ids, errors, all_docs = e.value
            n_chunks = len(all_docs)
            error = errors[0] if errors and errors[0] else None
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        return {
            "status": "ok" if error is None else "error",
            "file": path.name,
            "seconds": round(time.time() - s, 3),
            "chunks": n_chunks,
            "error": error,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/agent")
def agent_chat(req: AgentRequest):
    """ReAct Agent 问答：LangGraph StateGraph 驱动的多步检索 + 多轮记忆。"""
    from .langgraph_agent import LangGraphAgent

    agent = LangGraphAgent(top_k=req.top_k)
    res = agent.chat(req.question, session_id=req.session_id)
    return {
        "answer": res["answer"],
        "latency_ms": res["latency_ms"],
        "session_id": req.session_id,
        "trace": res["trace"],
        "engine": "langgraph",
    }


@app.post("/agent/stream")
def agent_chat_stream(req: AgentRequest):
    """ReAct Agent 流式问答：LangGraph 驱动，逐 token SSE 推送推理过程。"""
    from .langgraph_agent import LangGraphAgent

    agent = LangGraphAgent(top_k=req.top_k)
    return StreamingResponse(
        agent.stream_chat(req.question, session_id=req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
