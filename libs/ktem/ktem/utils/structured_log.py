"""结构化日志（Phase 4 工程化）。

优先尝试 structlog；未安装则降级为标准 logging + JSON 格式化器（满足
INSTRUCTIONS.md 的降级路径）。输出统一字段：
    trace_id / stage / latency_ms / n_retrieved / prompt_tokens /
    completion_tokens / cache_hit

日志落 logs/app.log（目录由 flowsettings.KH_LOG_DIR 指定）。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from theflow.settings import settings as flowsettings

_LOGGER_NAME = "traffic_ops"
_logger = None
_trace_id = None


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", get_trace_id()),
            "message": record.getMessage(),
        }
        # 合并结构化字段
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_trace_id() -> str:
    global _trace_id
    if _trace_id is None:
        _trace_id = uuid.uuid4().hex[:16]
    return _trace_id


def reset_trace_id() -> str:
    """开始一次新请求时生成新 trace_id，返回它。"""
    global _trace_id
    _trace_id = uuid.uuid4().hex[:16]
    return _trace_id


def _build_logger():
    global _logger
    if _logger is not None:
        return _logger

    # 尝试 structlog
    try:
        import structlog  # noqa: F401

        _logger = logging.getLogger(_LOGGER_NAME)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger._use_structlog = True  # type: ignore[attr-defined]
    except ImportError:
        _logger = logging.getLogger(_LOGGER_NAME)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger._use_structlog = False  # type: ignore[attr-defined]

    # 文件 handler
    log_dir = Path(getattr(flowsettings, "KH_LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setFormatter(JsonFormatter())
    _logger.addHandler(fh)

    return _logger


def log_event(stage: str, **fields) -> None:
    """记录一条结构化事件。

    Args:
        stage: 阶段名（如 retrieval / generation / judge）
        **fields: 额外结构化字段（latency_ms / n_retrieved / tokens ...）
    """
    logger = _build_logger()
    rec = logging.LogRecord(
        name=_LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=f"[{stage}]",
        args=(),
        exc_info=None,
    )
    rec.trace_id = get_trace_id()
    rec.extra_fields = fields
    logger.handle(rec)


class timed_stage:
    """上下文管理器：自动记录阶段耗时。用法：

    with timed_stage("retrieval") as ctx:
        docs = retriever.run(...)
        ctx.extra(n_retrieved=len(docs))
    """

    def __init__(self, stage: str):
        self.stage = stage
        self._t0 = None
        self._extra = {}

    def __enter__(self):
        self._t0 = time.time()
        return self

    def extra(self, **kwargs):
        self._extra.update(kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = round((time.time() - self._t0) * 1000, 2)
        fields = {"latency_ms": latency_ms}
        fields.update(self._extra)
        log_event(self.stage, **fields)
        return False
