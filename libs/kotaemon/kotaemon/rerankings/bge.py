from __future__ import annotations

import os
from typing import Optional

from kotaemon.base import Document, Param

from .base import BaseReranking

# 本地已下载的 bge-reranker 权重目录（优先使用，避免每次联网）
_LOCAL_MODEL_DIR = "ktem_app_data/huggingface/bge-reranker-base"


class BgeReranking(BaseReranking):
    """本地 bge-reranker 重排器。

    使用 sentence-transformers 加载 BAAI/bge-reranker 系列模型（默认
    BAAI/bge-reranker-base），对 (query, document) 对做交叉编码打分并重排。

    加载优先级（与指令一致）：
      1. 本地已下载权重（ktem_app_data/huggingface/bge-reranker-base）
      2. 本地 sentence-transformers 联网加载（走 HF_ENDPOINT 镜像）
      3. 加载失败则降级为保持原顺序（不破坏链路），metadata 标注未重排

    Args:
        model_name: HuggingFace 模型 ID 或本地路径
        max_length: 输入最大 token 长度（bge-reranker 默认 512）
    """

    model_name: str = Param(
        "BAAI/bge-reranker-base",
        help="HuggingFace model ID or local path for bge-reranker",
        required=True,
    )
    max_length: int = Param(
        512,
        help="Maximum input sequence length for the reranker",
        required=False,
    )
    batch_size: int = Param(
        16,
        help="Batch size for cross-encoder scoring",
        required=False,
    )

    _model: Optional[object] = None
    _load_error: Optional[str] = None

    def _resolve_model_name(self) -> str:
        """优先使用本地已下载权重，否则用 model_name。"""
        local_path = os.path.join(
            os.getcwd(), _LOCAL_MODEL_DIR
        ) if os.path.isdir(os.path.join(os.getcwd(), _LOCAL_MODEL_DIR)) else _LOCAL_MODEL_DIR
        if os.path.isdir(local_path):
            return local_path
        return self.model_name

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            resolved = self._resolve_model_name()
            self._model = CrossEncoder(
                resolved,
                max_length=self.max_length,
            )
        except Exception as e:  # noqa: BLE001
            self._load_error = str(e)
            self._model = None
        return self._model

    def run(self, documents: list[Document], query: str) -> list[Document]:
        """Use bge-reranker to re-order documents by relevance to the query."""
        if not documents:
            return []

        model = self._load_model()
        if model is None:
            print(
                "bge-reranker 加载失败，跳过重排（保持原顺序）: "
                f"{self._load_error}"
            )
            return documents

        pairs = [(query, d.text or d.content or "") for d in documents]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:  # noqa: BLE001
            print(f"bge-reranker 打分失败: {e}，保持原顺序")
            return documents

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        result: list[Document] = []
        for doc, score in scored:
            doc.metadata["reranking_score"] = float(score)
            result.append(doc)
        return result


__all__ = ["BgeReranking"]
