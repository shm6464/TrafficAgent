from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional, Sequence

from kotaemon.base import BaseComponent, Document, RetrievedDocument
from kotaemon.embeddings import BaseEmbeddings
from kotaemon.storages import BaseDocumentStore, BaseVectorStore


def _char_bigram_tokens(text: str) -> list[str]:
    """将中文文本切分为 char-bigram 词元。

    中文没有天然空格分词，这里用「字符二元组」近似词元，兼顾单字与双字
    组合的召回能力，纯 Python 无外部依赖。对英文/数字保留原样并按空白切分。

    Args:
        text: 输入文本

    Returns:
        词元列表（小写、去空白）
    """
    text = (text or "").lower()
    # 英文/数字段与中文字符混合处理：逐字符遍历，中文走 bigram，连续 ASCII 走单词
    tokens: list[str] = []
    i = 0
    n = len(text)
    buf = ""
    while i < n:
        ch = text[i]
        if ch.isascii() and (ch.isalnum()):
            buf += ch
            i += 1
            continue
        # 非 ASCII 字母数字（中文等 CJK 字符）
        if buf:
            tokens.append(buf)
            buf = ""
        if ch.isspace():
            i += 1
            continue
        # 单字符本身 + 与前一个 CJK 字符组成 bigram
        tokens.append(ch)
        if i + 1 < n and text[i + 1] >= "\u4e00" and text[i + 1] <= "\u9fff":
            tokens.append(ch + text[i + 1])
        i += 1
    if buf:
        tokens.append(buf)
    return tokens


class OkapiBM25:
    """纯 Python Okapi BM25 实现，无外部依赖。

    参考标准 BM25 公式：
        score(D, Q) = Σ IDF(q_i) * [ f(q_i,D) * (k1+1) ]
                        / [ f(q_i,D) + k1 * (1 - b + b * |D| / avgdl) ]
    其中 IDF(q_i) = ln(1 + (N - n(q_i) + 0.5) / (n(q_i) + 0.5))
    """

    def __init__(
        self,
        corpus: Iterable[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus: list[list[str]] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self._n_docs = 0

        self._initialize(corpus)

    def _initialize(self, corpus: Iterable[str]):
        nd: dict[str, int] = {}
        num_doc = 0
        for text in corpus:
            tokens = _char_bigram_tokens(text)
            self.corpus.append(tokens)
            self.doc_len.append(len(tokens))
            num_doc += 1
            for tok in set(tokens):
                nd[tok] = nd.get(tok, 0) + 1

        self._n_docs = num_doc
        self.doc_freqs = nd
        self.avgdl = (
            sum(self.doc_len) / num_doc if num_doc else 0.0
        )

        # 计算 IDF
        self.idf = {}
        for word, freq in nd.items():
            idf = math.log(1 + (num_doc - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
        self.epsilon = 0.25

    def get_scores(self, query: str) -> list[float]:
        q_tokens = _char_bigram_tokens(query)
        scores = [0.0] * self._n_docs

        for q in q_tokens:
            q_freq = self.idf.get(q, 0.0)
            if q_freq == 0.0:
                continue
            for i, doc_tokens in enumerate(self.corpus):
                freq = doc_tokens.count(q)
                if freq == 0:
                    continue
                dl = self.doc_len[i]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * dl / (self.avgdl or 1.0)
                )
                scores[i] += q_freq * (freq * (self.k1 + 1)) / denom

        return scores


class BM25Retriever(BaseComponent):
    """基于纯 Python OkapiBM25 的检索器。

    从 docstore 全量拉取文档文本构建 BM25 索引（语料规模在运维知识库场景下
    完全可接受），对 query 计算 BM25 分数并返回 Top-K。避免了 lancedb FTS
    的 en_stem 英文分词器对中文失效、以及多线程调用在 Windows 上卡死的问题。

    Args:
        doc_store: 文档存储（用于获取语料文本与文档对象）
    """

    doc_store: BaseDocumentStore
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 5

    _corpus: Optional[list[str]] = None
    _corpus_docs: Optional[list[Document]] = None
    _bm25: Optional[OkapiBM25] = None

    def _build_index(self):
        docs = self.doc_store.get_all()
        # get_all 可能未实现，回退到空
        if docs is None:
            docs = []
        self._corpus_docs = docs
        self._corpus = [d.text for d in docs]
        self._bm25 = OkapiBM25(self._corpus, k1=self.k1, b=self.b)

    def run(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> list[RetrievedDocument]:
        if top_k is None:
            top_k = self.top_k
        if self._bm25 is None:
            self._build_index()

        if not self._corpus_docs:
            return []

        scores = self._bm25.get_scores(query)
        # 按分数降序取 top_k（过滤 0 分）
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        result: list[RetrievedDocument] = []
        for i in ranked:
            doc = self._corpus_docs[i]
            score = scores[i]
            if score <= 0:
                continue
            result.append(
                RetrievedDocument(**doc.to_dict(), score=float(score))
            )
        return result


class HybridFusionRetriever(BaseComponent):
    """向量检索 + BM25 检索的混合融合检索器。

    使用 RRF（Reciprocal Rank Fusion）融合两路 Top-K 结果，也支持加权分数
    融合（alpha * 归一化向量分 + (1-alpha) * 归一化 BM25 分）。

    Args:
        embedding: 向量 embedding 模型
        vector_store: 向量存储
        doc_store: 文档存储（BM25 语料来源）
        bm25_retriever: BM25 检索器（可选，默认内部构建）
        top_k: 最终返回数量
        fusion_method: "rrf" 或 "weighted"
        alpha: 加权融合时向量分数的权重（默认 0.5）
        rrf_k: RRF 中的常数 k（默认 60）
    """

    embedding: BaseEmbeddings
    vector_store: BaseVectorStore
    doc_store: BaseDocumentStore
    bm25_retriever: Optional[BM25Retriever] = None
    rerankers: Sequence = []
    top_k: int = 5
    fusion_method: str = "rrf"  # rrf | weighted
    alpha: float = 0.5
    rrf_k: int = 60

    _cached_bm25: Optional[BM25Retriever] = None

    def _get_bm25(self) -> BM25Retriever:
        """返回 BM25 检索器，内部缓存实例以避免每次查询重建索引。

        之前的实现每次 run 都新建 BM25Retriever，导致每道查询都对全量语料
        重新做 char-bigram 分词 + 倒排统计，CPU 爆满（Phase 3 评测暴露）。
        """
        if self.bm25_retriever is not None:
            return self.bm25_retriever
        if self._cached_bm25 is None:
            self._cached_bm25 = BM25Retriever(
                doc_store=self.doc_store, top_k=self.top_k * 2
            )
        return self._cached_bm25

    def _vector_search(
        self, query: str, top_k: int
    ) -> list[RetrievedDocument]:
        emb = self.embedding.run(query)[0].embedding
        _, scores, ids = self.vector_store.query(embedding=emb, top_k=top_k)
        docs = self.doc_store.get(ids)
        return [
            RetrievedDocument(**doc.to_dict(), score=float(score))
            for doc, score in zip(docs, scores)
        ]

    def _rrf_fusion(
        self,
        vector_docs: list[RetrievedDocument],
        bm25_docs: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}
        for rank, doc in enumerate(vector_docs):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            doc_map[doc.doc_id] = doc
        for rank, doc in enumerate(bm25_docs):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + 1.0 / (
                self.rrf_k + rank + 1
            )
            doc_map.setdefault(doc.doc_id, doc)

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        result = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id]
            doc_dict = doc.to_dict()
            doc_dict.pop("score", None)
            doc_dict["metadata"] = dict(doc_dict.get("metadata") or {})
            doc_dict["metadata"]["rrf_score"] = scores[doc_id]
            result.append(RetrievedDocument(**doc_dict, score=scores[doc_id]))
        return result

    def _weighted_fusion(
        self,
        vector_docs: list[RetrievedDocument],
        bm25_docs: list[RetrievedDocument],
        top_k: int,
    ) -> list[RetrievedDocument]:
        def _norm(docs: list[RetrievedDocument]) -> dict[str, float]:
            if not docs:
                return {}
            mx = max(d.score for d in docs) or 1.0
            return {d.doc_id: d.score / mx for d in docs}

        v_norm = _norm(vector_docs)
        b_norm = _norm(bm25_docs)

        merged: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}
        for doc in vector_docs:
            doc_map[doc.doc_id] = doc
        for doc in bm25_docs:
            doc_map.setdefault(doc.doc_id, doc)

        for doc_id in doc_map:
            merged[doc_id] = self.alpha * v_norm.get(doc_id, 0.0) + (
                1 - self.alpha
            ) * b_norm.get(doc_id, 0.0)

        ranked_ids = sorted(merged, key=lambda i: merged[i], reverse=True)[:top_k]
        result = []
        for doc_id in ranked_ids:
            doc = doc_map[doc_id]
            doc_dict = doc.to_dict()
            doc_dict.pop("score", None)
            result.append(RetrievedDocument(**doc_dict, score=merged[doc_id]))
        return result

    def run(
        self, query: str, top_k: Optional[int] = None, **kwargs
    ) -> list[RetrievedDocument]:
        if top_k is None:
            top_k = self.top_k
        # 两路各取 top_k*2 再融合，保证召回充足
        candidate_k = top_k * 2

        vector_docs = self._vector_search(query, candidate_k)
        bm25_docs = self._get_bm25().run(query, top_k=candidate_k)

        if self.fusion_method == "weighted":
            result = self._weighted_fusion(vector_docs, bm25_docs, top_k)
        else:
            result = self._rrf_fusion(vector_docs, bm25_docs, top_k)

        # 两阶段重排：融合后追加 reranker（如 bge-reranker）
        if self.rerankers and query:
            for reranker in self.rerankers:
                result = reranker.run(documents=result, query=query)
        return result


__all__ = ["OkapiBM25", "BM25Retriever", "HybridFusionRetriever"]
