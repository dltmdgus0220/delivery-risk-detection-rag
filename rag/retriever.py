"""
하이브리드 검색 — 벡터 검색(numpy) + BM25 키워드 검색 + RRF 병합.

벡터 검색: review_chunks 임베딩을 메모리에 로드 후 numpy 내적으로 코사인 유사도 계산.
BM25: rank_bm25로 청크 텍스트 메모리 인덱스 구성.
RRF: 두 결과를 순위 기반으로 병합 (k=60).

사용 예시:
    from rag.retriever import hybrid_search
    results = hybrid_search("배달이 너무 늦어요", top_k=20)
"""

import logging
import os

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sqlalchemy import create_engine, text

from agents.embedding.embedder import embed

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# 모듈 레벨 캐시 — DB에서 한 번만 로드
_chunks_cache: list[dict] | None = None
_bm25_cache: BM25Okapi | None = None
_doc_vecs_cache: np.ndarray | None = None


# ── 청크 로드 ───────────────────────────────────────────────

def _parse_embedding(raw) -> np.ndarray:
    """pgvector에서 반환된 embedding을 np.ndarray로 변환."""
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32)
    if isinstance(raw, list):
        return np.array(raw, dtype=np.float32)
    # string 형태 "[0.1, 0.2, ...]" 대응
    import ast
    return np.array(ast.literal_eval(str(raw)), dtype=np.float32)


def _load_chunks() -> list[dict]:
    """review_chunks 전체를 메모리에 로드 (id, raw_review_id, chunk_index, chunk_text, embedding)."""
    global _chunks_cache, _doc_vecs_cache
    if _chunks_cache is not None:
        return _chunks_cache

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, raw_review_id, chunk_index, chunk_text, embedding
            FROM review_chunks
            ORDER BY id
        """)).fetchall()

    _chunks_cache = [
        {
            "id": row.id,
            "raw_review_id": row.raw_review_id,
            "chunk_index": row.chunk_index,
            "chunk_text": row.chunk_text,
            "embedding": _parse_embedding(row.embedding),
        }
        for row in rows
    ]
    _doc_vecs_cache = np.stack([c["embedding"] for c in _chunks_cache])
    logger.info(f"review_chunks 로드 완료: {len(_chunks_cache)}개")
    return _chunks_cache


# ── 검색 ────────────────────────────────────────────────────

# 이렇게 쿼리를 통해서도 hnsw 인덱스 기반으로 가까운 벡터 조회 가능. 
# rows = conn.execute(text("""
#     SELECT id, chunk_text
#     FROM review_chunks
#     ORDER BY embedding <-> :query_embedding
#     LIMIT 5
# """), {"query_embedding": query_embedding}).fetchall()

def _vector_search(query: str, top_k: int = 20) -> list[int]:
    """쿼리 임베딩과 문서 임베딩의 코사인 유사도(내적)로 top-k 인덱스 반환."""
    _load_chunks()
    q_vec = embed(EMBEDDING_MODEL, [query], is_query=True)[0]
    scores = _doc_vecs_cache @ q_vec
    return list(np.argsort(scores)[::-1][:top_k])


def _bm25_search(query: str, top_k: int = 20) -> list[int]:
    """BM25 키워드 검색으로 top-k 인덱스 반환."""
    global _bm25_cache
    if _bm25_cache is None:
        _load_chunks()
        tokenized = [c["chunk_text"].split() for c in _chunks_cache]
        _bm25_cache = BM25Okapi(tokenized)
        logger.info("BM25 인덱스 구성 완료")

    scores = _bm25_cache.get_scores(query.split())
    return list(np.argsort(scores)[::-1][:top_k])


def _rrf(results_a: list[int], results_b: list[int], k: int = 60) -> list[int]:
    """
    Reciprocal Rank Fusion으로 두 랭킹 병합.

    RRF(d) = Σ 1 / (k + rank(d))
    k=60: 상위 랭크에 지나친 가중치 쏠림 방지. 실험적으로 검증된 기본값.
    두 결과 모두에 등장한 문서는 점수가 더 높아짐.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(results_a, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(results_b, 1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── 메인 ────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 20) -> list[dict]:
    """
    하이브리드 검색 메인 함수.

    1. 벡터 검색 (코사인 유사도) → top-k
    2. BM25 키워드 검색 → top-k
    3. RRF 병합 → 최종 top-k 반환

    Args:
        query  : 검색 쿼리 (자연어)
        top_k  : 반환할 청크 수 (기본 20)

    Returns:
        청크 dict 리스트 (id, raw_review_id, chunk_index, chunk_text, embedding 포함)
    """
    _load_chunks()

    vec_results = _vector_search(query, top_k=top_k)
    bm25_results = _bm25_search(query, top_k=top_k)
    merged = _rrf(vec_results, bm25_results)[:top_k]

    return [_chunks_cache[i] for i in merged]
