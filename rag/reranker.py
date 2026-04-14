"""
리랭커 구현 — Cross-encoder / BGE-Reranker / MMR / Cohere.

하이브리드 검색 top-20을 입력받아 top-5로 압축.

사용 예시:
    from rag.reranker import rerank
    reranked = rerank("cross-encoder", query, candidates, top_n=5)
"""

import logging
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_RERANKERS = ["cross-encoder", "cross-encoder-mmarco", "albert-kor", "mmr", "cohere"]
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# 모델 캐시
_ce_cache = None          # 영어: cross-encoder/ms-marco-MiniLM-L6-v2
_ce_mmarco_cache = None   # 다국어: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
_albert_kor_cache = None  # 한국어 전용: bongsoo/albert-small-kor-cross-encoder-v1


# ── Cross-encoder ────────────────────────────────────────────

def _get_cross_encoder():
    """영어 Cross-encoder 로드 (캐시). MS MARCO(영어) 데이터셋으로 학습."""
    global _ce_cache
    if _ce_cache is None:
        from sentence_transformers import CrossEncoder
        _ce_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder (영어) 모델 로드 완료")
    return _ce_cache


def _get_cross_encoder_mmarco():
    """다국어 Cross-encoder 로드 (캐시). mMARCO(MS MARCO 다국어 번역, 한국어 포함)로 학습."""
    global _ce_mmarco_cache
    if _ce_mmarco_cache is None:
        from sentence_transformers import CrossEncoder
        _ce_mmarco_cache = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        logger.info("Cross-encoder (mMARCO 다국어) 모델 로드 완료")
    return _ce_mmarco_cache


def _rerank_cross_encoder(query: str, candidates: list[dict], top_n: int, ko: bool = False) -> list[dict]:
    """
    Cross-encoder로 (query, chunk) 쌍 직접 스코어링.
    쿼리와 문서를 함께 입력해 관련성을 직접 계산 → bi-encoder보다 정확.

    ko=True: 한국어 특화 모델(bongsoo) 사용
    """
    ce = _get_cross_encoder_ko() if ko else _get_cross_encoder()
    pairs = [(query, c["chunk_text"]) for c in candidates] # (쿼리, 문서) 쌍 
    scores = ce.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


# ── MMR (Maximal Marginal Relevance) ────────────────────────

def _rerank_mmr(query: str, candidates: list[dict], top_n: int, lambda_: float = 0.5) -> list[dict]:
    """
    MMR — 관련성과 다양성을 동시에 최적화.

    각 단계에서 아래를 최대화하는 문서 선택:
        λ × relevance(d, query) - (1-λ) × max_similarity(d, selected)

    λ=0.5: 관련성과 다양성 동등 반영.
    유사 리뷰가 top-5를 독점하는 문제 방지.
    """
    from agents.embedding.embedder import embed

    q_vec = embed(EMBEDDING_MODEL, [query], is_query=True)[0]
    doc_vecs = np.stack([c["embedding"] for c in candidates])

    # 코사인 유사도 (이미 L2 정규화된 벡터 → 내적 = 코사인 유사도)
    relevance = doc_vecs @ q_vec

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_n, len(candidates))):
        if not selected_indices:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            selected_vecs = doc_vecs[selected_indices]
            mmr_scores = {
                i: lambda_ * relevance[i] - (1 - lambda_) * float(np.max(doc_vecs[i] @ selected_vecs.T)) # 관련성 - 중복도
                for i in remaining
            }
            best = max(remaining, key=lambda i: mmr_scores[i])

        selected_indices.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_indices]


# ── Cohere ───────────────────────────────────────────────────

def _rerank_cohere(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Cohere Rerank API — 성능 기준선(SOTA).
    COHERE_API_KEY 필요.
    """
    import cohere
    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    docs = [c["chunk_text"] for c in candidates]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs,
        top_n=top_n,
    )
    return [candidates[r.index] for r in response.results]


# ── 메인 ────────────────────────────────────────────────────

def rerank(reranker_name: str, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """
    리랭킹 메인 함수.

    Args:
        reranker_name : "cross-encoder" | "mmr" | "cohere"
        query         : 검색 쿼리
        candidates    : hybrid_search 결과 (chunk dict 리스트, top-20)
        top_n         : 최종 반환 수 (기본 5)

    Returns:
        리랭킹된 청크 dict 리스트 (top_n개)
    """
    if reranker_name == "cross-encoder":
        return _rerank_cross_encoder(query, candidates, top_n, ko=False)
    elif reranker_name == "cross-encoder-ko":
        return _rerank_cross_encoder(query, candidates, top_n, ko=True)
    elif reranker_name == "mmr":
        return _rerank_mmr(query, candidates, top_n)
    elif reranker_name == "cohere":
        return _rerank_cohere(query, candidates, top_n)
    else:
        raise ValueError(f"지원하지 않는 리랭커: {reranker_name}. 선택 가능: {SUPPORTED_RERANKERS}")
