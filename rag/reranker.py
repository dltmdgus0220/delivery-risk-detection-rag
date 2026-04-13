"""
리랭커 구현 — Cross-encoder / MMR / Cohere.

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

SUPPORTED_RERANKERS = ["cross-encoder", "cross-encoder-ko", "mmr", "cohere"]
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Cross-encoder 모델 캐시
_ce_cache = None       # 영어: cross-encoder/ms-marco-MiniLM-L-6-v2
_ce_ko_cache = None    # 한국어: bongsoo/mmarco-mMiniLMv2-L12-H384-v1


# ── Cross-encoder ────────────────────────────────────────────

def _get_cross_encoder():
    """영어 Cross-encoder 로드 (캐시). MS MARCO(영어) 데이터셋으로 학습."""
    global _ce_cache
    if _ce_cache is None:
        from sentence_transformers import CrossEncoder
        _ce_cache = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Cross-encoder (영어) 모델 로드 완료")
    return _ce_cache


def _get_cross_encoder_ko():
    """한국어 Cross-encoder 로드 (캐시). mMARCO(MS MARCO 다국어 번역, 한국어 포함)로 학습."""
    global _ce_ko_cache
    if _ce_ko_cache is None:
        from sentence_transformers import CrossEncoder
        _ce_ko_cache = CrossEncoder("bongsoo/mmarco-mMiniLMv2-L12-H384-v1")
        logger.info("Cross-encoder (한국어) 모델 로드 완료")
    return _ce_ko_cache

