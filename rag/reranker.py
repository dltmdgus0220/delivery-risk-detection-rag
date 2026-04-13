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

SUPPORTED_RERANKERS = ["cross-encoder", "mmr", "cohere"]
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Cross-encoder 모델 캐시
_ce_cache = None

