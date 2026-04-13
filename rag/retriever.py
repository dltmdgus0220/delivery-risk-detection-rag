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

