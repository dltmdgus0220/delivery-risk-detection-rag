"""
End-to-End RAG 파이프라인.

쿼리 → 하이브리드 검색(top-20) → 리랭킹(top-5) → 결과 반환.

사용 예시:
    python -m rag.pipeline --query "배달이 너무 늦어요"
    python -m rag.pipeline --query "앱 오류가 자꾸 나요" --reranker mmr
"""

import argparse
import logging

from dotenv import load_dotenv

from rag.reranker import SUPPORTED_RERANKERS, rerank
from rag.retriever import hybrid_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "cross-encoder"

