"""
RAG Tool 노드.

rag/pipeline.py의 run_pipeline()을 그대로 연결.
하이브리드 검색(벡터 + BM25 + RRF) → 리랭킹(albert-kor) → top-5 청크 반환.
"""

import logging

from agents.chatbot.state import AgentStateDict
from config import RERANKER
from rag.pipeline import run_pipeline

logger = logging.getLogger(__name__)

