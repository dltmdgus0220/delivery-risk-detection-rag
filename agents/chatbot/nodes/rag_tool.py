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


def run_rag(state: AgentStateDict) -> AgentStateDict:
    """
    RAG Tool 노드.

    state["query"]로 하이브리드 검색 + 리랭킹을 수행해 top-5 청크를 반환한다.
    리랭커는 .env의 RERANKER 값 사용 (기본: albert-kor).
    label_filter가 있으면 해당 label의 청크만 검색 대상으로 한정.
    """
    query = state["query"]
    label_filter = state.get("label_filter")
    logger.info(f"RAG Tool 시작: '{query}' | 리랭커: {RERANKER} | label_filter: {label_filter}")

    rag_result = run_pipeline(query, reranker_name=RERANKER, top_n=5, label_filter=label_filter)
    logger.info(f"RAG Tool 완료: {len(rag_result)}개 청크 반환")

    return {"rag_result": rag_result}
