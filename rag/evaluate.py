"""
리랭커 비교 평가 (MRR@5, NDCG@5, Latency P50/P95).

하이브리드 검색 top-20 → 각 리랭커 → top-5 결과를 비교.
Ground truth: 하이브리드 검색 결과에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
결정 기준: NDCG@5 차이 5% 미만이면 속도가 빠른 것 선택.

MRR 사용 이유:
  벡터 검색/BM25는 "관련 있는 것만" 뽑는 게 아니라 "유사도/키워드 점수 높은 순" 으로 뽑음.
  → top-20 안에 실제 관련 없는 문서가 섞임 (키워드만 겹치거나 의미상 가까운 무관 문서).
  → 리랭커가 관련 있는 것을 top-5 안에서 얼마나 앞으로 올렸는지 MRR로 측정 가능.

사용 예시:
    python -m rag.evaluate
    → eval_report_reranker.json 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
from dotenv import load_dotenv

from agents.embedding.embedder import openai_client
from rag.reranker import SUPPORTED_RERANKERS, rerank
from rag.retriever import hybrid_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_PATH = "eval_report_reranker.json"
EVAL_RERANKERS = ["cross-encoder", "cross-encoder-ko", "mmr"]  # cohere는 비용으로 기본 제외 (필요 시 추가)

EVAL_QUERIES: list[str] = [
    "배달이 너무 늦어요",
    "음식이 식어서 왔어요",
    "앱이 자꾸 오류가 나요",
    "배달비가 너무 비싸요",
    "고객센터가 불친절해요",
    "쿠폰 할인이 마음에 들어요",
    "앱 삭제하겠습니다",
    "다른 배달앱으로 갈아탈게요",
    "주문 취소가 안 돼요",
    "결제 오류가 발생해요",
    "포장이 엉망이에요",
    "음식 퀄리티가 좋아요",
    "배달원이 불친절했어요",
    "앱 UI가 불편해요",
    "환불 처리가 안 돼요",
    "리뷰 조작이 의심돼요",
    "배달 예상 시간이 너무 달라요",
    "최소 주문 금액이 너무 높아요",
    "적립금 사용이 복잡해요",
    "자주 쓰는 편리한 앱이에요",
]


# ── Ground truth 생성 (LLM judge) ──────────────────────────

JUDGE_SYSTEM = """너는 정보 검색 품질 평가 전문가야.
주어진 쿼리와 리뷰 청크를 보고, 청크가 쿼리에 답하는 데 관련이 있는지 판단해.

관련 있음(1): 청크 내용이 쿼리 주제와 직접적으로 연관된 경우
관련 없음(0): 청크 내용이 쿼리와 무관한 경우

반드시 JSON 형식으로만 응답: {"relevant": 0 또는 1}"""


def judge_relevance(query: str, chunk_text: str, max_retries: int = 3) -> int:
    """GPT-4o-mini로 (query, chunk) 관련성 판단. 실패 시 0 반환."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"쿼리: {query}\n\n청크: {chunk_text}"},
                ],
                temperature=0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json")
            result = json.loads(raw.strip())
            return int(result.get("relevant", 0))
        except Exception as e:
            logger.warning(f"Judge 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return 0


# ── 평가 지표 ───────────────────────────────────────────────

def compute_mrr(ranked_chunks: list[dict], relevance: dict[int, int], k: int = 5) -> float:
    """MRR@k 계산. 첫 번째 관련 청크가 등장하는 순위의 역수."""
    for i, chunk in enumerate(ranked_chunks[:k], 1):
        if relevance.get(chunk["id"], 0) == 1:
            return 1.0 / i
    return 0.0


def compute_ndcg(ranked_chunks: list[dict], relevance: dict[int, int], k: int = 5) -> float:
    """NDCG@k 계산. relevance 키는 chunk id."""
    dcg = sum(
        relevance.get(c["id"], 0) / math.log2(i + 2)
        for i, c in enumerate(ranked_chunks[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

