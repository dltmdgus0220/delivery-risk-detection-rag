"""
임베딩 모델 비교 평가 (MRR@10, NDCG@10).

3종 모델로 샘플 리뷰를 임베딩한 뒤 미리 정의된 쿼리 20개를 사용해
검색 품질을 비교하고 리포트를 저장한다.

Ground truth 생성: 3종 모델 top-10 union에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
같은 ground truth를 모든 모델에 적용해 공정하게 비교.

사용 예시:
    python -m agents.embedding.evaluate
    → eval_report_embedding.json 저장
    → embedding_eval_results 테이블에 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from agents.embedding.chunker import chunk_review
from agents.embedding.embedder import SUPPORTED_MODELS, embed, openai_client

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

EVAL_MODELS = SUPPORTED_MODELS
REPORT_PATH = "eval_report_embedding.json"

# 한국어 쿼리 20개 (유형별: 배달/음식/앱/결제/CS)
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


# ── 샘플링 ─────────────────────────────────────────────────

def sample_and_chunk(n: int = 200) -> list[dict]:
    """processed_reviews에서 층화 샘플링 후 청킹."""
    per_rating = n // 5
    chunks: list[dict] = []

    with engine.connect() as conn:
        for rating in range(1, 6):
            rows = conn.execute(text("""
                SELECT p.raw_review_id, p.cleaned_text
                FROM processed_reviews p
                JOIN raw_reviews r ON p.raw_review_id = r.id
                WHERE r.rating = :rating
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"rating": rating, "limit": per_rating}).fetchall()

            for row in rows:
                for idx, chunk_text in enumerate(chunk_review(row.cleaned_text)):
                    chunks.append({
                        "raw_review_id": row.raw_review_id,
                        "chunk_index": idx,
                        "chunk_text": chunk_text,
                    })

    logger.info(f"샘플링 완료: {n}건 → 청크 {len(chunks)}개")
    return chunks


# ── Ground truth 생성 (LLM judge) ─────────────────────────

JUDGE_SYSTEM = """너는 정보 검색 품질 평가 전문가야.
주어진 쿼리와 리뷰 청크를 보고, 청크가 쿼리에 답하는 데 관련이 있는지 판단해.

관련 있음(1): 청크 내용이 쿼리 주제와 직접적으로 연관된 경우
관련 없음(0): 청크 내용이 쿼리와 무관한 경우

반드시 JSON 형식으로만 응답: {"relevant": 0 또는 1}"""

