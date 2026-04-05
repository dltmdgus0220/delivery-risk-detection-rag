"""
분류 파이프라인.

evaluate.py로 최적 모델 선정 후 아래 명령으로 전체 분류 실행.

사용 예시:
    python -m agents.classification.run --model gpt-4o-mini
    python -m agents.classification.run --model claude-haiku-4-5-20251001
    python -m agents.classification.run --model gemini-2.5-flash
"""

import argparse
import json
import logging
import os
import time

import anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

CLASSIFY_SYSTEM = """너는 배달앱 리뷰 분류 전문가야.
주어진 리뷰를 읽고 아래 기준으로 분류해.

[label: 리뷰의 주요 의도 1가지 선택]
- churn: 앱 삭제·이탈 의사가 명시적으로 드러나는 리뷰 (예: "삭제할게요", "다른 앱 쓸게요")
- complaint: 이탈의도가 명시적으로 드러나지 않지만 서비스·앱·배달에 대한 불만이나 문제 제기 (예: 배달 지연, 앱 오류, 고객센터 불만)
- positive: 긍정적인 경험이나 만족 표현 (예: "좋아요", "편리해요", "최고입니다")

[is_suggestion: 개선 제안 포함 여부]
- true: 개선 요청이나 기능 제안이 포함된 경우 (예: "이런 기능 추가해주세요", "UI를 바꿔주세요")
- false: 그렇지 않은 경우

[중요]
- label은 반드시 churn / complaint / positive 중 하나
- is_suggestion은 label과 무관하게 독립적으로 판단
- 반드시 JSON 형식으로만 응답:
{"label": "churn|complaint|positive", "is_suggestion": bool}"""

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
]


def get_unclassified_reviews() -> list[dict]:
    """review_labels에 없는 processed_reviews 전체 반환."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.raw_review_id, p.cleaned_text
            FROM processed_reviews p
            LEFT JOIN review_labels l ON p.raw_review_id = l.raw_review_id
            WHERE l.id IS NULL
            ORDER BY p.raw_review_id DESC
        """)).fetchall()

    result = [{"id": row.raw_review_id, "cleaned_text": row.cleaned_text} for row in rows]
    logger.info(f"미분류 리뷰: {len(result)}건")
    return result


def sample_reviews(n: int = 200) -> list[dict]:
    """
    별점 기반 층화 샘플링으로 n건 추출.
    evaluate.py에서 모델 비교용으로 사용.
    """
    result: list[dict] = []
    per_rating = n // 5

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

            result.extend({"id": row.raw_review_id, "cleaned_text": row.cleaned_text} for row in rows)

    logger.info(f"샘플링 완료: {len(result)}건")
    return result

