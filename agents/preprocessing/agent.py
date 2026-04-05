"""
전처리 Agent.

평가(evaluate.py)로 최적 모델 선정 후 아래 명령으로 전체 전처리 실행.

사용 예시:
    python -m agents.preprocessing.agent --model gpt-4o-mini
    python -m agents.preprocessing.agent --model claude-haiku-4-5-20251001
    python -m agents.preprocessing.agent --model gemini-2.0-flash
"""

import argparse
import logging
import os
import time

import anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PREPROCESS_SYSTEM = """너는 앱 리뷰 텍스트 전처리 전문가야.
주어진 리뷰에서 노이즈를 제거하고 핵심 내용만 남겨줘.

[제거 대상]
- 반복 한글 자음: ㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠ, ㅜㅜ 등
- 이모지/이모티콘: 😊 ❤️ ^^ :) 등
- 과도한 반복 특수문자: !!!! → !, .... → .
- URL

[절대 금지]
- 원문 내용 추가나 변경
- 맞춤법·띄어쓰기 수정
- 문장 재구성

[중요]
전처리된 텍스트만 반환. 설명 문구 절대 금지."""

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
    "gemini-2.0-flash",
]


def sample_reviews(n: int = 200) -> list[dict]:
    """
    별점 기반 층화 샘플링으로 n건 추출.
    별점 1~5에서 각 n//5건씩 랜덤 추출 → 별점 분포 유지.
    """
    result: list[dict] = []
    per_rating = n // 5

    with engine.connect() as conn:
        for rating in range(1, 6):
            rows = conn.execute(text("""
                SELECT id, review_text FROM raw_reviews
                WHERE rating = :rating
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"rating": rating, "limit": per_rating}).fetchall()

            result.extend({"id": row.id, "review_text": row.review_text} for row in rows)

    logger.info(f"샘플링 완료: {len(result)}건")
    return result


def get_unprocessed_reviews() -> list[dict]:
    """processed_reviews에 없는 raw_reviews 전체 반환."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT r.id, r.review_text
            FROM raw_reviews r
            LEFT JOIN processed_reviews p ON r.id = p.raw_review_id
            WHERE p.id IS NULL
            ORDER BY r.review_date DESC
        """)).fetchall()

    result = [{"id": row.id, "review_text": row.review_text} for row in rows]
    logger.info(f"미처리 리뷰: {len(result)}건")
    return result


def preprocess_one(review_text: str, model_name: str) -> str:
    """LLM 1종으로 리뷰 1건 전처리."""
    user_msg = f"리뷰: {review_text}"

    if model_name == "gpt-4o-mini":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PREPROCESS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    if model_name == "claude-haiku-4-5-20251001":
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=PREPROCESS_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return response.content[0].text.strip()

    if model_name == "gemini-2.0-flash":
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=PREPROCESS_SYSTEM,
        )
        response = model.generate_content(
            user_msg,
            generation_config=genai.GenerationConfig(temperature=0),
        )
        return response.text.strip()

    raise ValueError(f"지원하지 않는 모델: {model_name}")


def preprocess_batch(
    reviews: list[dict], model_name: str, delay: float = 0.3
) -> list[dict]:
    """
    리뷰 목록 배치 전처리.
    실패 시 원문 그대로 유지 (downstream에서 원문으로 처리됨).
    """
    results = []
    total = len(reviews)

    for i, review in enumerate(reviews, 1):
        try:
            cleaned = preprocess_one(review["review_text"], model_name)
            results.append({"id": review["id"], "cleaned_text": cleaned})
        except Exception as e:
            logger.error(f"전처리 실패 (id={review['id']}): {e}")
            results.append({"id": review["id"], "cleaned_text": review["review_text"]})

        if i % 50 == 0:
            logger.info(f"[{model_name}] {i}/{total}건 처리")
        time.sleep(delay)

    return results

