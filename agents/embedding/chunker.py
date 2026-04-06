"""
청킹 유틸리티.

기본값은 단일 청크(리뷰 전체 = 1개 청크).
Aspect 기반 청킹은 별도 함수로 구현되어 있으며,
config.CHUNKER_MODE = "aspect"로 설정하면 자동 파이프라인에서도 적용된다.

chunk_review()가 파이프라인 진입점 — config.CHUNKER_MODE에 따라 자동 분기.
"""

import json
import logging
import os
import time

import openai
from dotenv import load_dotenv

from config import CHUNKER_MODE

load_dotenv()
logger = logging.getLogger(__name__)

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ASPECT_SYSTEM = """너는 배달앱 리뷰 분석 전문가야.
주어진 리뷰를 읽고 아래 5가지 측면(aspect) 중 언급된 내용을 각각 짧은 문장으로 추출해줘.

[측면]
- 배달속도: 배달 시간, 지연, 빠름/느림
- 음식품질: 맛, 온도, 신선도, 포장 상태
- 앱UX: 앱 사용성, 오류, UI, 기능
- 가격: 배달비, 할인, 최소주문금액, 적립금
- CS: 고객센터, 환불, 응대, 처리

[규칙]
- 언급된 측면만 추출 (없는 측면은 생략)
- 원문 의미를 바꾸지 말고 간결하게 표현
- 반드시 JSON 배열로만 응답: ["문장1", "문장2", ...]
- 측면이 하나도 없으면 원문 그대로 배열에 담아 반환"""


def chunk(text: str) -> list[str]:
    """리뷰 전체를 1개 청크로 반환 (단일 청크 방식)."""
    if not text or not text.strip():
        return []
    return [text.strip()]


def chunk_by_aspect(text: str, max_retries: int = 3) -> list[str]:
    """
    GPT-4o-mini로 배달속도/음식품질/앱UX/가격/CS 단위로 분리.
    실패 시 단일 청크로 fallback.
    """
    if not text or not text.strip():
        return []

    for attempt in range(1, max_retries + 1):
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": ASPECT_SYSTEM},
                    {"role": "user", "content": f"리뷰: {text.strip()}"},
                ],
                temperature=0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json")
            parsed = json.loads(raw.strip())

            # {"chunks": [...]} 또는 [...] 형태 모두 대응
            if isinstance(parsed, list):
                chunks = parsed
            else:
                chunks = next(
                    (v for v in parsed.values() if isinstance(v, list)), None
                )
                if chunks is None:
                    raise ValueError(f"예상치 못한 응답 형식: {raw}")

            chunks = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]
            return chunks if chunks else [text.strip()]

        except Exception as e:
            logger.warning(f"Aspect 청킹 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error("Aspect 청킹 최종 실패 — 단일 청크로 fallback")
    return [text.strip()]


def chunk_review(text: str) -> list[str]:
    """
    파이프라인 진입점.
    config.CHUNKER_MODE에 따라 chunk() 또는 chunk_by_aspect() 호출.
    """
    if CHUNKER_MODE == "aspect":
        return chunk_by_aspect(text)
    return chunk(text)
