"""
SQL Tool 노드.

자연어 질문 → Text-to-SQL → SELECT 실행 → 결과 반환.

보안:
  - SELECT만 허용 (DDL/DML 차단)
  - 허용 테이블 화이트리스트: raw_reviews, processed_reviews, review_labels, review_chunks
"""

import json
import logging
import os
import re

from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

from agents.chatbot.state import AgentStateDict

logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

ALLOWED_TABLES = {"raw_reviews", "processed_reviews", "review_labels", "review_chunks"}

SQL_SYSTEM = """너는 배달앱 리뷰 분석 챗봇의 SQL 생성기야.
사용자 질문을 보고 PostgreSQL SELECT 쿼리를 생성해.

DB 스키마:
  raw_reviews (id, app_id, platform, reviewer_name, review_date, rating, thumbs_up_count, review_text, app_version, collected_at)
  processed_reviews (id, raw_review_id, cleaned_text, processed_by, processed_at)
  review_labels (id, raw_review_id, label, is_suggestion, classified_by, human_reviewed, reviewed_at)
    - label: 'churn'(이탈) | 'complaint'(불만) | 'positive'(긍정)
    - is_suggestion: TRUE이면 건의사항
  review_chunks (id, raw_review_id, chunk_index, chunk_text, model_name, chunked_at)

규칙:
  - SELECT만 생성 (INSERT/UPDATE/DELETE/DROP 절대 금지)
  - 허용 테이블: raw_reviews, processed_reviews, review_labels, review_chunks
  - 결과는 최대 100행으로 제한 (LIMIT 100)
  - 반드시 JSON 형식으로만 응답: {"sql": "SELECT ..."}"""


_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm

