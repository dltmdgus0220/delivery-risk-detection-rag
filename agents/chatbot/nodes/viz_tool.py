"""
Viz Tool 노드.

SQL 결과 → plotly 차트 → base64 PNG 반환.

흐름:
  - ["sql", "viz"] 조합: sql_tool이 먼저 실행되어 state에 sql_result가 있음 → 그대로 사용
  - ["viz"] 단독: state에 sql_result 없음 → 내부에서 SQL 생성 + 실행 후 차트 생성
"""

import base64
import io
import json
import logging
import os
import re

import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

from agents.chatbot.state import AgentStateDict

logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

ALLOWED_TABLES = {"raw_reviews", "processed_reviews", "review_labels", "review_chunks"}

VIZ_SYSTEM = """너는 배달앱 리뷰 분석 챗봇의 시각화 생성기야.
사용자 질문과 데이터를 보고 차트 명세를 JSON으로 반환해.

DB 스키마 (SQL이 필요한 경우):
  raw_reviews (id, review_date, rating, review_text, app_version)
  review_labels (id, raw_review_id, label, is_suggestion)
    - label: 'churn' | 'complaint' | 'positive'
  processed_reviews (id, raw_review_id, cleaned_text)

응답 형식:
{
  "chart_type": "bar" | "line" | "pie",
  "title": "차트 제목",
  "x_col": "x축 컬럼명",
  "y_col": "y축 컬럼명",
  "sql": "SELECT ... (데이터가 없을 때만 포함, 있으면 생략)"
}

규칙:
- 이미 데이터가 제공된 경우 sql 필드 생략
- sql이 필요한 경우 SELECT만 작성, LIMIT 100 포함
- 허용 테이블: raw_reviews, processed_reviews, review_labels, review_chunks"""


_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm

