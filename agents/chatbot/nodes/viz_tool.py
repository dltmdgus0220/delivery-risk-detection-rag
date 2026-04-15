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
  "chart_type": "bar" | "line" | "pie" | "scatter" | "histogram" | "heatmap",
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


def _validate_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"SELECT만 허용됩니다. 생성된 쿼리: {sql[:100]}")
    used_tables = set(re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE))
    disallowed = used_tables - ALLOWED_TABLES
    if disallowed:
        raise ValueError(f"허용되지 않은 테이블: {disallowed}")
    return sql


def _to_base64_png(fig: go.Figure) -> str:
    """plotly Figure → base64 PNG 문자열 변환."""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=800, height=500)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _build_chart(chart_type: str, title: str, x_col: str, y_col: str, data: list[dict]) -> go.Figure:
    """데이터와 명세로 plotly Figure 생성."""
    x = [row.get(x_col) for row in data]
    y = [row.get(y_col) for row in data]

    if chart_type == "pie":
        fig = go.Figure(go.Pie(labels=x, values=y))
    elif chart_type == "line":
        fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
    elif chart_type == "scatter":
        fig = go.Figure(go.Scatter(x=x, y=y, mode="markers"))
    elif chart_type == "histogram":
        fig = go.Figure(go.Histogram(x=x))
    elif chart_type == "heatmap":
        fig = go.Figure(go.Heatmap(z=[y], x=x))
    else:  # bar (기본)
        fig = go.Figure(go.Bar(x=x, y=y))

    fig.update_layout(title=title, template="plotly_white")
    return fig

