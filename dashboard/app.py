"""
배달앱 리뷰 분석 챗봇 — Streamlit 대시보드.

탭:
  💬 챗봇   — 자연어 질문 → FastAPI /chat → 답변 + citation + 차트
  🏷️ HITL  — label IS NULL 리뷰 수동 분류 (churn / complaint / positive)
"""

import base64
import os
import uuid
from io import BytesIO

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="배달앱 리뷰 분석",
    page_icon="🍔",
    layout="wide",
)

# ── 세션 초기화 ────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # {"role": "user"|"assistant", "content": str, "citations": [...], "chart": str|None}
    st.session_state.messages = []

