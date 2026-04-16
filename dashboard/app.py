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


# ── 탭 ────────────────────────────────────────────────────────

tab_chat, tab_hitl = st.tabs(["💬 챗봇", "🏷️ HITL"])


# ══════════════════════════════════════════════════════════════
# 챗봇 탭
# ══════════════════════════════════════════════════════════════

with tab_chat:
    st.title("💬 리뷰 분석 챗봇")
    st.caption("배달앱 리뷰에 대해 자유롭게 질문하세요.")

    # 대화 히스토리 출력
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # 차트 렌더링
            if msg.get("chart"):
                img_bytes = base64.b64decode(msg["chart"])
                st.image(BytesIO(img_bytes), use_container_width=True)

            # citation 접기/펴기
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📎 인용 리뷰 {len(citations)}건"):
                    for c in citations:
                        st.markdown(
                            f"**[리뷰 #{c['review_id']}]** {c['excerpt']}"
                        )
