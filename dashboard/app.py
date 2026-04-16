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

    # 입력창
    if prompt := st.chat_input("질문을 입력하세요…"):
        # 사용자 메시지 즉시 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # API 호출
        with st.chat_message("assistant"):
            with st.spinner("분석 중…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"message": prompt, "session_id": st.session_state.session_id},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
                    st.stop()
                except Exception as e:
                    st.error(f"오류: {e}")
                    st.stop()

            answer = data.get("answer", "")
            citations = data.get("citations", [])
            chart = data.get("chart")
            intent = data.get("intent", [])

            # intent 배지
            badge_map = {"sql": "🔢 SQL", "rag": "🔍 RAG", "viz": "📊 시각화", "chat": "💬 대화"}
            badges = " · ".join(badge_map.get(i, i) for i in intent)
            if badges:
                st.caption(badges)

            st.markdown(answer)

            if chart:
                img_bytes = base64.b64decode(chart)
                st.image(BytesIO(img_bytes), use_container_width=True)

            if citations:
                with st.expander(f"📎 인용 리뷰 {len(citations)}건"):
                    for c in citations:
                        st.markdown(
                            f"**[리뷰 #{c['review_id']}]** {c['excerpt']}"
                        )

        # 히스토리에 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "chart": chart,
        })

