"""
FastAPI 앱 — 챗봇 + HITL 엔드포인트.

엔드포인트:
  POST /chat                챗봇 (message, session_id → answer, intent, citations, chart)
  GET  /labels/pending      HITL 대기 리뷰 목록 (label IS NULL)
  PATCH /labels/{label_id}  HITL 수동 분류 수정
  GET  /health              헬스체크
"""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from agents.chatbot.graph import chatbot

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_engine = create_engine(os.environ["DATABASE_URL"])

app = FastAPI(title="배달앱 리뷰 분석 챗봇", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 url 허용. 운영에서는 위험하지만 개발에서는 편함.
    allow_methods=["*"], # 모든 HTTP 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)


# ── 요청 / 응답 스키마 ────────────────────────────────────────

class ChatRequest(BaseModel): # /chat으로 들어오는 요청 구조. 즉 사용자가 입력하는 것.
    message: str
    session_id: str = "default"


class Citation(BaseModel): # 인용 출처 구조
    review_id: int
    excerpt: str


class ChatResponse(BaseModel): # /chat이 반환하는 응답 구조. 즉 질문에 대한 답변.
    answer: str
    intent: list[str]
    citations: list[Citation]
    chart: str | None  # base64 PNG or None


# ── 세션 히스토리 (인메모리) ──────────────────────────────────
# 실서비스에서는 Redis 등 외부 저장소로 교체
_sessions: dict[str, list] = {}


# ── 엔드포인트 ────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    챗봇 메인 엔드포인트.

    LangGraph chatbot을 실행해 answer, intent, citations, chart를 반환한다.
    """
    logger.info(f"[{req.session_id}] 질문: '{req.message}'")

    history = _sessions.get(req.session_id, [])

    try:
        result = chatbot.invoke({
            "query": req.message,
            "messages": [HumanMessage(content=req.message)] + history,
            "session_id": req.session_id,
        })
    except Exception as e:
        logger.error(f"챗봇 실행 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 세션 히스토리 업데이트
    _sessions[req.session_id] = result.get("messages", [])

    citations = [
        Citation(review_id=c["review_id"], excerpt=c["excerpt"])
        for c in result.get("citations", [])
    ]

    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", []),
        citations=citations,
        chart=result.get("chart"),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── HITL 스키마 ───────────────────────────────────────────────

class PendingReview(BaseModel):
    label_id: int
    raw_review_id: int
    review_text: str
    review_date: str | None
    rating: int | None
    classified_by: str | None


class LabelUpdate(BaseModel):
    label: str  # churn | complaint | positive


# ── HITL 엔드포인트 ──────────────────────────────────────────

VALID_LABELS = {"churn", "complaint", "positive"}


@app.get("/labels/pending", response_model=list[PendingReview])
async def get_pending_labels(limit: int = 50):
    """
    label IS NULL인 리뷰 목록 반환 (HITL 대기열).
    """
    sql = text("""
        SELECT
            rl.id          AS label_id,
            rl.raw_review_id,
            rr.review_text,
            rr.review_date::text AS review_date,
            rr.rating,
            rl.classified_by
        FROM review_labels rl
        JOIN raw_reviews rr ON rr.id = rl.raw_review_id
        WHERE rl.label IS NULL
        ORDER BY rr.review_date DESC
        LIMIT :limit
    """)
    with _engine.connect() as conn:
        rows = conn.execute(sql, {"limit": limit}).mappings().fetchall()

    return [PendingReview(**dict(row)) for row in rows]


@app.patch("/labels/{label_id}")
async def update_label(label_id: int, body: LabelUpdate):
    """
    HITL 수동 분류 수정 — label 업데이트 + human_reviewed = TRUE 기록.
    """
    if body.label not in VALID_LABELS:
        raise HTTPException(status_code=422, detail=f"label은 {VALID_LABELS} 중 하나여야 합니다.")

    sql = text("""
        UPDATE review_labels
        SET label          = :label,
            human_reviewed = TRUE,
            reviewed_at    = :now
        WHERE id = :label_id
        RETURNING id
    """)
    with _engine.begin() as conn:
        result = conn.execute(sql, {
            "label": body.label,
            "now": datetime.now(timezone.utc),
            "label_id": label_id,
        }).fetchone()

    if result is None:
        raise HTTPException(status_code=404, detail=f"label_id={label_id} 를 찾을 수 없습니다.")

    return {"label_id": label_id, "label": body.label, "human_reviewed": True}
