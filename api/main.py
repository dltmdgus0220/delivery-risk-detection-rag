"""
FastAPI 앱 — 챗봇 엔드포인트.

엔드포인트:
  POST /chat        챗봇 (message, session_id → answer, intent, citations, chart)
  GET  /health      헬스체크
"""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agents.chatbot.graph import chatbot

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="배달앱 리뷰 분석 챗봇", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 url 허용. 운영에서는 위험하지만 개발에서는 편함.
    allow_methods=["*"], # 모든 HTTP 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

