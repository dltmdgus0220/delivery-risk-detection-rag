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
