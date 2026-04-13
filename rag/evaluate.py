"""
리랭커 비교 평가 (NDCG@5, Latency P50/P95).

하이브리드 검색 top-20 → 각 리랭커 → top-5 결과를 비교.
Ground truth: 하이브리드 검색 결과에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
결정 기준: NDCG@5 차이 5% 미만이면 속도가 빠른 것 선택.

사용 예시:
    python -m rag.evaluate
    → eval_report_reranker.json 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
from dotenv import load_dotenv

from agents.embedding.embedder import openai_client
from rag.reranker import SUPPORTED_RERANKERS, rerank
from rag.retriever import hybrid_search

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORT_PATH = "eval_report_reranker.json"
EVAL_RERANKERS = ["cross-encoder", "cross-encoder-ko", "mmr"]  # cohere는 비용으로 기본 제외 (필요 시 추가)

EVAL_QUERIES: list[str] = [
    "배달이 너무 늦어요",
    "음식이 식어서 왔어요",
    "앱이 자꾸 오류가 나요",
    "배달비가 너무 비싸요",
    "고객센터가 불친절해요",
    "쿠폰 할인이 마음에 들어요",
    "앱 삭제하겠습니다",
    "다른 배달앱으로 갈아탈게요",
    "주문 취소가 안 돼요",
    "결제 오류가 발생해요",
    "포장이 엉망이에요",
    "음식 퀄리티가 좋아요",
    "배달원이 불친절했어요",
    "앱 UI가 불편해요",
    "환불 처리가 안 돼요",
    "리뷰 조작이 의심돼요",
    "배달 예상 시간이 너무 달라요",
    "최소 주문 금액이 너무 높아요",
    "적립금 사용이 복잡해요",
    "자주 쓰는 편리한 앱이에요",
]

