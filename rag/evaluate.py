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

