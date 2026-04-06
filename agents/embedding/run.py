"""
임베딩 파이프라인 — 전체 리뷰 청킹 + 임베딩 후 review_chunks 저장.

evaluate.py로 최적 모델 선정 후 아래 명령으로 전체 실행.

사용 예시:
    python -m agents.embedding.run --model text-embedding-3-small
    python -m agents.embedding.run --model BAAI/bge-m3
    python -m agents.embedding.run --model intfloat/multilingual-e5-large
"""

import argparse
import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from agents.embedding.chunker import chunk_review
from agents.embedding.embedder import MODEL_DIM, SUPPORTED_MODELS, embed

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

