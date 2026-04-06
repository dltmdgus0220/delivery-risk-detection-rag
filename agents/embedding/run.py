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


# ── DB 초기화 ──────────────────────────────────────────────

def init_table(model_name: str):
    """review_chunks 테이블 생성 (없을 때만). 차원은 모델에 따라 결정."""
    dim = MODEL_DIM[model_name]
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS review_chunks (
                id             SERIAL PRIMARY KEY,
                raw_review_id  INT REFERENCES raw_reviews(id) ON DELETE CASCADE,
                chunk_index    INT,
                chunk_text     TEXT,
                embedding      vector({dim}),
                model_name     VARCHAR(100),
                chunked_at     TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_review_chunks_embedding
            ON review_chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))
    logger.info(f"review_chunks 테이블 준비 완료 (vector({dim}))")


# ── 데이터 조회 ────────────────────────────────────────────

def get_unembedded_reviews() -> list[dict]:
    """review_chunks에 없는 processed_reviews 반환."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT p.raw_review_id, p.cleaned_text
            FROM processed_reviews p
            LEFT JOIN review_chunks c ON p.raw_review_id = c.raw_review_id
            WHERE c.id IS NULL
            ORDER BY p.raw_review_id
        """)).fetchall()

    result = [{"id": row.raw_review_id, "cleaned_text": row.cleaned_text} for row in rows]
    logger.info(f"임베딩 대상 리뷰: {len(result)}건")
    return result


# ── 저장 ───────────────────────────────────────────────────

def save_chunks(raw_review_id: int, chunks: list[str], embeddings, model_name: str):
    with engine.begin() as conn:
        for idx, (chunk_text, vec) in enumerate(zip(chunks, embeddings)):
            conn.execute(text("""
                INSERT INTO review_chunks
                    (raw_review_id, chunk_index, chunk_text, embedding, model_name)
                VALUES
                    (:raw_review_id, :chunk_index, :chunk_text, :embedding, :model_name)
            """), {
                "raw_review_id": raw_review_id,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "embedding": str(vec.tolist()),
                "model_name": model_name,
            })

