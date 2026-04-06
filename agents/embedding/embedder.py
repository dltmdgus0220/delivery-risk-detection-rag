"""
임베딩 공통 모듈 — run.py / evaluate.py 공유.

모델별 임베딩 로직과 상수를 한 곳에서 관리한다.
SentenceTransformer 모델은 모듈 레벨 캐시로 한 번만 로드한다.
"""

import os
import time

import numpy as np
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SUPPORTED_MODELS = [
    "text-embedding-3-small",          # OpenAI API, 1536차원
    "BAAI/bge-m3",                     # HuggingFace, 1024차원
    "intfloat/multilingual-e5-large",  # HuggingFace, 1024차원
]

MODEL_DIM = {
    "text-embedding-3-small": 1536,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
}

# SentenceTransformer 모델 캐시 (배치마다 재로딩 방지)
# 한 번 로드한 모델은 계속 재사용하기 위함.
_st_cache: dict[str, SentenceTransformer] = {}


def _get_st(model_name: str) -> SentenceTransformer:
    if model_name not in _st_cache:
        _st_cache[model_name] = SentenceTransformer(model_name)
    return _st_cache[model_name]


def embed(model_name: str, texts: list[str], is_query: bool = False) -> np.ndarray:
    """
    텍스트 리스트를 임베딩 벡터(np.ndarray)로 변환.

    Args:
        model_name : SUPPORTED_MODELS 중 하나
        texts      : 임베딩할 텍스트 리스트
        is_query   : True면 쿼리용 접두어 적용 (e5-large 전용)

    Returns:
        shape (len(texts), dim), float32, L2 정규화 완료
    """
    if model_name == "text-embedding-3-small":
        vectors = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            vectors.extend([d.embedding for d in resp.data])
            time.sleep(0.1)
        vecs = np.array(vectors, dtype=np.float32)

    elif model_name == "BAAI/bge-m3":
        st = _get_st("BAAI/bge-m3")
        vecs = st.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        vecs = np.array(vecs, dtype=np.float32)

