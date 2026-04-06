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

