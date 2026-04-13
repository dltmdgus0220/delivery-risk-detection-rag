"""
리랭커 구현 — Cross-encoder / MMR / Cohere.

하이브리드 검색 top-20을 입력받아 top-5로 압축.

사용 예시:
    from rag.reranker import rerank
    reranked = rerank("cross-encoder", query, candidates, top_n=5)
"""

import logging
import os

import numpy as np
from dotenv import load_dotenv

