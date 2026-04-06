"""
임베딩 모델 비교 평가 (MRR@10, NDCG@10).

3종 모델로 샘플 리뷰를 임베딩한 뒤 미리 정의된 쿼리 20개를 사용해
검색 품질을 비교하고 리포트를 저장한다.

Ground truth 생성: 3종 모델 top-10 union에 대해 GPT-4o-mini judge로 관련성(0/1) 레이블링.
같은 ground truth를 모든 모델에 적용해 공정하게 비교.

사용 예시:
    python -m agents.embedding.evaluate
    → eval_report_embedding.json 저장
    → embedding_eval_results 테이블에 저장
"""

import json
import logging
import math
import os
import time

import numpy as np
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

from agents.embedding.chunker import chunk_review

