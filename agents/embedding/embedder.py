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

