"""
전처리 품질 평가 (LLM judge).

3종 모델(gpt-4o-mini / claude-haiku / gemini-2.0-flash)로 샘플 200건을 전처리한 뒤
GPT-4o-mini를 judge로 사용해 품질을 비교하고 리포트를 저장한다.

사용 예시:
    python -m agents.preprocessing.evaluate
    → eval_report_preprocessing.json 저장
"""

import json
import logging
import os
import time

import openai
from dotenv import load_dotenv

from agents.preprocessing.agent import preprocess_batch, sample_reviews

