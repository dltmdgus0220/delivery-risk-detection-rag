"""
분류 품질 평가 (LLM judge).

3종 모델(gpt-4o-mini / claude-haiku-4-5-20251001 / gemini-2.5-flash)로 샘플 200건을 분류한 뒤
gemini-2.5-flash-lite를 judge로 사용해 품질을 비교하고 리포트를 저장한다.

사용 예시:
    python -m agents.classification.evaluate
    → eval_report_classification.json 저장
"""

import json
import logging
import os
import time

import google.generativeai as genai
from dotenv import load_dotenv

from agents.classification.run import classify_batch, sample_reviews
