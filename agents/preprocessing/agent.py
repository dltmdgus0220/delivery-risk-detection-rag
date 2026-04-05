"""
전처리 Agent.

평가(evaluate.py)로 최적 모델 선정 후 아래 명령으로 전체 전처리 실행.

사용 예시:
    python -m agents.preprocessing.agent --model gpt-4o-mini
    python -m agents.preprocessing.agent --model claude-haiku-4-5-20251001
    python -m agents.preprocessing.agent --model gemini-2.0-flash
"""

import argparse
import logging
import os
import time

import anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PREPROCESS_SYSTEM = """너는 앱 리뷰 텍스트 전처리 전문가야.
주어진 리뷰에서 노이즈를 제거하고 핵심 내용만 남겨줘.

[제거 대상]
- 반복 한글 자음: ㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠㅠ, ㅜㅜ 등
- 이모지/이모티콘: 😊 ❤️ ^^ :) 등
- 과도한 반복 특수문자: !!!! → !, .... → .
- URL

[절대 금지]
- 원문 내용 추가나 변경
- 맞춤법·띄어쓰기 수정
- 문장 재구성

[중요]
전처리된 텍스트만 반환. 설명 문구 절대 금지."""

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine(os.environ["DATABASE_URL"])

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "claude-haiku-4-5-20251001",
    "gemini-2.0-flash",
]

