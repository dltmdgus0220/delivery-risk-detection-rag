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


JUDGE_SYSTEM = """너는 텍스트 전처리 품질 평가 전문가야.
원문 리뷰와 전처리 결과를 비교해서 아래 3가지를 각 1~5점으로 평가해.

1. noise_removal: 반복 문자·이모지·불필요한 특수문자가 잘 제거됐는가
   5=완벽히 제거 / 3=일부 남음 / 1=거의 제거 안 됨

2. meaning_preserved: 원문의 핵심 내용과 감정이 유지됐는가
   5=완벽히 보존 / 3=일부 변형 / 1=핵심 손실

3. no_over_processing: 필요한 내용까지 삭제되지 않았는가
   5=과도한 처리 없음 / 3=일부 불필요하게 삭제 / 1=너무 많이 삭제됨

반드시 JSON 형식으로만 응답:
{"noise_removal": 점수, "meaning_preserved": 점수, "no_over_processing": 점수}"""


JUDGE_USER = """원문: {original}

전처리 결과: {preprocessed}"""

