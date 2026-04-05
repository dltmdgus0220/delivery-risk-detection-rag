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

JUDGE_SYSTEM = """너는 텍스트 분류 품질 평가 전문가야.
배달앱 리뷰와 멀티라벨 분류 결과를 보고 아래 3가지를 각 1~5점으로 평가해.

[라벨 정의]
- is_churn: 앱 삭제·이탈 의사가 명시적으로 드러나는 리뷰
- is_complaint: 서비스·앱·배달에 대한 불만이나 문제 제기
- is_suggestion: 개선 요청이나 기능 제안
- is_positive: 긍정적인 경험이나 만족 표현

[평가 기준]
1. label_accuracy: 붙은 라벨이 리뷰 내용과 실제로 일치하는가
   5=모든 라벨이 정확 / 3=일부 오분류 / 1=대부분 틀림

2. label_completeness: 해당되는 라벨이 빠짐없이 붙었는가
   5=누락 없음 / 3=일부 누락 / 1=대부분 누락

3. no_false_positive: 해당 안 되는 라벨이 잘못 붙지 않았는가
   5=오탐 없음 / 3=일부 오탐 / 1=오탐 다수

반드시 JSON 형식으로만 응답:
{"label_accuracy": 점수, "label_completeness": 점수, "no_false_positive": 점수}"""

JUDGE_USER = """리뷰: {review}

분류 결과: {labels}"""
