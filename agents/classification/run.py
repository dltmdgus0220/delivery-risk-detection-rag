"""
분류 파이프라인.

evaluate.py로 최적 모델 선정 후 아래 명령으로 전체 분류 실행.

사용 예시:
    python -m agents.classification.run --model gpt-4o-mini
    python -m agents.classification.run --model claude-haiku-4-5-20251001
    python -m agents.classification.run --model gemini-2.5-flash
"""

import argparse
import json
import logging
import os
import time

import anthropic
import google.generativeai as genai
import openai
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

CLASSIFY_SYSTEM = """너는 배달앱 리뷰 분류 전문가야.
주어진 리뷰를 읽고 아래 기준으로 분류해.

[label: 리뷰의 주요 의도 1가지 선택]
- churn: 앱 삭제·이탈 의사가 명시적으로 드러나는 리뷰 (예: "삭제할게요", "다른 앱 쓸게요")
- complaint: 서비스·앱·배달에 대한 불만이나 문제 제기 (예: 배달 지연, 앱 오류, 고객센터 불만)
- positive: 긍정적인 경험이나 만족 표현 (예: "좋아요", "편리해요", "최고입니다")

[is_suggestion: 개선 제안 포함 여부]
- true: 개선 요청이나 기능 제안이 포함된 경우 (예: "이런 기능 추가해주세요", "UI를 바꿔주세요")
- false: 그렇지 않은 경우

[중요]
- label은 반드시 churn / complaint / positive 중 하나
- is_suggestion은 label과 무관하게 독립적으로 판단
- 반드시 JSON 형식으로만 응답:
{"label": "churn|complaint|positive", "is_suggestion": bool}"""

