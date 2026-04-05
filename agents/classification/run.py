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

