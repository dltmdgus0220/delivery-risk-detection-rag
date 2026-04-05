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

