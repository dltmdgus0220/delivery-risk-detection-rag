"""
청킹 유틸리티.

기본값은 단일 청크(리뷰 전체 = 1개 청크).
Aspect 기반 청킹은 별도 함수로 구현되어 있으며,
config.CHUNKER_MODE = "aspect"로 설정하면 자동 파이프라인에서도 적용된다.

chunk_review()가 파이프라인 진입점 — config.CHUNKER_MODE에 따라 자동 분기.
"""

import json
import logging
import os
import time

import openai
from dotenv import load_dotenv

from config import CHUNKER_MODE
