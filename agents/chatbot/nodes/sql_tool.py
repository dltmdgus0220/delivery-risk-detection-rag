"""
SQL Tool 노드.

자연어 질문 → Text-to-SQL → SELECT 실행 → 결과 반환.

보안:
  - SELECT만 허용 (DDL/DML 차단)
  - 허용 테이블 화이트리스트: raw_reviews, processed_reviews, review_labels, review_chunks
"""

import json
import logging
import os
import re

from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

from agents.chatbot.state import AgentStateDict

