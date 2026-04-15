"""
Viz Tool 노드.

SQL 결과 → plotly 차트 → base64 PNG 반환.

흐름:
  - ["sql", "viz"] 조합: sql_tool이 먼저 실행되어 state에 sql_result가 있음 → 그대로 사용
  - ["viz"] 단독: state에 sql_result 없음 → 내부에서 SQL 생성 + 실행 후 차트 생성
"""

import base64
import io
import json
import logging
import os
import re

import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text

from agents.chatbot.state import AgentStateDict

