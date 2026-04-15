"""
Answer Generator 노드.

모든 툴 실행 결과를 취합해 최종 답변과 citation을 생성한다.

intent별 동작:
  - ["chat"]        : 툴 결과 없이 LLM이 대화로 응답
  - ["rag"]         : RAG 청크를 컨텍스트로 LLM이 요약 응답 + citation
  - ["sql"]         : SQL 결과를 바탕으로 LLM이 수치 요약 응답
  - ["viz"]         : 차트 생성 완료 안내 응답
  - ["sql", "rag"]  : SQL 수치 + RAG 사례를 합산해 응답 + citation
"""

import json
import logging

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from agents.chatbot.state import AgentStateDict

logger = logging.getLogger(__name__)

