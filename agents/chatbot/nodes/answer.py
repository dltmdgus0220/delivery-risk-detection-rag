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

ANSWER_SYSTEM = """너는 배달앱 리뷰 분석 챗봇이야.
사용자 질문과 아래 제공된 데이터를 바탕으로 친절하고 명확하게 답변해.

답변 규칙:
- 제공된 데이터에 근거해서만 답변해. 없는 내용을 지어내지 마.
- 리뷰를 인용할 때는 쿼리의 의도와 감성이 맞는 것만 인용해. (예: "배달이 늦어요" 쿼리에 "배달이 빨랐어요" 리뷰 인용 금지)
- SQL 수치가 있으면 구체적인 숫자를 포함해서 답변해.
- 자연스러운 한국어로 답변해."""

