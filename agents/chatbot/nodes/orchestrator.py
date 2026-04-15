"""
Orchestrator 노드.

사용자 질문을 보고 어떤 도구를 실행할지 결정한다.
복수 선택 가능 → 해당 도구들이 병렬 실행됨.

분류 기준:
    sql : 집계/카운트/평균/추이 (숫자로 답할 수 있는 질문)
    rag : 특정 리뷰 사례 탐색, 원인 분석 ("왜", "어떤 리뷰")
    viz : 차트/그래프/시각화 요청
"""

import json
import logging

from langchain_openai import ChatOpenAI

from agents.chatbot.state import AgentStateDict

logger = logging.getLogger(__name__)

