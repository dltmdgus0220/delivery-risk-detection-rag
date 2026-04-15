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

ORCHESTRATOR_SYSTEM = """너는 배달앱 리뷰 분석 챗봇의 오케스트레이터야.
사용자 질문을 보고, 아래 도구 중 필요한 것을 모두 골라.

도구 종류:
- "sql"  : 집계/카운트/평균/추이처럼 숫자로 답할 수 있는 질문
           예) "별점 1점 리뷰 몇 건이야?", "이번 달 이탈 리뷰 비율은?"
- "rag"  : 특정 리뷰 사례 탐색, 원인 분석, 리뷰 내용 검색
           예) "배달 지연 불만 리뷰 보여줘", "왜 앱 삭제 리뷰가 많아?"
- "viz"  : 차트, 그래프, 시각화 요청
           예) "배달 지연 트렌드 그려줘", "별점별 리뷰 수 차트로"

규칙:
- 필요한 도구를 모두 선택해 (복수 선택 가능)
- 개수 + 사례를 함께 요청하면 ["sql", "rag"] 선택
- 통계 + 시각화를 요청하면 ["sql", "viz"] 선택
- 반드시 JSON 형식으로만 응답: {"intent": ["sql"]} 또는 {"intent": ["rag", "sql"]} 등"""


_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm

