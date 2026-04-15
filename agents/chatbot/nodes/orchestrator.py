"""
Orchestrator 노드.

사용자 질문을 보고 어떤 도구를 실행할지 결정한다.
복수 선택 가능 → 해당 도구들이 병렬 실행됨.

분류 기준:
    sql  : 집계/카운트/평균/추이 (숫자로 답할 수 있는 질문)
    rag  : 특정 리뷰 사례 탐색, 원인 분석 ("왜", "어떤 리뷰")
    viz  : 차트/그래프/시각화 요청
    chat : 인사, 도움말, 일반 대화 (툴 없이 LLM이 직접 응답)
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
- "chat" : 인사, 도움말, 기능 문의 등 툴 없이 바로 답할 수 있는 일반 대화
           예) "안녕", "뭘 물어볼 수 있어?", "도움말"

규칙:
- 필요한 도구를 모두 선택해 (복수 선택 가능)
- 개수 + 사례를 함께 요청하면 ["sql", "rag"] 선택
- 시각화 요청은 ["viz"] 단독 선택 (viz가 내부적으로 SQL을 처리함)
- 일반 대화는 ["chat"] 단독 선택 (다른 도구와 함께 쓰지 않음)
- 반드시 JSON 형식으로만 응답: {"intent": ["sql"]} 또는 {"intent": ["rag", "sql"]} 등"""


_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


def orchestrate(state: AgentStateDict) -> AgentStateDict:
    """
    Orchestrator 노드.

    사용자 질문(state["query"])을 보고 실행할 도구 목록(intent)을 결정해 반환한다.
    """
    query = state["query"]
    logger.info(f"Orchestrator 시작: '{query}'")

    llm = _get_llm()
    response = llm.invoke([
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        {"role": "user", "content": query},
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json")

    result = json.loads(raw.strip())
    intent: list[str] = result.get("intent", ["rag"])

    # 유효하지 않은 값 필터링 → 아무것도 없으면 chat으로 폴백
    valid = {"sql", "rag", "viz", "chat"}
    intent = [i for i in intent if i in valid] or ["chat"]

    logger.info(f"Orchestrator 결정: {intent}")
    return {"intent": intent}
