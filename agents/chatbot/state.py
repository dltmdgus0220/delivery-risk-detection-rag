"""
AgentStateDict — LangGraph StateGraph에 직접 전달하는 상태 타입.

모든 노드는 AgentStateDict를 입력받아 필요한 필드만 채워서 반환한다.
LangGraph는 반환된 dict를 기존 상태에 병합(merge)한다.

사용 예시:
    from langgraph.graph import StateGraph
    from agents.chatbot.state import AgentStateDict

    graph = StateGraph(AgentStateDict)
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentStateDict(TypedDict, total=False):
    """
    StateGraph(AgentStateDict)로 그래프를 선언할 때 사용.

    Fields:
        query      : 사용자 원문 질문
        intent     : Intent Classifier 분류 결과. 복수 선택 가능 → 병렬 실행
                     예) ["sql"] | ["rag"] | ["viz"] | ["sql", "rag"]
        sql_result : SQL Tool 실행 결과 (행 리스트)
        rag_result : RAG Tool 검색 결과 (청크 dict 리스트)
        chart      : Viz Tool 생성 차트 (base64 PNG 문자열, 없으면 None)
        answer     : Answer Generator 최종 답변
        citations  : 인용 출처 리스트 (review_id, rating, excerpt)
        messages   : 멀티턴 대화 히스토리. add_messages로 자동 누적 (덮어쓰지 않음)
        session_id : 세션 식별자

    total=False: 모든 필드가 선택적 → 각 노드가 자신이 채운 필드만 반환 가능.
    messages만 Annotated[..., add_messages]로 선언해 자동 누적.
    """

    query: str
    intent: list[str]
    sql_result: list[dict[str, Any]]
    rag_result: list[dict[str, Any]]
    chart: str | None
    answer: str
    citations: list[dict[str, Any]]
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
