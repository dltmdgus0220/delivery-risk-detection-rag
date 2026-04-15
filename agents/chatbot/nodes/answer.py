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


_llm: ChatOpenAI | None = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    return _llm


def _build_context(state: AgentStateDict) -> str:
    """툴 결과를 LLM에 넘길 컨텍스트 문자열로 변환."""
    parts = []

    sql_result = state.get("sql_result", [])
    if sql_result:
        parts.append(f"[SQL 결과] ({len(sql_result)}행)\n" +
                     json.dumps(sql_result[:20], ensure_ascii=False, default=str)) # json.dumps: python 객체(dict, list 등) -> json 문자열로 반환. 반대는 json.loads

    rag_result = state.get("rag_result", [])
    if rag_result:
        chunks = "\n".join(
            f"[리뷰 {i+1}] review_id={c['raw_review_id']} | {c['chunk_text']}"
            for i, c in enumerate(rag_result)
        )
        parts.append(f"[관련 리뷰]\n{chunks}")

    if state.get("chart"):
        parts.append("[차트] 차트가 생성되었습니다.")

    return "\n\n".join(parts)


def _build_citations(rag_result: list[dict]) -> list[dict]:
    """RAG 결과에서 citation 리스트 생성."""
    return [
        {
            "review_id": c["raw_review_id"],
            "excerpt": c["chunk_text"][:100],
        }
        for c in rag_result
    ]

