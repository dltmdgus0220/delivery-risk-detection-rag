"""
LangGraph 워크플로우 조립.

흐름:
  START → orchestrator → (intent에 따라 병렬 실행) → answer → END

라우팅:
  ["chat"]       → answer (툴 없이 바로)
  ["sql"]        → sql_tool → answer
  ["rag"]        → rag_tool → answer
  ["viz"]        → viz_tool → answer
  ["sql", "rag"] → sql_tool + rag_tool (병렬) → answer
"""

from langgraph.graph import END, START, StateGraph

from agents.chatbot.nodes.answer import generate_answer
from agents.chatbot.nodes.orchestrator import orchestrate
from agents.chatbot.nodes.rag_tool import run_rag
from agents.chatbot.nodes.sql_tool import run_sql
from agents.chatbot.nodes.viz_tool import run_viz
from agents.chatbot.state import AgentStateDict


def _route(state: AgentStateDict) -> list[str] | str:
    """
    orchestrator 결과(intent)에 따라 다음 노드를 결정.

    - "chat" → 툴 없이 바로 answer
    - 그 외  → 해당 툴 노드들을 리스트로 반환 (LangGraph가 병렬 실행)
    """
    intent = state.get("intent", ["chat"])

    if "chat" in intent:
        return "answer"

    node_map = {"sql": "sql_tool", "rag": "rag_tool", "viz": "viz_tool"}
    next_nodes = [node_map[i] for i in intent if i in node_map]
    return next_nodes if next_nodes else "answer"

