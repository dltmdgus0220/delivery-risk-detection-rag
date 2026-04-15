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

    - "chat" → 툴 없이 바로 answer_generator
    - 그 외  → 해당 툴 노드들을 리스트로 반환 (LangGraph가 병렬 실행)
    """
    intent = state.get("intent", ["chat"])

    if "chat" in intent:
        return "answer_generator"

    node_map = {"sql": "sql_tool", "rag": "rag_tool", "viz": "viz_tool"}
    next_nodes = [node_map[i] for i in intent if i in node_map]
    return next_nodes if next_nodes else "answer_generator"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentStateDict)

    # 노드 등록
    graph.add_node("orchestrator", orchestrate)
    graph.add_node("sql_tool", run_sql)
    graph.add_node("rag_tool", run_rag)
    graph.add_node("viz_tool", run_viz)
    graph.add_node("answer_generator", generate_answer)

    # 엣지
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", _route)  # intent → 병렬 or 단일 라우팅

    # 각 툴 → answer_generator (병렬 실행 후 모두 완료되면 수렴)
    graph.add_edge("sql_tool", "answer_generator")
    graph.add_edge("rag_tool", "answer_generator")
    graph.add_edge("viz_tool", "answer_generator")

    graph.add_edge("answer_generator", END)

    return graph.compile()


# 모듈 레벨 캐시 — 앱 시작 시 한 번만 컴파일
chatbot = build_graph()
