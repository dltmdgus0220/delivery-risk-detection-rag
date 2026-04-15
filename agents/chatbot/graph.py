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

