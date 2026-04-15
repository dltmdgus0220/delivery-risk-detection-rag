"""
AgentState — LangGraph 노드 간 공유 상태 객체.

모든 노드는 AgentState를 입력받아 필요한 필드만 채워서 반환한다.
LangGraph는 반환된 dict를 기존 상태에 병합(merge)한다.
"""

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

