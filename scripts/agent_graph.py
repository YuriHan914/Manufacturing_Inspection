from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, TypedDict

import streamlit as st
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from scripts.langchain_gemma import ChatLocalGemma, extract_date_range
from scripts.llm_tools import (
    _parse_log_type_answer,
    _parse_yes_no_answer,
    _validate_log_date,
    dashboard_set_query_setting,
    list_defect_types,
    log_read,
    summary_download_report,
    summary_read_inference_results,
)

# ---------------------------------------------------------------------------
# LangGraph tools: thin, conversational wrappers around scripts.llm_tools'
# pure business-logic functions. Each multi-turn tool calls interrupt() for
# whatever it still needs — LangGraph's checkpointer (below) pauses the graph
# there and resumes the SAME function call, with the paused interrupt()
# returning the user's answer, once handle_user_command() is called again
# with Command(resume=...). No hand-rolled session_state loop bookkeeping.
# ---------------------------------------------------------------------------


@tool
def dashboard_set_query_setting_tool(table: str = "", start_date: str = "", end_date: str = "") -> dict[str, Any]:
    """Change the Dashboard query settings: which table to query and the start_date/end_date
    (YYYY-MM-DD) period to load. Use this whenever the user wants to change what data the
    Dashboard queries — a new date range, a single day, or a different table."""
    if not start_date and not end_date:
        answer = interrupt("조회할 기간을 알려주세요. (예: 2026-01-01 ~ 2026-01-31)")
        start_date, end_date = extract_date_range(str(answer))
        if start_date is None:
            return {
                "status": "error",
                "tool": "dashboard_set_query_setting",
                "message": "날짜를 인식하지 못했습니다. 다시 알려주세요. (예: 2026-01-01 ~ 2026-01-31)",
                "requires_rerun": False,
            }

    if not table:
        from scripts.utils import SUPABASE_IMAGE_TABLE

        valid_tables = (SUPABASE_IMAGE_TABLE,)
        current_table = str(st.session_state.get("dashboard_query_table", SUPABASE_IMAGE_TABLE)).strip() or SUPABASE_IMAGE_TABLE
        confirmed = interrupt(
            f"{current_table} 테이블에서 {start_date}부터 {end_date}까지 조회하는 것이 맞나요? (예/아니오로 답해주세요.)"
        )
        if _parse_yes_no_answer(str(confirmed)):
            table = current_table
        else:
            chosen_table = str(interrupt(f"어떤 table을 조회하시겠어요? ({', '.join(valid_tables)} 중에서 선택해주세요.)")).strip()
            if not chosen_table:
                return {
                    "status": "error",
                    "tool": "dashboard_set_query_setting",
                    "message": "테이블명이 필요합니다.",
                    "requires_rerun": False,
                }
            table = chosen_table

    invoked_args = {"table": table, "start_date": start_date, "end_date": end_date}
    return {**dashboard_set_query_setting(table=table, start_date=start_date, end_date=end_date), "invoked_args": invoked_args}


@tool
def summary_download_report_tool() -> dict[str, Any]:
    """Generate the Summary page PDF report FILE and open the Summary page. Use this ONLY when
    the user explicitly asks to download, generate, export, or publish a report/PDF document
    (e.g. "리포트 만들어줘", "리포트 다운로드", "보고서 발행해줘"). Do NOT use this for questions
    that just ask for a number, count, or status (e.g. "불량 몇개야", "불량데이터 개수는?",
    "OK/NG 몇개야?") — those go to summary_read_inference_results_tool instead."""
    return {**summary_download_report(), "invoked_args": {}}


@tool
def summary_read_inference_results_tool(
    start_date: str = "", end_date: str = "", label: str = "", metric: str = "", limit: int = 10
) -> dict[str, Any]:
    """Answer questions about how many OK/NG (good/defect/불량) images there are, or show recent
    inference records — optionally filtered by date range (YYYY-MM-DD) or label. Use this for
    any count/status question such as "불량데이터 몇개야", "불량 개수는?", "정상 몇개야",
    "최근 결과 보여줘". This only reads data — it does NOT generate or download any report/PDF.
    `metric` narrows the answer to just one number: "good" (정상/양품/OK), "bad" (불량/NG), or
    "total" (전체) — leave it empty for the full OK/NG breakdown."""
    invoked_args = {"start_date": start_date, "end_date": end_date, "label": label, "metric": metric, "limit": limit}
    result = summary_read_inference_results(
        start_date=start_date or None,
        end_date=end_date or None,
        label=label or None,
        metric=metric or None,
        limit=limit,
    )
    return {**result, "invoked_args": invoked_args}


@tool
def list_defect_types_tool() -> dict[str, Any]:
    """List the classification model's defect (NG) categories — every class label other than
    "Normal" (the OK class). Use this whenever the user asks what defect/불량 types or
    categories exist (e.g. "불량 유형은 뭐가 있지?", "불량 종류 알려줘")."""
    return {**list_defect_types(), "invoked_args": {}}


@tool
def log_read_tool(date_value: str = "", log_type: str = "") -> dict[str, Any]:
    """Read entries from the app's /log directory for one date, optionally filtered by log
    type (error, Emergency, Warning, start, done, System done, System start, Model update).
    Use this whenever the user wants to see, filter, or review application logs."""
    if not date_value:
        answer = interrupt("어떤 날짜의 log를 보여 드릴까요?")
        date_value, _end = extract_date_range(str(answer))
        if date_value is None:
            return {
                "status": "error",
                "tool": "log_read",
                "message": "날짜를 인식하지 못했습니다. 다시 알려주세요.",
                "requires_rerun": False,
            }

    validation_message = _validate_log_date(date_value)
    if validation_message:
        return {"status": "error", "tool": "log_read", "message": validation_message, "requires_rerun": False}

    if not log_type:
        from scripts.utils import SEVERITY_ORDER

        valid_types = ", ".join(SEVERITY_ORDER.keys())
        answer = interrupt(
            "전체 로그 타입을 보여 드릴까요? 아니면 특정 로그 타입만 보여드릴까요? "
            f"({valid_types} 중 선택, 전체를 원하시면 '전체'라고 답해주세요.)"
        )
        log_type = _parse_log_type_answer(str(answer))

    invoked_args = {"date_value": date_value, "log_type": log_type, "limit": 20}
    return {**log_read(date_value=date_value, log_type=log_type, limit=20), "invoked_args": invoked_args}


TOOLS = (
    dashboard_set_query_setting_tool,
    summary_download_report_tool,
    summary_read_inference_results_tool,
    log_read_tool,
    list_defect_types_tool,
)
TOOLS_BY_NAME = {bound_tool.name: bound_tool for bound_tool in TOOLS}

UNSUPPORTED_RESULT: dict[str, Any] = {
    "status": "unsupported",
    "tool": None,
    "message": "요청하신 기능은 제공하지 않습니다.",
    "requires_rerun": False,
}


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_result: dict[str, Any] | None


def _tools_node(state: GraphState) -> dict[str, Any]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if not tool_calls:
        return {"tool_result": dict(UNSUPPORTED_RESULT)}

    call = tool_calls[0]
    tool_fn = TOOLS_BY_NAME.get(call["name"])
    if tool_fn is None:
        return {"tool_result": dict(UNSUPPORTED_RESULT)}

    result = tool_fn.invoke(call["args"])
    tool_message = ToolMessage(
        content=json.dumps(result, ensure_ascii=False, default=str),
        tool_call_id=call["id"],
        name=call["name"],
    )
    return {"messages": [tool_message], "tool_result": result}


def _agent_node(state: GraphState) -> dict[str, Any]:
    # Fixed generation settings — not configurable via Streamlit or by the LLM itself.
    # The Streamlit sidebar's own model/temperature/max_tokens controls are unrelated
    # display/config UI and are intentionally not threaded into this call.
    llm = ChatLocalGemma().bind_tools(list(TOOLS))
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def _build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", _tools_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=MemorySaver())


@st.cache_resource
def _get_graph():
    return _build_graph()


THREAD_ID_KEY = "agent_thread_id"
AWAITING_INPUT_KEY = "agent_awaiting_input"


def _new_thread_id() -> str:
    return uuid.uuid4().hex


def handle_user_command(user_prompt: str) -> dict[str, Any]:
    """Route + execute one user command against the four app tools via the LangGraph agent
    (ChatLocalGemma tool-selection + interrupt()-based tools above).

    Returns:
      - the tool's result dict once a tool actually runs to completion (status "ok"/"error")
      - the tool's own clarifying question while an interrupt is still pending
        (status "needs_input") — call this again with the user's answer as `user_prompt`
      - {"status": "unsupported", ...} when the model didn't select any of the four tools

    In every case except "needs_input", the conversation thread is rotated so the next
    command starts fresh: LangGraph's checkpointer *is* the short-term history here, and it
    is only meant to live for the duration of one open interrupt loop.
    """
    graph = _get_graph()
    thread_id = st.session_state.get(THREAD_ID_KEY) or _new_thread_id()
    st.session_state[THREAD_ID_KEY] = thread_id
    config = {"configurable": {"thread_id": thread_id}}

    if st.session_state.get(AWAITING_INPUT_KEY):
        invoke_input: Any = Command(resume=user_prompt)
    else:
        invoke_input = {"messages": [HumanMessage(content=user_prompt)], "tool_result": None}

    result = graph.invoke(invoke_input, config=config)

    pending_interrupts = result.get("__interrupt__")
    if pending_interrupts:
        st.session_state[AWAITING_INPUT_KEY] = True
        return {
            "status": "needs_input",
            "tool": None,
            "message": str(pending_interrupts[0].value),
            "requires_rerun": False,
        }

    st.session_state[AWAITING_INPUT_KEY] = False
    st.session_state[THREAD_ID_KEY] = _new_thread_id()

    return result.get("tool_result") or dict(UNSUPPORTED_RESULT)
