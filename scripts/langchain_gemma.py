from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from scripts.app_mcp import _extract_json_object
from scripts.local_gemma_model import generate_response
from scripts.tool_router import route_tool

UNSUPPORTED_MESSAGE = "요청하신 기능은 제공하지 않습니다."

# Tools whose arguments include a date — pre-filled from the user's initial message (if a
# date is present in it) so a query like "1월 데이터 몇개야" doesn't need a follow-up question.
_DATE_ARG_FIELDS: dict[str, tuple[str, ...]] = {
    "dashboard_set_query_setting_tool": ("start_date", "end_date"),
    "summary_read_inference_results_tool": ("start_date", "end_date"),
    "log_read_tool": ("date_value",),
}

# summary_read_inference_results_tool: which single number the question is actually asking
# for, so the answer states that number instead of always dumping the full OK/NG breakdown.
_METRIC_ARG_TOOLS = {"summary_read_inference_results_tool"}
_GOOD_METRIC_KEYWORDS = ("양품", "정상", "굿", "ok")
_BAD_METRIC_KEYWORDS = ("불량", "ng", "디펙트", "defect")
_TOTAL_METRIC_KEYWORDS = ("전체", "총")


def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _infer_metric(user_text: str) -> str:
    """Detect whether the question asks specifically for the good/bad/total count.

    Substring matching (no word boundaries), which suits Korean here since these are all
    short, fixed domain terms. Returns "" (no single metric — answer with the full
    breakdown) when the question is ambiguous or mentions more than one of good/bad/total.
    """
    normalized = user_text.strip().lower()
    matched = {
        metric
        for metric, keywords in (
            ("good", _GOOD_METRIC_KEYWORDS),
            ("bad", _BAD_METRIC_KEYWORDS),
            ("total", _TOTAL_METRIC_KEYWORDS),
        )
        if any(keyword in normalized for keyword in keywords)
    }
    return matched.pop() if len(matched) == 1 else ""


def _extract_tool_arguments(tool_name: str, user_text: str, model_dir: Optional[str]) -> dict[str, Any]:
    arguments: dict[str, Any] = {}

    date_fields = _DATE_ARG_FIELDS.get(tool_name)
    if date_fields:
        start_date, end_date = extract_date_range(user_text, model_dir=model_dir)
        if start_date:
            if date_fields == ("date_value",):
                arguments["date_value"] = start_date
            else:
                arguments["start_date"] = start_date
                arguments["end_date"] = end_date

    if tool_name in _METRIC_ARG_TOOLS:
        metric = _infer_metric(user_text)
        if metric:
            arguments["metric"] = metric

    return arguments


class ChatLocalGemma(BaseChatModel):
    """Minimal LangChain BaseChatModel wrapper around the local Gemma model.

    Tool selection is rule-based: scripts.tool_router.route_tool embeds the user's message
    with the local model's own embeddings (scripts.local_gemma_model.embed_text) and picks
    the tool whose example phrases are closest by cosine similarity. There is no free-form
    LLM JSON generation for the routing decision itself, and no keyword/regex fallback — if
    nothing clears the similarity threshold, tool_calls is left empty and the caller treats
    that as "no tool matched".
    """

    model_dir: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "local-gemma-tool-router"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> Any:
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools: list[BaseTool] = kwargs.get("tools") or []
        valid_names = [candidate_tool.name for candidate_tool in tools]
        user_text = _latest_human_text(messages)

        tool_name, _score = route_tool(user_text, valid_names, model_dir=self.model_dir)

        tool_calls: list[dict[str, Any]] = []
        if tool_name:
            tool_calls.append(
                {
                    "name": tool_name,
                    "args": _extract_tool_arguments(tool_name, user_text, self.model_dir),
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                }
            )

        message = AIMessage(content="" if tool_calls else UNSUPPORTED_MESSAGE, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def extract_date_range(text: str, *, model_dir: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Ask the local LLM to normalize free text into a (start_date, end_date) YYYY-MM-DD pair.

    Used both to pre-fill date arguments right after tool routing (ChatLocalGemma._generate,
    via _extract_tool_arguments) and to interpret a reply typed after a clarifying question
    (e.g. "어떤 날짜의 log를 보여 드릴까요?"). No regex/keyword fallback — if the model can't
    produce a valid ISO date, both values come back None.
    """
    prompt = f"""
Extract a date or date range from the text below and reply with exactly one JSON object:
{{"start_date": "YYYY-MM-DD"|null, "end_date": "YYYY-MM-DD"|null}}
Normalize any date format you see, including Korean-style dates like "26년 2월 3일", to
YYYY-MM-DD. If the text names a single day, use that same date for both start_date and
end_date. If no date can be found, reply with {{"start_date": null, "end_date": null}}.
Reply with only the JSON object and nothing else.

Text: {text}
""".strip()

    raw_response = generate_response(
        prompt=prompt,
        system_prompt="You extract dates from text and reply with strict JSON only.",
        max_new_tokens=64,
        temperature=0.0,
        model_dir=model_dir,
    )

    parsed = _extract_json_object(raw_response)
    if not parsed:
        return None, None

    start_date = parsed.get("start_date")
    end_date = parsed.get("end_date")
    start_date = start_date if isinstance(start_date, str) and start_date.strip() else None
    end_date = end_date if isinstance(end_date, str) and end_date.strip() else None
    if start_date and not end_date:
        end_date = start_date
    if end_date and not start_date:
        start_date = end_date
    return start_date, end_date
