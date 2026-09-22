from __future__ import annotations

import math
from typing import Optional, Sequence

from scripts.local_gemma_model import embed_text

# ---------------------------------------------------------------------------
# Rule-based tool routing: cosine similarity between the user's message and a
# handful of example phrases per tool, using embeddings from the same local
# Gemma model (scripts.local_gemma_model.embed_text) — no separate embedding
# model, and no free-form LLM JSON generation for the routing decision itself.
#
# Multiple synonyms/phrasings are listed per tool so uncommon wording (e.g.
# "양품" instead of "정상") still lands near the right anchor.
# ---------------------------------------------------------------------------

TOOL_ANCHOR_PHRASES: dict[str, tuple[str, ...]] = {
    "dashboard_set_query_setting_tool": (
        "대시보드 조회 기간을 변경해줘",
        "다른 기간으로 데이터를 조회하고 싶어",
        "2026년 1월 데이터로 조회 설정을 바꿔줘",
        "조회할 테이블을 변경해줘",
        "1월 1일부터 1월 31일까지 기간으로 설정해줘",
    ),
    "summary_download_report_tool": (
        "리포트 다운로드 해줘",
        "요약 보고서 생성해줘",
        "PDF 리포트 발행해줘",
        "인스펙션 리포트 만들어줘",
        "보고서 내보내기",
        "리포트 저장해줘",
        "보고서 저장해줘",
    ),
    "summary_read_inference_results_tool": (
        "현재 불량데이터는 몇개야?",
        "불량품 개수 알려줘",
        "양품 데이터가 몇개야?",
        "정상 제품 몇개야?",
        "OK NG 개수 알려줘",
        "최근 추론 결과 보여줘",
        "불량률이 어떻게 돼?",
    ),
    "log_read_tool": (
        "로그 기록 보여줘",
        "오늘 로그 보여줘",
        "에러 로그 확인해줘",
        "특정 날짜 로그 조회해줘",
        "경고 로그만 보여줘",
        "시스템 시작 로그 확인",
        "로그 내역 알려줘",
    ),
    "list_defect_types_tool": (
        "불량 유형은 뭐가 있지?",
        "불량 종류 알려줘",
        "결함 유형이 뭐야?",
        "어떤 불량이 있어?",
        "defect 종류 알려줘",
        "분류 클래스가 뭐가 있어?",
    ),
}

DEFAULT_SIMILARITY_THRESHOLD = 0.55

# Anchor phrase embeddings never change for a given (tool, model_dir), so compute them once.
_anchor_embedding_cache: dict[tuple[str, Optional[str]], list[list[float]]] = {}


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_anchor_embeddings(tool_name: str, model_dir: Optional[str]) -> list[list[float]]:
    cache_key = (tool_name, model_dir)
    cached = _anchor_embedding_cache.get(cache_key)
    if cached is not None:
        return cached

    embeddings = [embed_text(phrase, model_dir=model_dir) for phrase in TOOL_ANCHOR_PHRASES[tool_name]]
    _anchor_embedding_cache[cache_key] = embeddings
    return embeddings


def route_tool(
    user_text: str,
    valid_tool_names: Sequence[str],
    model_dir: Optional[str] = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[Optional[str], float]:
    """Pick the tool whose anchor phrases are most similar to `user_text`.

    Returns (tool_name, best_score). tool_name is None if no tool clears `threshold`
    (or if `user_text` is empty), matching the "no tool matched" behavior the caller
    already expects from an empty tool_calls list.
    """
    normalized = user_text.strip()
    if not normalized:
        return None, 0.0

    query_embedding = embed_text(normalized, model_dir=model_dir)
    if not query_embedding:
        return None, 0.0

    best_tool: Optional[str] = None
    best_score = -1.0
    for tool_name in valid_tool_names:
        if tool_name not in TOOL_ANCHOR_PHRASES:
            continue
        anchor_embeddings = _get_anchor_embeddings(tool_name, model_dir)
        tool_score = max(_cosine_similarity(query_embedding, anchor) for anchor in anchor_embeddings)
        if tool_score > best_score:
            best_score = tool_score
            best_tool = tool_name

    if best_tool is None or best_score < threshold:
        return None, best_score

    return best_tool, best_score
