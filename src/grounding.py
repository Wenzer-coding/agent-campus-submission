"""引用保真校验与观察账本（docs/02-solution.md §6）。

核心思想：不信任模型对引用的自证——只信任本会话工具真实返回过的原文。
校验器只查 ObservationLedger，不扫描消息文本，也不访问语料库原始文件。
"""
from __future__ import annotations

from .corpus import normalize

MODEL_STATUSES = {"answered", "needs_user_confirmation", "not_in_corpus", "tool_failure"}
LOOP_STATUSES = {"step_limit", "grounding_failed"}
ALL_STATUSES = MODEL_STATUSES | LOOP_STATUSES
NO_CITATION_STATUSES = {"not_in_corpus", "tool_failure", "step_limit", "grounding_failed"}


class ObservationLedger:
    """记录本会话中真实观察到的文档原文片段（引用校验的唯一证据来源）。"""

    def __init__(self) -> None:
        self.spans: list[dict] = []

    def add_span(self, doc_id: str, version_date: str, text: str, source_tool: str) -> None:
        if text is None:
            return
        self.spans.append(
            {"doc_id": doc_id, "version_date": version_date, "text": str(text), "source_tool": source_tool}
        )

    def add_from_tool_result(self, result: dict) -> None:
        tool = result.get("tool")
        if "error" in result:
            return  # 错误观察只作负证据，不进入账本
        if tool == "doc_search":
            for r in result.get("results", []):
                self.add_span(r.get("doc_id", ""), r.get("version_date", ""), r.get("snippet", ""), "doc_search")
        elif tool == "doc_read":
            for sec in result.get("sections", []):
                self.add_span(result.get("doc_id", ""), result.get("version_date", ""), sec.get("heading", ""), "doc_read")
                self.add_span(result.get("doc_id", ""), result.get("version_date", ""), sec.get("text", ""), "doc_read")

    def observed_docs(self) -> dict[str, set[str]]:
        """doc_id -> 观察到的全部版本日期集合。"""
        out: dict[str, set[str]] = {}
        for s in self.spans:
            out.setdefault(s["doc_id"], set()).add(s["version_date"])
        return out

    def find_span_containing(self, doc_id: str, quote: str) -> str | None:
        nq = normalize(quote)
        if not nq:
            return None
        for s in self.spans:
            if s["doc_id"] != doc_id:
                continue
            if nq in normalize(s["text"]):
                return s["text"]
        return None


def validate_payload(payload: dict) -> list[str]:
    """schema 校验；返回错误列表（空列表 = 通过）。"""
    errs: list[str] = []
    if not isinstance(payload, dict):
        return ["payload 必须是 JSON 对象"]
    status = payload.get("status")
    if status not in MODEL_STATUSES:
        errs.append(f"status 非法: {status!r}（只能是 {sorted(MODEL_STATUSES)}）")
    answer_text = payload.get("answer_text")
    if not isinstance(answer_text, str) or not answer_text.strip():
        errs.append("answer_text 必须是非空字符串")
    citations = payload.get("citations")
    if not isinstance(citations, list):
        errs.append("citations 必须是数组")
        citations = []
    for i, c in enumerate(citations):
        if not isinstance(c, dict):
            errs.append(f"citations[{i}] 必须是对象")
            continue
        for key in ("doc_id", "version_date", "quote"):
            v = c.get(key)
            if not isinstance(v, str) or not v.strip():
                errs.append(f"citations[{i}].{key} 缺失或为空")
    if status in ("answered", "needs_user_confirmation") and not citations:
        errs.append(f"status={status} 时 citations 不能为空")
    if status == "not_in_corpus":
        gaps = payload.get("gaps")
        if not isinstance(gaps, list) or not gaps:
            errs.append("status=not_in_corpus 时 gaps 必须是非空数组")
    if status == "needs_user_confirmation":
        fq = payload.get("followup_question")
        if not isinstance(fq, str) or not fq.strip():
            errs.append("status=needs_user_confirmation 时必须提供 followup_question")
    return errs


def check_citations(payload: dict, ledger: ObservationLedger) -> list[str]:
    """引用保真校验；返回失败明细（空列表 = 全部通过）。"""
    errs: list[str] = []
    status = payload.get("status")
    if status in NO_CITATION_STATUSES:
        return errs
    citations = payload.get("citations") or []
    observed = ledger.observed_docs()
    for i, c in enumerate(citations):
        if not isinstance(c, dict):
            continue
        doc_id = c.get("doc_id", "")
        version = c.get("version_date", "")
        quote = c.get("quote", "")
        if doc_id not in observed:
            errs.append(f"citations[{i}]: 文档 {doc_id!r} 未在本会话以内容形式被观察到（不能引用未读内容）")
            continue
        if version not in observed[doc_id]:
            errs.append(
                f"citations[{i}]: 版本日期 {version!r} 与已观察到的 {doc_id} 版本 {sorted(observed[doc_id])} 不一致"
            )
        if ledger.find_span_containing(doc_id, quote) is None:
            errs.append(f"citations[{i}]: quote 在 {doc_id} 的已观察原文中找不到（原文={quote[:50]!r}…）")
    return errs


def validate_submission(payload: dict, ledger: ObservationLedger) -> tuple[bool, list[str]]:
    """schema + 引用保真 双重校验。"""
    errs = validate_payload(payload)
    if errs:
        return False, errs
    errs = check_citations(payload, ledger)
    return (not errs), errs


def normalize_payload(payload: dict) -> dict:
    """补齐默认字段，保证交付结构固定。"""
    out = {
        "status": payload.get("status"),
        "answer_text": payload.get("answer_text", ""),
        "citations": payload.get("citations") or [],
        "gaps": payload.get("gaps") or [],
        "followup_question": payload.get("followup_question") or None,
        "incomplete_reason": payload.get("incomplete_reason") or None,
    }
    return out
