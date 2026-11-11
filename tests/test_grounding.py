"""引用保真校验与交付 schema 测试。"""
from __future__ import annotations

from src.grounding import (
    ObservationLedger,
    normalize_payload,
    validate_payload,
    validate_submission,
)


def _ledger_with_d05() -> ObservationLedger:
    ledger = ObservationLedger()
    ledger.add_span("D05", "2024-01-10", "全职员工累计工作满一年后，每年享有 5 天带薪年假。", "doc_read")
    ledger.add_span("D05", "2024-01-10", "试用期员工不享有年假，转正后按上述规定执行。", "doc_read")
    return ledger


def _payload(**overrides) -> dict:
    base = {
        "status": "answered",
        "answer_text": "满一年后每年 5 天。",
        "citations": [
            {"doc_id": "D05", "version_date": "2024-01-10", "quote": "全职员工累计工作满一年后，每年享有 5 天带薪年假。"}
        ],
    }
    base.update(overrides)
    return normalize_payload(base)


def test_valid_citation_passes():
    ok, errs = validate_submission(_payload(), _ledger_with_d05())
    assert ok, errs


def test_forged_quote_rejected():
    p = _payload(
        citations=[{"doc_id": "D05", "version_date": "2024-01-10", "quote": "每年享有 10 天带薪年假。"}]
    )
    ok, errs = validate_submission(p, _ledger_with_d05())
    assert not ok
    assert any("找不到" in e for e in errs)


def test_unobserved_doc_rejected():
    p = _payload(
        citations=[{"doc_id": "D01", "version_date": "2024-06-01", "quote": "差旅费用报销申请应在费用发生后 30 天内提交"}]
    )
    ok, errs = validate_submission(p, _ledger_with_d05())
    assert not ok
    assert any("未在本会话" in e for e in errs)


def test_wrong_version_rejected():
    p = _payload(
        citations=[{"doc_id": "D05", "version_date": "2025-01-01", "quote": "全职员工累计工作满一年后，每年享有 5 天带薪年假。"}]
    )
    ok, errs = validate_submission(p, _ledger_with_d05())
    assert not ok
    assert any("版本日期" in e for e in errs)


def test_cross_sentence_concat_rejected():
    # 拼接两个真实句子：归一化保留标点，拼接结果不是任何单个 span 的子串
    quote = "全职员工累计工作满一年后，每年享有 5 天带薪年假。试用期员工不享有年假"
    p = _payload(citations=[{"doc_id": "D05", "version_date": "2024-01-10", "quote": quote}])
    ok, errs = validate_submission(p, _ledger_with_d05())
    assert not ok


def test_error_observation_is_not_evidence():
    ledger = ObservationLedger()
    ledger.add_from_tool_result(
        {"tool": "doc_read", "error": "DOC_NOT_FOUND", "message": "文档不存在"}
    )
    ledger.add_from_tool_result(
        {"tool": "doc_search", "results": [], "best_score": 0.0}
    )
    assert ledger.observed_docs() == {}


def test_schema_answered_requires_citations():
    p = {"status": "answered", "answer_text": "x", "citations": []}
    errs = validate_payload(p)
    assert any("citations" in e for e in errs)


def test_schema_rejects_loop_status():
    p = {"status": "step_limit", "answer_text": "x", "citations": []}
    assert validate_payload(p)


def test_schema_not_in_corpus_requires_gaps():
    p = {"status": "not_in_corpus", "answer_text": "无资料", "citations": [], "gaps": []}
    assert any("gaps" in e for e in validate_payload(p))


def test_schema_confirmation_requires_followup():
    p = {
        "status": "needs_user_confirmation",
        "answer_text": "冲突",
        "citations": [
            {"doc_id": "D01", "version_date": "2024-06-01", "quote": "q1"},
            {"doc_id": "D02", "version_date": "2024-08-15", "quote": "q2"},
        ],
    }
    assert any("followup_question" in e for e in validate_payload(p))


def test_no_citation_statuses_skip_grounding():
    p = normalize_payload({"status": "tool_failure", "answer_text": "工具失败", "citations": []})
    ok, errs = validate_submission(p, ObservationLedger())
    assert ok, errs
