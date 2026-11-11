"""工具层与故障注入测试。"""
from __future__ import annotations

import pytest

from src.tools import ToolRegistry, parse_fault_spec


def test_parse_fault_spec_ok():
    f = parse_fault_spec("doc_read:1:timeout;doc_search:2")
    assert f == {("doc_read", 1): "timeout", ("doc_search", 2): "timeout"}


@pytest.mark.parametrize(
    "spec",
    ["doc_read", "doc_read:0", "doc_read:1:boom", "submit_answer:1", "nope:1"],
)
def test_parse_fault_spec_invalid(spec):
    with pytest.raises(ValueError):
        parse_fault_spec(spec)


def test_fault_ordinal_semantics(corpus):
    reg = ToolRegistry(corpus, "doc_search:2:timeout")
    r1 = reg.dispatch("doc_search", {"query": "年假"})
    assert "error" not in r1
    r2 = reg.dispatch("doc_search", {"query": "年假"})
    assert r2["error"] == "TOOL_TIMEOUT"  # 第 2 次调用注入
    r3 = reg.dispatch("doc_search", {"query": "年假"})
    assert "error" not in r3  # 后续恢复
    assert reg.fault_hits == [{"tool": "doc_search", "ordinal": 2, "error_kind": "timeout"}]


def test_io_fault_shape(corpus):
    reg = ToolRegistry(corpus, "doc_read:1:io")
    r = reg.dispatch("doc_read", {"doc_id": "D05"})
    assert r["error"] == "TOOL_IO_ERROR"
    assert r["tool"] == "doc_read"
    assert "message" in r


def test_doc_read_missing_doc(corpus):
    reg = ToolRegistry(corpus)
    r = reg.dispatch("doc_read", {"doc_id": "D99"})
    assert r["error"] == "DOC_NOT_FOUND"


def test_empty_query_rejected(corpus):
    reg = ToolRegistry(corpus)
    r = reg.dispatch("doc_search", {"query": "  "})
    assert r["error"] == "INVALID_ARGUMENTS"


def test_submit_not_injectable(corpus):
    with pytest.raises(ValueError):
        ToolRegistry(corpus, "submit_answer:1")


def test_doc_list_shape(corpus):
    reg = ToolRegistry(corpus)
    r = reg.dispatch("doc_list", {})
    assert len(r["docs"]) == 9
    assert {"doc_id", "title", "version_date", "topic"} <= set(r["docs"][0])
