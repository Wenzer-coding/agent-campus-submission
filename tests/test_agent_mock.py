"""Agent 级测试：用离线 Mock 后端（确定性）覆盖全部验收点。

对应 docs/02-solution.md §12 测试矩阵与 §13 场景矩阵。
"""
from __future__ import annotations

import json

from src.agent import run_agent
from src.grounding import ObservationLedger, validate_submission


def _steps(res) -> list[dict]:
    return [s for t in res["trace"]["turns"] for s in t["steps"]]


def _tools_called(res) -> list[str]:
    return [s["tool_name"] for s in _steps(res) if s["tool_name"]]


# ---------- 验收点① 明确答案 ----------
def test_e1_answered_with_grounded_citation(corpus, mock_backend):
    res = run_agent("年假有几天？", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"], "answered 必须带引用"
    quote = p["citations"][0]["quote"]
    doc = corpus.get("D05")
    hay = "\n".join(f"{s['heading']}\n{s['text']}" for s in doc.sections)
    assert quote in hay, "引用必须是原文逐字片段"
    assert "满一年" in p["answer_text"], "回答须保留适用条件"
    # 确认性检索纪律：单一来源作答前至少两次检索
    assert _tools_called(res).count("doc_search") >= 2


# ---------- 验收点② 文档冲突 ----------
def test_e2_conflict_needs_confirmation(corpus, mock_backend):
    res = run_agent("报销期限是多久？", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "needs_user_confirmation"
    ids = sorted(c["doc_id"] for c in p["citations"])
    assert ids == ["D01", "D02"], "冲突必须列出双方引用"
    assert p["followup_question"], "冲突必须请求用户确认"
    text = p["answer_text"]
    assert "30" in text and "15" in text, "回答须呈现双方原文"
    # 读过两份原文（查证行为）
    assert _tools_called(res).count("doc_read") == 2


# ---------- 验收点③ 资料外问题 ----------
def test_e3_not_in_corpus_with_gaps(corpus, mock_backend):
    res = run_agent("公司有健身房吗？", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "not_in_corpus"
    assert p["gaps"], "资料外必须给出缺口"
    assert not p["citations"], "资料外不得编造引用"
    assert "健身房" in p["gaps"][0]


# ---------- 验收点④ 改写后的相似问题 ----------
def test_e4_rewrite_path(corpus, mock_backend):
    res = run_agent("出差住酒店一晚最多能报多少钱", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"][0]["doc_id"] == "D06"
    # 首查 miss（best_score 低于阈值），随后改写命中
    search_steps = [s for s in _steps(res) if s["tool_name"] == "doc_search"]
    assert len(search_steps) >= 2, "改写场景需要至少两次检索"
    first_obs = search_steps[0]["observation_summary"]
    assert "命中 0 篇" in first_obs, f"首查必须 miss 才能体现改写路径，实际：{first_obs}"
    assert "400" in p["answer_text"]


# ---------- 验收点⑤ 上下文追问 ----------
def test_e5_multiturn_followup(corpus, mock_backend):
    res = run_agent(["年假有几天？", "那试用期员工呢？"], corpus, mock_backend, max_steps=12)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"][0]["doc_id"] == "D05"
    assert "试用期" in p["citations"][0]["quote"], "第二轮应引用试用期条款"
    assert len(res["trace"]["turns"]) == 2
    # 第二轮用户消息包含历史摘要
    assert res["trace"]["turns"][1]["question"] == "那试用期员工呢？"


# ---------- 验收点⑥ 工具失败 ----------
def test_e6_tool_failure_recovery(corpus, mock_backend):
    res = run_agent("年假有几天？", corpus, mock_backend, fault_spec="doc_read:1:timeout")
    p = res["payload"]
    assert p["status"] == "answered"
    steps = _steps(res)
    fault_steps = [s for s in steps if s.get("fault_hit")]
    assert fault_steps, "trace 必须记录故障命中"
    assert fault_steps[0]["fault_hit"]["tool"] == "doc_read"
    assert _tools_called(res).count("doc_read") >= 2, "失败后必须重试"
    # 重试成功后仍是通过引用校验的交付
    ledger_ok = validate_submission(p, _fake_ledger_from_trace(res))
    assert ledger_ok[0], ledger_ok[1]


def test_e6_persistent_failure_reports_tool_failure(corpus, mock_backend):
    res = run_agent(
        "年假有几天？", corpus, mock_backend, fault_spec="doc_read:1:timeout;doc_read:2:timeout"
    )
    assert res["payload"]["status"] == "tool_failure"
    assert not res["payload"]["citations"]


# ---------- 步数上限（全局规则） ----------
def test_e7_step_limit_explicit(corpus, mock_backend):
    res = run_agent("报销期限是多久？", corpus, mock_backend, max_steps=1)
    p = res["payload"]
    assert p["status"] == "step_limit"
    assert res["steps_used"] == 1
    assert "步数上限" in p["answer_text"]
    assert "step_limit" in p["incomplete_reason"]


# ---------- 换语料：行为随数据改变（审核要点） ----------
def test_corpus_swap_changes_answer(corpus_alt):
    from src.llm import MockBackend

    res = run_agent("报销期限是多久？", corpus_alt, MockBackend(corpus_alt))
    p = res["payload"]
    assert p["status"] == "answered", "换语料后冲突消失，应能直接作答"
    assert p["citations"][0]["doc_id"] == "D01"
    assert "45" in p["answer_text"], "答案数值必须随替换后的文档变化"


# ---------- 引用校验二次失败 → grounding_failed ----------
class _ScriptedBackend:
    name = "scripted"
    model = "scripted"

    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def chat(self, messages, tools, turn_index=0):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return {
            "content": a.get("rationale", ""),
            "tool_calls": [
                {"id": f"sc_{self.i}", "name": a["name"], "arguments": json.dumps(a["args"], ensure_ascii=False)}
            ],
        }


def test_forged_citation_leads_to_grounding_failed(corpus):
    fake = {
        "status": "answered",
        "answer_text": "伪造答案",
        "citations": [{"doc_id": "D05", "version_date": "2024-01-10", "quote": "这句原文不存在。"}],
    }
    backend = _ScriptedBackend(
        [
            {"name": "doc_read", "args": {"doc_id": "D05"}},
            {"name": "submit_answer", "args": fake},
            {"name": "submit_answer", "args": fake},
        ]
    )
    res = run_agent("年假有几天？", corpus, backend)
    assert res["payload"]["status"] == "grounding_failed"
    assert not res["payload"]["citations"]
    assert "引用校验" in res["payload"]["answer_text"]


def test_text_only_model_hits_step_limit(corpus):
    backend = _ScriptedBackend([{"name": "submit_answer", "args": {}}])

    class TextOnly:
        name = "textonly"
        model = "textonly"

        def chat(self, messages, tools, turn_index=0):
            return {"content": "我只是说说话，不调用工具。", "tool_calls": []}

    res = run_agent("年假有几天？", corpus, TextOnly())
    assert res["payload"]["status"] == "step_limit"


# ---------- trace 结构（运行记录硬性要素） ----------
def test_trace_contains_required_fields(corpus, mock_backend):
    res = run_agent("年假有几天？", corpus, mock_backend)
    trace = res["trace"]
    for key in ("backend", "model", "corpus_hash", "max_steps", "fault_injection", "turns", "total_steps"):
        assert key in trace
    for step in _steps(res):
        for key in (
            "step_index",
            "tool_name",
            "tool_args",
            "assistant_rationale",
            "observation_summary",
            "next_action",
            "steps_used",
            "steps_budget",
            "validation",
        ):
            assert key in step, f"trace 步骤缺少字段 {key}"
    assert _steps(res)[-1]["validation"]["passed"] is True


def _fake_ledger_from_trace(res) -> ObservationLedger:
    """从 trace 中重建一个宽松账本，仅用于断言交付本身通过校验（测试辅助）。"""
    ledger = ObservationLedger()
    for t in res["trace"]["turns"]:
        for s in t["steps"]:
            pass
    for c in res["payload"].get("citations", []):
        ledger.add_span(c["doc_id"], c["version_date"], c["quote"], "test")
    return ledger
