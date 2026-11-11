"""Agent 级测试：用离线 Mock 后端（确定性）覆盖全部验收点。

对应 docs/02-solution.md §12 测试矩阵与 §13 场景矩阵。
"""
from __future__ import annotations

import json

from src.agent import run_agent


def _steps(res) -> list[dict]:
    return [s for t in res["trace"]["turns"] for s in t["steps"]]


def _tools_called(res) -> list[str]:
    return [s["tool_name"] for s in _steps(res) if s["tool_name"]]


def _doc_haystack(corpus, doc_id: str) -> str:
    doc = corpus.get(doc_id)
    return doc.title + "\n" + "\n".join(f"{s['heading']}\n{s['text']}" for s in doc.sections)


# ---------- 验收点① 明确答案 ----------
def test_e1_answered_with_grounded_citation(corpus, mock_backend):
    res = run_agent("年假有几天？", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"], "answered 必须带引用"
    quote = p["citations"][0]["quote"]
    assert quote in _doc_haystack(corpus, "D05"), "引用必须是原文逐字片段"
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
    # 引用逐字可核
    for c in p["citations"]:
        assert c["quote"] in _doc_haystack(corpus, c["doc_id"])


# ---------- 验收点③ 资料外问题 ----------
def test_e3_not_in_corpus_with_gaps(corpus, mock_backend):
    res = run_agent("公司有健身房吗？", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "not_in_corpus"
    assert p["gaps"], "资料外必须给出缺口"
    assert not p["citations"], "资料外不得携带引用"
    assert "健身房" in p["gaps"][0]


# ---------- 验收点④ 改写后的相似问题 ----------
def test_e4_rewrite_path(corpus, mock_backend):
    res = run_agent("出差住酒店一晚最多能报多少钱", corpus, mock_backend)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"][0]["doc_id"] == "D06"
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
    # 最终引用必须逐字出自原文（用真实语料核对，而非从待校验 payload 自身造证据）
    for c in p["citations"]:
        assert c["quote"] in _doc_haystack(corpus, c["doc_id"])
    assert steps[-1]["validation"]["passed"] is True


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
    assert "已完成动作摘要" in p["answer_text"], "step_limit 交付应附已完成的观察摘要"


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
        self.actions = actions
        self.i = 0
        self.last_messages: list[dict] | None = None

    def chat(self, messages, tools, turn_index=0):
        self.last_messages = messages
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        if a.get("text_only"):
            return {"content": a.get("content", "（纯文本）"), "tool_calls": []}
        calls = a.get("calls") or [{"name": a["name"], "args": a.get("args", {})}]
        return {
            "content": a.get("rationale", ""),
            "tool_calls": [
                {
                    "id": f"sc_{self.i}_{j}",
                    "name": c["name"],
                    "arguments": json.dumps(c["args"], ensure_ascii=False),
                }
                for j, c in enumerate(calls)
            ],
        }


VALID_D05_PAYLOAD = {
    "status": "answered",
    "answer_text": "满一年后每年 5 天。",
    "citations": [
        {
            "doc_id": "D05",
            "version_date": "2024-01-10",
            "quote": "全职员工累计工作满一年后，每年享有 5 天带薪年假。",
        }
    ],
}


def _assert_pairing(messages: list[dict]) -> None:
    """消息不变量：每条 assistant tool_calls 中的每个 id 都有配对 tool 响应。"""
    tool_ids = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                assert tc["id"] in tool_ids, f"未配对的 tool_call: {tc['id']}"


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


def test_multi_call_batch_keeps_pairing(corpus):
    """同一条 assistant 消息里 submit_answer 成功 + 多余调用：必须补齐配对响应，且多轮可继续。"""
    backend = _ScriptedBackend(
        [
            # 第一轮：读取 + 清单（同一批两个调用）
            {"calls": [{"name": "doc_read", "args": {"doc_id": "D05"}}, {"name": "doc_list", "args": {}}]},
            # 第二轮批次：提交成功 + 一个多余调用
            {"calls": [{"name": "submit_answer", "args": VALID_D05_PAYLOAD}, {"name": "doc_list", "args": {}}]},
            # 第二轮追问：再提交一次（验证消息序列合法、可继续调用模型）
            {"name": "submit_answer", "args": VALID_D05_PAYLOAD},
        ]
    )
    res = run_agent(["年假有几天？", "还有别的规定吗？"], corpus, backend)
    assert res["payload"]["status"] == "answered"
    assert backend.last_messages is not None
    _assert_pairing(backend.last_messages)


def test_text_nudge_resets_after_tool_call(corpus):
    """纯文本催促只统计"连续"次数：text → tool → text → submit 不应触发 step_limit。"""
    backend = _ScriptedBackend(
        [
            {"text_only": True, "content": "我先想想。"},
            {"name": "doc_read", "args": {"doc_id": "D05"}},
            {"text_only": True, "content": "再想想。"},
            {"name": "submit_answer", "args": VALID_D05_PAYLOAD},
        ]
    )
    res = run_agent("年假有几天？", corpus, backend)
    assert res["payload"]["status"] == "answered", "不连续的两条纯文本不应触发未完成"


def test_no_citation_statuses_reject_smuggled_citations(corpus):
    """not_in_corpus/tool_failure 不得携带引用（防伪造引用借道绕过校验）。"""
    smuggled = {
        "status": "not_in_corpus",
        "answer_text": "无资料",
        "gaps": ["x"],
        "citations": [{"doc_id": "D99", "version_date": "2099-01-01", "quote": "完全编造"}],
    }
    backend = _ScriptedBackend([{"name": "submit_answer", "args": smuggled}])
    res = run_agent("年假有几天？", corpus, backend)
    # 第一次校验失败进入修复，修复后仍非法 → grounding_failed，绝不会带出伪造引用
    assert res["payload"]["status"] == "grounding_failed"
    assert res["payload"]["citations"] == []


# ---------- trace 结构（运行记录硬性要素） ----------
def test_trace_contains_required_fields(corpus, mock_backend):
    res = run_agent("年假有几天？", corpus, mock_backend)
    trace = res["trace"]
    for key in (
        "backend",
        "model",
        "corpus_hash",
        "max_steps",
        "fault_injection",
        "dependencies",
        "turns",
        "total_steps",
    ):
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
