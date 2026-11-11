"""真实模型（DeepSeek）冒烟测试。

无 DEEPSEEK_API_KEY 时自动跳过——离线测试集仍完整覆盖全部验收点。
"""
from __future__ import annotations

import os

import pytest

from src.agent import run_agent

pytestmark = pytest.mark.real_model

requires_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"), reason="需要 DEEPSEEK_API_KEY 才能运行真实模型测试"
)


@requires_key
def test_deepseek_full_qa(corpus):
    from src.llm import DeepSeekBackend

    res = run_agent("年假有几天？", corpus, DeepSeekBackend(), max_steps=8)
    p = res["payload"]
    assert p["status"] == "answered"
    assert p["citations"], "真实模型交付也必须带引用且通过保真校验"
    assert res["steps_used"] >= 2


@requires_key
def test_deepseek_conflict(corpus):
    from src.llm import DeepSeekBackend

    res = run_agent("报销期限是多久？", corpus, DeepSeekBackend(), max_steps=10)
    p = res["payload"]
    # 验收点②：未裁决冲突必须请求确认、双引用，不得擅自选边
    assert p["status"] == "needs_user_confirmation"
    assert sorted(c["doc_id"] for c in p["citations"]) == ["D01", "D02"]
    assert p["followup_question"]
