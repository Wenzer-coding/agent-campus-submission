"""ReAct 主循环（docs/02-solution.md §4）。

循环只负责：执行模型选择的动作、步数预算、观察记账、交付校验、trace 记录。
不含任何"检索到 X 就做 Y"的预设分支——每一步动作由后端（真实模型或 Mock 决策器）给出。
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

from .grounding import ObservationLedger, normalize_payload, validate_submission
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS, ToolRegistry

TIMEOUT_NUDGE = "请使用工具完成任务；最终交付必须调用 submit_answer 工具。"
HISTORY_CAP = 1200
REPO_ROOT = Path(__file__).resolve().parents[1]


def _display_path(path_str: str) -> str:
    """trace 中展示仓库内相对路径（可移植），仓库外路径保持原样。"""
    try:
        return Path(path_str).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path_str)


def _json_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _summarize(result: dict) -> str:
    if "error" in result:
        return f"error={result.get('error')}: {str(result.get('message', ''))[:160]}"
    tool = result.get("tool")
    if tool == "doc_search":
        docs = [r.get("doc_id") for r in result.get("results", [])]
        return f"命中 {len(docs)} 篇 {docs}；best_score={result.get('best_score')}"
    if tool == "doc_read":
        return f"读取 {result.get('doc_id')} 成功（{len(result.get('sections', []))} 小节，版本 {result.get('version_date')}）"
    if tool == "doc_list":
        return f"文档清单 {len(result.get('docs', []))} 篇"
    return json.dumps(result, ensure_ascii=False)[:200]


def _step_limit_payload(
    steps_used: int, max_steps: int, registry: ToolRegistry, reason: str = "", recent_obs: list[str] | None = None
) -> dict:
    counts = "、".join(f"{k}×{v}" for k, v in sorted(registry.dispatch_counts.items())) or "无"
    text = f"未能在步数上限（{max_steps} 步）内完成任务。"
    if reason:
        text += reason
    summary = ""
    if recent_obs:
        summary = " 已完成动作摘要：" + "；".join(recent_obs[-2:])[:240]
    return {
        "status": "step_limit",
        "answer_text": text + summary,
        "citations": [],
        "gaps": [],
        "followup_question": None,
        "incomplete_reason": f"step_limit：已执行 {steps_used}/{max_steps} 步；工具调用统计：{counts}。",
    }


def _grounding_failed_payload(errors: list[str]) -> dict:
    return {
        "status": "grounding_failed",
        "answer_text": "生成内容未通过引用校验，已不予交付。",
        "citations": [],
        "gaps": [],
        "followup_question": None,
        "incomplete_reason": ("；".join(errors))[:500],
    }


def _parse_arguments(raw: str) -> tuple[dict | None, str | None]:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, f"arguments 不是合法 JSON: {exc}"
    if not isinstance(args, dict):
        return None, "arguments 必须是 JSON 对象"
    return args, None


def _update_history(prev: str, question: str, payload: dict) -> str:
    cites = "；".join(
        f"[{c.get('doc_id')} v{c.get('version_date')}]“{c.get('quote')}”" for c in payload.get("citations", [])
    )
    entry = f"问：{question}\n答（{payload.get('status')}）：{str(payload.get('answer_text', ''))[:300]}"
    if cites:
        entry += f"\n引用：{cites[:400]}"
    merged = (prev + "\n" + entry) if prev else entry
    return merged[-HISTORY_CAP:]


def run_agent(
    turns: str | list[str],
    corpus,
    backend,
    max_steps: int = 8,
    fault_spec: str = "",
    scenario_id: str = "",
    system_prompt: str | None = None,
) -> dict:
    """运行一次会话（单轮或 turns=多轮）。返回 {payload, trace, steps_used}。"""
    if isinstance(turns, str):
        turns = [turns]
    turns = list(turns)
    registry = ToolRegistry(corpus, fault_spec)
    ledger = ObservationLedger()
    messages: list[dict] = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    steps_used = 0
    repair_used = False
    text_nudges = 0
    history = ""
    final_payload: dict | None = None
    recent_obs: list[str] = []

    deps: dict[str, str] = {}
    try:
        import openai as _openai

        deps["openai"] = getattr(_openai, "__version__", "?")
    except ImportError:  # pragma: no cover
        pass
    try:
        import pytest as _pytest

        deps["pytest"] = getattr(_pytest, "__version__", "?")
    except ImportError:  # pragma: no cover
        pass

    trace = {
        "scenario_id": scenario_id,
        "backend": backend.name,
        "model": backend.model,
        "corpus_dir": _display_path(str(corpus.docs_dir)),
        "corpus_hash": corpus.corpus_hash(),
        "fault_injection": fault_spec,
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "dependencies": deps,
        "turns": [],
    }

    for turn_index, question in enumerate(turns):
        text_nudges = 0  # 只统计"连续"纯文本回复
        user_content = question if not history else f"[对话历史摘要]\n{history}\n[当前问题]\n{question}"
        messages.append({"role": "user", "content": user_content})
        turn_trace = {"turn_index": turn_index, "question": question, "steps": [], "final": None}
        trace["turns"].append(turn_trace)
        turn_final: dict | None = None

        while turn_final is None:
            if steps_used >= max_steps:
                turn_final = _step_limit_payload(steps_used, max_steps, registry, recent_obs=recent_obs)
                break
            resp = backend.chat(messages, TOOLS, turn_index)
            content = (resp.get("content") or "").strip()
            calls = resp.get("tool_calls") or []

            if not calls:
                # 纯文本且无工具调用：催促；连续 2 次按未完成处理
                text_nudges += 1
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": TIMEOUT_NUDGE})
                steps_used += 1
                turn_trace["steps"].append(
                    {
                        "step_index": steps_used,
                        "turn_index": turn_index,
                        "tool_name": None,
                        "tool_args": None,
                        "assistant_rationale": content,
                        "observation_summary": "模型未调用工具，已催促",
                        "observation_sha256": None,
                        "next_action": "催促重试" if text_nudges < 2 else "终止",
                        "latency_ms": 0,
                        "steps_used": steps_used,
                        "steps_budget": max_steps,
                        "fault_hit": None,
                        "validation": None,
                    }
                )
                if text_nudges >= 2:
                    turn_final = _step_limit_payload(
                        steps_used,
                        max_steps,
                        registry,
                        reason="模型连续两次未按要求调用工具。",
                        recent_obs=recent_obs,
                    )
                continue

            text_nudges = 0  # 有工具调用即打断"连续纯文本"计数

            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ],
                }
            )

            budget_exhausted_in_batch = False
            responded_ids: set[str] = set()
            for call in calls:
                if turn_final is not None:
                    break
                if steps_used >= max_steps:
                    # 保持消息不变量：每个 tool_call 必须有配对响应
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {"tool": call["name"], "error": "STEP_BUDGET_EXHAUSTED", "message": "步数预算耗尽，该调用未执行"},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    responded_ids.add(call["id"])
                    budget_exhausted_in_batch = True
                    continue
                steps_used += 1
                step_entry = {
                    "step_index": steps_used,
                    "turn_index": turn_index,
                    "tool_name": call["name"],
                    "tool_args": None,
                    "assistant_rationale": content,
                    "observation_summary": "",
                    "observation_sha256": None,
                    "next_action": None,
                    "latency_ms": 0,
                    "steps_used": steps_used,
                    "steps_budget": max_steps,
                    "fault_hit": None,
                    "validation": None,
                }
                t0 = time.perf_counter()

                if call["name"] == "submit_answer":
                    args, parse_err = _parse_arguments(call["arguments"])
                    step_entry["tool_args"] = args if args is not None else {"raw": call["arguments"][:300]}
                    if parse_err:
                        ok, errors = False, [parse_err]
                        normalized = None
                    else:
                        normalized = normalize_payload(args)
                        ok, errors = validate_submission(normalized, ledger)
                    latency = int((time.perf_counter() - t0) * 1000)
                    step_entry["latency_ms"] = latency
                    if ok:
                        # 消息不变量：每个 tool_call_id 必须有配对响应（含成功的 submit_answer）
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps({"tool": "submit_answer", "accepted": True}, ensure_ascii=False),
                            }
                        )
                        responded_ids.add(call["id"])
                        recent_obs.append("submit_answer 校验通过")
                        step_entry["observation_summary"] = "submit_answer 校验通过，任务完成"
                        step_entry["validation"] = {"passed": True, "failed_checks": []}
                        turn_trace["steps"].append(step_entry)
                        turn_final = normalized
                        break
                    # 校验失败：一次修复机会
                    step_entry["observation_summary"] = "交付校验失败：" + "；".join(errors[:3])
                    step_entry["validation"] = {"passed": False, "failed_checks": errors}
                    turn_trace["steps"].append(step_entry)
                    recent_obs.append("submit_answer 校验失败")
                    if not repair_used:
                        repair_used = True
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    {"tool": "submit_answer", "error": "VALIDATION_FAILED", "errors": errors},
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        responded_ids.add(call["id"])
                        continue
                    turn_final = _grounding_failed_payload(errors)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(
                                {"tool": "submit_answer", "accepted": False, "errors": errors},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    responded_ids.add(call["id"])
                    break

                # 普通工具
                args, parse_err = _parse_arguments(call["arguments"])
                step_entry["tool_args"] = args if args is not None else {"raw": call["arguments"][:300]}
                faults_before = len(registry.fault_hits)
                if parse_err:
                    result = {"tool": call["name"], "error": "INVALID_ARGUMENTS", "message": parse_err}
                else:
                    result = registry.dispatch(call["name"], args)
                if len(registry.fault_hits) > faults_before:
                    step_entry["fault_hit"] = registry.fault_hits[-1]
                if "error" not in result:
                    ledger.add_from_tool_result(result)
                latency = int((time.perf_counter() - t0) * 1000)
                step_entry["latency_ms"] = latency
                payload_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
                step_entry["observation_summary"] = _summarize(result)[:300]
                step_entry["observation_sha256"] = _json_sha(payload_json)
                turn_trace["steps"].append(step_entry)
                recent_obs.append(f"{call['name']} → {step_entry['observation_summary'][:80]}")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": payload_json})
                responded_ids.add(call["id"])

            if turn_final is not None:
                # 批次提前终止时，为同批剩余调用补配对响应（消息序列不变量）
                for call in calls:
                    if call["id"] not in responded_ids:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    {"tool": call["name"], "error": "NOT_EXECUTED", "message": "同批已产生交付/终止，该调用未执行"},
                                    ensure_ascii=False,
                                ),
                            }
                        )
                break
            if budget_exhausted_in_batch:
                turn_final = _step_limit_payload(steps_used, max_steps, registry, recent_obs=recent_obs)
                break
            # 本轮所有工具调用执行完毕，继续循环让后端决定下一步

        turn_trace["final"] = turn_final
        final_payload = turn_final
        history = _update_history(history, question, turn_final)
        # 记录本轮后下一步动作（供 trace 展示"后续动作"）
        _fill_next_actions(turn_trace)

    trace["total_steps"] = steps_used
    return {"payload": final_payload, "trace": trace, "steps_used": steps_used}


def _fill_next_actions(turn_trace: dict) -> None:
    steps = turn_trace.get("steps") or []
    for i, step in enumerate(steps):
        if step.get("next_action") in (None, "催促重试"):
            if i + 1 < len(steps):
                nxt = steps[i + 1]["tool_name"] or "（催促后重试）"
                step["next_action"] = nxt
            else:
                final_status = (turn_trace.get("final") or {}).get("status", "结束")
                step["next_action"] = f"终止并交付（{final_status}）"
