"""模型接入层（docs/02-solution.md §10）。

- DeepSeekBackend：OpenAI 兼容 function calling，SDK 自带超时与重试。
- MockBackend：离线规则决策器。只读"观察特征"（最近工具结果、本回合计数、历史问题），
  按优先级规则选择动作与参数；不读取场景 ID，不对问句做字面等值分支。
  边界：Mock 只验证流程、schema、校验器与边界分支的可运行性，
  不证明真实模型决策效果——决策效果证据以 DeepSeek 真实运行 trace 为准。
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from .corpus import bigrams, normalize, split_sentences

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class Backend:
    name = "base"
    model = "unknown"

    def chat(self, messages: list[dict], tools: list[dict], turn_index: int = 0) -> dict:
        """返回 {"content": str, "tool_calls": [{"id","name","arguments"(JSON字符串)}]}"""
        raise NotImplementedError


class DeepSeekBackend(Backend):
    name = "deepseek"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        from openai import OpenAI

        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填写")
        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        # 传输层健壮性：SDK 自带指数退避重试，不叠加自写外层重试
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)

    def chat(self, messages: list[dict], tools: list[dict], turn_index: int = 0) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.1,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                }
            )
        return {"content": msg.content or "", "tool_calls": calls}


class MockBackend(Backend):
    """离线 Mock 决策器（非固定脚本：决策依赖观察结果特征）。"""

    name = "mock"
    model = "mock-rule-decider"

    def __init__(self, corpus, rewrite_map_path: str | Path | None = None):
        self.corpus = corpus
        p = Path(rewrite_map_path) if rewrite_map_path else (DATA_DIR / "rewrite_map.json")
        self.rewrite_map = json.loads(p.read_text(encoding="utf-8-sig")) if p.is_file() else {}
        self._turn = -1
        self.questions: list[str] = []
        self._reset_turn()

    # ---------- 状态 ----------
    def _reset_turn(self) -> None:
        self.search_payloads: list[dict] = []
        self.reads: dict[str, dict] = {}
        self.rewrite_idx = 0
        self.listed = False
        self.last_error: dict | None = None
        self.retried: set[str] = set()
        self.submit_error: dict | None = None
        self.last_submit_payload: dict | None = None
        self.last_action: dict | None = None

    # ---------- 接口 ----------
    def chat(self, messages: list[dict], tools: list[dict], turn_index: int = 0) -> dict:
        if turn_index != self._turn:
            self._turn = turn_index
            self._reset_turn()
        last = messages[-1] if messages else None
        if last and last.get("role") == "user":
            content = last.get("content") or ""
            question = content.split("[当前问题]")[-1].strip() if "[当前问题]" in content else content.strip()
            if question and (not self.questions or self.questions[-1] != question):
                self.questions.append(question)
        elif last and last.get("role") == "tool":
            try:
                payload = json.loads(last.get("content") or "{}")
            except json.JSONDecodeError:
                payload = {"tool": "unknown", "error": "BAD_PAYLOAD", "message": "无法解析工具结果"}
            self._observe(payload)

        action = self._decide()
        self.last_action = action
        if action["name"] == "submit_answer":
            self.last_submit_payload = action.get("args", {})
        tool_id = f"mock_{uuid.uuid4().hex[:8]}"
        return {
            "content": action.get("rationale", ""),
            "tool_calls": [
                {
                    "id": tool_id,
                    "name": action["name"],
                    "arguments": json.dumps(action.get("args", {}), ensure_ascii=False),
                }
            ],
        }

    # ---------- 观察 ----------
    def _observe(self, payload: dict) -> None:
        tool = payload.get("tool")
        if "error" in payload:
            self.last_error = payload
            if tool == "submit_answer":
                self.submit_error = payload
            return
        self.last_error = None
        # 成功即打断"连续失败"计数（只对连续失败才降级 tool_failure）
        self.retried = {k for k in self.retried if not str(k).startswith(f"{tool}:")}
        if tool == "doc_search":
            self.search_payloads.append(payload)
        elif tool == "doc_read":
            self.reads[payload.get("doc_id", "")] = payload
        elif tool == "doc_list":
            self.listed = True

    # ---------- 决策（按优先级） ----------
    def _rewrites(self, q: str) -> list[str]:
        for key in (q, q.strip("？?！!。.，, ")):
            if key in self.rewrite_map:
                return self.rewrite_map[key]
        return []

    def _decide(self) -> dict:
        q = self.questions[-1] if self.questions else ""
        # 多轮：追问需要结合上一轮问题做指代消解（特征：存在历史问题）
        prev = self.questions[-2] if len(self.questions) > 1 else ""
        search_query = f"{prev} {q}".strip() if prev else q

        # 1) 交付校验失败 → 修正后重新提交
        if self.submit_error:
            err = self.submit_error
            self.submit_error = None
            fixed = self._fix_submission()
            return {
                "name": "submit_answer",
                "args": fixed,
                "rationale": f"观察到交付校验失败（{str(err.get('errors'))[:120]}），按错误明细修正后重新提交。",
            }

        # 2) 工具失败 → 重试一次；仍失败则如实返回 tool_failure
        if self.last_error:
            err = self.last_error
            key = f"{err.get('tool')}:{err.get('error')}"
            if err.get("tool") != "submit_answer":
                if key not in self.retried:
                    self.retried.add(key)
                    prev = self.last_action or {"name": err.get("tool"), "args": {"/": "/"}}
                    return {
                        "name": prev.get("name", err.get("tool")),
                        "args": prev.get("args", {}),
                        "rationale": f"观察到 {err.get('tool')} 返回 {err.get('error')}（{str(err.get('message'))[:80]}），按策略重试一次该调用。",
                    }
                return {
                    "name": "submit_answer",
                    "args": {
                        "status": "tool_failure",
                        "answer_text": f"工具 {err.get('tool')} 连续失败，无法完成本次查证。",
                        "citations": [],
                        "gaps": [],
                        "incomplete_reason": f"{err.get('error')}: {str(err.get('message'))[:200]}",
                    },
                    "rationale": "同一工具重试后仍失败，如实返回 tool_failure，不编造结果。",
                }

        # 3) 尚未检索
        if not self.search_payloads:
            note = "（含上一轮问题，用于指代消解）" if prev else ""
            return {
                "name": "doc_search",
                "args": {"query": search_query},
                "rationale": f"尚未检索，先对问题进行文档检索{note}。",
            }

        last_search = self.search_payloads[-1]
        if not last_search.get("results"):
            # 4a) 空结果 → 表驱动改写（最多 2 次）
            rewrites = self._rewrites(q)
            if self.rewrite_idx < min(len(rewrites), 2):
                rw = rewrites[self.rewrite_idx]
                self.rewrite_idx += 1
                return {
                    "name": "doc_search",
                    "args": {"query": rw},
                    "rationale": f"检索无结果（best_score={last_search.get('best_score')}），将查询改写为制度用语后重试：{rw}",
                }
            # 4b) 仍无结果 → 列文档核实
            if not self.listed:
                return {"name": "doc_list", "args": {}, "rationale": "多次改写检索仍无结果，列出全部文档核实资料是否存在。"}
            # 4c) 确认资料外
            return {
                "name": "submit_answer",
                "args": {
                    "status": "not_in_corpus",
                    "answer_text": "现有制度文档库中没有与该问题相关的可依据资料。",
                    "citations": [],
                    "gaps": [q],
                },
                "rationale": "文档库清单中也没有相关资料，判定资料外，返回资料缺口而不编造。",
            }

        results = last_search["results"]

        # 5) 冲突查证：前两名得分接近说明可能是同一问题的多个来源，先读原文再判定
        top2 = results[:2]
        if len(top2) == 2 and abs(top2[0]["score"] - top2[1]["score"]) <= 0.05:
            missing = [r["doc_id"] for r in top2 if r["doc_id"] not in self.reads]
            if missing:
                return {
                    "name": "doc_read",
                    "args": {"doc_id": missing[0]},
                    "rationale": f"两份文档得分接近（{top2[0]['doc_id']}={top2[0]['score']}, {top2[1]['doc_id']}={top2[1]['score']}），读取 {missing[0]} 原文查证是否构成冲突。",
                }
            conflict = self._extract_conflict(q, [r["doc_id"] for r in top2])
            if conflict:
                return conflict

        # 6) 单一来源 → 确认性检索（排除未发现的冲突文档）
        unique_docs = {r["doc_id"] for p in self.search_payloads for r in p.get("results", [])}
        if len(unique_docs) == 1 and len(self.search_payloads) == 1:
            return {
                "name": "doc_search",
                "args": {"query": search_query, "top_k": 10},
                "rationale": "目前仅单一文档命中，按纪律先做一次扩大范围的确认性检索，排除未发现的冲突文档。",
            }

        # 7) 读取最优文档
        top_doc = results[0]["doc_id"]
        if top_doc not in self.reads:
            return {
                "name": "doc_read",
                "args": {"doc_id": top_doc},
                "rationale": f"检索结果显示 {top_doc} 最相关（score={results[0]['score']}），读取原文以确认并引用。",
            }

        # 8) 依据已读原文作答
        return self._answered(q, top_doc)

    # ---------- 构造交付 ----------
    def _best_sentence(self, q: str, read: dict) -> tuple[str, str] | None:
        qb = bigrams(q)
        best: tuple[tuple, str, str] | None = None
        for sec in read.get("sections", []):
            for s in split_sentences(sec.get("text", "")):
                ov = len(qb & bigrams(s))
                if ov == 0:
                    continue
                score = (ov, bool(re.search(r"\d", s)), -len(s))
                if best is None or score > best[0]:
                    best = (score, sec.get("heading", ""), s)
        return (best[1], best[2]) if best else None

    def _extract_conflict(self, q: str, doc_ids: list[str]) -> dict | None:
        sents: dict[str, tuple[str, str, dict]] = {}
        for doc_id in doc_ids:
            read = self.reads.get(doc_id)
            if not read:
                continue
            best = self._best_sentence(q, read)
            if not best:
                continue
            heading, sentence = best
            digits = set(re.findall(r"\d+", sentence))
            if digits:
                sents[doc_id] = (sentence, heading, read)
        if len(sents) < 2:
            return None
        items = list(sents.items())[:2]
        (d1, (s1, _h1, r1)), (d2, (s2, _h2, r2)) = items[0], items[1]
        nums1 = tuple(re.findall(r"\d+(?:\.\d+)?", s1))
        nums2 = tuple(re.findall(r"\d+(?:\.\d+)?", s2))
        if nums1 == nums2:
            return None
        return {
            "name": "submit_answer",
            "args": {
                "status": "needs_user_confirmation",
                "answer_text": (
                    f"检索发现两份制度对同一问题说法不一致：{r1.get('title')}（{d1}，版本 {r1.get('version_date')}）规定“{s1}”；"
                    f"{r2.get('title')}（{d2}，版本 {r2.get('version_date')}）规定“{s2}”。"
                    "两份文档均未注明适用范围与优先级，无法确定应适用哪一份制度，需要您确认。"
                ),
                "citations": [
                    {"doc_id": d1, "version_date": r1.get("version_date"), "quote": s1},
                    {"doc_id": d2, "version_date": r2.get("version_date"), "quote": s2},
                ],
                "followup_question": f"请问您适用《{r1.get('title')}》还是《{r2.get('title')}》？",
            },
            "rationale": "两份文档对同一问题给出不同数值且均无适用范围说明，判定未裁决冲突，列出双方引用并请求用户确认。",
        }

    def _answered(self, q: str, doc_id: str) -> dict:
        read = self.reads[doc_id]
        best = self._best_sentence(q, read)
        if best:
            heading, sentence = best
            return {
                "name": "submit_answer",
                "args": {
                    "status": "answered",
                    "answer_text": f"根据《{read.get('title')}》（{doc_id}，版本 {read.get('version_date')}）“{heading}”一节：{sentence}",
                    "citations": [
                        {"doc_id": doc_id, "version_date": read.get("version_date"), "quote": sentence}
                    ],
                },
                "rationale": f"已读取 {doc_id} 且未发现冲突来源，依据原文作答并附逐字引用。",
            }
        first = read.get("sections", [{}])[0]
        return {
            "name": "submit_answer",
            "args": {
                "status": "answered",
                "answer_text": f"根据《{read.get('title')}》（{doc_id}）原文：{first.get('text', '')}",
                "citations": [
                    {"doc_id": doc_id, "version_date": read.get("version_date"), "quote": first.get("text", "")}
                ],
            },
            "rationale": "已有原文但未匹配到具体问句片段，按最相关小节原文作答。",
        }

    def _fix_submission(self) -> dict:
        payload = self.last_submit_payload or {
            "status": "not_in_corpus",
            "answer_text": "无法完成本次交付。",
            "citations": [],
            "gaps": ["无法完成本次交付"],
        }
        fixed = dict(payload)
        fixed["gaps"] = fixed.get("gaps") or []
        # 用已读原文过滤引用（Mock 引用均来自原文，此路径通常不触发）
        kept = []
        for c in payload.get("citations") or []:
            read = self.reads.get(c.get("doc_id", ""))
            if not read:
                continue
            hay = "\n".join(
                f"{sec.get('heading', '')}\n{sec.get('text', '')}" for sec in read.get("sections", [])
            )
            if c.get("quote", "") and normalize(c["quote"]) in normalize(hay):
                kept.append(c)
        fixed["citations"] = kept
        if fixed.get("status") in ("answered", "needs_user_confirmation") and not kept:
            fixed["status"] = "not_in_corpus"
            fixed["gaps"] = fixed.get("gaps") or ["引用修正后无可交付依据"]
        return fixed
