"""工具注册、调度与故障注入（docs/02-solution.md §8）。

约定：所有工具结果（含错误）都是统一形状的 dict，且包含 "tool" 字段；
错误对象为 {"tool": ..., "error": CODE, "message": ...}，绝不向循环抛异常。
"""
from __future__ import annotations

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "doc_search",
            "description": "在企业制度文档库中检索。返回命中文档的 ID、标题、版本日期、命中原文句子与得分；得分低于阈值的文档不会返回。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，建议使用制度用语"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5，上限 10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "doc_read",
            "description": "按文档 ID 读取完整的制度原文（按小节返回）。引用之前应先读取原文确认。",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string", "description": "文档 ID，如 D01"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "doc_list",
            "description": "列出文档库中全部文档的 ID、标题、版本日期与主题，用于判断资料是否存在。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "提交最终回答（唯一交付方式）。status 只能取 answered / needs_user_confirmation / not_in_corpus / tool_failure。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["answered", "needs_user_confirmation", "not_in_corpus", "tool_failure"],
                        "description": "任务状态",
                    },
                    "answer_text": {"type": "string", "description": "面向员工的回答或说明（中文）"},
                    "citations": {
                        "type": "array",
                        "description": "引用列表；answered 至少 1 条，needs_user_confirmation 应列出冲突双方",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_id": {"type": "string"},
                                "version_date": {"type": "string"},
                                "quote": {"type": "string", "description": "原文逐字片段"},
                            },
                            "required": ["doc_id", "version_date", "quote"],
                        },
                    },
                    "gaps": {"type": "array", "items": {"type": "string"}, "description": "资料缺口，not_in_corpus 时必填"},
                    "followup_question": {"type": "string", "description": "需要用户确认时的追问"},
                    "incomplete_reason": {"type": "string", "description": "未能完成时的原因说明"},
                },
                "required": ["status", "answer_text", "citations"],
            },
        },
    },
]

CONTROL_TOOLS = {"submit_answer"}
KNOWN_TOOLS = {"doc_search", "doc_read", "doc_list"}


def parse_fault_spec(spec: str) -> dict[tuple[str, int], str]:
    """解析 FAULT_INJECTION：`<tool>:<ordinal>[:<kind>]`，多条以 ; 分隔。

    submit_answer 不接受注入（被忽略）。非法片段直接报 ValueError，避免静默失效。
    """
    faults: dict[tuple[str, int], str] = {}
    if not spec or not spec.strip():
        return faults
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) < 2 or len(bits) > 3:
            raise ValueError(f"故障注入片段非法: {part!r}（应为 tool:ordinal[:timeout|io]）")
        tool, ordinal_s = bits[0].strip(), bits[1].strip()
        kind = bits[2].strip() if len(bits) == 3 else "timeout"
        if tool in CONTROL_TOOLS:
            raise ValueError(f"控制类工具不允许注入故障: {tool}")
        if tool not in KNOWN_TOOLS:
            raise ValueError(f"未知工具: {tool}")
        try:
            ordinal = int(ordinal_s)
        except ValueError:
            raise ValueError(f"故障注入序号非法: {part!r}") from None
        if ordinal < 1:
            raise ValueError(f"故障注入序号必须从 1 开始: {part!r}")
        if kind not in ("timeout", "io"):
            raise ValueError(f"故障类型非法: {kind!r}（应为 timeout 或 io）")
        faults[(tool, ordinal)] = kind
    return faults


class ToolRegistry:
    """本地确定性业务工具 + 故障注入。"""

    def __init__(self, corpus, fault_spec: str = ""):
        self.corpus = corpus
        self.faults = parse_fault_spec(fault_spec)
        self.dispatch_counts: dict[str, int] = {}
        self.fault_hits: list[dict] = []

    # ---- 调度 ----
    def dispatch(self, name: str, args: dict) -> dict:
        if name in CONTROL_TOOLS:
            raise ValueError("submit_answer 由主循环处理，不应进入 ToolRegistry.dispatch")
        ordinal = self.dispatch_counts.get(name, 0) + 1
        self.dispatch_counts[name] = ordinal
        kind = self.faults.get((name, ordinal))
        if kind:
            self.fault_hits.append({"tool": name, "ordinal": ordinal, "error_kind": kind})
            code = "TOOL_TIMEOUT" if kind == "timeout" else "TOOL_IO_ERROR"
            label = "超时" if kind == "timeout" else "IO 异常"
            return {
                "tool": name,
                "error": code,
                "message": f"{name} 第 {ordinal} 次调用模拟{label}（故障注入，后续调用将恢复）",
                "fault_injected": True,
            }
        try:
            if name == "doc_search":
                return self._doc_search(args)
            if name == "doc_read":
                return self._doc_read(args)
            if name == "doc_list":
                return self._doc_list()
            return {"tool": name, "error": "UNKNOWN_TOOL", "message": f"未知工具: {name}"}
        except Exception as exc:  # noqa: BLE001 - 统一错误形状，绝不向循环抛出
            return {"tool": name, "error": "TOOL_EXCEPTION", "message": str(exc)}

    # ---- 工具实现 ----
    def _doc_search(self, args: dict) -> dict:
        query = str(args.get("query", "") or "").strip()
        if not query:
            return {"tool": "doc_search", "error": "INVALID_ARGUMENTS", "message": "query 不能为空"}
        top_k = args.get("top_k")
        if top_k is not None and not isinstance(top_k, int):
            top_k = None
        res = self.corpus.search(query, top_k)
        return {"tool": "doc_search", "query": query, **res}

    def _doc_read(self, args: dict) -> dict:
        doc_id = str(args.get("doc_id", "") or "").strip()
        doc = self.corpus.get(doc_id)
        if doc is None:
            return {
                "tool": "doc_read",
                "error": "DOC_NOT_FOUND",
                "message": f"文档 {doc_id!r} 不存在，可先用 doc_list 查看全部文档 ID",
            }
        return {
            "tool": "doc_read",
            "doc_id": doc.doc_id,
            "title": doc.title,
            "version_date": doc.version_date,
            "sections": [dict(s) for s in doc.sections],
        }

    def _doc_list(self) -> dict:
        return {"tool": "doc_list", "docs": self.corpus.list_docs()}
