"""CLI 入口：ask（单轮）/ batch（批量场景）。

运行：
  python -m src.cli ask "报销期限是多久？"
  python -m src.cli ask "年假有几天？" --backend mock
  python -m src.cli batch examples/inputs --outdir examples/runs --answers-dir examples/outputs
  python -m src.cli ask "报销期限是多久？" --corpus-dir examples/corpus_alt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from src.agent import run_agent
    from src.corpus import Corpus
    from src.llm import DeepSeekBackend, MockBackend
else:
    from .agent import run_agent
    from .corpus import Corpus
    from .llm import DeepSeekBackend, MockBackend


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if (stream.encoding or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def build_backend(kind: str, corpus):
    if kind == "mock":
        return MockBackend(corpus)
    if kind == "deepseek":
        return DeepSeekBackend()
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekBackend()
    print(
        "[warn] 未检测到 DEEPSEEK_API_KEY，自动回落到离线 Mock 后端"
        "（Mock 仅验证流程与边界分支，不证明真实模型决策效果）。",
        file=sys.stderr,
    )
    return MockBackend(corpus)


def render_answer_md(res: dict, scenario_id: str) -> str:
    payload = res["payload"] or {}
    trace = res["trace"]
    lines = [
        f"# {scenario_id}",
        "",
        f"- 后端：{trace['backend']}（{trace['model']}）",
        f"- 状态：{payload.get('status')}",
        f"- 步数：{trace['total_steps']} / {trace['max_steps']}",
        f"- 语料目录：{trace['corpus_dir']}（sha256 `{trace['corpus_hash'][:16]}…`）",
        f"- 故障注入：{trace['fault_injection'] or '无'}",
        f"- 运行时间：{trace['timestamp']}",
        f"- Trace：examples/runs/{scenario_id}.trace.json",
        "",
        "## 回答",
        "",
        payload.get("answer_text", ""),
        "",
    ]
    citations = payload.get("citations") or []
    if citations:
        lines += ["## 引用", "", "| 文档 | 版本日期 | 原文片段 |", "|---|---|---|"]
        for c in citations:
            lines.append(f"| {c.get('doc_id')} | {c.get('version_date')} | {c.get('quote')} |")
        lines.append("")
    gaps = payload.get("gaps") or []
    if gaps:
        lines += ["## 资料缺口", ""] + [f"- {g}" for g in gaps] + [""]
    if payload.get("followup_question"):
        lines += ["## 追问", "", payload["followup_question"], ""]
    if payload.get("incomplete_reason"):
        lines += ["## 未完成原因", "", payload["incomplete_reason"], ""]
    return "\n".join(lines)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_ask(args) -> int:
    corpus = Corpus(args.corpus_dir)
    backend = build_backend(args.backend, corpus)
    fault = args.fault or os.getenv("FAULT_INJECTION", "")
    res = run_agent(
        args.question,
        corpus=corpus,
        backend=backend,
        max_steps=args.max_steps,
        fault_spec=fault,
        scenario_id="ask",
    )
    payload = res["payload"]
    print(f"状态：{payload.get('status')}    步数：{res['steps_used']}/{args.max_steps}    后端：{backend.name}")
    print("─" * 60)
    print(payload.get("answer_text", ""))
    for c in payload.get("citations") or []:
        print(f"  引用 [{c.get('doc_id')} v{c.get('version_date')}]：{c.get('quote')}")
    if payload.get("gaps"):
        print(f"  资料缺口：{'；'.join(payload['gaps'])}")
    if payload.get("followup_question"):
        print(f"  追问：{payload['followup_question']}")
    if payload.get("incomplete_reason"):
        print(f"  未完成原因：{payload['incomplete_reason']}")
    if args.trace_out:
        _write_json(Path(args.trace_out), res["trace"])
        print(f"\nTrace 已写入：{args.trace_out}")
    return 0 if payload.get("status") not in ("grounding_failed",) else 2


def cmd_batch(args) -> int:
    inputs_dir = Path(args.inputs_dir)
    files = sorted(inputs_dir.glob("*.json"))
    if not files:
        print(f"[error] 输入目录没有 JSON 场景：{inputs_dir}", file=sys.stderr)
        return 1
    outdir, answers_dir = Path(args.outdir), Path(args.answers_dir)
    results = []
    for f in files:
        spec = json.loads(f.read_text(encoding="utf-8-sig"))
        sid = spec.get("scenario_id") or f.stem
        turns = spec.get("turns")
        if not turns:
            print(f"[error] {f} 缺少 turns 数组", file=sys.stderr)
            return 1
        corpus = Corpus(spec.get("corpus_dir") or args.corpus_dir)
        backend = build_backend(spec.get("backend") or args.backend, corpus)
        fault = spec.get("fault_injection") or args.fault or os.getenv("FAULT_INJECTION", "")
        res = run_agent(
            turns,
            corpus=corpus,
            backend=backend,
            max_steps=int(spec.get("max_steps") or args.max_steps),
            fault_spec=fault,
            scenario_id=sid,
        )
        _write_json(outdir / f"{sid}.trace.json", res["trace"])
        answers_dir.mkdir(parents=True, exist_ok=True)
        (answers_dir / f"{sid}.answer.md").write_text(render_answer_md(res, sid) + "\n", encoding="utf-8")
        results.append((sid, res["payload"].get("status"), res["steps_used"], backend.name))
        print(f"[{sid}] status={res['payload'].get('status')} steps={res['steps_used']} backend={backend.name}")
    print(f"\n共 {len(results)} 个场景 → trace: {outdir} / answers: {answers_dir}")
    return 0


def main(argv=None) -> int:
    _ensure_utf8_stdio()
    _load_env()
    parser = argparse.ArgumentParser(prog="python -m src.cli", description="企业知识问答 Agent（题3）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="单轮提问")
    p_ask.add_argument("question", help="要问的问题")
    p_ask.add_argument("--trace-out", default="", help="将 trace 写入指定 JSON 文件")

    p_batch = sub.add_parser("batch", help="批量运行 examples/inputs 场景")
    p_batch.add_argument("inputs_dir", help="场景输入目录（*.json）")
    p_batch.add_argument("--outdir", default=str(REPO_ROOT / "examples" / "runs"), help="trace 输出目录")
    p_batch.add_argument(
        "--answers-dir", default=str(REPO_ROOT / "examples" / "outputs"), help="答案 markdown 输出目录"
    )

    for p in (p_ask, p_batch):
        p.add_argument("--backend", choices=["auto", "deepseek", "mock"], default="auto")
        p.add_argument("--corpus-dir", default=str(REPO_ROOT / "data" / "docs"))
        p.add_argument("--max-steps", type=int, default=8)
        p.add_argument("--fault", default="", help='故障注入，如 "doc_read:1:timeout"')

    args = parser.parse_args(argv)
    try:
        if args.cmd == "ask":
            return cmd_ask(args)
        return cmd_batch(args)
    except Exception as exc:  # noqa: BLE001 - 网络类异常给出可操作提示
        if exc.__class__.__name__ in ("APIConnectionError", "APITimeoutError", "APIStatusError"):
            print(
                "[error] 无法访问 DeepSeek API（网络异常或服务不可用）。"
                "离线评审请显式使用 --backend mock。",
                file=sys.stderr,
            )
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
