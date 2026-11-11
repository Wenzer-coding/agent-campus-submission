# 企业知识问答 Agent（小影校招 AI Agent 开发考核 · 题3）

一个基于 **模型驱动 ReAct 循环** 的企业制度问答 Agent：检索 → 读取 → 观察 → 动态调整（改写查询 / 继续查证 / 追问 / 拒答 / 交付）。

- **动态决策**：每一步动作（检索、读取、列清单、交付）由模型根据工具返回结果自主选择，循环骨架不含预设分支；
- **引用保真**：代码级校验——交付中的引用必须是本会话工具真实观察到的逐字原文（伪造 / 拼接 / 错版本一律拒绝，二次失败不交付）；
- **冲突纪律**：制度冲突且无裁决依据时，列出双方引用并请求用户确认，不擅自选边；
- **可离线复现**：Mock 后端无密钥可跑（仅验证流程，边界见下文）；
- **确定性测试**：`pytest` 49 条测试全离线可复现；8 个真实模型运行记录见 `examples/`。

---

## 快速开始

```bash
# 1. 安装（Python 3.10+，实测 3.12.10）
pip install -r requirements.txt -r requirements-dev.txt

# 2. 配置真实模型（可选；不配置会自动回落 Mock）
cp .env.example .env        # 填写 DEEPSEEK_API_KEY（Windows 可用 copy .env.example .env）

# 3. 运行
python -m src.cli ask "年假有几天？"                    # 真实模型
python -m src.cli ask "报销期限是多久？" --backend mock  # 离线 Mock（无需密钥）

# 4. 批量复跑全部示例场景（生成 trace 与答案）
python -m src.cli batch examples/inputs --outdir examples/runs --answers-dir examples/outputs

# 5. 测试
pytest -v
```

**换语料验证**（审核要点：替换文档内容后行为改变）：

```bash
python -m src.cli ask "报销期限是多久？" --corpus-dir examples/corpus_alt
# 预期：语料中 D01 为 45 天、D02 已删除 → 不再冲突，直接 answered
```

## 评审入口（10 分钟路线）

1. `docs/03-evaluation.md` —— 8 个场景的预期 vs 实际结果总表（含文件引用）；
2. `examples/outputs/E2.answer.md` + `examples/runs/E2.trace.json` —— 冲突处理路径（双引用 + 追问，逐步骤决策依据）；
3. `examples/outputs/E4.answer.md` + `examples/runs/E4.trace.json` —— 检索改进路径（miss → 改写查询 → 命中）；
4. `pytest -v` —— 49 条离线测试（覆盖全部验收点）；
5. `docs/02-solution.md` —— 设计取舍与关键契约（步数口径、校验状态机、Mock 边界）。

## 目录结构

```
├── src/
│   ├── agent.py        # ReAct 主循环：步数预算 / 观察写回 / trace 记账
│   ├── grounding.py    # 引用保真校验（ObservationLedger + schema 校验）
│   ├── llm.py          # DeepSeek（OpenAI 兼容）与离线 Mock 决策器
│   ├── corpus.py       # 语料加载 + 确定性 bigram 检索
│   ├── prompts.py      # system prompt（引用/冲突/缺口/失败纪律）
│   ├── tools/          # doc_search / doc_read / doc_list + 故障注入
│   └── cli.py          # ask / batch 命令行入口
├── tests/              # pytest（离线确定性 + 真实模型冒烟）
├── examples/
│   ├── inputs/         # E1–E8 场景输入
│   ├── outputs/        # 实际答案（markdown，可读版）
│   ├── runs/           # 实际执行 trace（JSON，含每步参数/返回/依据）
│   └── corpus_alt/     # 替换语料（D01→45 天，删除 D02）
├── data/
│   ├── docs/           # 9 篇合成制度文档（含参考实例的 D01/D02 冲突对）
│   ├── rules.md        # 数据规范与检索/注入规则
│   └── rewrite_map.json# Mock 改写表（真实模型不使用）
└── docs/               # 01-requirements ~ 05-retrospective
```

## 状态枚举

| status | 含义 | 由谁产生 |
|---|---|---|
| `answered` | 有依据的明确回答（citations 均通过保真校验） | 模型 |
| `needs_user_confirmation` | 未裁决冲突/多义，列双方引用 + 追问 | 模型 |
| `not_in_corpus` | 资料外问题，给出 gaps，不编造 | 模型 |
| `tool_failure` | 工具持续失败，如实返回失败 | 模型 |
| `step_limit` | 达到步数上限，明确未完成 | 循环层 |
| `grounding_failed` | 引用校验二次失败，不交付未验证内容 | 循环层 |

## 运行模式与 Mock 边界

- `--backend deepseek`：真实模型（需 `DEEPSEEK_API_KEY`）；
- `--backend mock`：离线规则决策器（**只验证流程、schema、校验器与边界分支可运行，不证明真实模型决策效果**；真实决策效果的证据以 `examples/runs/` 中的 DeepSeek trace 为准）；
- `--backend auto`（默认）：有 key 用 DeepSeek，无 key 自动回落 Mock 并打印边界告警。

## 执行记录（trace）字段

每个场景的 trace（`examples/runs/E*.trace.json`）包含：后端/模型/语料哈希/故障注入配置/步数预算；每步含 `step_index`、`tool_name`、`tool_args`、`assistant_rationale`（决策依据）、`observation_summary`（返回摘要）、`observation_sha256`、`next_action`（后续动作）、`fault_hit`、`validation`（引用校验结果）与步数计数；每轮末尾附最终交付全文。

## 已知限制

检索为词法 bigram（同义改写依赖模型）；引用校验保证"引用真实"，语义支撑由场景测试与模型纪律保障；Mock 不证明决策效果；详见 `docs/03-evaluation.md §5`。

## 数据声明

`data/` 与 `examples/corpus_alt/` 均为**合成虚构数据**，不含任何真实企业制度或个人隐私；密钥不进入仓库（`.env` 已 gitignore）。
