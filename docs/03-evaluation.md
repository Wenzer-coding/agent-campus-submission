# 03-evaluation.md — 测试预期、实际结果、失败与限制

## 1. 测试与运行环境

| 项 | 值 |
|---|---|
| 环境 | Windows 11 / Python 3.12.10 / openai SDK 2.54 / pytest 9.2 |
| 真实模型 | DeepSeek `deepseek-chat`（OpenAI 兼容 function calling，temperature=0.1，SDK 自带 60s 超时与 2 次重试） |
| 离线测试结果 | `pytest -q` → **52 passed, 2 deselected**（真实模型用例默认不跑），记录见 `examples/test-results.txt` |
| 真实模型测试 | `pytest -q -m real_model` → **2 passed**（无 key 环境自动 skip），同一文件有记录 |
| 真实运行记录 | `python -m src.cli batch examples/inputs --backend deepseek`，8 个场景 trace 见 `examples/runs/`，答案见 `examples/outputs/` |

## 2. 场景结果总表（实际运行，非模拟）

| 场景 | 输入 | 预期 | **实际结果** | 步数 | 关键证据文件 |
|---|---|---|---|---|---|
| E1 明确答案 | "年假有几天？" | answered，引用 D05，含适用条件，作答复读确认性检索 | ✅ answered；检索→清单→读取→换措辞确认性检索（miss）→交付；3 条 D05 逐字引用，含"累计工作满一年后"条件 | 5/8 | `examples/runs/E1.trace.json`、`examples/outputs/E1.answer.md` |
| E2 文档冲突 | "报销期限是多久？" | needs_user_confirmation，列 D01(30天)/D02(15天) 双引用 + 追问，不选边 | ✅ 完全符合：命中双文档→读取双方→2 次确认性检索→交付冲突说明 + 双引用 + 追问；明确拒绝以版本新旧选边，并给"按较严格期限把握"的处理建议 | 7/8 | `examples/runs/E2.trace.json`、`examples/outputs/E2.answer.md` |
| E3 资料外 | "公司有健身房吗？" | not_in_corpus，gaps 非空，不编造 | ✅ not_in_corpus；2 次不同措辞检索均 miss + 文档清单核实 → 判定资料外；gaps 列出 2 项缺口 | 4/8 | `examples/runs/E3.trace.json`、`examples/outputs/E3.answer.md` |
| E4 改写相似问题 | "出差住酒店一晚最多能报多少钱" | 口语问题经改写后命中依据并作答 | ✅ answered：S1 模型将口语问法改写为"出差住宿费标准 每晚限额"检索并命中 D06；读取 D06/D01 查证；S5/S6 两次确认性检索 miss → 判定无其他来源 → 作答（引用 D06 原文 400/300 元） | 7/8 | `examples/runs/E4.trace.json`、`examples/outputs/E4.answer.md` |
| E5 上下文追问 | ["年假有几天？", "那试用期员工呢？"] | 第二轮结合历史，引用 D05 试用期条款 | ✅ 第二轮直接复用上轮已读原文（doc_read）→ 检索"试用期 年假"（仅 D05 命中）→ 引用 D05"试用期员工不享有年假…"作答；全程 D05 三条引用逐字可核 | 8/12 | `examples/runs/E5.trace.json`、`examples/outputs/E5.answer.md` |
| E6 工具失败 | E1 问句 + `doc_read:1:timeout` | 观察到超时后重试或兜底，如实交付 | ✅ trace 记录 fault_hit（doc_read 第 1 次超时）→ 模型换措辞补检索 → 重试 doc_read 成功 → 正常作答 | 6/8 | `examples/runs/E6.trace.json`、`examples/outputs/E6.answer.md` |
| E7 步数上限 | E2 问句 + `max_steps=1` | step_limit 明确未完成 | ✅ step_limit；交付附"已完成动作摘要"+ 工具调用统计（doc_search×1） | 1/1 | `examples/runs/E7.trace.json`、`examples/outputs/E7.answer.md` |
| E8 换语料（审核要点自证） | E2 问句 + `--corpus-dir examples/corpus_alt`（D01→45 天、删除 D02） | 答案与动作随数据改变 | ✅ answered，引用 D01 原文"45 天内提交"；冲突场景随 D02 删除而消失 | 5/8 | `examples/runs/E8_corpus_alt.trace.json`、`examples/outputs/E8_corpus_alt.answer.md` |

**全局规则 G1（两条执行路径）证据对照**：
- **E2（冲突路径）**：S1 命中双文档 → S3/S4 读取双方原文 → S5/S6 确认性检索 → S7 交付冲突 + 追问：模型依据"结果冲突"调整动作；
- **E6（故障路径）**：S3 doc_read 超时（fault_hit）→ S4 换措辞补检索 → S5 重试 doc_read 成功 → S6 作答：模型依据"工具失败"调整动作；
- 补充路径：E4（查询精化 + 确认性检索 miss 后调整判定）、E3（miss → 改写 → 清单核实 → 拒答）。
每段 trace 的每一步都含 `assistant_rationale`（决策依据）、`tool_args`、`observation_summary` 与 `next_action`。

## 3. 自动化测试覆盖（tests/，离线确定性）

| 测试文件 | 覆盖 | 数量 | 结果 |
|---|---|---|---|
| `test_corpus.py` | 语料完整性、冲突双命中、单源命中、口语问题 miss、改写后命中、**命中片段相关性与逐字性**、top_k 边界与非法值、归一化保留标点、换语料内容 | 13 | 通过 |
| `test_grounding.py` | 真引用通过 / 伪造 quote 拒绝 / 未观察文档拒绝 / 版本不符拒绝 / **跨句拼接拒绝** / 错误观察不作证据 / schema 各状态约束 / **无引用状态携带引用被拒** / **冲突须双引用** | 13 | 通过 |
| `test_fault.py` | 注入文法校验、序号语义（第 2 次命中、之后恢复）、io/timeout 形状、DOC_NOT_FOUND、空 query、submit 不可注入 | 12 | 通过 |
| `test_agent_mock.py` | E1–E7 全部验收点（Mock 后端）、持久故障→tool_failure、伪造引用二次失败→grounding_failed、**批量多 tool_calls 的配对不变量**、**纯文本催促只算连续**、**无引用状态携带伪造引用→拒绝**、换语料行为改变、trace 必需字段 | 14 | 通过 |
| `test_real_model.py` | DeepSeek 真实冒烟 ×2（默认 deselected；`-m real_model` 时执行） | 2 | 通过 |

## 4. 关键机制的实际验证

1. **引用保真校验真实生效**：`test_forged_citation_leads_to_grounding_failed` 让模型连续两次提交不存在的"原文"，循环依次执行"错误观察写回 → 一次修复 → 二次失败 → `grounding_failed`（不交付未验证内容）"。
2. **冲突纪律真实生效**：E2 由真实模型自主产生 `needs_user_confirmation`，双引用均通过代码级校验；回答明确写出"不能仅凭版本日期新旧判断"。
3. **步数上限真实生效**：E7 在预算 1 步时停止并交付 `step_limit`，附已完成动作摘要，未执行任何未授权调用。
4. **换语料真实生效**：E8 与 E2 同一问题同一代码，仅语料不同，结果从"冲突待确认"变为"依据 D01 回答 45 天"——直接回应"替换文档内容核对答案与动作是否改变"。
5. **消息协议不变量**：真实批量运行曾暴露"成功交付后未补 tool 响应导致多轮 400"的缺陷，修复后新增 `test_multi_call_batch_keeps_pairing` 与 `test_text_nudge_resets_after_tool_call` 回归测试，防止复发。

## 5. 失败、限制与已知问题（如实记录）

1. **非确定性**：DeepSeek temperature=0.1 仍存在波动。E5 两次运行分别为 8 步与 9 步（内容等价）；E2 追问措辞每次略有不同。步数预算须留余量（E5 配置 12 步）。
2. **检索能力边界**：检索为字符 bigram 词法匹配，同义改写完全依赖模型（E4 即为证明——口语"住酒店"与制度"住宿费"词面无交集，原始问法检索 miss=0.0513；模型改写后命中）。E4 的确认性检索两次 miss 也说明：词法检索对"确认无其他来源"的支持有限，当前由模型的多轮检索 + 原文核对弥补。
3. **Mock 的边界**：Mock 是"观察特征 → 动作"的规则决策器（改写表为 `data/rewrite_map.json` 精确匹配），只验证流程、schema、校验器与边界分支可运行；**不证明真实模型决策效果**。所有决策效果证据以 DeepSeek trace 为准。
4. **引用校验的边界**：校验保证"quote 确实来自本会话观察到的原文"，但不判断该引用在语义上是否足以支撑结论（语义支撑由模型纪律 + 场景测试约束）。跨句拼接、伪造、错版本均被拒绝；`not_in_corpus`/`tool_failure` 状态的引用被强制为空，防止伪造引用借道。
5. **submit schema 的约束**：`answered`/`needs_user_confirmation` 必须带引用、冲突须双引用、冲突须追问；模型首次交付不合规时会消耗一次修复回合（scripted 测试覆盖该路径；真实运行 8 个场景均首次通过）。
6. **网络依赖**：无 key 时 CLI 自动回落 Mock 并打印边界告警；有 key 但网络异常时 CLI 输出可操作错误提示，不会静默降级——离线评审请显式 `--backend mock`。
7. **语料规模**：9 篇合成文档足以覆盖验收场景，但不足以代表真实企业知识库的召回噪声。

## 6. 结论

题面 6 项"必须验收"全部由真实模型运行记录 + 离线确定性测试双重覆盖（见 §2 表格与 tests/）；全局规则的步数上限、未完成状态、双执行路径、Mock 边界声明、替换语料验证均已落实并有对应文件。已知限制集中在检索词法与语义支撑判断，均有明确的后续改进方向（见 05-retrospective）。
