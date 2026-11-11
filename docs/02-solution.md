# 02-solution.md — 设计方案（题3：企业知识问答 Agent）

> 状态：v2 定稿。v1 经两路独立评审（考核规范合规 / 工程可行性）共 19 条意见修订，关键契约已全部定死。
> 本文档是实现契约：代码、测试、数据、运行记录均以此为准；实现与文档出现偏差时，以 03-evaluation.md 记录的实际行为为准并说明原因。

## 1. 选题理由与需求回顾

四个题目结构相似（工具 + 动态决策循环 + 边界场景），选**题3 企业知识问答 Agent**，理由：

| 维度 | 题3 表现 |
|---|---|
| 验收点数量 | 6 项（明确答案 / 文档冲突 / 资料外问题 / 改写相似问题 / 上下文追问 / 工具失败），四题最多 |
| 工具与数据复杂度 | 最低：检索类只读工具 + 本地 JSON 文档集，出错面最小 |
| ReAct 路径丰富度 | 最高：检索改进（改写查询）、读取原文、冲突查证、上下文追问、拒答、工具失败兜底，天然形成多条不同执行路径 |
| 行为定义精确度 | 最难的"文档冲突"场景，题目参考实例给出了精确期望行为 |
| 审核要点应对 | "替换文档内容，核对答案与动作是否改变"——本设计全部行为数据驱动，并提供 `--corpus-dir` 换库入口与参数化测试 |

需求回顾（题面）：帮助员工找到制度或产品问题的可靠答案；输入为问题及可选对话历史、带文档 ID 与版本日期的合成制度；交付回答、引用（文档 ID + 原文片段）、资料缺口与任务状态；模型根据检索结果动态选择补充检索、读取原文、追问或停止，资料冲突时说明冲突并查证，**不凭空选一个答案**；必须验收六项：明确答案、文档冲突、资料外问题、改写后的相似问题、上下文追问、工具失败。

## 2. 总体架构

```
CLI (src/cli.py: ask / batch, --backend deepseek|mock, --corpus-dir, --max-steps, --fault)
        │
AgentLoop (src/agent.py)
  state: messages[] · steps_used · ObservationLedger · repair_quota
  循环: 模型决策(tool_calls 或文本) → 预算检查 → 执行工具 → 观察写回 → trace 记账
  终止: submit_answer 通过校验 / 步数耗尽(step_limit) / 引用校验二次失败(grounding_failed)
        │
  ┌─────┴──────────────┬─────────────────────┐
LLM 适配层 src/llm.py  工具层 src/tools/        校验器 src/grounding.py
 · DeepSeek(OpenAI兼容)  · doc_search 词法检索    · 引用保真校验(基于 Ledger)
 · Mock 规则决策器        · doc_read 结构化读取    · schema 校验
                        · doc_list 文档清单
                        · submit_answer 终止信号
        │
数据层 src/corpus.py ← data/docs/*.json · data/rules.md · data/rewrite_map.json
```

## 3. 运行形态与技术取舍

| 决策 | 选择 | 理由 | 放弃的选项 |
|---|---|---|---|
| 运行形态 | **CLI** | 可复制命令最直接、trace 落盘便于评审、零服务依赖 | HTTP API / Notebook（增加与考核无关的层） |
| 框架 | **原生实现** | 评分核心即"模型决策循环"本身；原生循环把决策点完全暴露给评审；依赖最少 | LangGraph/ADK（框架遮蔽决策点；规则明示"框架名称不加分，能解释取舍才有价值"） |
| 模型接入 | **OpenAI 兼容协议 + DeepSeek** | 支持 function calling、价格低、openai SDK 为事实标准 | 私有协议直连（无收益） |
| Mock | **观察特征规则决策器** | 考核要求离线入口；与真实后端共用循环、工具、校验，仅替换决策函数 | 录制回放（不能应对改写场景） |
| 检索 | **确定性字符 bigram 覆盖率评分** | 无依赖、可复现、可离线断言；检索弱点由模型"改写查询"动态弥补（正是要考察的路径） | 向量检索（引入不可复现依赖，对合成语料无收益） |
| 语言 | **Python 3.12**（Windows 实测） | 环境既有、评审友好 | — |

依赖：`requirements.txt` = `openai>=1.40`、`python-dotenv>=1.0`；开发依赖 `requirements-dev.txt` = `pytest>=8.0`。

## 4. ReAct 主循环（src/agent.py）——步数与写回语义（定死）

### 4.1 步数计数（唯一口径）

- **1 步 = 1 次工具调用**，`submit_answer` 同样计 1 步。`MAX_STEPS` 默认 8，CLI `--max-steps` 可调。
- 一条 assistant 消息可携带多个 `tool_calls`（DeepSeek 可能并行下发）：按顺序逐个执行、逐个扣预算；预算不足时只执行前 k 个，其余以错误观察写回。
- **预算归零后不再执行任何工具调用（含 submit_answer）**，立即终止并返回 `status="step_limit"` 交付，附已完成的观察摘要与未完成原因。
- 不依赖 API 的 `parallel_tool_calls` 参数（DeepSeek 支持情况不明确），预算一律代码层执行。

### 4.2 消息写回状态机

```
(a) 普通工具结果 → role="tool", tool_call_id=对应调用，content=JSON 结果
(b) submit_answer 的 schema 违例或引用校验失败
    → 以【同一个 submit_answer 的 tool_call_id】写回 role="tool" 错误明细 JSON
    → repair_quota=1（仅一次）；修复轮内允许普通工具调用（正常扣预算）
    → 第二次 submit 仍失败 → 直接落终态 grounding_failed，不再修复
(c) 模型返回纯文本、无 tool_calls → 无 id 可配对，写 role="user" 催促消息
    （"请调用工具，最终必须用 submit_answer 交付"），计 1 步；连续 2 次 → step_limit
```

不变量：**会话中每条 assistant tool_calls 消息里的每个 tool_call_id 都必须有配对的 tool 响应**（含成功的 submit_answer），否则多轮消息序列非法。

### 4.3 决策依据与 trace

system prompt 要求模型在每次工具调用前用一两句话说明选择依据（写在 assistant message content 中），trace 原样记录为 `assistant_rationale`——满足"步骤、工具参数、返回摘要、**后续动作**及简要决策依据"，不含模型内部思维过程。

循环骨架只负责执行、预算、记账、校验，**不含任何"检索到 X 就做 Y"的预设分支**；每一步工具选择均由模型给出。

## 5. 交付结构（submit_answer schema）

```json
{
  "status": "answered | needs_user_confirmation | not_in_corpus | tool_failure",
  "answer_text": "面向员工的回答或说明",
  "citations": [{"doc_id": "D01", "version_date": "2024-06-01", "quote": "逐字原文片段"}],
  "gaps": ["未找到关于健身房的规定"],
  "followup_question": "请问适用总部制度还是分公司细则？",
  "incomplete_reason": null
}
```

- **模型可填 4 值**；`step_limit` 与 `grounding_failed` 由**循环层**追加，不接受模型自填。
- `answered`：citations 非空且全部通过保真校验。
- `needs_user_confirmation`：文档冲突/多义，列双方引用 + `followup_question`，不选边。
- `not_in_corpus`：资料外，`gaps` 非空，不编造。
- `tool_failure`：**同一工具连续 ≥2 次失败**（含注入故障重试后仍失败），如实返回失败事实。
- `grounding_failed`（循环层）：修复后仍校验失败 → citations 置空、answer_text 改为"生成内容未通过引用校验，已不予交付"、`incomplete_reason` 记录失败规则明细。
- **无 citations 的状态（not_in_corpus / tool_failure / step_limit / grounding_failed）的 citations 必须为空**——携带引用即判校验失败，防止伪造引用借道绕过保真校验。
- `needs_user_confirmation` 须至少携带两条引用（冲突双方），由 schema 校验强制。

## 6. 引用保真校验（src/grounding.py）——代码级防线

### 6.1 ObservationLedger

工具层每次成功观察后由循环记账：`{doc_id, version_date, spans[], source_tool, msg_index}`。校验器**只查 Ledger，不扫消息文本**。

### 6.2 校验规则

1. `doc_id` 必须出现在**携带内容的成功观察**中（search 命中片段或 read 成功）；错误观察（DOC_NOT_FOUND / 超时）只作负证据。
2. `quote` 必须逐字出现在该文档被观察到的某个 span 中。匹配 = `norm(quote)` 是 `norm(span)` 的子串；`norm` = NFKC + casefold + Unicode 空白折叠为单空格 + 去零宽字符，**不删标点**（防跨句拼接）。
3. `version_date` 必须与观察到的版本日期一致。
4. 无 citations 的交付跳过引用校验，且 schema 强制这些状态的引用为空（见 §5），杜绝伪造引用借道。

失败 → 错误明细（规则号、双方归一化文本、doc_id）写回 → 修复一轮（§4.2b）；再失败 → `grounding_failed`。

### 6.3 snippet 规则

`doc_search` 的 snippet 为**整句截取**（按 `。！？…\n` 分句，取命中句原样返回），不做字符截断、不加省略号——否则合法引用会被误杀。`doc_read` 只返回 `sections`（heading + text），不重复提供 `content` 字段；quote 必须落在单一 section（heading 或 text）内。

## 7. 冲突处理策略（对应参考实例）

**作答纪律（写入 system prompt，由模型执行，测试固化）**：

1. 两来源对同一问题给出不同事实、且**均未声明适用范围、优先级或替代关系** → 未裁决冲突。版本日期先后 ≠ 优先级，不得以新压旧。
2. 交付 `needs_user_confirmation`：列出双方说法与各自引用，说明无法确定适用哪份，`followup_question` 请用户确认（如"请问您适用《差旅费用报销管理办法》还是《费用报销操作指引》？"）。
3. **禁止**无依据选边。
4. **确认性检索纪律**：以单一来源作 `answered` 前，必须至少再执行一次确认性检索（换词或提高 top_k）以排除未发现的冲突文档——防止"只搜到 D01 就答 30 天"的首查即答反例（题目明令禁止的方向）。
5. 若检索到裁决性信息（某文档声明替代关系/适用范围），以裁决信息收敛为 `answered` 并引用裁决原文。

## 8. 工具层（src/tools/）——接口级规格

| 工具 | 参数 | 返回 |
|---|---|---|
| `doc_search` | `query: str, top_k: int = 5` | `{results: [{doc_id, title, version_date, snippet, score}], best_score, empty_reason?}` |
| `doc_read` | `doc_id: str` | `{doc_id, title, version_date, sections: [{heading, text}]}` 或 `{error: "DOC_NOT_FOUND"}` |
| `doc_list` | 无 | `{docs: [{doc_id, title, version_date, topic}]}` |
| `submit_answer` | §5 schema | 终止信号（不返回内容） |

### 8.1 doc_search 评分（确定性）

- 索引文本 = 标题 + 全部 `heading + text`；query 与索引文本均做 NFKC + 去空白后取字符 bigram 集合。
- `score = (body_coverage + 2 × title_coverage) / 3`，coverage = |query bigrams ∩ 字段 bigrams| / |query bigrams|，值域 [0,1]。
- **阈值 T = 0.2**（`data/docs/search_config.json` 可配置）：`score < T` 不返回；响应始终附 `best_score` 供模型判断。
- query 归一化后 bigram 集为空（单字/纯空白）→ `{results: [], best_score: 0, empty_reason: "query too short"}`。
- `top_k` 默认 5、上限 10，非法值（0/负数/非整数）纠正为默认；排序键 `(-score, doc_id)` 保证全序确定。

### 8.2 故障注入

- 文法：`FAULT_INJECTION="<tool>:<fail_on_ordinal>[:<error_kind>]"`，多条用 `;` 分隔；`error_kind ∈ {timeout, io}`，默认 timeout。
- 语义：本次 agent 运行内该工具第 N 次真实派发（1 起始，跨轮累计）抛出该故障，后续调用恢复正常；`submit_answer` 不接受注入。
- 故障以**统一 error 对象**返回而非抛异常：`{"error": "TOOL_TIMEOUT", "tool": "doc_read", "message": "..."}`；循环捕获一切工具异常转换为 `{"error": "TOOL_EXCEPTION", ...}`，保证模型观察形状一致。
- trace 记录注入配置与命中序号。

## 9. 数据集设计（data/）

### 9.1 文档结构

```json
{
  "doc_id": "D01",
  "title": "差旅费用报销管理办法",
  "version_date": "2024-06-01",
  "topic": "差旅报销",
  "sections": [{"heading": "报销期限", "text": "差旅费用报销申请应在费用发生后 30 天内提交，逾期不予受理。"}]
}
```

### 9.2 文档集（全部合成虚构，data/rules.md 声明）

| 文档 | 主题 | 设计意图 |
|---|---|---|
| D01 差旅费用报销管理办法 v2024-06-01 | 正文含报销期限 30 天（未声明适用范围/优先级） | 参考实例冲突方一 |
| D02 费用报销操作指引 v2024-08-15 | 正文含报销期限 15 天（未声明适用范围/优先级/替代关系） | 参考实例冲突方二 |
| D03 考勤与作息管理规定 v2024-01-15 | 作息 9:00-18:00、打卡 | 干扰项 |
| D04 发票开具与领取说明 v2024-05-10 | 发票抬头税号流程 | 干扰项（不含"报销"bigram） |
| D05 年假管理制度 v2024-01-10 | 满一年 5 天年假；试用期员工不享有 | 明确答案 + 上下文追问 + E1 适用条件 |
| D06 差旅住宿标准 v2024-03-20 | 一线 400 元/晚、其他 300 元/晚 | 改写相似问题目标（口语问法 vs 制度措辞） |
| D07 新员工入职与转正规定 v2024-02-01 | 试用期 3 个月、转正流程（标题避开"试用期"防污染 E5 检索） | 干扰项 |
| D08 办公设备申领流程 v2024-04-08 | 电脑外设申领审批 | 干扰项（不含"公司"字样） |
| D09 培训活动组织规范 v2024-07-22 | 培训报备与费用列支（措辞避开"报销"防污染 E2） | 干扰项 |
| （刻意缺席：健身房、班车、公积金） | — | 资料外问题 |

### 9.3 场景可触发性（bigram 覆盖率预验证，阈值 T=0.2）

| 场景 query | 首查结果（实测） | 依据 |
|---|---|---|
| "报销期限是多久" | D01=0.2778、D02=0.2778（双命中） | 两文档均含"报销期限"字面 |
| "年假有几天" | 仅 D05=0.25 | D07 无"年假"bigram 被滤除 |
| "出差住酒店一晚最多能报多少钱" | **无命中（best_score=0.0513 < 0.2）** | 口语措辞与制度措辞词面差异是场景引擎 |
| 改写后"差旅住宿标准 每晚限额 报销" | D06=0.4545 > D01=0.2121 | 标题权重使目标文档显著领先 |
| "试用期 年假"（E5 T2） | D05=0.4167 单独命中 | D07 低于阈值被滤除 |

`examples/corpus_alt/`：D01 的 30 天改为 45 天、删除 D02 的替换语料，用于演示与测试"替换文档内容 → 答案与动作随之改变"（`--corpus-dir` 切换）。

## 10. 模型接入与 Mock 边界（src/llm.py）

- **DeepSeek 后端**：OpenAI 兼容 `/chat/completions`，`tools=[...]` function calling；`OpenAI(timeout=60, max_retries=2)` 用 SDK 自带重试（不叠加自写外层重试）；环境变量 `DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL`，`.env` 加载，`.env.example` 提供模板，密钥不入库。
- **无 key 且未指定 `--backend`**：告警并自动回落 mock，保证零配置可运行。
- **Mock 后端**：与真实后端同一循环、工具层、校验器；决策函数为**观察特征向量 + 优先级策略**的纯函数——特征取自最近工具结果/消息状态（`last_search_results、best_score、是否已 read、同槽位数值冲突、pending 校验错误、故障未重试、last_action、历史问题`），按优先级出动作；**不读取场景 ID、不对问句做等值分支**；查询改写用 `data/rewrite_map.json` **精确匹配**（未入表的问法不做通用兜底，Mock 只覆盖入表变体，不声称通用改写能力）；Mock 同样产出 `assistant_rationale`，trace schema 与真实后端一致。
- **边界声明**（README、04-ai-usage 同步）：Mock 只验证流程、schema、校验器与边界分支的可运行性，**不证明真实模型决策效果**；ReAct 决策效果证据以 DeepSeek 真实运行 trace 为准。

## 11. 多轮对话

- 输入 schema：`{"scenario_id": "E5", "turns": ["...", "..."]}`，单轮也为长度 1 的数组。
- **历史压缩为代码级抽取式**（不调模型）：保留最近 2 轮的 user 问句原文 + `answer_text` + `status` + citations（含 quote 原文），总长封顶 1200 字符；统一放在 **user 消息前导**（system 保持稳定利于前缀缓存）。
- ObservationLedger 按 session 累计不随轮清空——T2 可引用 T1 观察过的原文；system prompt 要求"追问若引用上轮结论，逐字复用其 quote，否则重新 doc_read"。
- CLI `ask` 为单轮入口；多轮演示走 `batch` 输入文件。

## 12. 测试策略（tests/，离线确定性）

| 层 | 测试 | 断言 |
|---|---|---|
| 单元 | 语料加载校验 | 9 文档字段完整、doc_id 唯一 |
| 单元 | 检索排序 | "报销期限是多久" top_k 同时含 D01/D02；"年假有几天"仅 D05；E4 首查 best_score < T |
| 单元 | 归一化 | NFKC 全半角、空白折叠、不删标点；quote 跨句拼接被拒 |
| 单元 | 引用校验 | 真引用过 / 伪造 quote 拒 / doc_id 未观察拒 / version_date 不符拒 / 错误观察不算证据 |
| 单元 | 步数预算 | max_steps=1 时首个工具后 submit 被拒 → step_limit，附观察摘要 |
| 单元 | 故障注入文法 | 序号语义、跨轮累计、submit 不可注入 |
| Agent 级（mock） | 六场景（正常/冲突/资料外/改写/多轮/工具失败） | 各自 status、citations、gaps 断言 |
| Agent 级（mock） | 换语料 | `--corpus-dir examples/corpus_alt` 后 D01 改 45 天、冲突消失 |
| Agent 级（real） | DeepSeek 冒烟（`@pytest.mark.real_model`，无 key skip） | 一次完整问答 + 引用校验通过 |

## 13. 示例场景矩阵（examples/，真实模型 DeepSeek 运行）

每组产出：`examples/inputs/E*.json` + `examples/outputs/E*.answer.md` + `examples/runs/E*.trace.json`。trace 头部：`backend、model、corpus_hash(data/docs sha256)、fault_injection、max_steps、timestamp、dependency versions`；每步：`turn_index、step_index、tool_name、tool_args、observation_summary(≤300字符) + observation_sha256、assistant_rationale、next_action、latency_ms、steps_used/budget、fault_hit、validation{passed, failed_checks}`；每轮末尾附 submit payload 全文。

| 场景 | 输入 | 期望行为 | 覆盖验收点 | 规则条款 |
|---|---|---|---|---|
| E1 正常问答 | "年假有几天？" | answered，引用 D05；answer_text 含适用条件（"累计工作满一年"）；trace 含 ≥2 次检索（确认性检索纪律） | 明确答案 | 必须验收① |
| E2 文档冲突 | "报销期限是多久？" | needs_user_confirmation，列 D01(30天)/D02(15天) 双引用 + 追问；不选边 | 文档冲突 | 必须验收②、参考实例 |
| E3 资料外 | "公司有健身房吗？" | not_in_corpus，gaps 指明缺口，不编造 | 资料外问题 | 必须验收③ |
| E4 改写相似问题 | "出差住酒店一晚最多能报多少钱" | 口语问法经模型改写为制度用语检索并命中 D06；两次确认性检索 miss 后判定无其他来源 → answered（真实运行路径，见 03；离线 Mock 测试另覆盖"首查 miss→改写→命中"机制） | 改写后的相似问题（检索改进） | 必须验收④、审核要点"检索改进" |
| E5 上下文追问 | turns: ["年假有几天？", "那试用期员工呢？"] | T2 结合历史检索"试用期 年假"→ 引用 D05 试用期条款 | 上下文追问 | 必须验收⑤ |
| E6 工具失败 | E1 问句 + `--fault doc_read:1:timeout` | 观察到 TOOL_TIMEOUT → 重试或改用检索片段兜底 → 如实交付 | 工具失败 | 必须验收⑥ |
| E7 步数上限 | E2 问句 + `--max-steps 1` | step_limit 明确未完成 + 已观察摘要 | 步数上限 | 全局规则 |
| E8 换语料 | E2 问句 + `--corpus-dir examples/corpus_alt`（D01→45 天、删除 D02） | answered 引用 D01 原文 45 天，冲突消失 | 替换文档内容验证 | 审核要点 |

**两条执行路径证据**：E2（双命中→读取双方原文→确认性检索→请求确认）与 E6（超时→重试成功）形成最鲜明的"模型根据不同工具结果调整动作"对照；E4（查询精化 + 确认性检索 miss 后调整判定）、E3（两次 miss→清单→拒答）、E1/E5/E7/E8 提供额外路径。03-evaluation.md 逐文件引用。

## 14. 交付物清单与规则映射

| 规则要求 | 交付物 |
|---|---|
| README（环境/安装/运行/评审入口） | README.md |
| src/ | src/（corpus.py, llm.py, agent.py, grounding.py, tools/, cli.py） |
| tests/ 可执行测试 | tests/（pytest，离线确定性） |
| examples/（输入、实际输出、Trace） | examples/（inputs/outputs/runs/corpus_alt） |
| data/（合成资料和规则） | data/docs、data/rules.md、data/rewrite_map.json |
| docs 01-requirements | 用户、目标、范围、成功标准 |
| docs 02-solution | 本文档 |
| docs 03-evaluation | 测试预期、实际结果、失败与限制，引用 examples/ 文件 |
| docs 04-ai-usage | AI 辅助范围、个人贡献、验证结果 |
| docs 05-retrospective | 最难问题、学习收获、下一步、**实际用时与未完成部分** |
| 依赖文件 | requirements.txt、requirements-dev.txt |
| .env.example | 是（README 注明变量用途；无密钥提交） |
| 运行命令可复制 | README §运行 |
| Mock 边界声明 | README、docs/02 §10、docs/04 |
| 步数上限 + 未完成状态 | §4 + E7 trace |
| 至少 3 组输入覆盖全部验收点 | E1–E8（8 组） |

## 15. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型捏造引用/数值 | submit_answer 结构化 + Ledger 保真校验 + 修复一轮 + grounding_failed 终态 |
| 模型冲突场景擅自选边 | 纪律入 system prompt + 确认性检索纪律 + E2 断言 + 测试固化 |
| E4 检索路径随模型波动 | 模型自主选择查询；实测于首次检索前完成口语→制度用语改写并命中，确认性检索 miss 后判定无其他来源。离线测试覆盖"首查 miss→改写→命中"机制，实际路径如实记录于 03 |
| 多轮历史撑爆上下文 | 抽取式压缩 ≤1200 字符 |
| DeepSeek 波动 | SDK 重试 + mock 兜底 + 失败如实记录 |
| 评审替换语料 | 全数据驱动 + `--corpus-dir` + corpus_alt 演示 + 参数化测试 |

## 16. 运行命令（验证后定稿）

```bash
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env          # 填 DEEPSEEK_API_KEY（无 key 时 mock 自动回落）

python -m src.cli ask "报销期限是多久？"
python -m src.cli ask "年假有几天？" --backend mock
python -m src.cli batch examples/inputs --backend deepseek
# 注意：上面这条命令会覆盖 examples/runs 与 examples/outputs 中已提交的真实运行证据，
# 复跑前请确认 .env 已配置有效 key；离线演示请改用 --backend mock 并指定其他输出目录。
python -m src.cli ask "报销期限是多久？" --corpus-dir examples/corpus_alt
pytest -q                      # 52 条离线测试；真实模型冒烟：pytest -m real_model
```
