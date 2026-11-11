# data/rules.md — 数据规范与检索规则

## 1. 数据性质声明

- `data/docs/` 下全部文档均为**合成虚构数据**，用于考核演示，不代表任何真实企业的制度。
- 文档集刻意构造了：一对未裁决冲突（D01/D02 报销期限）、单源可答题（D05）、口语/制度措辞差异（D06）、以及刻意缺席的主题（健身房、班车、公积金）。
- 不包含任何真实个人隐私或业务数据。

## 2. 文档结构

每个 `D*.json`：

| 字段 | 含义 |
|---|---|
| `doc_id` | 文档唯一 ID（D01~D09） |
| `title` | 标题 |
| `version_date` | 版本日期（YYYY-MM-DD），引用时必须一致 |
| `topic` | 主题标签（供 doc_list 使用） |
| `sections[].heading` | 小节标题（参与检索索引） |
| `sections[].text` | 小节正文（参与检索索引；引用 quote 必须逐字取自 heading 或 text） |

`search_config.json`：检索参数（见 §3），不属于文档。

## 3. 检索评分规则（doc_search）

- 索引文本 = 标题 + 全部 heading + 全部 text。
- query 与索引文本均经 NFKC 归一化、去空白后取**字符 bigram 集合**。
- `coverage = |query bigrams ∩ 字段 bigrams| / |query bigrams|`
- `score = (body_coverage + title_weight × title_coverage) / (1 + title_weight)`，默认 `title_weight = 2.0`。
- 阈值 `threshold`（默认 0.2）：`score < threshold` 的文档不返回；响应始终包含全库 `best_score` 供模型判断"最近命中程度"。
- 排序键 `(-score, doc_id)`，保证结果全序确定；`top_k` 默认 5、上限 10。
- 评分是**确定性纯函数**，无随机性、无外部模型依赖，可离线复现。

## 4. 故障注入规则（演示与测试用）

- 文法：`FAULT_INJECTION="<tool>:<fail_on_ordinal>[:<error_kind>]"`，多条以 `;` 分隔。
- `fail_on_ordinal` 从 1 起始，按本次 agent 运行内该工具的**真实派发次数**累计（跨轮次累计）。
- `error_kind` ∈ {`timeout`, `io`}，默认 `timeout`；`submit_answer` 不接受注入。
- 故障以统一错误对象返回（`{"error": "TOOL_TIMEOUT", ...}`），后续调用恢复正常。

## 5. rewrite_map.json（仅 Mock 后端使用）

- 供**离线 Mock 决策器**在检索零命中时做表驱动查询改写；Mock 不读取场景 ID、不对问句做等值分支，只做"观察特征 → 动作"的规则决策。
- 真实模型后端不读取此表，由模型自行决定是否及如何改写查询。
- Mock 边界：只验证流程与边界分支，不证明真实模型决策效果。

## 6. 替换语料（审核要点演示）

`examples/corpus_alt/` 是替换语料示例：D01 的报销期限改为 45 天、删除 D02。
用 `--corpus-dir examples/corpus_alt` 运行同一问题，可核对：答案数值随文档变化、冲突场景消失（变为 `answered`）——用于证明行为完全由数据驱动。
