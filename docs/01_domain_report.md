# Phase 1 领域化报告：城市轨道交通运维知识库

> 生成时间：2026-10-02
> 本报告记录语料构成、文档导入结果与 3 组真实运维问答样例。

## 一、语料构成统计

| 项 | 值 |
|---|---|
| 语料目录 | `data/traffic_docs/` |
| 文档总数 | 19 份（Markdown 格式） |
| 文档类型分布 | 规章 8 份、技术文档 8 份、规范 2 份、手册 1 份 |
| 含表格文档 | 11 份 |
| 条款级引用素材 | 6 份（含真实规章条款号，如第三十四~三十七条、第九~十五条） |

语料覆盖四大类场景：
1. **规章条款**：《城市轨道交通行车组织管理办法》《城市轨道交通设施设备运行维护管理办法》等（含真实条款号）；
2. **技术原理**：CBTC/ATP/ATO/ATS、牵引供电、列车定位测速等；
3. **检修规程**：车辆检修修程与周期、信号/通信/轨道维修分级周期；
4. **应急预案**：运营突发事件分级、区间疏散、火灾处置等。

完整清单见 `data/traffic_docs/MANIFEST.md`。

## 二、文档导入结果

**导入方式：脚本直调（未启动 Web UI）**

执行 `scripts/ingest_traffic.py`，复用 kotaemon/ktem 的 `FileIndex` 与 `IndexDocumentPipeline` 管线，将 19 份文档批量导入新建索引 `traffic_ops_kb`（FileIndex id=4，collection=`index_4`）。

| 项 | 值 |
|---|---|
| 待入库文档 | 19 份 |
| 成功入库 | 19 份 |
| 失败 | 0 份 |
| 生成 chunk 数 | 27 个 |
| 平均单文件耗时 | 约 0.2s（首份 0.99s，含模型预热） |

## 三、领域化改造

1. **领域 prompt**（新增 `libs/ktem/ktem/prompts/traffic.py`）：运维场景 system prompt + QA 模板，要求答案给出条款号、原文引用、文档出处，不确定时明确说明"未在知识库中检索到"。
2. **配置开关**（修改 `flowsettings.py`）：新增 `KH_REASONING_USE_TRAFFIC_PROMPT`（默认开启，`USE_TRAFFIC_PROMPT` 环境变量可切回默认 prompt）。
3. **管线注入**（修改 `libs/ktem/ktem/reasoning/simple.py`）：在 `get_pipeline()` 中根据开关注入领域 prompt。
4. **UI 文案**（修改 `chat_panel.py`、`chat_suggestion.py`）：欢迎语与示例问题替换为交通运维场景。

## 四、3 组运维问答样例（`scripts/query_cli.py` 直调）

### 问题 1：列车在区间发生故障时的乘客疏散流程是什么？有哪些限速要求？

- **检索命中**：`01_行车组织管理办法_区间疏散与限速.md`（score 0.60）等 5 个片段
- **条款级引用**：✅ 正确引用《城市轨道交通行车组织管理办法》**第三十五条**
- **答案要点**：扣停后续列车 → 选择步行/接驳疏散 → 步行疏散需明确方向、接触轨停电、启动环控、乘客引导、通道监控 → 限速 25 km/h

### 问题 2：地铁列车架修和大修的周期分别是多少？

- **检索命中**：`04_车辆检修修程与周期.md`、`02_设施设备运行维护管理办法_检修周期.md` 等
- **条款级引用**：✅ 引用**第十三条（一）**
- **答案要点**：架修不超过 6 年或 80 万车公里；大修不超过 12 年或 160 万车公里（时间与里程以先到者为准）；并诚实区分了规章"架修/大修"与市域快线规范"架修A/B/厂修"的命名差异，明确"未检索到 >120km/h 的专门数值"

### 问题 3：CBTC 降级为点式 ATP 的条件与操作要点？

- **检索命中**：`03_CBTC信号系统原理.md`（score 0.51）等 5 个片段
- **条款级引用**：⚠️ 技术文档无专门规章条款号（诚实标注"未检索到专门规定降级的规章条款号"）
- **答案要点**：降级条件为无线通信中断或 ZC 故障 → 点式 ATP/后备模式限速运行（示例 25 km/h）→ 调度扣停后续列车 → 恢复后重新初始化定位

## 五、关键工程修复（Windows 环境）

入库过程中定位并修复 2 处 **theflow 框架层 `__call__` 在 Windows 上卡死**的问题：

| 文件 | 修复 | 说明 |
|---|---|---|
| `libs/ktem/ktem/index/file/pipelines.py` | `self.splitter(text_docs)` → `self.splitter.run(text_docs)` | TokenSplitter 绕开框架层 |
| `libs/kotaemon/kotaemon/indices/vectorindex.py` | `self.embedding(docs)` → `self.embedding.run(docs)` | embedding 绕开框架层 |

根因：theflow `Function.__call__` 会触发共享缓存/上下文管理，在 Windows 上卡死；`.run()` 直接调用底层实现，稳定可用。

## 六、结论

Phase 1 目标达成：项目从"通用 RAG Demo"转变为"城市轨道交通运维知识库"，语料 19 份、脚本化入库 100% 成功、领域 prompt 生效、问答答案含条款级引用与出处。

**入库方式：脚本方式（`scripts/ingest_traffic.py`），全程未启动 Web UI。**
