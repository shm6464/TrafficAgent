# 05_cs_retrofit_plan.md — 智能客服换皮改造方案（A 层）

> 目标：把「城市轨道交通运维知识库问答系统」换皮为「电商/零售智能客服系统」。
> 范围：仅 A 层（换定位 + 换语料 + 换 prompt + 换工具分类 + 换文案），
> **不改检索/重排/缓存/评测/FastAPI 服务化的任何核心代码**，工程化成果全保留。
> 约束：不代写完整代码，只给命令级步骤 + 代码骨架，由用户自行敲入并回传输出。

---

## 0. 改造总览

| 层 | 改动内容 | 涉及文件 | 是否破坏现有能力 |
|---|---|---|---|
| 语料 | 交通 19 份 → 电商客服语料 | `data/cs_docs/`（新增）+ `scripts/ingest_cs.py`（新增） | 否（原语料/脚本保留） |
| prompt | `traffic.py` → 新增 `cs.py` | `libs/ktem/ktem/prompts/cs.py`（新增） | 否 |
| 工具分类 | 规章/设备/检修/应急 → 商品/售后/物流/账号/投诉 | `api/agent.py`（改 TOOLS + _classify_file） | 否 |
| 文案/标题 | 「交通运维」→「智能客服」 | `api/main.py`、`api/service.py`、README | 否（仅文案） |
| 索引 | 新增 `cs_kb` 索引（不动 traffic_ops_kb） | `api/service.py` 的 collection、`api/agent.py` | 否 |

**核心原则：全部用「新增」而非「覆盖」**，原有交通运维版本随时可切回（保留原索引、原 prompt、原语料）。

---

## 1. 新增电商客服语料（data/cs_docs/）

新建目录并写入 8~10 份 Markdown 语料（建议覆盖以下分类，便于后续工具分类）：

| 序号 | 建议文件名 | 内容 | 对应工具分类 |
|---|---|---|---|
| 01 | 01_商品与退换货政策.md | 7 天无理由退货、质量问题换货、运费承担规则 | 售后 |
| 02 | 02_物流配送时效说明.md | 发货时效、偏远地区、物流异常处理 | 物流 |
| 03 | 03_账号与会员体系.md | 注册登录、会员等级、积分、优惠券 | 账号 |
| 04 | 04_订单查询与修改.md | 订单状态、改地址、取消订单、发票 | 订单 |
| 05 | 05_支付方式与退款说明.md | 支付方式、退款到账时效、部分退款 | 售后 |
| 06 | 06_常见商品咨询FAQ.md | 尺码/材质/库存/使用方法等高频问答 | 商品 |
| 07 | 07_投诉与升级处理流程.md | 投诉受理、处理时限、升级规则、转人工标准 | 投诉 |
| 08 | 08_大促与活动规则.md | 满减、预售、定金尾款、活动有效期 | 商品 |

> 语料自写要点（贴合你已有的「条款号 + 原文引用」能力，可把政策写成带编号条款，
> 例如「第 1 条 7 天无理由退货…」，这样换皮后 prompt 里的"条款号定位"亮点仍成立）。
> 授权说明：语料为自编业务政策整理稿，非真实公司内部数据，仅用于技术验证。

---

## 2. 新增客服 prompt（libs/ktem/ktem/prompts/cs.py）

代码骨架（参考 `traffic.py`，但语气转为客服）：

```python
"""电商/零售智能客服系统的领域提示词。"""

CS_SYSTEM_PROMPT = (
    "你是一名电商平台智能客服助手，服务于购买商品、咨询售后、查询物流的顾客。"
    "请严格遵循以下要求回答问题：\n"
    "1. 语气亲切、专业，先共情/安抚再给结论（投诉场景尤其要先安抚）；\n"
    "2. 答案必须基于给定的上下文（商品政策、退换货规则、物流说明）；\n"
    "3. 若上下文中包含政策条款，请标注条款编号（如'第 1 条'）；\n"
    "4. 关键结论附带原文依据；\n"
    "5. 若上下文不足以回答，明确说明'抱歉，该问题需要转接人工客服为您处理'，"
    "严禁编造政策或承诺；\n"
    "6. 涉及金额、时效、运费等数值时，必须准确引用原文。"
)

CS_QA_TEXT_PROMPT = (
    "请基于以下上下文回答末尾的问题，回答要详细、礼貌且解释清晰。\n"
    "如果上下文中包含政策条款，请标注条款编号；关键结论请给出原文依据。"
    "如果你不知道答案，请说明需要转接人工客服，不要编造。"
    "用{lang}回答。\n\n"
    "{context}\n"
    "问题：{question}\n"
    "答案："
)

__all__ = ["CS_SYSTEM_PROMPT", "CS_QA_TEXT_PROMPT"]
```

---

## 3. 改 api/agent.py（工具分类换皮）

### 3.1 改 TOOLS（四类 → 五类客服工具）

```python
TOOLS = {
    "search_product": {
        "desc": "检索商品咨询类知识（商品FAQ、尺码材质库存、使用说明、活动规则）",
        "keywords": ("FAQ", "尺码", "材质", "库存", "活动", "满减", "预售"),
    },
    "search_aftersale": {
        "desc": "检索售后类知识（退换货政策、退款说明、发票）",
        "keywords": ("退货", "换货", "退款", "发票", "售后"),
    },
    "search_logistics": {
        "desc": "检索物流类知识（配送时效、物流异常、偏远地区）",
        "keywords": ("物流", "配送", "时效", "快递"),
    },
    "search_account": {
        "desc": "检索账号与订单类知识（注册登录、会员、积分优惠券、订单查询修改）",
        "keywords": ("账号", "会员", "积分", "优惠券", "订单", "注册"),
    },
    "search_complaint": {
        "desc": "检索投诉与升级类知识（投诉受理、处理时限、转人工标准）",
        "keywords": ("投诉", "升级", "转人工", "维权"),
    },
}
```

### 3.2 改 _classify_file（关键词映射到新分类）

```python
def _classify_file(file_name: str) -> str:
    if any(k in file_name for k in ("投诉", "升级", "维权")):
        return "search_complaint"
    if any(k in file_name for k in ("退", "换货", "退款", "发票", "售后")):
        return "search_aftersale"
    if any(k in file_name for k in ("物流", "配送", "快递", "时效")):
        return "search_logistics"
    if any(k in file_name for k in ("账号", "会员", "订单", "积分", "优惠券", "注册")):
        return "search_account"
    return "search_product"
```

### 3.3 改 prompt 导入

`api/agent.py` 顶部：

```python
# 改前
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT
# 改后
from ktem.prompts.cs import CS_QA_TEXT_PROMPT, CS_SYSTEM_PROMPT
```

并同步把 `kb_search` 内的 `TRAFFIC_QA_TEXT_PROMPT.format(...)` 和
`SystemMessage(content=TRAFFIC_SYSTEM_PROMPT)` 替换为 `CS_*`。

`REACT_TEMPLATE` / `FINAL_TEMPLATE` 里的「城市轨道交通运维 Agent」文案也改为
「电商智能客服 Agent」，转人工措辞改为「转接人工客服」。

---

## 4. 改 api/service.py（prompt 导入 + collection）

### 4.1 prompt 导入（同 agent.py）

```python
# 改前
from ktem.prompts.traffic import TRAFFIC_QA_TEXT_PROMPT, TRAFFIC_SYSTEM_PROMPT
# 改后
from ktem.prompts.cs import CS_QA_TEXT_PROMPT, CS_SYSTEM_PROMPT
```

并替换 `_build_prompt` 与 `query()` 内的 `TRAFFIC_*` 为 `CS_*`。

### 4.2 collection 改为新索引

> 注意：`collection = "index_4"` 是 `traffic_ops_kb` 索引对应的 collection。
> 换皮后应新建 `cs_kb` 索引，collection id 需在建库后从 DB 查得。
> 若想最小改动，可先在 `cs_kb` 建库后，把 service 的 collection 换成新 id。

```python
def __init__(self, collection: str = "index_5", ...):  # index_5 为示例，以实际为准
```

> 稳妥做法：先跑通「复用 index_4 检索 + CS prompt」，确认 prompt 换皮生效，
> 再单独建 `cs_kb` 索引切换 collection。两步解耦，避免一次改多处排错难。

---

## 5. 改 api/main.py（标题/文案）

| 位置 | 改前 | 改后 |
|---|---|---|
| `FastAPI(title=...)` | 交通运维知识库问答系统 | 电商智能客服系统 |
| `description` | 交通运维 RAG | 电商客服 RAG（混合检索 + bge 重排） |
| `healthz` 返回 | `traffic_ops_qa` | `cs_qa` |
| `/ingest` 的 `INDEX_NAME` | `traffic_ops_kb` | `cs_kb` |
| 首页 `agent.html` | 运维风 | 客服风（可选，`api/static/agent.html`） |

---

## 6. 新增入库脚本 scripts/ingest_cs.py

复制 `scripts/ingest_traffic.py`，改三处：

```python
INDEX_NAME = "cs_kb"                          # 改前 traffic_ops_kb
DOCS_DIR = ROOT / "data" / "cs_docs"          # 改前 data/traffic_docs
# 打印文案："交通运维语料批量入库" → "电商客服语料批量入库"
```

> 索引名 `cs_kb` 与 `api/main.py` 的 `/ingest` 保持一致。

---

## 7. 验证方式（按序执行，每步回传输出）

```powershell
cd D:\google\kotaemon-main\kotaemon-main
.\.venv\Scripts\activate

# 1. prompt 换皮生效检查（无网络依赖）
python -c "from ktem.prompts.cs import CS_SYSTEM_PROMPT; print(CS_SYSTEM_PROMPT[:80])"

# 2. 客服语料入库
python scripts\ingest_cs.py

# 3. 命令行问答（用客服问题自测）
python scripts\query_cli.py        # 问：7天无理由退货的条件是什么？

# 4. 接口测试
python -m pytest tests\test_api.py -v
```

---

## 8. 风险与注意

1. **collection id 不对**：换 `cs_kb` 后 `service.py` 的 `collection="index_4"` 必须换成
   新索引的真实 id，否则检索会查到旧的交通语料。两步解耦可避免此坑。
2. **DeepSeek 额度**：换皮不增加 LLM 调用量，风险与原来一致。
3. **原项目保留**：`traffic_docs`、`ingest_traffic.py`、`traffic.py`、`traffic_ops_kb`
   索引全部保留，随时切回。
4. **条款号亮点延续**：把客服政策写成带编号条款，原「条款定位 + 原文引用」的
   工程亮点在客服场景依然成立，简历叙事可无缝复用。
