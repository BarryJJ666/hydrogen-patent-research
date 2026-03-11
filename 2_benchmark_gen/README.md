# 氢能专利知识图谱 Benchmark 生成系统

基于氢能专利知识图谱（Neo4j），自动生成 Text-to-Cypher 问答对数据集，用于大模型 SFT 微调训练。

## 项目概述

本系统通过**模式组合 + 实体采样 + LLM润色**的方式，批量生成高质量的「自然语言问题 → Cypher查询」配对数据。生成的数据集以 LlamaFactory Alpaca 格式输出，可直接用于 Qwen 等模型的 SFT 训练。

### 核心流程（6步流水线）

```
Step 1: 模式组合生成 Cypher    →  23种MATCH × 17种RETURN = 391种组合
Step 2: 语法验证               →  过滤语法错误的 Cypher
Step 3: 执行验证               →  在 Neo4j 上执行，过滤空结果/超量结果
Step 4: 自然语言问题生成        →  LLM生成 或 模板生成
Step 5: 数据集划分             →  按类别分层抽样，保持分布一致
Step 6: 格式化输出             →  转为 LlamaFactory Alpaca 格式
```

## 项目结构

```
hydrogen_benchmark_gen/
├── main.py                         # 主入口
├── requirements.txt                # 依赖
├── config/
│   ├── settings.py                 # 全局配置（Neo4j、LLM、生成参数）
│   ├── match_patterns.json         # 23种 MATCH 模式定义
│   └── return_patterns.json        # 17种 RETURN 模式定义
├── sampler/
│   ├── neo4j_sampler.py            # 从 Neo4j 采样真实实体
│   └── entity_sampler.py           # 参数采样器（机构、地域、年份等）
├── generator/
│   ├── cypher_generator.py         # Cypher 查询实例化
│   └── question_generator.py       # 自然语言问题生成（LLM/模板）
├── validator/
│   ├── syntax_validator.py         # Cypher 语法校验
│   └── execution_validator.py      # Cypher 执行验证
├── formatter/
│   ├── llama_factory_formatter.py  # Alpaca 格式转换
│   └── dataset_splitter.py         # 分层数据集划分
├── pipeline/
│   └── full_pipeline.py            # 6步流水线编排
├── utils/
│   ├── llm_client.py               # 百度内网 LLM API 客户端
│   ├── neo4j_client.py             # Neo4j 驱动封装
│   └── logger.py                   # 日志配置
└── output/                         # 输出目录（运行后生成）
    ├── raw/                        # 中间结果
    ├── validated/                  # 原始验证数据
    └── sft_format/                 # 最终 SFT 训练数据
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
- `neo4j>=5.0.0` — Neo4j Python 驱动
- `requests>=2.28.0` — HTTP 客户端（调用 LLM API）
- `tqdm>=4.65.0` — 进度条

## 运行方式

### 基本运行

```bash
cd 2_benchmark_gen

# 使用 LLM 生成 5000 条数据（推荐）
python3 main.py --target 5000 --train-ratio 0.8

# 后台运行
nohup python3 -u main.py --target 5000 --train-ratio 0.8 > nohup.out 2>&1 &
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--target` / `--total-samples` | int | 5000 | 目标生成的 QA 对数量。实际会多生成 1.5 倍以弥补验证损耗 |
| `--train-ratio` | float | 0.8 | 训练集占比。0.8 表示 80% 训练、20% 测试 |
| `--no-llm` | flag | False | 不使用 LLM，仅用模板生成问题（速度快，但问题表述单一） |
| `--no-intermediate` | flag | False | 不保存中间结果文件（节省磁盘空间） |

### 使用示例

```bash
# 快速测试：100条，不用LLM，不保存中间文件
python3 main.py --target 100 --no-llm --no-intermediate

# 正式生成：5000条，LLM润色，8:2划分
python3 main.py --target 5000 --train-ratio 0.8

# 大规模生成：10000条，7:3划分
python3 main.py --target 10000 --train-ratio 0.7
```

## 配置说明

所有配置位于 `config/settings.py`，可直接修改：

### Neo4j 连接配置

所有敏感配置通过环境变量管理，请参考项目根目录的 `.env.example` 文件：

```python
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
}
```

### LLM API 配置

```python
LLM_CONFIG = {
    "api_url": os.getenv("LLM_API_URL", ""),
    "bot_id": os.getenv("LLM_BOT_ID", ""),
    "ak": os.getenv("LLM_API_KEY", ""),
    "sk": os.getenv("LLM_SECRET_KEY", ""),
    "timeout": 60,
    "max_retries": 5,
    "rate_limit": 20,
}
```

### Benchmark 生成配置

```python
BENCHMARK_CONFIG = {
    "target_count": 5000,           # 目标生成数量
    "train_ratio": 0.8,             # 训练集比例（可被命令行参数覆盖）
    "validation_timeout": 30,       # Cypher 执行超时，单位秒
    "max_result_count": 10000,      # 单条查询最大返回行数，超过则丢弃
    "min_result_count": 1,          # 单条查询最小返回行数，0行则丢弃
    "question_variants": 2,         # 每个问题的变体数量
}
```

### 问题类型分布

```python
QUESTION_TYPE_DISTRIBUTION = {
    "count": 0.25,     # 计数类："XX领域有多少专利？"
    "rank": 0.20,      # 排名类："哪些机构专利最多？"
    "list": 0.20,      # 列表类："列出XX机构的专利"
    "trend": 0.15,     # 趋势类："近5年专利变化趋势？"
    "combo": 0.15,     # 组合类："XX省XX领域的专利数？"
    "detail": 0.05,    # 详情类："XX专利的详细信息？"
}
```

## 查询模式说明

### MATCH 模式（23种）

| 类别 | 数量 | 示例 |
|------|------|------|
| basic | 1 | 全部专利 |
| domain | 1 | 按技术领域筛选 |
| org | 2 | 按机构名称/类型筛选 |
| region | 3 | 按省份/城市/国家筛选 |
| time | 2 | 按年份/年份范围筛选 |
| business | 5 | 含转让/许可/质押/诉讼 |
| combo | 9 | 机构+领域、省份+领域、领域+年份等多条件组合 |

### RETURN 模式（17种）

| 类别 | 数量 | 示例 |
|------|------|------|
| count | 4 | 总数统计、按年/领域/机构分组计数 |
| rank | 4 | 机构/省份/国家 Top-N 排名 |
| list | 3 | 专利列表、带详情列表、最新专利 |
| trend | 2 | 年度趋势、近N年趋势 |
| aggregate | 4 | 平均转让数、最大转让数、含转让/诉讼统计 |

两类模式组合可产生 **23 × 17 = 391** 种不同的查询模板，再乘以采样的实体参数，可生成大量多样化的数据。

## 输出文件

运行完成后，输出在 `output/` 目录下：

```
output/
├── raw/                            # 中间结果（加 --no-intermediate 可跳过）
│   ├── 01_generated.json           # Step 1: 原始生成的 Cypher 实例
│   ├── 02_validation_results.json  # Step 3: 执行验证结果
│   ├── 03_qa_pairs.json            # Step 4: QA 问答对
│   └── pipeline_stats.json         # 流水线统计信息
├── validated/                      # 原始训练/测试数据
│   ├── train_raw.json
│   └── test_raw.json
└── sft_format/                     # LlamaFactory 格式（最终输出）
    ├── train.json                  # 训练集
    ├── test.json                   # 测试集
    ├── dataset_info.json           # LlamaFactory 数据集描述
    └── statistics.json             # 数据集统计
```

### SFT 数据格式

每条数据为 LlamaFactory Alpaca 格式：

```json
{
    "instruction": "你是一个氢能专利知识图谱查询助手。根据用户的问题，生成对应的Neo4j Cypher查询语句。\n\n氢能专利知识图谱Schema:\n...(完整Schema描述)...",
    "input": "制氢技术领域近5年的专利申请趋势如何？",
    "output": "MATCH (p:Patent)-[:BELONGS_TO]->(d:TechDomain {name: '制氢技术'}) WITH substring(p.application_date, 0, 4) AS year, count(p) AS cnt WHERE toInteger(year) >= 2020 RETURN year, cnt ORDER BY year"
}
```

### 用于 LlamaFactory 训练

将 `dataset_info.json` 复制到 LlamaFactory 的 `data/` 目录下，即可直接引用：

```yaml
# LlamaFactory 训练配置
dataset: hydrogen_cypher_train
template: qwen
```

## 常见问题

**Q: LLM 调用报错 `gc err, header missing parameter`？**
A: 已修复。旧版使用了错误的 API 请求格式，新版已对齐 `hydrogen_patent_kg_v2` 项目的正确签名方式。

**Q: 生成速度慢？**
A: LLM 问题生成是主要耗时环节。可调整 `rate_limit`（每秒请求数）来平衡速度与稳定性。使用 `--no-llm` 可跳过 LLM 生成，速度提升约 10 倍，但问题表述为模板化。

**Q: 验证通过率低？**
A: 可调整 `BENCHMARK_CONFIG` 中的 `min_result_count`（降低至0允许空结果）和 `max_result_count`（提高上限）。当前预期验证通过率约 80%。

**Q: 如何增加新的查询模式？**
A: 在 `config/match_patterns.json` 中添加新的 MATCH 模式，在 `config/return_patterns.json` 中添加新的 RETURN 模式。格式参考已有条目。
