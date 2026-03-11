# 氢能专利知识图谱 模型评估系统

对比评估多种模型在氢能专利 Text-to-Cypher 任务上的表现，支持本地 vLLM 推理和远程 API 调用两种方式。

## 支持的模型

### 小模型（本地 vLLM，需GPU）

| 模型名 | 说明 | 评估模式 |
|--------|------|----------|
| `qwen3_4b` | Qwen3-4B 基座模型 | Direct |
| `qwen3_8b` | Qwen3-8B 基座模型 | Direct |
| `qwen3_4b_sft` | Qwen3-4B SFT 微调版 | Direct |
| `qwen3_8b_sft` | Qwen3-8B SFT 微调版 | Direct |

### 大模型 API（三种模式）

| 模型 | Text-to-Cypher | 一轮工具调用 | 多轮工具调用 |
|------|----------------|--------------|--------------|
| Qwen3.5-35B | `qwen35_35b_direct` | `qwen35_35b_tool_single` | `qwen35_35b_tool_multi` |
| Qwen3-235B | `qwen3_235b_direct` | `qwen3_235b_tool_single` | `qwen3_235b_tool_multi` |
| GLM-5 | `glm5_direct` | `glm5_tool_single` | `glm5_tool_multi` |
| DeepSeek R1 | `deepseek_r1_direct` | `deepseek_r1_tool_single` | `deepseek_r1_tool_multi` |
| DeepSeek V3.1 | `deepseek_v31_direct` | `deepseek_v31_tool_single` | `deepseek_v31_tool_multi` |
| DeepSeek V3.2 | `deepseek_v32_direct` | `deepseek_v32_tool_single` | `deepseek_v32_tool_multi` |
| ERNIE-4.5 | `ernie45_direct` | `ernie45_tool_single` | `ernie45_tool_multi` |

**共计 25 个模型配置**：4 个小模型 + 7 × 3 = 21 个大模型配置

## 快速开始

### 环境要求

- **Python 3.6+**（必须，因为使用了 f-string 等语法）
- 确保 `python3` 命令可用，或修改 `scripts/run_evaluation.sh` 中的 `PYTHON` 变量

### 安装依赖

```bash
pip install -r requirements.txt
```

### 查看可用模型

```bash
python3 main.py --list-models
```

### 快速测试（单个模型）

```bash
# 测试 DeepSeek V3.2 Direct 模式，50 条样本
python3 main.py --models deepseek_v32_direct --max-samples 50

# 测试 GLM-5 一轮工具调用
python3 main.py --models glm5_tool_single --max-samples 50
```

### 使用通配符过滤

```bash
# 所有 DeepSeek 模型
python3 main.py --model-filter "deepseek*"

# 所有 Direct 模式（Text-to-Cypher）
python3 main.py --model-filter "*_direct"

# 所有一轮工具调用模式
python3 main.py --model-filter "*_tool_single"

# 所有多轮工具调用模式
python3 main.py --model-filter "*_tool_multi"
```

### 批量运行脚本

```bash
# 评估所有大模型的 Text-to-Cypher 模式
./scripts/run_evaluation.sh direct

# 评估所有大模型的一轮工具调用
./scripts/run_evaluation.sh tool_single

# 评估所有大模型的多轮工具调用
./scripts/run_evaluation.sh tool_multi

# 评估本地小模型（需GPU）
./scripts/run_evaluation.sh local

# 分批全量评估（推荐）
./scripts/run_evaluation.sh batch

# 自定义模型组合
./scripts/run_evaluation.sh custom "deepseek_v32_direct glm5_tool_multi"

# 查看所有命令
./scripts/run_evaluation.sh help
```

## 结果管理

结果直接保存在 `results/` 目录下，每个模型独立保存：

```
results/
├── raw/
│   ├── deepseek_v32_direct_20260310_143025.jsonl
│   ├── glm5_tool_multi_20260310_143026.jsonl
│   └── ...
└── reports/
    ├── deepseek_v32_direct_20260310_143025.md
    ├── glm5_tool_multi_20260310_143026.md
    ├── comparison_20260310_143030.md    # 多模型对比报告（可选）
    └── comparison_20260310_143030.csv
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test-data` | str | `../hydrogen_benchmark_gen/output/sft_format/test.json` | 测试数据路径 |
| `--models` | str[] | `all` | 要评估的模型列表 |
| `--model-filter` | str | None | 模型过滤器（支持通配符） |
| `--max-samples` | int | None | 最大评估样本数 |
| `--batch-size` | int | 8 | 批量推理大小 |
| `--no-psjs` | flag | False | 跳过 PSJS 指标计算 |
| `--list-models` | flag | False | 列出所有可用模型 |

## 评估指标

### Text-to-Cypher 模式（Direct）

| 指标 | 说明 |
|------|------|
| `execution_accuracy` | 执行准确率：预测 Cypher 执行结果与金标完全一致 |
| `psjs` | PSJS：溯源子图 Jaccard 相似度 |
| `executable_rate` | 可执行率：Cypher 能成功执行 |
| `syntax_error_rate` | 语法错误率 |

### Tool Calling 模式

| 指标 | 说明 |
|------|------|
| `answer_accuracy` | 答案准确率：Agent 返回的答案正确 |

### 效率指标（通用）

| 指标 | 说明 |
|------|------|
| `avg_latency_ms` | 平均延迟 |
| `p50_latency_ms` / `p99_latency_ms` | P50 / P99 延迟 |
| `avg_input_tokens` / `avg_output_tokens` | 平均 Token 数 |

## 三种评估模式说明

### 1. Text-to-Cypher（Direct）

```
用户问题 → LLM 直接生成 Cypher → Neo4j 执行 → 与金标结果比对
```

适用场景：评估模型的 Cypher 生成能力

### 2. 一轮工具调用（Tool Single）

```
用户问题 → LLM 选择工具+参数 → 执行一次 → 返回结果
```

适用场景：简单查询，快速响应

### 3. 多轮工具调用（Tool Multi）

```
用户问题 → ReAct 循环 (思考→调用→观察) → 最多10轮 → 返回答案
```

适用场景：复杂查询，需要多步推理

## 配置说明

编辑 `config/settings.py` 修改配置：

### 添加新的大模型

```python
# 在 API_MODELS 中添加
API_MODELS = {
    ...
    "my_model": {"name": "My Custom Model", "bot_id": "9999"},
}
```

系统会自动生成 `my_model_direct`、`my_model_tool_single`、`my_model_tool_multi` 三个配置。

### Neo4j 连接

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
LLM_API_BASE = {
    "api_url": os.getenv("LLM_API_URL", ""),
    "ak": os.getenv("LLM_API_KEY", ""),
    "sk": os.getenv("LLM_SECRET_KEY", ""),
    "timeout": 120,
    "max_retries": 5,
}
```

## Tool Calling 工具链

Tool Calling 模式使用 ReAct Agent，可调用以下工具：

| 工具名 | 功能 |
|--------|------|
| `query_patents` | 查询专利列表 |
| `count_patents` | 统计专利数量（支持分组） |
| `rank_patents` | Top-N 排名 |
| `trend_patents` | 年度趋势 |
| `get_patent_detail` | 专利详情 |
| `search` | 全文搜索 |

### 筛选条件

```python
filters = {
    "org": "清华大学",       # 机构名称（模糊匹配）
    "org_type": "高校",      # 机构类型
    "domain": "制氢技术",    # 技术领域
    "region": "北京",        # 省份/城市
    "year": 2023,            # 年份
    "year_start": 2020,      # 起始年份
    "year_end": 2024,        # 结束年份
    "has_transfer": True,    # 含专利转让
}
```

## 项目结构

```
hydrogen_model_eval/
├── main.py                         # 主入口
├── scripts/
│   └── run_evaluation.sh           # 批量运行脚本
├── config/
│   └── settings.py                 # 全局配置
├── models/
│   ├── __init__.py                 # ModelFactory
│   ├── base_model.py               # BaseModel
│   ├── local_qwen.py               # 本地模型
│   ├── deepseek_api.py             # API 模型
│   └── tool_calling_wrapper.py     # Tool Calling 包装器
├── tools/
│   ├── meta_tools.py               # 知识图谱查询工具
│   └── react_agent.py              # ReAct Agent
├── evaluator/
│   ├── metrics.py                  # 指标定义
│   └── cypher_evaluator.py         # Cypher 评估器
├── runner/
│   └── batch_runner.py             # 批量运行器
├── reporter/
│   └── report_generator.py         # 报告生成
└── utils/
    ├── llm_client.py               # LLM API 客户端
    ├── neo4j_client.py             # Neo4j 客户端
    └── logger.py                   # 日志
```

## 常见问题

**Q: 只想评估 API 模型，不装 vLLM 行吗？**

可以。只要不评估本地模型即可：
```bash
python3 main.py --model-filter "*_direct"  # 只评估大模型
```

**Q: 如何快速测试某个模型？**

```bash
python3 main.py --models deepseek_v32_direct --max-samples 10
```

**Q: 如何对比一轮和多轮工具调用？**

```bash
# 同时评估
python3 main.py --models deepseek_v32_tool_single deepseek_v32_tool_multi
```

**Q: 结果会覆盖之前的吗？**

不会。每个模型每次运行都带时间戳，独立保存在 `results/raw/` 和 `results/reports/` 目录。

**Q: 如何添加新的大模型？**

在 `config/settings.py` 的 `API_MODELS` 中添加一条记录，系统自动生成三种模式的配置。
