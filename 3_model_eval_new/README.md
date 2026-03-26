# 氢能专利知识图谱 模型评估系统

对比评估多种模型在氢能专利 Text-to-Cypher 任务上的表现，支持本地 vLLM 推理和远程 API 调用两种方式。

## 评估模式

系统支持 **四种评估模式**：

| 后缀 | 模式名称 | 说明 |
|------|---------|------|
| `_single` | 单轮不带CoT | 直接输出 Cypher，一次生成 |
| `_single_cot` | 单轮带CoT | JSON格式输出 `{"thought": "...", "cypher": "..."}`，强制推理 |
| `_multi` | 多轮不带CoT | 执行→反馈→修正循环，模型自主判断结果正确性 |
| `_multi_cot` | 多轮带CoT | JSON格式 + 执行反馈循环，兼具推理和修正能力 |

## 支持的模型

### 本地模型（需GPU，仅支持单轮）

| 模型名 | 说明 |
|--------|------|
| `qwen3_4b` | Qwen3-4B 基座模型 |
| `qwen3_8b` | Qwen3-8B 基座模型 |
| `qwen3_4b_sft` | Qwen3-4B SFT 微调版 |
| `qwen3_8b_sft` | Qwen3-8B SFT 微调版 |
| `qwen25_3b` | Qwen2.5-3B-Instruct |
| `qwen25_7b` | Qwen2.5-7B-Instruct |
| `qwen25_3b_sft` | Qwen2.5-3B SFT 微调版 |
| `qwen25_7b_sft` | Qwen2.5-7B SFT 微调版 |

### 大模型 API（四种模式）

每个大模型自动生成四种配置：

| 模型 | 单轮不带CoT | 单轮带CoT | 多轮不带CoT | 多轮带CoT |
|------|-------------|-----------|-------------|-----------|
| DeepSeek V3.2 | `deepseek_v32_single` | `deepseek_v32_single_cot` | `deepseek_v32_multi` | `deepseek_v32_multi_cot` |
| DeepSeek V3.1 | `deepseek_v31_single` | `deepseek_v31_single_cot` | `deepseek_v31_multi` | `deepseek_v31_multi_cot` |
| DeepSeek R1 | `deepseek_r1_single` | `deepseek_r1_single_cot` | `deepseek_r1_multi` | `deepseek_r1_multi_cot` |
| Qwen3.5-35B | `qwen35_35b_single` | `qwen35_35b_single_cot` | `qwen35_35b_multi` | `qwen35_35b_multi_cot` |
| Qwen3-235B | `qwen3_235b_single` | `qwen3_235b_single_cot` | `qwen3_235b_multi` | `qwen3_235b_multi_cot` |
| GLM-5 | `glm5_single` | `glm5_single_cot` | `glm5_multi` | `glm5_multi_cot` |
| ERNIE-4.5 | `ernie45_single` | `ernie45_single_cot` | `ernie45_multi` | `ernie45_multi_cot` |

**共计 36+ 个模型配置**：8 个本地模型 + 7 × 4 = 28 个大模型配置

## 快速开始

### 环境要求

- **Python 3.6+**
- 确保 `python3` 命令可用

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
# 测试 DeepSeek V3.2 单轮不带CoT
python3 main.py --models deepseek_v32_single --max-samples 50

# 测试 DeepSeek V3.2 单轮带CoT
python3 main.py --models deepseek_v32_single_cot --max-samples 50

# 测试 DeepSeek V3.2 多轮不带CoT
python3 main.py --models deepseek_v32_multi --max-samples 50

# 测试 DeepSeek V3.2 多轮带CoT
python3 main.py --models deepseek_v32_multi_cot --max-samples 50
```

### 使用通配符过滤

```bash
# 所有 DeepSeek 模型
python3 main.py --model-filter "deepseek*"

# 所有单轮不带CoT模型
python3 main.py --model-filter "*_single"

# 所有单轮带CoT模型
python3 main.py --model-filter "*_single_cot"

# 所有多轮不带CoT模型
python3 main.py --model-filter "*_multi"

# 所有多轮带CoT模型
python3 main.py --model-filter "*_multi_cot"
```

### 批量运行脚本

```bash
# 编辑 MODEL_QUEUE 后运行
./scripts/run_baidu_models.sh
```

## 结果管理

结果保存在 `results/` 目录或 `output/` 目录下：

```
results/
├── raw/
│   ├── deepseek_v32_single_20260310_143025.jsonl
│   ├── deepseek_v32_multi_cot_20260310_143026.jsonl
│   └── ...
└── reports/
    ├── deepseek_v32_single_20260310_143025.md
    ├── comparison_20260310_143030.md    # 多模型对比报告
    └── comparison_20260310_143030.csv

output/  # 新的组织方式（按运行分组）
└── deepseek_v32_single+multi_20260310_143025/
    ├── raw/
    └── reports/
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--test-data` | str | `../hydrogen_benchmark_gen/.../test.json` | 测试数据路径 |
| `--models` | str[] | `all` | 要评估的模型列表 |
| `--model-filter` | str | None | 模型过滤器（支持通配符） |
| `--max-samples` | int | None | 最大评估样本数 |
| `--batch-size` | int | 8 | 批量推理大小 |
| `--no-psjs` | flag | False | 跳过 PSJS 指标计算 |
| `--list-models` | flag | False | 列出所有可用模型 |

## 评估指标

### 核心指标

| 指标 | 说明 |
|------|------|
| `execution_accuracy` | **执行准确率 (EX)**：预测 Cypher 执行结果与金标完全一致 |
| `semantic_accuracy` | **语义准确率**：召回率 ≥ 80% 即判定匹配（宽松评估） |
| `psjs` | **PSJS**：溯源子图 Jaccard 相似度 |
| `executable_rate` | **可执行率**：Cypher 能成功执行的比例 |
| `syntax_error_rate` | **语法错误率** |

### 效率指标

| 指标 | 说明 |
|------|------|
| `avg_latency_ms` | 平均延迟 |
| `p50_latency_ms` / `p99_latency_ms` | P50 / P99 延迟 |
| `avg_input_tokens` / `avg_output_tokens` | 平均 Token 数 |

## 四种模式详解

### 1. 单轮不带CoT (`_single`)

```
用户问题 → LLM 直接生成 Cypher → 返回
```

特点：快速、简洁，适合简单查询。

### 2. 单轮带CoT (`_single_cot`)

```
用户问题 → LLM 输出 JSON {"thought": "推理过程", "cypher": "查询"} → 返回
```

特点：强制模型先推理再生成，提高复杂查询准确率。

### 3. 多轮不带CoT (`_multi`)

```
用户问题 → LLM 生成 Cypher → 执行
    ↓
结果 + 原问题 → LLM 自主判断是否正确
    ↓
正确: 输出 DONE → 结束
错误: 输出修正后的 Cypher → 继续循环（最多3轮）
```

特点：模型自主判断执行结果是否正确回答了原问题，而非外层系统判断。

### 4. 多轮带CoT (`_multi_cot`)

```
用户问题 → LLM 输出 JSON {"thought": "...", "cypher": "..."} → 执行
    ↓
结果 + 原问题 → LLM 判断并输出 DONE 或修正的 JSON
    ↓
循环直到 DONE 或达到最大轮数
```

特点：兼具推理和修正能力，适合复杂查询。

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

系统会自动生成 `my_model_single`、`my_model_single_cot`、`my_model_multi`、`my_model_multi_cot` 四个配置。

### Neo4j 连接

```python
NEO4J_CONFIG = {
    "uri": "",
    "user": "neo4j",
    "password": "",
    "database": "neo4j",
}
```

### LLM API 配置

```python
LLM_API_BASE = {
    "api_url": "https://bep.baidu-int.com/plugins/api/v2/access",
    "ak": "...",
    "sk": "...",
    "timeout": 120,
    "max_retries": 5,
}
```

## 项目结构

```
hydrogen_model_eval/
├── main.py                         # 主入口
├── scripts/
│   └── run_baidu_models.sh         # 批量运行脚本
├── config/
│   └── settings.py                 # 全局配置
├── models/
│   ├── __init__.py                 # ModelFactory
│   ├── base_model.py               # BaseModel
│   ├── local_qwen.py               # 本地模型（vLLM）
│   ├── text2cypher.py              # Text2CypherModel（单轮/多轮不带CoT）
│   ├── text2cypher_cot.py          # Text2CypherCotModel（单轮/多轮带CoT）
│   └── litellm_model.py            # LiteLLM 包装器
├── evaluator/
│   ├── metrics.py                  # 指标定义
│   ├── cypher_evaluator.py         # Cypher 评估器
│   └── unified_evaluator.py        # 统一评估器
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
python3 main.py --model-filter "*_single"  # 只评估大模型单轮
```

**Q: 如何快速测试某个模型？**

```bash
python3 main.py --models deepseek_v32_single --max-samples 10
```

**Q: 如何对比四种模式的效果？**

```bash
# 同时评估同一模型的四种模式
python3 main.py --models deepseek_v32_single deepseek_v32_single_cot deepseek_v32_multi deepseek_v32_multi_cot
```

**Q: 结果会覆盖之前的吗？**

不会。每个模型每次运行都带时间戳，独立保存。

**Q: 如何添加新的大模型？**

在 `config/settings.py` 的 `API_MODELS` 中添加一条记录，系统自动生成四种模式的配置。

**Q: 多轮模式最多循环几轮？**

默认 3 轮，可在 `config/settings.py` 中修改 `max_rounds` 参数。
