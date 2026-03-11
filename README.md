# Hydrogen Patent Knowledge Graph Research

基于氢能专利知识图谱的 Text-to-Cypher 研究平台，包含知识图谱构建、Benchmark生成和模型评测三个核心模块。

## 项目概述

本项目面向氢能领域专利数据，构建了一套完整的知识图谱问答研究框架：

| 模块 | 说明 | 目录 |
|------|------|------|
| **知识图谱构建** | 从专利Excel数据构建Neo4j知识图谱，支持智能问答Agent | `1_knowledge_graph/` |
| **Benchmark生成** | 自动生成Text-to-Cypher训练数据（自然语言问题→Cypher查询配对） | `2_benchmark_gen/` |
| **模型评测** | 评估多种LLM在专利领域的Cypher生成能力 | `3_model_eval/` |

## 知识图谱Schema

### 节点类型（11种）

| 节点类型 | 属性 | 说明 |
|---------|------|------|
| Patent | application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count | 专利 |
| Organization | name, entity_type | 机构（公司/高校/研究机构） |
| Person | uid, name | 人物 |
| TechDomain | name, level | 技术领域 |
| IPCCode | code, section, class_code | IPC分类号 |
| Country | name | 国家 |
| LegalStatus | name | 法律状态 |
| Location | location_id, name, level, country, province, city | 地点 |
| PatentFamily | family_id | 专利族 |
| LitigationType | name | 诉讼类型 |

### 关系类型（17+种）

| 关系类型 | 方向 | 说明 |
|---------|------|------|
| APPLIED_BY | Patent → Organization/Person | 申请人（最常用） |
| OWNED_BY | Patent → Organization/Person | 权利人 |
| TRANSFERRED_FROM/TO | Patent → Organization/Person | 转让 |
| LICENSED_FROM/TO | Patent → Organization/Person | 许可 |
| PLEDGED_FROM/TO | Patent → Organization/Person | 质押 |
| LITIGATED_WITH | Patent → Organization/Person | 诉讼 |
| BELONGS_TO | Patent → TechDomain | 所属领域 |
| CLASSIFIED_AS | Patent → IPCCode | IPC分类 |
| LOCATED_IN | Organization → Location | 所在地 |
| PUBLISHED_IN | Patent → Country | 公开国家 |
| HAS_STATUS | Patent → LegalStatus | 法律状态 |

### 技术领域

```
氢能技术
├── 制氢技术
├── 储氢技术
│   ├── 物理储氢
│   ├── 合金储氢
│   ├── 无机储氢
│   └── 有机储氢
├── 氢燃料电池
└── 氢制冷
```

## 快速开始

### 环境要求

- Python 3.9+
- Neo4j 5.x（社区版或企业版）
- 8GB+ RAM

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/hydrogen-patent-research.git
cd hydrogen-patent-research

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际配置（见下方说明）
```

### 环境变量配置

编辑 `.env` 文件，填入以下必要配置：

```bash
# Neo4j 数据库配置（必填）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM API 配置（如需使用问答功能或Benchmark生成）
LLM_API_URL=https://your-llm-api-endpoint
LLM_API_KEY=your_api_key
LLM_SECRET_KEY=your_secret_key
LLM_BOT_ID=your_bot_id
```

## 模块使用说明

### 1. 知识图谱构建 (`1_knowledge_graph/`)

#### 导入预置数据

项目提供了预生成的Cypher脚本，可直接导入Neo4j：

```bash
# 进入Neo4j浏览器或使用cypher-shell
# 按顺序执行 data/cypher_scripts/ 目录下的脚本

# 1. 创建Schema和索引
cat data/cypher_scripts/00_schema.cypher | cypher-shell -u neo4j -p your_password
cat data/cypher_scripts/00_vector_index.cypher | cypher-shell -u neo4j -p your_password

# 2. 导入节点数据
cat data/cypher_scripts/01_tech_domains.cypher | cypher-shell -u neo4j -p your_password
cat data/cypher_scripts/02_ipc_codes.cypher | cypher-shell -u neo4j -p your_password
# ... 按顺序执行其他脚本

# 或使用批量导入脚本
for f in data/cypher_scripts/*.cypher; do
    echo "Importing $f..."
    cat "$f" | cypher-shell -u neo4j -p your_password
done
```

#### 从原始数据构建

如果需要从Excel数据重新构建知识图谱：

```bash
cd 1_knowledge_graph

# 运行完整构建流程
python main.py

# 或分步执行
python main.py --step load      # 加载数据
python main.py --step resolve   # 实体消解
python main.py --step generate  # 生成Cypher
python main.py --step import    # 导入Neo4j
```

#### 智能问答

```bash
cd 1_knowledge_graph

# 启动命令行问答界面
python query_cli.py
```

### 2. Benchmark生成 (`2_benchmark_gen/`)

自动生成Text-to-Cypher训练数据：

```bash
cd 2_benchmark_gen

# 生成5000条数据（默认）
python main.py --target 5000

# 快速测试（不使用LLM生成问题）
python main.py --target 100 --no-llm

# 自定义训练集比例
python main.py --target 5000 --train-ratio 0.8
```

生成的数据位于 `output/sft_format/` 目录：
- `train.json` - 训练集
- `test.json` - 测试集
- `dataset_info.json` - 数据集元信息

**数据格式示例**（LlamaFactory Alpaca格式）：

```json
{
  "instruction": "你是一个氢能专利知识图谱的Cypher查询专家。请根据用户的自然语言问题生成对应的Cypher查询语句。",
  "input": "制氢技术领域有多少专利？",
  "output": "MATCH (p:Patent)-[:BELONGS_TO]->(t:TechDomain {name: '制氢技术'}) RETURN count(DISTINCT p) AS patent_count"
}
```

### 3. 模型评测 (`3_model_eval/`)

评测多种LLM在专利领域的Cypher生成能力：

```bash
cd 3_model_eval

# 查看可用模型
python main.py --list-models

# 评测单个模型
python main.py --models deepseek_v32_direct --max-samples 100

# 评测多个模型
python main.py --models "deepseek_v32_direct,qwen3_235b_direct"

# 按模式筛选评测
python main.py --model-filter "*_direct"      # 所有Direct模式
python main.py --model-filter "*_tool_single" # 所有单轮工具调用
```

#### 评测模式说明

| 模式 | 说明 |
|------|------|
| `*_direct` | Text-to-Cypher：LLM直接生成Cypher查询 |
| `*_tool_single` | 单轮工具调用：一次调用知识图谱查询工具 |
| `*_tool_multi` | 多轮工具调用：ReAct循环推理 |

#### 评测指标

- **EX (Execution Accuracy)**: 执行准确率，生成的Cypher与标准答案执行结果是否一致
- **PSJS (Partial Set Jaccard Similarity)**: 部分集合Jaccard相似度
- **可执行率**: 生成的Cypher能否成功执行

## 项目结构

```
hydrogen-patent-research/
├── 1_knowledge_graph/          # 知识图谱构建模块
│   ├── config/                 # 配置文件
│   ├── data_pipeline/          # 数据处理流水线
│   ├── graph_db/               # Neo4j操作
│   ├── vector/                 # 向量检索
│   ├── langgraph_agent/        # 智能问答Agent
│   ├── utils/                  # 工具函数
│   ├── main.py                 # 主入口
│   └── query_cli.py            # 问答CLI
│
├── 2_benchmark_gen/            # Benchmark生成模块
│   ├── config/                 # 配置文件
│   │   ├── match_patterns.json # MATCH模式定义（23种）
│   │   └── return_patterns.json# RETURN模式定义（17种）
│   ├── sampler/                # 数据采样
│   ├── generator/              # Cypher/问题生成
│   ├── validator/              # 验证器
│   ├── formatter/              # 格式化输出
│   ├── pipeline/               # 完整流水线
│   └── main.py                 # 主入口
│
├── 3_model_eval/               # 模型评测模块
│   ├── config/                 # 配置文件
│   ├── models/                 # 模型适配器
│   ├── tools/                  # 知识图谱工具
│   ├── evaluator/              # 评测器
│   ├── runner/                 # 批量运行器
│   ├── reporter/               # 报告生成器
│   └── main.py                 # 主入口
│
├── data/                       # 数据目录
│   ├── cypher_scripts/         # Cypher导入脚本
│   └── benchmark/              # Benchmark数据集
│       ├── train.json          # 训练集
│       └── test.json           # 测试集
│
├── .env.example                # 环境变量模板
├── .gitignore                  # Git忽略规则
├── requirements.txt            # Python依赖
└── README.md                   # 本文档
```

## 数据说明

### 预置数据

| 数据类型 | 位置 | 说明 |
|---------|------|------|
| Cypher脚本 | `data/cypher_scripts/` | 可直接导入Neo4j的Cypher语句 |
| 训练集 | `data/benchmark/train.json` | 约4000条QA配对数据 |
| 测试集 | `data/benchmark/test.json` | 约1000条QA配对数据 |

### 统计信息

- 专利数量：~15000条
- 机构数量：~5000个
- 技术领域：8个细分领域
- Benchmark数据：~5000条QA配对

## 常见问题

### Neo4j连接失败

确保Neo4j服务已启动，并检查 `.env` 中的配置：
```bash
# 检查Neo4j状态
neo4j status

# 测试连接
cypher-shell -u neo4j -p your_password "RETURN 1"
```

### 向量模型下载慢

默认使用HuggingFace的 `BAAI/bge-m3`，可预先下载到本地：
```bash
# 使用huggingface-cli下载
huggingface-cli download BAAI/bge-m3 --local-dir /path/to/bge-m3

# 然后在.env中配置
EMBEDDING_MODEL_PATH=/path/to/bge-m3
```

### 依赖安装问题

```bash
# 安装PyTorch（根据CUDA版本选择）
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

## 贡献

欢迎提交Issue和Pull Request！

## License

MIT License

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{hydrogen-patent-kg,
  title={Hydrogen Patent Knowledge Graph Research Platform},
  author={Zhang Yuzhe},
  year={2026},
  url={https://github.com/your-username/hydrogen-patent-research}
}
```
