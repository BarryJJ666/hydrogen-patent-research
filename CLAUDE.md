# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hydrogen Patent Knowledge Graph Research Platform -- a Text-to-Cypher research system for hydrogen energy patents. Written in Python, using Neo4j as the graph database. The project is in Chinese (comments, prompts, UI) but code identifiers are in English.

Three independent modules executed sequentially:

1. **`1_knowledge_graph/`** - Builds a Neo4j knowledge graph from patent Excel data. Includes a LangGraph-based "think-and-search" QA agent.
2. **`2_benchmark_gen/`** - Generates Text-to-Cypher training data (natural language question + Cypher query pairs) via pattern combination + entity sampling + LLM refinement. Outputs LlamaFactory Alpaca format.
3. **`3_model_eval/`** - Evaluates LLMs on Cypher generation. Supports 25 model configs: 4 local vLLM + 7 API models x 3 modes (direct, tool_single, tool_multi).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Module 1: Knowledge Graph Construction
cd 1_knowledge_graph
python main.py                        # Full pipeline (steps 1-5)
python main.py --step 1 2 3           # Run specific steps
python main.py --no-cache             # Skip cache, rebuild from scratch
python main.py --clear                # Clear Neo4j before import
python query_cli.py                   # Interactive QA CLI

# Module 2: Benchmark Generation
cd 2_benchmark_gen
python main.py --target 5000          # Generate 5000 QA pairs (default)
python main.py --target 100 --no-llm  # Quick test without LLM
python main.py --target 5000 --train-ratio 0.8

# Module 3: Model Evaluation
cd 3_model_eval
python main.py --list-models
python main.py --models deepseek_v32_direct --max-samples 50
python main.py --model-filter "*_direct"
python main.py --models "deepseek_v32_direct" "qwen3_235b_direct"
./scripts/run_evaluation.sh direct    # Batch evaluate all direct models
```

## Architecture

### Shared Infrastructure
- All modules use `python-dotenv` and read from a root `.env` file (see `.env.example`)
- Each module has its own `config/settings.py`, `utils/logger.py`, and `utils/llm_client.py` -- these are NOT shared across modules
- Each module's `main.py` adds its own directory to `sys.path`; run scripts from within the module directory
- Neo4j connection config is duplicated in each module's `config/settings.py` (all read from same env vars)

### Knowledge Graph Schema (Neo4j)
- **11 node types**: Patent, Organization, Person, TechDomain, IPCCode, Country, LegalStatus, Location, PatentFamily, LitigationType
- **17+ relationship types**: APPLIED_BY, OWNED_BY, TRANSFERRED_FROM/TO, LICENSED_FROM/TO, PLEDGED_FROM/TO, LITIGATED_WITH, BELONGS_TO, CLASSIFIED_AS, LOCATED_IN, PUBLISHED_IN, HAS_STATUS
- `application_date` is a string `'YYYY-MM-DD'` -- extract year with `substring(p.application_date, 0, 4)`
- Tech domains: 制氢技术, 储氢技术 (with children: 物理储氢, 合金储氢, 无机储氢, 有机储氢), 氢燃料电池, 氢制冷

### Module 1: Knowledge Graph (`1_knowledge_graph/`)
- 5-step pipeline: Load Excel data -> Entity resolution (vector + LLM) -> Generate Cypher -> Build vector index -> Import to Neo4j
- `data_pipeline/`: DataLoader, EntityResolver, UnifiedEntityResolver (enhanced: vector coarse filter + LLM decision), CypherGenerator
- `graph_db/`: Neo4j connection, importer, query executor, statistics
- `vector/`: FAISS-based vector search using BAAI/bge-m3 embeddings (1024-dim, HNSW index)
- `langgraph_agent/`: "Think-and-search" agent using ReAct loop (max 10 steps). LLM calls high-level tools (Cypher, fulltext, vector, GraphRAG) instead of generating Cypher directly
- `utils/cache.py`: JSON-based caching for intermediate pipeline results

### Module 2: Benchmark Gen (`2_benchmark_gen/`)
- 6-step pipeline: Pattern combination -> Syntax validation -> Execution validation -> Question generation -> Dataset split -> Format output
- `config/match_patterns.json` (23 MATCH patterns) x `config/return_patterns.json` (17 RETURN patterns) = 391 template combinations
- `sampler/`: Samples real entities from Neo4j to instantiate Cypher templates
- `validator/`: Validates generated Cypher syntax and execution against Neo4j
- `formatter/`: Converts to LlamaFactory Alpaca format with stratified train/test split
- Output: `output/sft_format/{train,test,dataset_info}.json`

### Module 3: Model Eval (`3_model_eval/`)
- Three evaluation modes: `direct` (Text-to-Cypher), `tool_single` (one tool call), `tool_multi` (ReAct loop up to 10 steps)
- `models/`: ModelFactory pattern. `base_model.py` defines interface; `deepseek_api.py` for API models; `local_qwen.py` for local vLLM; `tool_calling_wrapper.py` wraps models with tool-calling capability
- `tools/meta_tools.py`: 6 knowledge graph query tools (query_patents, count_patents, rank_patents, trend_patents, get_patent_detail, search)
- `tools/react_agent.py`: ReAct agent for multi-step tool calling
- `evaluator/`: Metrics -- EX (Execution Accuracy), PSJS (Partial Set Jaccard Similarity), executable rate, syntax error rate
- `runner/batch_runner.py`: Orchestrates evaluation across models
- `reporter/`: Generates per-model reports and multi-model comparison (Markdown + CSV)
- Model configs auto-generated: 7 API models x 3 modes = 21 configs + 4 local models
- Results saved with timestamps: `results/raw/{model}_{timestamp}.jsonl`, `results/reports/{model}_{timestamp}.md`

### Data Layout
- `data/cypher_scripts/`: Pre-generated Cypher import scripts (ordered: `00_schema.cypher`, `01_tech_domains.cypher`, etc.)
- `data/benchmark/`: Pre-built benchmark dataset (`train.json`, `test.json`, ~5000 QA pairs)

## Environment Variables (Required)

```
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
LLM_API_URL, LLM_API_KEY, LLM_SECRET_KEY, LLM_BOT_ID
```

For model eval, each API model needs a bot ID env var: `MODEL_BOT_ID_QWEN35_35B`, `MODEL_BOT_ID_DEEPSEEK_V32`, etc.

## Key Conventions

- No test framework is configured; there are no unit tests
- Each module is self-contained with its own settings, logger, and LLM client
- LLM API uses a custom client (`utils/llm_client.py`) with HMAC signing, not standard OpenAI-compatible SDK
- Cypher queries in benchmarks use `substring(p.application_date, 0, 4)` for year extraction and `count(DISTINCT p)` to avoid duplicates
- Location queries use specific fields (`loc.province`, `loc.city`, `loc.country`), not `loc.name`
- Organization matching uses `CONTAINS` for fuzzy match: `WHERE o.name CONTAINS '某机构'`
