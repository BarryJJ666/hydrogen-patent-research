#!/usr/bin/env python
"""
OOT Benchmark Evaluation via OpenRouter API.

Usage:
  python scripts/oot_benchmark/eval_openrouter.py --model z-ai/glm-5
  python scripts/oot_benchmark/eval_openrouter.py --model deepseek/deepseek-chat-v3-0324
  python scripts/oot_benchmark/eval_openrouter.py --model qwen/qwen3-235b-a22b
"""
import json
import os
import re
import sys
import time
import argparse
import requests
from collections import Counter, defaultdict
from neo4j import GraphDatabase

# ─── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_KEY = os.getenv(
    "OPENROUTER_API_KEY",
    "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",
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PWD = "hydrogen2026"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "oot_benchmark_final.json")

# ─── Full GPG Schema (matches the paper) ──────────────────────────────────────
SCHEMA_PROMPT = """你是氢能专利知识图谱查询助手。请根据用户的问题，直接生成对应的Neo4j Cypher查询语句。

氢能专利知识图谱Schema:

节点类型:
- Patent: 专利（属性：application_no, title_cn, abstract_cn, application_date, patent_type, transfer_count, license_count, pledge_count, litigation_count, publication_no, publication_date, ipc_main, tech_domain, title_en, abstract_en）
- Organization: 机构（属性：name, entity_type）entity_type取值: 'company', 'university', 'institute'
- Person: 人物（属性：name）
- TechDomain: 技术领域（属性：name, level）3级层次：氢能技术→{制氢技术, 储氢技术, 氢燃料电池, 氢制冷}；储氢技术→{物理储氢, 合金储氢, 无机储氢, 有机储氢}
- IPCCode: IPC分类号（属性：code, subclass, section, class_code）
- Location: 地点（属性：name, level, country, province, city）查询地点时使用具体字段如loc.province='北京市', loc.city='深圳市', loc.country='中国'
- Country: 公开国家（属性：name）注意只有name属性，存储中文名如'中国','美国','日本','韩国','欧洲专利局','德国','英国'
- LegalStatus: 法律状态（属性：name）取值包括：授权, 驳回, 撤回, 公开, 实质审查, 全部无效, 权利终止, 期限届满等
- PatentFamily: 专利族（属性：family_id）
- LitigationType: 诉讼类型（属性：name）取值包括：侵权案件, 无效诉讼, 行政案件, 权属案件, 其他案件等

关系类型（及方向）:
- APPLIED_BY: Patent → Organization（申请人）
- OWNED_BY: Patent → Organization（权利人）
- BELONGS_TO: Patent → TechDomain（所属技术领域）
- CLASSIFIED_AS: Patent → IPCCode（IPC分类）
- PARENT_DOMAIN: TechDomain → TechDomain（子域→父域）
- HAS_STATUS: Patent → LegalStatus（法律状态）
- IN_FAMILY: Patent → PatentFamily（所属专利族）
- LOCATED_IN: Organization → Location（机构所在地）
- PUBLISHED_IN: Patent → Country（公开国家）
- PARENT_LOCATION: Location → Location（子地点→父地点）
- TRANSFERRED_FROM: Patent → Organization（转让来源方）
- TRANSFERRED_TO: Patent → Organization（受让方）
- LICENSED_FROM: Patent → Organization（许可方）
- LICENSED_TO: Patent → Organization（被许可方）
- PLEDGED_FROM: Patent → Organization（质押出方）
- PLEDGED_TO: Patent → Organization（质押入方）
- LITIGATED_WITH: Patent → Organization（诉讼对方，edge上有role属性）
- HAS_LITIGATION_TYPE: Patent → LitigationType（诉讼类型）

注意:
- application_date是字符串格式'YYYY-MM-DD'，提取年份使用substring(p.application_date, 0, 4)
- 查询机构的专利数量时使用APPLIED_BY关系
- 机构名使用CONTAINS模糊匹配: WHERE o.name CONTAINS '某机构'
- 计数时使用count(DISTINCT p)避免重复
- Country节点只有name属性（中文），没有code属性
- Patent的标题字段是title_cn，摘要字段是abstract_cn

只输出Cypher语句，不要添加任何解释、注释或Markdown格式。"""

# ─── Neo4j ────────────────────────────────────────────────────────────────────
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


def execute_cypher(cypher, timeout=15):
    """Execute Cypher, return sorted tuple list or None on error."""
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = [tuple(str(v) for v in record.values()) for record in result]
            return sorted(records)
    except Exception:
        return None


def results_match(gold, pred):
    if gold is None or pred is None:
        return False
    if gold == pred:
        return True
    if len(gold) == len(pred) and set(gold) == set(pred):
        return True
    return False


# ─── OpenRouter ───────────────────────────────────────────────────────────────
def call_openrouter(question, model, temperature=0.1, max_retries=3):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_KEY}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": f"用户问题: {question}\n\nCypher查询:"},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=120
            )
            if resp.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, 0, 0
    return None, 0, 0


def clean_cypher(text):
    if not text:
        return ""
    t = text.strip()
    # Remove markdown code blocks
    if t.startswith("```cypher"):
        t = t[9:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    # Remove leading comments
    lines = t.strip().split("\n")
    cleaned = [l for l in lines if not l.strip().startswith("//") and not l.strip().startswith("--")]
    return "\n".join(cleaned).strip()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OOT Benchmark Eval via OpenRouter")
    parser.add_argument("--model", default="z-ai/glm-5", help="OpenRouter model ID")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input benchmark JSON")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--output-dir", default=os.path.join(SCRIPT_DIR, "eval_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_label = args.model.replace("/", "_")

    with open(args.input, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if args.max_samples:
        questions = questions[: args.max_samples]

    print(f"Model: {args.model}")
    print(f"Questions: {len(questions)}")
    print()

    results = []
    correct = 0
    executable = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        gold_cypher = q["cypher"]

        # Call LLM
        raw_response, in_tok, out_tok = call_openrouter(question, args.model)
        pred_cypher = clean_cypher(raw_response)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        # Execute
        gold_result = execute_cypher(gold_cypher)
        pred_result = execute_cypher(pred_cypher) if pred_cypher else None

        is_executable = pred_result is not None
        is_correct = results_match(gold_result, pred_result)

        if is_executable:
            executable += 1
        if is_correct:
            correct += 1

        results.append({
            "id": qid,
            "question": question,
            "gold_cypher": gold_cypher,
            "pred_cypher": pred_cypher,
            "correct": is_correct,
            "executable": is_executable,
            "family": q.get("family"),
            "n_dims": q.get("n_dims"),
            "dims": q.get("dims"),
            "gold_empty": gold_result is not None and len(gold_result) == 0,
        })

        status = "OK" if is_correct else ("EXEC" if is_executable else "FAIL")
        running_ex = correct / (i + 1) * 100
        print(
            f"[{i+1:3d}/{len(questions)}] {qid} [{status:4s}] "
            f"EX={running_ex:.1f}%  ({correct}/{i+1})"
        )

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    # ─── Report ───────────────────────────────────────────────────────────────
    total = len(questions)
    ex_pct = correct / total * 100
    exec_pct = executable / total * 100

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total: {total}")
    print(f"EX: {correct}/{total} = {ex_pct:.1f}%")
    print(f"Executable: {executable}/{total} = {exec_pct:.1f}%")
    print(f"Tokens: {total_input_tokens} in / {total_output_tokens} out")

    # By family
    print(f"\nBy family:")
    family_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        fam = r["family"]
        family_stats[fam]["total"] += 1
        if r["correct"]:
            family_stats[fam]["correct"] += 1
    for fam in sorted(family_stats):
        s = family_stats[fam]
        ex = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {fam:20s}: {s['correct']:2d}/{s['total']:2d} = {ex:5.1f}%")

    # By n_dims
    print(f"\nBy n_dims:")
    dim_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        nd = r["n_dims"]
        dim_stats[nd]["total"] += 1
        if r["correct"]:
            dim_stats[nd]["correct"] += 1
    for nd in sorted(dim_stats):
        s = dim_stats[nd]
        ex = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {nd}-dim: {s['correct']:2d}/{s['total']:2d} = {ex:5.1f}%")

    # By empty/non-empty gold
    print(f"\nBy gold result:")
    empty_correct = sum(1 for r in results if r["gold_empty"] and r["correct"])
    empty_total = sum(1 for r in results if r["gold_empty"])
    nonempty_correct = sum(1 for r in results if not r["gold_empty"] and r["correct"])
    nonempty_total = sum(1 for r in results if not r["gold_empty"])
    print(f"  empty:     {empty_correct}/{empty_total} = {empty_correct/max(empty_total,1)*100:.1f}%")
    print(f"  non-empty: {nonempty_correct}/{nonempty_total} = {nonempty_correct/max(nonempty_total,1)*100:.1f}%")

    # Save
    out_path = os.path.join(args.output_dir, f"{model_label}_oot_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "total": total,
                "ex": ex_pct,
                "executable_rate": exec_pct,
                "by_family": {k: dict(v) for k, v in family_stats.items()},
                "by_dims": {str(k): dict(v) for k, v in dim_stats.items()},
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
    driver.close()
