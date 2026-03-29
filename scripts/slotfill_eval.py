#!/usr/bin/env python
"""SlotFill → Compiler baseline evaluation."""
import json, sys, time
from openai import OpenAI
from neo4j import GraphDatabase
from cypher_compiler import compile_cypher

BASE = '/home/v-zezhouwang/hydrogen-patent-research'

client = OpenAI(
    base_url="https://labds.bdware.cn:21041/v1",
    api_key="sk-nShBYQmXOXAW2Se6ixe-Lg",
)
MODEL = "Qwen3-Max"

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'hydrogen2026'))

SLOT_FILL_PROMPT = """你是氢能专利查询意图提取器。从用户问题中提取结构化约束，输出JSON。

JSON字段说明：
- task: "list"(列表) / "count"(计数) / "rank"(排名) / "trend"(趋势)
- tech_domain: 技术领域（制氢技术/储氢技术/物理储氢/合金储氢/无机储氢/有机储氢/氢燃料电池/氢制冷），null表示不限
- legal_status: 法律状态（授权/驳回/撤回/实质审查/公开/权利终止/未缴年费），null表示不限
- province: 省份（如"广东省"），null表示不限
- country: 国家（如"中国"/"美国"/"日本"），null表示不限
- org_name: 机构名称关键词（模糊匹配），null表示不限
- org_type: 机构类型（"公司"/"高校"/"研究机构"），null表示不限
- year_from: 起始年份（如"2018"），null表示不限
- year_to: 结束年份（如"2024"），null表示不限
- year_exact: 精确年份（如"2020"），null表示不限
- has_transfer: 是否涉及专利转让，true/false/null
- has_litigation: 是否涉及诉讼，true/false/null
- has_license: 是否涉及许可，true/false/null
- has_pledge: 是否涉及质押，true/false/null
- transferee_name: 受让方名称关键词，null表示不限
- group_by: 分组依据（"year"/"domain"/"org"/"province"/"country"），null表示不分组
- top_n: 返回前N个结果，null表示默认

示例1:
问题: "制氢技术领域有多少专利？"
输出: {"task":"count","tech_domain":"制氢技术","legal_status":null,"province":null,"country":null,"org_name":null,"org_type":null,"year_from":null,"year_to":null,"year_exact":null,"has_transfer":null,"has_litigation":null,"has_license":null,"has_pledge":null,"transferee_name":null,"group_by":null,"top_n":null}

示例2:
问题: "广东省的公司在氢燃料电池领域有哪些专利？"
输出: {"task":"list","tech_domain":"氢燃料电池","legal_status":null,"province":"广东省","country":null,"org_name":null,"org_type":"公司","year_from":null,"year_to":null,"year_exact":null,"has_transfer":null,"has_litigation":null,"has_license":null,"has_pledge":null,"transferee_name":null,"group_by":null,"top_n":null}

示例3:
问题: "2018年以来合金储氢领域的专利申请趋势如何？"
输出: {"task":"trend","tech_domain":"合金储氢","legal_status":null,"province":null,"country":null,"org_name":null,"org_type":null,"year_from":"2018","year_to":null,"year_exact":null,"has_transfer":null,"has_litigation":null,"has_license":null,"has_pledge":null,"transferee_name":null,"group_by":null,"top_n":null}

请直接输出JSON，不要添加任何解释或markdown标记。"""


def execute_cypher(cypher):
    try:
        with driver.session() as s:
            result = s.run(cypher)
            records = [tuple(sorted((k, str(v)) for k, v in record.items())) for record in result]
            return sorted(records)
    except:
        return None


def extract_slots(question):
    """Use LLM to extract JSON constraints from question."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SLOT_FILL_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=512,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
        return json.loads(text)
    except Exception as e:
        return None


# Load test data
with open(f'{BASE}/data/benchmark/test.json') as f:
    test_data = json.load(f)

MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"SlotFill → Compiler Baseline ({MAX_SAMPLES} samples, model={MODEL})")
print("=" * 60)

correct = 0
executable = 0
slot_errors = 0
compile_errors = 0
total = 0

for i, item in enumerate(test_data[:MAX_SAMPLES]):
    question = item['input']
    gold_cypher = item['output']

    # Step 1: Extract slots
    slots = extract_slots(question)
    if slots is None:
        slot_errors += 1
        total += 1
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{MAX_SAMPLES}] EX={correct}/{total}={correct/total*100:.1f}%")
        time.sleep(0.3)
        continue

    # Step 2: Compile to Cypher
    try:
        pred_cypher = compile_cypher(slots)
    except Exception:
        compile_errors += 1
        total += 1
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{MAX_SAMPLES}] EX={correct}/{total}={correct/total*100:.1f}%")
        time.sleep(0.3)
        continue

    # Step 3: Execute and compare
    gold_result = execute_cypher(gold_cypher)
    pred_result = execute_cypher(pred_cypher)

    is_executable = pred_result is not None
    is_correct = gold_result is not None and pred_result is not None and gold_result == pred_result
    # Also try value-only comparison
    if not is_correct and gold_result is not None and pred_result is not None:
        gold_vals = sorted([tuple(v for _, v in row) for row in gold_result])
        pred_vals = sorted([tuple(v for _, v in row) for row in pred_result])
        if gold_vals == pred_vals:
            is_correct = True

    total += 1
    if is_executable:
        executable += 1
    if is_correct:
        correct += 1

    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] EX={correct}/{total}={correct/total*100:.1f}% "
              f"exec={executable}/{total}={executable/total*100:.1f}% "
              f"slot_err={slot_errors} compile_err={compile_errors}")

    time.sleep(0.3)

print(f"\n{'='*60}")
print(f"FINAL: EX={correct}/{total}={correct/total*100:.1f}%")
print(f"  Executable: {executable}/{total}={executable/total*100:.1f}%")
print(f"  Slot extraction errors: {slot_errors}")
print(f"  Compile errors: {compile_errors}")

driver.close()
