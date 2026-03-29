#!/usr/bin/env python
"""Exp3: Few-shot evaluation with Kimi K2.5 via OpenAI-compatible API."""
import json, os, sys, time
from openai import OpenAI
from neo4j import GraphDatabase

BASE = '/home/v-zezhouwang/hydrogen-patent-research'

# API config
client = OpenAI(
    base_url=os.environ.get("KIMI_API_BASE", "https://labds.bdware.cn:21041/v1"),
    api_key=os.environ["KIMI_API_KEY"],
)
MODEL = "bailian-Kimi-k2.5"

# Neo4j
driver = GraphDatabase.driver(
    os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
)

def execute_cypher(cypher):
    try:
        with driver.session() as s:
            result = s.run(cypher)
            records = [tuple(sorted((k, str(v)) for k, v in record.items())) for record in result]
            return sorted(records)
    except:
        return None

# Load test data
with open(f'{BASE}/data/benchmark/test.json') as f:
    test_data = json.load(f)

# Get system prompt from first test item (it contains schema)
system_prompt = test_data[0]['instruction']

# Select 3 diverse few-shot examples from TRAINING set
with open(f'{BASE}/data/benchmark/train.json') as f:
    train_data = json.load(f)

# Pick 3 examples: 1 simple, 1 medium, 1 complex
examples = [
    train_data[0],   # simple
    train_data[100], # medium
    train_data[500], # complex
]

few_shot_messages = []
for ex in examples:
    few_shot_messages.append({"role": "user", "content": ex['input']})
    few_shot_messages.append({"role": "assistant", "content": ex['output']})

MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"Exp3: Kimi K2.5 few-shot evaluation ({MAX_SAMPLES} samples)")
print(f"Model: {MODEL}")
print(f"Few-shot examples: {len(examples)}")
print("="*60)

correct = 0
executable = 0
syntax_errors = 0
total = 0
results = []

for i, item in enumerate(test_data[:MAX_SAMPLES]):
    question = item['input']
    gold_cypher = item['output']

    # Call API
    messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [{"role": "user", "content": question}]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        pred_cypher = response.choices[0].message.content.strip()
        # Strip markdown code fence if present
        if pred_cypher.startswith("```"):
            lines = pred_cypher.split("\n")
            pred_cypher = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    except Exception as e:
        print(f"  [{i+1}] API error: {e}")
        pred_cypher = ""
        time.sleep(2)
        continue

    # Execute and compare
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
    else:
        syntax_errors += 1
    if is_correct:
        correct += 1

    results.append({
        'qid': f'q{i:04d}',
        'question': question,
        'gold_cypher': gold_cypher,
        'pred_cypher': pred_cypher,
        'correct': is_correct,
        'executable': is_executable,
    })

    if (i+1) % 10 == 0:
        print(f"  [{i+1}/{MAX_SAMPLES}] EX={correct}/{total}={correct/total*100:.1f}% "
              f"exec={executable}/{total}={executable/total*100:.1f}%")

    time.sleep(0.5)  # Rate limiting

print(f"\n{'='*60}")
print(f"FINAL: EX={correct}/{total}={correct/total*100:.1f}%")
print(f"  Executable: {executable}/{total}={executable/total*100:.1f}%")
print(f"  Syntax errors: {syntax_errors}/{total}={syntax_errors/total*100:.1f}%")

# Save results
with open(f'{BASE}/scripts/kimi_fewshot_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

driver.close()
