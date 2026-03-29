#!/usr/bin/env python
"""E4+E5: Execute gold/pred Cypher against Neo4j, compute real EX, stratify by complexity."""
import json, re, os, sys
from collections import defaultdict
from neo4j import GraphDatabase

BASE = '/home/v-zezhouwang/hydrogen-patent-research'
NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = 'neo4j'
NEO4J_PWD = 'hydrogen2026'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

def execute_cypher(cypher, timeout=10):
    """Execute Cypher and return result as sorted list of tuples, or None on error."""
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = []
            for record in result:
                records.append(tuple(str(v) for v in record.values()))
            return sorted(records)
    except Exception as e:
        return None

def results_match(gold_result, pred_result):
    """CypherBench-style flexible matching."""
    if gold_result is None or pred_result is None:
        return False
    if gold_result == pred_result:
        return True
    # Try column permutations for single-row results
    if len(gold_result) == len(pred_result):
        gold_sets = set(gold_result)
        pred_sets = set(pred_result)
        if gold_sets == pred_sets:
            return True
    return False

# --- Dimension detection ---
DIMENSION_PATTERNS = {
    'technical': [r':TechDomain', r':IPCCode', r'BELONGS_TO', r'CLASSIFIED_AS', r'tech_domain'],
    'legal': [r':LegalStatus', r':PatentFamily', r'HAS_STATUS', r'IN_FAMILY', r'legal_status'],
    'geographic': [r':Location', r':Country', r'LOCATED_IN', r'PUBLISHED_IN', r'\.province', r'\.city', r'\.country'],
    'organizational': [r':Organization', r':Person', r'APPLIED_BY', r'OWNED_BY', r'entity_type'],
    'business': [r'TRANSFERRED', r'LICENSED', r'PLEDGED', r'LITIGATED', r'transfer_count', r'license_count', r'pledge_count', r'litigation_count', r':LitigationType'],
}

def detect_dimensions(cypher):
    dims = set()
    for dim, patterns in DIMENSION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, cypher, re.IGNORECASE):
                dims.add(dim)
                break
    return dims

def count_hops(cypher):
    match_part = cypher.split('RETURN')[0] if 'RETURN' in cypher else cypher
    rels = re.findall(r'\[:\w+\]', match_part)
    return max(len(rels), 1)

def detect_query_type(cypher):
    if re.search(r'ORDER BY.*LIMIT', cypher, re.IGNORECASE):
        return 'ranking'
    ret_part = cypher.split('RETURN')[-1] if 'RETURN' in cypher else ''
    if re.search(r'count\s*\(', ret_part, re.IGNORECASE):
        if re.search(r'year|substring.*0.*4', cypher, re.IGNORECASE):
            return 'trend'
        return 'count'
    return 'list'

# --- Load data ---
# Load eval results
eval_files = {
    'sft': f'{BASE}/3_model_eval_new/output/qwen25_7b_sft_single_20260324_011629/raw.jsonl',
    'rl': f'{BASE}/3_model_eval_new/output/qwen25_7b_rl_single_20260324_022628/raw.jsonl',
}

results = {}
for model, path in eval_files.items():
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    results[model] = entries
    print(f"Loaded {model}: {len(entries)} entries")

# --- Execute and compare (sample first 100 for speed, then full) ---
MAX_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"\nEvaluating {MAX_SAMPLES} samples per model against Neo4j...")

model_results = {}
for model, entries in results.items():
    by_ndims = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_hops = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_qtype = defaultdict(lambda: {'total': 0, 'correct': 0})

    case_candidates = []

    for i, entry in enumerate(entries[:MAX_SAMPLES]):
        gold_cypher = entry['gold_cypher']
        pred_cypher = entry.get('pred_cypher', '')

        gold_result = execute_cypher(gold_cypher)
        pred_result = execute_cypher(pred_cypher)
        correct = results_match(gold_result, pred_result)

        dims = detect_dimensions(gold_cypher)
        hops = count_hops(gold_cypher)
        qtype = detect_query_type(gold_cypher)

        by_ndims[len(dims)]['total'] += 1
        by_ndims[len(dims)]['correct'] += int(correct)
        by_hops[min(hops, 4)]['total'] += 1
        by_hops[min(hops, 4)]['correct'] += int(correct)
        by_qtype[qtype]['total'] += 1
        by_qtype[qtype]['correct'] += int(correct)

        entry['_correct'] = correct
        entry['_dims'] = sorted(dims)
        entry['_ndims'] = len(dims)
        entry['_hops'] = hops
        entry['_qtype'] = qtype

        if (i+1) % 20 == 0:
            total_correct = sum(d['correct'] for d in by_ndims.values())
            total = sum(d['total'] for d in by_ndims.values())
            print(f"  {model} [{i+1}/{MAX_SAMPLES}] EX so far: {total_correct}/{total} = {total_correct/total*100:.1f}%")

    model_results[model] = {
        'by_ndims': dict(by_ndims),
        'by_hops': dict(by_hops),
        'by_qtype': dict(by_qtype),
        'entries': entries[:MAX_SAMPLES],
    }

# --- Print E4 results ---
print("\n" + "="*80)
print("E4: QUERY COMPLEXITY STRATIFICATION (Real EX via Neo4j execution)")
print("="*80)

for model in ['sft', 'rl']:
    mr = model_results[model]
    total_c = sum(d['correct'] for d in mr['by_ndims'].values())
    total_t = sum(d['total'] for d in mr['by_ndims'].values())
    print(f"\n--- {model.upper()} Model (Overall EX: {total_c}/{total_t} = {total_c/total_t*100:.1f}%) ---")

    print("\nBy #pragmatic dimensions:")
    print(f"  {'#Dims':<8} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(mr['by_ndims'].keys()):
        d = mr['by_ndims'][k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<8} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

    print("\nBy #hops:")
    print(f"  {'#Hops':<8} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(mr['by_hops'].keys()):
        d = mr['by_hops'][k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<8} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

    print("\nBy query type:")
    print(f"  {'Type':<12} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(mr['by_qtype'].keys()):
        d = mr['by_qtype'][k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<12} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

# --- E5: Case study candidates ---
print("\n" + "="*80)
print("E5: CASE STUDY CANDIDATES (RL correct, SFT incorrect, dims >= 2)")
print("="*80)

if 'sft' in model_results and 'rl' in model_results:
    sft_entries = {e['question']: e for e in model_results['sft']['entries']}
    rl_entries = {e['question']: e for e in model_results['rl']['entries']}

    candidates = []
    for q in rl_entries:
        if q not in sft_entries:
            continue
        rl_e = rl_entries[q]
        sft_e = sft_entries[q]
        if rl_e.get('_correct') and not sft_e.get('_correct') and rl_e.get('_ndims', 0) >= 2:
            candidates.append({
                'question': q,
                'n_dims': rl_e['_ndims'],
                'dimensions': rl_e['_dims'],
                'query_type': rl_e['_qtype'],
                'hops': rl_e['_hops'],
                'gold_cypher': rl_e['gold_cypher'],
                'sft_pred': sft_e.get('pred_cypher', 'N/A'),
                'rl_pred': rl_e.get('pred_cypher', 'N/A'),
            })

    candidates.sort(key=lambda x: (-x['n_dims'], -x['hops']))
    print(f"\nFound {len(candidates)} candidates")
    for i, c in enumerate(candidates[:10]):
        print(f"\n--- Candidate {i+1} ({c['n_dims']} dims, {c['hops']} hops, {c['query_type']}) ---")
        print(f"  Dims: {', '.join(c['dimensions'])}")
        print(f"  Q: {c['question'][:120]}")
        print(f"  Gold: {c['gold_cypher'][:150]}")
        print(f"  SFT:  {c['sft_pred'][:150]}")
        print(f"  RL:   {c['rl_pred'][:150]}")

driver.close()
