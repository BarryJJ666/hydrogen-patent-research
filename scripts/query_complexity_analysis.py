#!/usr/bin/env python
"""E4: Query complexity stratification + E5: Case study selection."""
import json, re, os, glob
from collections import defaultdict

BASE = '/home/v-zezhouwang/hydrogen-patent-research'

# --- Load data ---
with open(f'{BASE}/data/benchmark/test.json') as f:
    test_data = json.load(f)

# Build qid -> test item mapping (qid = q0000, q0001, ...)
test_by_input = {}
for i, item in enumerate(test_data):
    test_by_input[item['input'].strip()] = {
        'qid': f'q{i:04d}',
        'question': item['input'].strip(),
        'gold_cypher': item['output'].strip(),
    }

# Load eval results
eval_dirs = sorted(glob.glob(f'{BASE}/3_model_eval_new/output/*/raw.jsonl'))
print(f"Found {len(eval_dirs)} eval result files:")
for d in eval_dirs:
    model_name = d.split('/')[-2]
    print(f"  {model_name}")

results_by_model = {}
for path in eval_dirs:
    model_name = path.split('/')[-2]
    # Extract model type (sft or rl)
    if '_sft_' in model_name:
        model_type = 'sft'
    elif '_rl_' in model_name:
        model_type = 'rl'
    else:
        model_type = 'unknown'

    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Use first run of each type
    key = model_type
    if key not in results_by_model:
        results_by_model[key] = entries
        print(f"  Using {model_name} as '{key}' ({len(entries)} entries)")

# --- E4: Classify queries by complexity ---

# Pragmatic dimension detection from Cypher
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
    """Count relationship traversals in MATCH clause."""
    match_part = cypher.split('RETURN')[0] if 'RETURN' in cypher else cypher
    rels = re.findall(r'-\[.*?\]->', match_part) + re.findall(r'<-\[.*?\]-', match_part)
    # Also count shorthand like -[:TYPE]->
    rels2 = re.findall(r'-\[:?\w+\]->', match_part)
    return max(len(rels), len(rels2), 1)

def detect_query_type(cypher):
    ret_part = cypher.split('RETURN')[-1] if 'RETURN' in cypher else ''
    if re.search(r'ORDER BY.*LIMIT', cypher, re.IGNORECASE):
        return 'ranking'
    if re.search(r'count\s*\(', ret_part, re.IGNORECASE):
        if re.search(r'year|substring.*0.*4', cypher, re.IGNORECASE):
            return 'trend'
        return 'count'
    return 'list'

# Classify all test items
classifications = {}
for item in test_data:
    q = item['input'].strip()
    cypher = item['output'].strip()
    dims = detect_dimensions(cypher)
    hops = count_hops(cypher)
    qtype = detect_query_type(cypher)

    classifications[q] = {
        'dimensions': dims,
        'n_dims': len(dims),
        'hops': min(hops, 4),
        'query_type': qtype,
        'gold_cypher': cypher,
    }

# Match eval results to classifications
def compute_stratified_ex(model_entries, classifications):
    """Compute EX stratified by various axes."""
    by_ndims = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_hops = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_qtype = defaultdict(lambda: {'total': 0, 'correct': 0})

    matched = 0
    for entry in model_entries:
        q = entry['question'].strip()
        if q not in classifications:
            continue
        matched += 1
        cls = classifications[q]
        # Use Cypher string match as proxy for EX (pred == gold)
        pred = entry.get('pred_cypher', '').strip()
        gold = entry.get('gold_cypher', cls['gold_cypher']).strip()
        correct = 1 if pred == gold else 0

        by_ndims[cls['n_dims']]['total'] += 1
        by_ndims[cls['n_dims']]['correct'] += correct

        by_hops[cls['hops']]['total'] += 1
        by_hops[cls['hops']]['correct'] += correct

        by_qtype[cls['query_type']]['total'] += 1
        by_qtype[cls['query_type']]['correct'] += correct

    return by_ndims, by_hops, by_qtype, matched

print("\n" + "="*80)
print("E4: QUERY COMPLEXITY STRATIFICATION")
print("="*80)

for model_type in ['sft', 'rl']:
    if model_type not in results_by_model:
        continue
    entries = results_by_model[model_type]
    by_ndims, by_hops, by_qtype, matched = compute_stratified_ex(entries, classifications)

    print(f"\n--- {model_type.upper()} Model ({matched} matched samples) ---")

    print("\nBy number of pragmatic dimensions:")
    print(f"  {'#Dims':<8} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(by_ndims.keys()):
        d = by_ndims[k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<8} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

    print("\nBy number of hops:")
    print(f"  {'#Hops':<8} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(by_hops.keys()):
        d = by_hops[k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<8} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

    print("\nBy query type:")
    print(f"  {'Type':<12} {'Total':<8} {'Correct':<10} {'EX%':<8}")
    for k in sorted(by_qtype.keys()):
        d = by_qtype[k]
        ex = d['correct'] / d['total'] * 100 if d['total'] > 0 else 0
        print(f"  {k:<12} {d['total']:<8} {d['correct']:<10} {ex:.1f}%")

# --- E5: Case Study Selection ---
print("\n" + "="*80)
print("E5: CASE STUDY CANDIDATES")
print("="*80)

if 'sft' in results_by_model and 'rl' in results_by_model:
    sft_entries = {e['question'].strip(): e for e in results_by_model['sft']}
    rl_entries = {e['question'].strip(): e for e in results_by_model['rl']}

    candidates = []
    for q in rl_entries:
        if q not in sft_entries or q not in classifications:
            continue
        cls = classifications[q]
        rl_e = rl_entries[q]
        sft_e = sft_entries[q]

        # RL correct, SFT incorrect (based on Cypher string match)
        rl_correct = rl_e.get('pred_cypher', '').strip() == cls['gold_cypher']
        sft_correct = sft_e.get('pred_cypher', '').strip() == cls['gold_cypher']

        if rl_correct and not sft_correct and cls['n_dims'] >= 2:
            candidates.append({
                'question': q,
                'n_dims': cls['n_dims'],
                'dimensions': sorted(cls['dimensions']),
                'query_type': cls['query_type'],
                'hops': cls['hops'],
                'gold_cypher': cls['gold_cypher'],
                'sft_pred': sft_e.get('pred_cypher', 'N/A'),
                'rl_pred': rl_e.get('pred_cypher', 'N/A'),
            })

    # Sort by number of dimensions (descending)
    candidates.sort(key=lambda x: (-x['n_dims'], -x['hops']))

    print(f"\nFound {len(candidates)} candidates (RL correct + SFT incorrect + dims>=2)")
    for i, c in enumerate(candidates[:10]):
        print(f"\n--- Candidate {i+1} ({c['n_dims']} dims, {c['hops']} hops, {c['query_type']}) ---")
        print(f"  Dimensions: {', '.join(c['dimensions'])}")
        print(f"  Question: {c['question'][:100]}...")
        print(f"  Gold:     {c['gold_cypher'][:120]}...")
        print(f"  SFT pred: {c['sft_pred'][:120]}...")
        print(f"  RL pred:  {c['rl_pred'][:120]}...")
else:
    print("Need both SFT and RL results for case study comparison.")
