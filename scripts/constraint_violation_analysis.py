#!/usr/bin/env python
"""E1-替代: Constraint violation analysis.
For each pragmatic dimension, relax gold Cypher by removing that dimension's
constraints, execute both versions on full GPG, and measure over-retrieval / violation rate.
"""
import json, re, sys
from collections import defaultdict
from neo4j import GraphDatabase

BASE = '/home/v-zezhouwang/hydrogen-patent-research'
NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = 'neo4j'
NEO4J_PWD = 'hydrogen2026'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

def execute_cypher(cypher, timeout=30):
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = [tuple(str(v) for v in record.values()) for record in result]
            return records
    except Exception as e:
        return None

# --- Dimension detection (reuse from eval_with_neo4j.py) ---
DIMENSION_PATTERNS = {
    'technical': [r':TechDomain', r':IPCCode', r'BELONGS_TO', r'CLASSIFIED_AS', r'tech_domain'],
    'legal': [r':LegalStatus', r':PatentFamily', r'HAS_STATUS', r'IN_FAMILY', r'legal_status'],
    'geographic': [r':Location', r'LOCATED_IN', r'\.province', r'\.city', r'\.country'],
    'organizational': [r'entity_type'],  # Only entity_type filter, not APPLIED_BY (too fundamental)
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

# --- Relaxation functions: remove dimension-specific constraints ---
def relax_technical(cypher):
    """Remove TechDomain constraints."""
    # Remove MATCH clauses involving TechDomain
    relaxed = re.sub(r',?\s*\(p\)-\[:BELONGS_TO\]->\(td:TechDomain[^)]*\)', '', cypher)
    relaxed = re.sub(r'MATCH\s+\(p\)-\[:BELONGS_TO\]->\(td:TechDomain[^)]*\)\s*', '', relaxed)
    # Remove WHERE clauses referencing td
    relaxed = re.sub(r'\s*AND\s+td\.\w+\s*=\s*\'[^\']*\'', '', relaxed)
    relaxed = re.sub(r'WHERE\s+td\.\w+\s*=\s*\'[^\']*\'\s*AND\s+', 'WHERE ', relaxed)
    relaxed = re.sub(r'WHERE\s+td\.\w+\s*=\s*\'[^\']*\'\s*', '', relaxed)
    return relaxed

def relax_legal(cypher):
    """Remove LegalStatus constraints."""
    relaxed = re.sub(r',?\s*\(p\)-\[:HAS_STATUS\]->\(s:LegalStatus[^)]*\)', '', cypher)
    relaxed = re.sub(r'MATCH\s+\(p\)-\[:HAS_STATUS\]->\(s:LegalStatus[^)]*\)\s*', '', relaxed)
    # Remove WHERE on legal_status property
    relaxed = re.sub(r'\s*AND\s+p\.legal_status\s*=\s*\'[^\']*\'', '', relaxed)
    relaxed = re.sub(r'WHERE\s+p\.legal_status\s*=\s*\'[^\']*\'\s*AND\s+', 'WHERE ', relaxed)
    return relaxed

def relax_geographic(cypher):
    """Remove Location/geographic constraints."""
    # Remove LOCATED_IN traverse
    relaxed = re.sub(r'-\[:LOCATED_IN\]->\(loc:Location[^)]*\)', '', cypher)
    # Remove WHERE on loc.province/city/country
    relaxed = re.sub(r"\s*AND\s+loc\.\w+\s*=\s*'[^']*'", '', relaxed)
    relaxed = re.sub(r"WHERE\s+loc\.\w+\s*=\s*'[^']*'\s*AND\s+", 'WHERE ', relaxed)
    relaxed = re.sub(r"WHERE\s+loc\.\w+\s*=\s*'[^']*'\s*", '', relaxed)
    return relaxed

def relax_business(cypher):
    """Remove business dynamics constraints (transfer/license/pledge/litigation counts)."""
    relaxed = re.sub(r'\s*AND\s+p\.\w+_count\s*>\s*\d+', '', cypher)
    relaxed = re.sub(r'WHERE\s+p\.\w+_count\s*>\s*\d+\s*AND\s+', 'WHERE ', relaxed)
    relaxed = re.sub(r'WHERE\s+p\.\w+_count\s*>\s*\d+\s*', '', relaxed)
    # Remove TRANSFERRED_TO/FROM etc matches
    relaxed = re.sub(r',?\s*\(p\)-\[:TRANSFERRED_\w+\]->\([^)]*\)', '', relaxed)
    relaxed = re.sub(r'MATCH\s+\(p\)-\[:TRANSFERRED_\w+\]->\([^)]*\)\s*', '', relaxed)
    return relaxed

RELAXATION_FNS = {
    'technical': relax_technical,
    'geographic': relax_geographic,
    'legal': relax_legal,
    'business': relax_business,
}

# --- Load test data ---
with open(f'{BASE}/data/benchmark/test.json') as f:
    test_data = json.load(f)

MAX_PER_DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 30

print("E1-替代: Constraint Violation Analysis")
print("="*70)

results = {}
for dim, relax_fn in RELAXATION_FNS.items():
    # Find queries involving this dimension
    candidates = []
    for item in test_data:
        cypher = item['output'].strip()
        dims = detect_dimensions(cypher)
        if dim in dims and len(dims) >= 2:  # Must involve this dim + at least one other
            candidates.append({'question': item['input'], 'gold_cypher': cypher, 'dims': dims})

    if not candidates:
        print(f"\n{dim}: No multi-dim queries found, skipping")
        continue

    # Sample
    sample = candidates[:MAX_PER_DIM]
    print(f"\n--- {dim.upper()} dimension ({len(sample)}/{len(candidates)} queries) ---")

    total = 0
    violations = 0
    inflation_ratios = []
    errors = 0

    for item in sample:
        gold = item['gold_cypher']
        relaxed = relax_fn(gold)

        # Skip if relaxation didn't change anything
        if relaxed.strip() == gold.strip():
            continue

        gold_result = execute_cypher(gold)
        relaxed_result = execute_cypher(relaxed)

        if gold_result is None or relaxed_result is None:
            errors += 1
            continue

        total += 1
        gold_set = set(gold_result)
        relaxed_set = set(relaxed_result)

        # Over-retrieved = in relaxed but not in gold
        over_retrieved = relaxed_set - gold_set
        if len(relaxed_set) > 0:
            violation_rate = len(over_retrieved) / len(relaxed_set)
            violations += 1 if len(over_retrieved) > 0 else 0

        # Inflation ratio
        if len(gold_set) > 0:
            inflation = len(relaxed_set) / len(gold_set)
            inflation_ratios.append(inflation)

    if total > 0:
        avg_inflation = sum(inflation_ratios) / len(inflation_ratios) if inflation_ratios else 0
        violation_pct = violations / total * 100
        print(f"  Queries evaluated: {total} (errors: {errors})")
        print(f"  Queries with violations: {violations}/{total} ({violation_pct:.1f}%)")
        print(f"  Avg result inflation: {avg_inflation:.1f}x")
        results[dim] = {
            'total': total,
            'violations': violations,
            'violation_pct': violation_pct,
            'avg_inflation': avg_inflation,
        }
    else:
        print(f"  No valid queries after relaxation (errors: {errors})")

# --- Summary ---
print("\n" + "="*70)
print("SUMMARY TABLE (for LaTeX)")
print("="*70)
print(f"{'Dimension':<20} {'Queries':<10} {'Violation%':<15} {'Inflation':<12}")
for dim in ['technical', 'legal', 'geographic', 'business']:
    if dim in results:
        r = results[dim]
        print(f"{dim:<20} {r['total']:<10} {r['violation_pct']:.1f}%{'':<10} {r['avg_inflation']:.1f}x")

driver.close()
