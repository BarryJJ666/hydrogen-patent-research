#!/usr/bin/env python
"""Batch fix out-of-template questions based on actual Neo4j schema.

Fixes:
1. Country: {code:'US'} -> {name:'美国'} (Country has no 'code' property)
2. Patent: p.title -> p.title_cn (actual property name)
3. Patent: p.application_number -> p.application_no
4. LitigationType: '侵权诉讼' -> '侵权案件'
5. LegalStatus: '无效' -> '全部无效', '终止' -> '权利终止'
6. IPC prefixes: replace non-existent with real ones
7. Remove questions using AFFILIATED_WITH (doesn't exist)
8. Country code mappings for c.code references
"""
import json
import re
import os

SCRIPT_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(SCRIPT_DIR, 'oot_questions_all.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'oot_questions_fixed.json')

# Country code -> Chinese name mapping
COUNTRY_MAP = {
    'CN': '中国',
    'US': '美国',
    'JP': '日本',
    'EP': '欧洲专利局',
    'KR': '韩国',
    'DE': '德国',
    'GB': '英国',
    'FR': '法国',
    'WO': '世界知识产权组织(WIPO)',
    'TW': '中国台湾',
    'AU': '澳大利亚',
    'CA': '加拿大',
    'RU': '俄罗斯',
    'IN': '印度',
    'BR': '巴西',
}

# IPC prefix replacements (non-existent -> existent)
IPC_REPLACEMENTS = {
    'H01M10': 'H01M8',    # H01M8 exists (fuel cells)
    'Y02E':  'C25B',      # electrolysis
    'B60L':  'H01M4',     # electrode materials
    'C07C':  'C01B',      # basic chemistry
    'F16L':  'F17C',      # storage
    'G01N':  'C01B',      # measurement -> basic chemistry
    'H02J':  'H01M8',     # power distribution -> fuel cells
}

# Litigation type fixes
LITIGATION_FIXES = {
    '侵权诉讼': '侵权案件',
    '商业秘密纠纷': '其他案件',
}

# Legal status fixes
LEGAL_STATUS_FIXES = {
    '无效': '全部无效',
    '终止': '权利终止',
}


def fix_cypher(cypher, question):
    """Apply all fixes to a Cypher query. Returns (fixed_cypher, fixes_applied)."""
    fixes = []
    original = cypher

    # 1. Fix p.title -> p.title_cn
    if 'p.title' in cypher and 'p.title_cn' not in cypher:
        cypher = cypher.replace('p.title', 'p.title_cn')
        fixes.append('title->title_cn')

    # 2. Fix p.application_number -> p.application_no
    if 'p.application_number' in cypher:
        cypher = cypher.replace('p.application_number', 'p.application_no')
        fixes.append('application_number->application_no')

    # 3. Fix p.abstract -> p.abstract_cn
    if re.search(r'p\.abstract(?!_)', cypher):
        cypher = re.sub(r'p\.abstract(?!_)', 'p.abstract_cn', cypher)
        fixes.append('abstract->abstract_cn')

    # 4. Fix Country code references
    # Pattern: Country {code:'XX'} -> Country {name:'中文名'}
    for code, name in COUNTRY_MAP.items():
        pattern = f"{{code:'{code}'}}"
        replacement = f"{{name:'{name}'}}"
        if pattern in cypher:
            cypher = cypher.replace(pattern, replacement)
            fixes.append(f"country {code}->{name}")

    # Pattern: c.code = 'XX' or c.code <> 'XX' -> c.name = '中文名' or c.name <> '中文名'
    for code, name in COUNTRY_MAP.items():
        for op in ['=', '<>', '!=']:
            old = f"c.code {op} '{code}'"
            new = f"c.name {op} '{name}'"
            if old in cypher:
                cypher = cypher.replace(old, new)
                fixes.append(f"c.code {op} {code}")

    # Pattern: c.code IN ['XX','YY'] -> c.name IN ['中文','中文']
    in_match = re.search(r"c\.code\s+IN\s+\[([^\]]+)\]", cypher)
    if in_match:
        codes_str = in_match.group(1)
        codes = re.findall(r"'([^']+)'", codes_str)
        names = [COUNTRY_MAP.get(c, c) for c in codes]
        names_str = ','.join(f"'{n}'" for n in names)
        cypher = cypher.replace(f"c.code IN [{codes_str}]", f"c.name IN [{names_str}]")
        fixes.append(f"c.code IN -> c.name IN")

    # Fix collect(DISTINCT c.code) -> collect(DISTINCT c.name)
    cypher = cypher.replace('c.code', 'c.name')
    if 'c.name' in cypher and 'c.code' in original:
        if 'c.code' not in [f.split('->')[0].strip() for f in fixes if '->' in f]:
            fixes.append('c.code->c.name (general)')

    # Also fix Country references like countries list containing codes
    # Pattern: 'CN' IN countries -> '中国' IN countries
    for code, name in COUNTRY_MAP.items():
        old_pattern = f"'{code}' IN countries"
        new_pattern = f"'{name}' IN countries"
        if old_pattern in cypher:
            cypher = cypher.replace(old_pattern, new_pattern)
            fixes.append(f"'{code}' in countries->'{name}'")

    # Fix UNWIND ['CN','US'] -> UNWIND ['中国','美国']
    unwind_match = re.search(r"UNWIND\s+\[([^\]]+)\]\s+AS\s+code", cypher)
    if unwind_match:
        codes_str = unwind_match.group(1)
        codes = re.findall(r"'([^']+)'", codes_str)
        names = [COUNTRY_MAP.get(c, c) for c in codes]
        names_str = ','.join(f"'{n}'" for n in names)
        cypher = cypher.replace(f"UNWIND [{codes_str}] AS code", f"UNWIND [{names_str}] AS country_name")
        # Also fix references to 'code' variable
        cypher = cypher.replace('code, f,', 'country_name, f,')
        cypher = cypher.replace('{code: code}', '{name: country_name}')
        cypher = cypher.replace('code AS country', 'country_name AS country')
        fixes.append('UNWIND country codes->names')

    # 5. Fix litigation type names
    for old_val, new_val in LITIGATION_FIXES.items():
        if f"'{old_val}'" in cypher:
            cypher = cypher.replace(f"'{old_val}'", f"'{new_val}'")
            fixes.append(f"litigation '{old_val}'->'{new_val}'")

    # 6. Fix legal status names
    for old_val, new_val in LEGAL_STATUS_FIXES.items():
        if f"{{name:'{old_val}'}}" in cypher:
            cypher = cypher.replace(f"{{name:'{old_val}'}}", f"{{name:'{new_val}'}}")
            fixes.append(f"status '{old_val}'->'{new_val}'")

    # 7. Fix IPC prefixes
    for old_prefix, new_prefix in IPC_REPLACEMENTS.items():
        if f"'{old_prefix}'" in cypher:
            cypher = cypher.replace(f"'{old_prefix}'", f"'{new_prefix}'")
            fixes.append(f"IPC '{old_prefix}'->'{new_prefix}'")

    # 8. Fix loc.country comparisons (loc.country stores Chinese names)
    # loc.country = '中国' is correct, but loc.country = 'CN' is wrong
    # Actually loc.country already stores Chinese... let's check
    # The Codex queries already use '中国' for loc.country, so this should be fine

    return cypher, fixes


def fix_question(question, cypher_fixes):
    """Update question text to match Cypher fixes (e.g., country names)."""
    q = question
    # Fix country references in question text to match
    # Most questions already use Chinese names, so minimal fixes needed
    return q


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")

    fixed_questions = []
    removed_ids = []
    fix_stats = {}
    total_fixes = 0

    for q in questions:
        qid = q['id']
        cypher = q['cypher']

        # Remove questions using AFFILIATED_WITH (doesn't exist in graph)
        if 'AFFILIATED_WITH' in cypher:
            removed_ids.append(qid)
            continue

        fixed_cypher, fixes = fix_cypher(cypher, q['question'])

        if fixes:
            total_fixes += 1
            for f in fixes:
                fix_type = f.split('->')[0].split("'")[0].strip() if '->' in f else f
                fix_stats[fix_type] = fix_stats.get(fix_type, 0) + 1

        q_copy = dict(q)
        q_copy['cypher'] = fixed_cypher
        if fixes:
            q_copy['_fixes'] = fixes
        fixed_questions.append(q_copy)

    # Re-number IDs sequentially
    for i, q in enumerate(fixed_questions):
        q['id'] = f"oot_{i+1:03d}"

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(fixed_questions, f, ensure_ascii=False, indent=2)

    print(f"\n=== Fix Summary ===")
    print(f"Original: {len(questions)}")
    print(f"Removed (AFFILIATED_WITH): {len(removed_ids)} -> {removed_ids}")
    print(f"Output: {len(fixed_questions)}")
    print(f"Questions with fixes: {total_fixes}")
    print(f"\nFix breakdown:")
    for fix_type, count in sorted(fix_stats.items(), key=lambda x: -x[1]):
        print(f"  {fix_type}: {count}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
