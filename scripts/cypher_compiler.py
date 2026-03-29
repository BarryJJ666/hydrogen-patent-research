#!/usr/bin/env python
"""Cypher Compiler: JSON constraints → deterministic Cypher query."""


def compile_cypher(constraints: dict) -> str:
    """Compile a JSON constraint dict into a valid Cypher query."""
    match_clauses = ["(p:Patent)"]
    where_clauses = []
    with_clauses = []
    return_fields = []
    order_by = None
    limit = None

    task = constraints.get("task", "list")
    tech_domain = constraints.get("tech_domain")
    legal_status = constraints.get("legal_status")
    province = constraints.get("province")
    country = constraints.get("country")
    org_name = constraints.get("org_name")
    org_type = constraints.get("org_type")
    year_from = constraints.get("year_from")
    year_to = constraints.get("year_to")
    year_exact = constraints.get("year_exact")
    has_transfer = constraints.get("has_transfer")
    has_litigation = constraints.get("has_litigation")
    has_license = constraints.get("has_license")
    has_pledge = constraints.get("has_pledge")
    transferee_name = constraints.get("transferee_name")
    group_by = constraints.get("group_by")
    top_n = constraints.get("top_n")

    # --- MATCH clauses ---
    if tech_domain:
        match_clauses.append(f"(p)-[:BELONGS_TO]->(td:TechDomain {{name: '{tech_domain}'}})")

    if legal_status:
        match_clauses.append(f"(p)-[:HAS_STATUS]->(ls:LegalStatus {{name: '{legal_status}'}})")

    needs_org = org_name or org_type or province
    if needs_org:
        org_props = ""
        if org_type:
            org_props = f" {{entity_type: '{org_type}'}}"
        match_clauses.append(f"(p)-[:APPLIED_BY]->(o:Organization{org_props})")
        if org_name:
            where_clauses.append(f"o.name CONTAINS '{org_name}'")

    if province:
        match_clauses.append("(o)-[:LOCATED_IN]->(loc:Location)")
        where_clauses.append(f"loc.province = '{province}'")

    if country and not province:
        match_clauses.append("(p)-[:PUBLISHED_IN]->(c:Country)")
        where_clauses.append(f"c.name = '{country}'")

    if transferee_name:
        match_clauses.append("(p)-[:TRANSFERRED_TO]->(tgt:Organization)")
        where_clauses.append(f"tgt.name CONTAINS '{transferee_name}'")
    elif has_transfer:
        match_clauses.append("(p)-[:TRANSFERRED_TO]->(tgt:Organization)")

    if has_litigation:
        match_clauses.append("(p)-[:HAS_LITIGATION_TYPE]->(lt:LitigationType)")

    if has_license:
        match_clauses.append("(p)-[:LICENSED_TO]->(lic:Organization)")

    if has_pledge:
        match_clauses.append("(p)-[:PLEDGED_TO]->(pl:Organization)")

    # --- WHERE clauses (time) ---
    if year_exact:
        where_clauses.append(f"substring(p.application_date, 0, 4) = '{year_exact}'")
    else:
        if year_from:
            where_clauses.append(f"substring(p.application_date, 0, 4) >= '{year_from}'")
        if year_to:
            where_clauses.append(f"substring(p.application_date, 0, 4) <= '{year_to}'")

    # Business count filters (fallback if no edge match)
    if has_transfer and not transferee_name and "(p)-[:TRANSFERRED_TO]->(tgt:Organization)" not in match_clauses:
        where_clauses.append("p.transfer_count > 0")
    if has_litigation and "(p)-[:HAS_LITIGATION_TYPE]->(lt:LitigationType)" not in match_clauses:
        where_clauses.append("p.litigation_count > 0")

    # --- Build MATCH ---
    match_str = "MATCH " + ", ".join(match_clauses)
    where_str = ""
    if where_clauses:
        where_str = " WHERE " + " AND ".join(where_clauses)

    # --- Build RETURN based on task + group_by ---
    if task == "count" and group_by:
        if group_by == "year":
            return f"{match_str}{where_str} WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count RETURN year, count ORDER BY year"
        elif group_by == "domain":
            if not tech_domain:
                match_str += ", (p)-[:BELONGS_TO]->(td:TechDomain)" if "td:TechDomain" not in match_str else ""
            return f"{match_str}{where_str} WITH td.name AS domain, count(DISTINCT p) AS count RETURN domain, count ORDER BY count DESC"
        elif group_by == "org":
            if not needs_org:
                match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
            return f"{match_str}{where_str} WITH o.name AS org, count(DISTINCT p) AS count RETURN org, count ORDER BY count DESC"
        elif group_by == "province":
            if not province:
                if not needs_org:
                    match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
                match_str += ", (o)-[:LOCATED_IN]->(loc:Location)" if "loc:Location" not in match_str else ""
            return f"{match_str}{where_str} WITH loc.province AS province, count(DISTINCT p) AS count WHERE province IS NOT NULL AND province <> '' RETURN province, count ORDER BY count DESC"
        elif group_by == "country":
            if "c:Country" not in match_str:
                if not needs_org:
                    match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
                match_str += ", (o)-[:LOCATED_IN]->(loc:Location)" if "loc:Location" not in match_str else ""
            return f"{match_str}{where_str} WITH loc.country AS country, count(DISTINCT p) AS count RETURN country, count ORDER BY count DESC"
        else:
            return f"{match_str}{where_str} RETURN count(DISTINCT p) AS total"

    elif task == "count":
        return f"{match_str}{where_str} RETURN count(DISTINCT p) AS total"

    elif task == "trend":
        year_filter = ""
        if year_from:
            year_filter = f" WHERE toInteger(year) >= {year_from}"
        return f"{match_str}{where_str} WITH substring(p.application_date, 0, 4) AS year, count(DISTINCT p) AS count{year_filter} RETURN year, count ORDER BY year"

    elif task == "rank":
        n = top_n or 10
        if group_by == "org":
            if not needs_org:
                match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
            return f"{match_str}{where_str} WITH o.name AS org, count(DISTINCT p) AS count RETURN org, count ORDER BY count DESC LIMIT {n}"
        elif group_by == "province":
            if not province:
                if not needs_org:
                    match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
                match_str += ", (o)-[:LOCATED_IN]->(loc:Location)" if "loc:Location" not in match_str else ""
            return f"{match_str}{where_str} WITH loc.province AS province, count(DISTINCT p) AS count WHERE province IS NOT NULL AND province <> '' RETURN province, count ORDER BY count DESC LIMIT {n}"
        elif group_by == "country":
            if "c:Country" not in match_str:
                if not needs_org:
                    match_str += ", (p)-[:APPLIED_BY]->(o:Organization)"
                match_str += ", (o)-[:LOCATED_IN]->(loc:Location)" if "loc:Location" not in match_str else ""
            return f"{match_str}{where_str} WITH loc.country AS country, count(DISTINCT p) AS count RETURN country, count ORDER BY count DESC LIMIT {n}"
        elif group_by == "domain":
            if not tech_domain:
                match_str += ", (p)-[:BELONGS_TO]->(td:TechDomain)" if "td:TechDomain" not in match_str else ""
            return f"{match_str}{where_str} WITH td.name AS domain, count(DISTINCT p) AS count RETURN domain, count ORDER BY count DESC LIMIT {n}"
        else:
            return f"{match_str}{where_str} RETURN p.application_no AS app_no, p.title_cn AS title, p.application_date AS date ORDER BY p.application_date DESC LIMIT {n}"

    else:  # list
        limit_n = top_n or 50
        return f"{match_str}{where_str} RETURN p.application_no AS app_no, p.title_cn AS title, substring(p.application_date, 0, 4) AS year ORDER BY p.application_date DESC LIMIT {limit_n}"


if __name__ == "__main__":
    # Quick test
    test_cases = [
        {"task": "count", "tech_domain": "制氢技术"},
        {"task": "list", "tech_domain": "物理储氢", "province": "广东省"},
        {"task": "rank", "org_type": "公司", "tech_domain": "氢燃料电池", "group_by": "org", "top_n": 10},
        {"task": "trend", "tech_domain": "合金储氢", "year_from": "2018"},
        {"task": "list", "has_transfer": True, "tech_domain": "氢燃料电池"},
    ]
    for tc in test_cases:
        cypher = compile_cypher(tc)
        print(f"Input: {tc}")
        print(f"Cypher: {cypher}")
        print()
