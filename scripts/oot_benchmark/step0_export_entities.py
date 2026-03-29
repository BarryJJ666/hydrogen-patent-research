#!/usr/bin/env python
"""Step 0: Export entity dictionary from Neo4j for out-of-template benchmark generation."""
import json
import os
from neo4j import GraphDatabase

NEO4J_URI = 'bolt://localhost:7687'
NEO4J_USER = 'neo4j'
NEO4J_PWD = 'hydrogen2026'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'entity_dict')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_query(cypher):
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def export_all():
    entities = {}

    # 1. TechDomain (all 9)
    print("Exporting TechDomains...")
    entities['tech_domains'] = run_query("""
        MATCH (td:TechDomain)
        OPTIONAL MATCH (td)-[:PARENT_DOMAIN]->(parent:TechDomain)
        RETURN td.name AS name, td.level AS level, parent.name AS parent_name
        ORDER BY td.level, td.name
    """)

    # 2. LegalStatus (all, with counts)
    print("Exporting LegalStatuses...")
    entities['legal_statuses'] = run_query("""
        MATCH (ls:LegalStatus)<-[:HAS_STATUS]-(p:Patent)
        RETURN ls.name AS name, count(p) AS patent_count
        ORDER BY patent_count DESC
    """)

    # 3. Top provinces by org count
    print("Exporting top provinces...")
    entities['provinces'] = run_query("""
        MATCH (o:Organization)-[:LOCATED_IN]->(loc:Location)
        WHERE loc.province IS NOT NULL AND loc.province <> ''
        RETURN loc.province AS province, count(DISTINCT o) AS org_count
        ORDER BY org_count DESC LIMIT 30
    """)

    # 4. Top cities by org count
    print("Exporting top cities...")
    entities['cities'] = run_query("""
        MATCH (o:Organization)-[:LOCATED_IN]->(loc:Location)
        WHERE loc.city IS NOT NULL AND loc.city <> ''
        RETURN loc.city AS city, loc.province AS province, count(DISTINCT o) AS org_count
        ORDER BY org_count DESC LIMIT 50
    """)

    # 5. Countries (top 20)
    print("Exporting countries...")
    entities['countries'] = run_query("""
        MATCH (p:Patent)-[:PUBLISHED_IN]->(c:Country)
        RETURN c.name AS name, count(p) AS patent_count
        ORDER BY patent_count DESC LIMIT 20
    """)

    # 6. Top orgs by patent count
    print("Exporting top organizations (by patent count)...")
    entities['top_orgs_by_patents'] = run_query("""
        MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
        RETURN o.name AS name, o.entity_type AS entity_type, count(DISTINCT p) AS patent_count
        ORDER BY patent_count DESC LIMIT 100
    """)

    # 7. Top orgs by transfer count (as transferee)
    print("Exporting top organizations (by transfers received)...")
    entities['top_orgs_by_transfers'] = run_query("""
        MATCH (p:Patent)-[:TRANSFERRED_TO]->(o:Organization)
        RETURN o.name AS name, o.entity_type AS entity_type, count(DISTINCT p) AS transfer_count
        ORDER BY transfer_count DESC LIMIT 50
    """)

    # 8. Top orgs by license count
    print("Exporting top organizations (by licenses)...")
    entities['top_orgs_by_licenses'] = run_query("""
        MATCH (p:Patent)-[:LICENSED_TO]->(o:Organization)
        RETURN o.name AS name, o.entity_type AS entity_type, count(DISTINCT p) AS license_count
        ORDER BY license_count DESC LIMIT 50
    """)

    # 9. Top orgs by pledge count
    print("Exporting top organizations (by pledges)...")
    entities['top_orgs_by_pledges'] = run_query("""
        MATCH (p:Patent)-[:PLEDGED_TO]->(o:Organization)
        RETURN o.name AS name, o.entity_type AS entity_type, count(DISTINCT p) AS pledge_count
        ORDER BY pledge_count DESC LIMIT 50
    """)

    # 10. LitigationType (all)
    print("Exporting LitigationTypes...")
    entities['litigation_types'] = run_query("""
        MATCH (lt:LitigationType)<-[:HAS_LITIGATION_TYPE]-(p:Patent)
        RETURN lt.name AS name, count(DISTINCT p) AS patent_count
        ORDER BY patent_count DESC
    """)

    # 11. IPC codes (top 30 by frequency, first 4 chars)
    print("Exporting top IPC codes...")
    entities['ipc_codes'] = run_query("""
        MATCH (p:Patent)-[:CLASSIFIED_AS]->(ipc:IPCCode)
        RETURN ipc.code AS code, ipc.description AS description, count(DISTINCT p) AS patent_count
        ORDER BY patent_count DESC LIMIT 30
    """)

    # 12. PatentFamily stats
    print("Exporting PatentFamily stats...")
    entities['patent_family_stats'] = run_query("""
        MATCH (pf:PatentFamily)<-[:IN_FAMILY]-(p:Patent)
        WITH pf, count(p) AS member_count
        RETURN min(member_count) AS min_members,
               max(member_count) AS max_members,
               avg(member_count) AS avg_members,
               count(pf) AS total_families
    """)

    # 13. Sample patent families with multiple members
    print("Exporting sample multi-member families...")
    entities['sample_families'] = run_query("""
        MATCH (pf:PatentFamily)<-[:IN_FAMILY]-(p:Patent)
        WITH pf, count(p) AS member_count
        WHERE member_count >= 3
        RETURN pf.family_id AS family_id, member_count
        ORDER BY member_count DESC LIMIT 20
    """)

    # 14. Year distribution
    print("Exporting year distribution...")
    entities['year_distribution'] = run_query("""
        MATCH (p:Patent)
        WITH substring(p.application_date, 0, 4) AS year
        WHERE year IS NOT NULL
        RETURN year, count(*) AS count
        ORDER BY year DESC LIMIT 20
    """)

    # 15. Entity type distribution
    print("Exporting entity type distribution...")
    entities['entity_type_distribution'] = run_query("""
        MATCH (o:Organization)
        WHERE o.entity_type IS NOT NULL
        RETURN o.entity_type AS entity_type, count(o) AS count
        ORDER BY count DESC
    """)

    # 16. Relationship counts (for validation)
    print("Exporting relationship counts...")
    entities['relationship_counts'] = run_query("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)

    # Save
    output_path = os.path.join(OUTPUT_DIR, 'entity_dictionary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nEntity dictionary saved to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for key, data in entities.items():
        if isinstance(data, list):
            print(f"  {key}: {len(data)} entries")
        else:
            print(f"  {key}: {data}")


if __name__ == '__main__':
    export_all()
    driver.close()
