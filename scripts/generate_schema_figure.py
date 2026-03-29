#!/usr/bin/env python
"""Generate GPG schema visualization figure using graphviz."""
import graphviz

OUT_PATH = 'dpmlretriever/paper/figures/gpg_schema'

dot = graphviz.Digraph('GPG', format='pdf')
dot.attr(rankdir='LR', fontname='Helvetica', fontsize='11', bgcolor='white')
dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
dot.attr('edge', fontname='Helvetica', fontsize='8')

# Color scheme by pragmatic dimension
TECH = '#BBDEFB'     # blue - technical
LEGAL = '#FFCDD2'    # red - legal
GEO = '#C8E6C9'     # green - geographic
ORG = '#FFE0B2'     # orange - organizational
BIZ = '#E1BEE7'     # purple - business dynamics
CORE = '#FFF9C4'    # yellow - core entity

# Core node
dot.node('Patent', 'Patent\n(core entity)', fillcolor=CORE, penwidth='2')

# Technical dimension
dot.node('TechDomain', 'TechDomain\n(3-level hierarchy)', fillcolor=TECH)
dot.node('IPCCode', 'IPCCode', fillcolor=TECH)

# Legal dimension
dot.node('LegalStatus', 'LegalStatus', fillcolor=LEGAL)
dot.node('PatentFamily', 'PatentFamily', fillcolor=LEGAL)

# Geographic dimension
dot.node('Location', 'Location\n(4-level hierarchy)', fillcolor=GEO)
dot.node('Country', 'Country', fillcolor=GEO)

# Organizational dimension
dot.node('Organization', 'Organization\n(company/university/institute)', fillcolor=ORG)
dot.node('Person', 'Person', fillcolor=ORG)

# Business dynamics dimension
dot.node('LitigationType', 'LitigationType', fillcolor=BIZ)

# --- Edges ---

# Technical
dot.edge('Patent', 'TechDomain', 'BELONGS_TO', color='#1565C0')
dot.edge('Patent', 'IPCCode', 'CLASSIFIED_AS', color='#1565C0')
dot.edge('TechDomain', 'TechDomain', 'PARENT_DOMAIN', style='dashed', color='#1565C0')

# Legal
dot.edge('Patent', 'LegalStatus', 'HAS_STATUS', color='#C62828')
dot.edge('Patent', 'PatentFamily', 'IN_FAMILY', color='#C62828')

# Geographic
dot.edge('Patent', 'Country', 'PUBLISHED_IN', color='#2E7D32')
dot.edge('Organization', 'Location', 'LOCATED_IN', color='#2E7D32')
dot.edge('Location', 'Location', 'PARENT_LOCATION', style='dashed', color='#2E7D32')

# Organizational
dot.edge('Patent', 'Organization', 'APPLIED_BY', color='#E65100')
dot.edge('Patent', 'Person', 'APPLIED_BY', color='#E65100', style='dotted')
dot.edge('Patent', 'Organization', 'OWNED_BY', color='#E65100')

# Business dynamics
dot.edge('Patent', 'Organization', 'TRANSFERRED_TO', color='#6A1B9A')
dot.edge('Patent', 'Organization', 'LICENSED_TO', color='#6A1B9A', style='dashed')
dot.edge('Patent', 'Organization', 'PLEDGED_TO', color='#6A1B9A', style='dotted')
dot.edge('Patent', 'Organization', 'LITIGATED_WITH', color='#6A1B9A', dir='both')
dot.edge('Patent', 'LitigationType', 'HAS_LITIGATION_TYPE', color='#6A1B9A')

# Legend subgraph
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='Pragmatic Dimensions', style='rounded', color='gray')
    c.node('leg_tech', 'Technical', fillcolor=TECH, shape='box')
    c.node('leg_legal', 'Legal', fillcolor=LEGAL, shape='box')
    c.node('leg_geo', 'Geographic', fillcolor=GEO, shape='box')
    c.node('leg_org', 'Organizational', fillcolor=ORG, shape='box')
    c.node('leg_biz', 'Business Dynamics', fillcolor=BIZ, shape='box')

dot.render(OUT_PATH, cleanup=True)
print(f'Saved to {OUT_PATH}.pdf')
