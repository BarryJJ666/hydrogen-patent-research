// 技术领域节点
MERGE (t:TechDomain {name: '氢能技术'}) SET t.level = 1, t.parent_name = '';
MERGE (t:TechDomain {name: '储氢技术'}) SET t.level = 2, t.parent_name = '氢能技术';
MERGE (t:TechDomain {name: '制氢技术'}) SET t.level = 2, t.parent_name = '氢能技术';
MERGE (t:TechDomain {name: '氢燃料电池'}) SET t.level = 2, t.parent_name = '氢能技术';
MERGE (t:TechDomain {name: '氢制冷'}) SET t.level = 2, t.parent_name = '氢能技术';
MERGE (t:TechDomain {name: '物理储氢'}) SET t.level = 3, t.parent_name = '储氢技术';
MERGE (t:TechDomain {name: '合金储氢'}) SET t.level = 3, t.parent_name = '储氢技术';
MERGE (t:TechDomain {name: '无机储氢'}) SET t.level = 3, t.parent_name = '储氢技术';
MERGE (t:TechDomain {name: '有机储氢'}) SET t.level = 3, t.parent_name = '储氢技术';

// 技术领域层级关系
MATCH (child:TechDomain {name: '储氢技术'}) MATCH (parent:TechDomain {name: '氢能技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '制氢技术'}) MATCH (parent:TechDomain {name: '氢能技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '氢燃料电池'}) MATCH (parent:TechDomain {name: '氢能技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '氢制冷'}) MATCH (parent:TechDomain {name: '氢能技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '物理储氢'}) MATCH (parent:TechDomain {name: '储氢技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '合金储氢'}) MATCH (parent:TechDomain {name: '储氢技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '无机储氢'}) MATCH (parent:TechDomain {name: '储氢技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);
MATCH (child:TechDomain {name: '有机储氢'}) MATCH (parent:TechDomain {name: '储氢技术'}) MERGE (child)-[:PARENT_DOMAIN]->(parent);