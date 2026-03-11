# -*- coding: utf-8 -*-
"""
Prompt模板集中管理
"""

# ==============================================================================
# Schema 描述（供 LLM 生成 Cypher 时参考）
# ==============================================================================
SCHEMA_DESCRIPTION = """
Neo4j 知识图谱 Schema（绿色专利 - 氢能领域）：

节点标签及关键属性：
- Patent: application_no(唯一), title_cn, title_en, abstract_cn, abstract_en, application_date, publication_date, publication_no, patent_type, legal_status, ipc_main, tech_domain, transfer_count, license_count, pledge_count, litigation_count, full_text, embedding
- Organization: name(唯一), name_aliases, entity_type (公司/高校/研究机构/机构)
- Person: uid(唯一, 格式"姓名@机构_领域"), name, affiliated_org
- TechDomain: name(唯一), level (1=氢能技术, 2=储氢技术等, 3=物理储氢等), parent_name
- IPCCode: code(唯一), section, class_code, subclass
- Country: name(唯一)
- LegalStatus: name(唯一)
- PatentFamily: family_id(唯一), members
- Location: location_id(唯一), name, level, country, province, city, district, full_path
  # level: 1=国家, 2=省/直辖市, 3=市, 4=区县
  # 中国地址：详细到省-市-区，full_path如"中国/广东省/深圳市/南山区"
  # 外国地址：只到国家，full_path如"日本"，province/city/district为空
  # 特殊地区：中国香港、中国澳门、中国台湾（必须带"中国"前缀）

关系类型：
- (Patent)-[:APPLIED_BY]->(Organization|Person)  申请人
- (Patent)-[:OWNED_BY]->(Organization|Person)    当前权利人
- (Patent)-[:BELONGS_TO]->(TechDomain)           所属技术领域
- (Patent)-[:CLASSIFIED_AS]->(IPCCode)           IPC分类
- (Patent)-[:PUBLISHED_IN]->(Country)            公开国别
- (Patent)-[:HAS_STATUS]->(LegalStatus)          法律状态
- (Patent)-[:IN_FAMILY]->(PatentFamily)          同族专利
- (Patent)-[:TRANSFERRED_TO {transferor}]->(Organization|Person)  转让
- (Patent)-[:LICENSED_TO {licensor}]->(Organization|Person)       许可
- (Patent)-[:PLEDGED_TO {pledgor}]->(Organization|Person)         质押
- (Patent)-[:LITIGATED_WITH {role, litigation_type}]->(Organization|Person) 诉讼
- (Organization)-[:LOCATED_IN]->(Location)       机构所在地点
- (Patent)-[:ORIGINATED_FROM]->(Location)        专利来源地点
- (Location)-[:PARENT_LOCATION]->(Location)      地点层级（仅中国）
- (TechDomain)-[:PARENT_DOMAIN]->(TechDomain)   技术领域层级
- (IPCCode)-[:PARENT_IPC]->(IPCCode)            IPC层级

技术领域层级：
  氢能技术
  ├── 制氢技术
  ├── 储氢技术
  │   ├── 物理储氢
  │   ├── 合金储氢
  │   ├── 无机储氢
  │   └── 有机储氢
  ├── 氢燃料电池
  └── 氢制冷

全文索引：
- patent_fulltext: 支持对 Patent 的 title_cn, title_en, abstract_cn, abstract_en, full_text 全文搜索
- org_fulltext: 支持对 Organization 的 name, name_aliases 全文搜索
"""

# ==============================================================================
# Text-to-Cypher Few-shot 示例（模式化设计 - 无固定实体映射）
# 说明：示例中的 {占位符} 表示需要根据用户问题动态替换的内容
# ==============================================================================
FEWSHOT_EXAMPLES = {
    "factual": [
        {
            "question": "查询某个机构/企业的专利列表",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构名}'
OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
RETURN p.title_cn AS title, p.application_no AS app_no, td.name AS tech_domain, p.application_date AS date
ORDER BY p.application_date DESC LIMIT 20""",
            "pattern": "机构查询：用CONTAINS模糊匹配机构名，支持中英文"
        },
        {
            "question": "统计某机构在某技术领域的专利数量",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构名}'
MATCH (p)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
RETURN count(p) AS patent_count""",
            "pattern": "机构+领域联合过滤"
        },
        {
            "question": "根据专利申请号查询详情",
            "cypher": """MATCH (p:Patent {application_no: '{申请号}'})
OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
OPTIONAL MATCH (p)-[:HAS_STATUS]->(ls:LegalStatus)
RETURN p.title_cn AS title, p.abstract_cn AS abstract, p.application_date AS date,
       collect(DISTINCT COALESCE(applicant.name, applicant.uid)) AS applicants,
       td.name AS tech_domain, ls.name AS legal_status""",
            "pattern": "精确查询：用申请号精确匹配"
        },
        {
            "question": "查询被转让过的专利",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构名}'
AND p.transfer_count > 0
OPTIONAL MATCH (p)-[:TRANSFERRED_TO]->(transferee)
RETURN p.title_cn AS title, p.application_no AS app_no, p.transfer_count AS transfers,
       collect(DISTINCT COALESCE(transferee.name, transferee.uid)) AS transferees
ORDER BY p.transfer_count DESC LIMIT 20""",
            "pattern": "专利交易查询"
        },
        {
            "question": "根据IPC分类查询专利",
            "cypher": """MATCH (p:Patent)-[:CLASSIFIED_AS]->(ipc:IPCCode)
WHERE ipc.code STARTS WITH '{IPC前缀}'
OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
OPTIONAL MATCH (p)-[:APPLIED_BY]->(applicant)
RETURN p.title_cn AS title, p.application_no AS app_no, ipc.code AS ipc,
       td.name AS tech_domain, collect(DISTINCT COALESCE(applicant.name, applicant.uid))[..3] AS applicants
ORDER BY p.application_date DESC LIMIT 20""",
            "pattern": "IPC分类查询：用STARTS WITH匹配IPC前缀"
        },
    ],
    "statistical": [
        {
            "question": "统计各技术领域的专利数量",
            "cypher": """MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain)
RETURN td.name AS tech_domain, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "领域统计：按技术领域分组计数"
        },
        {
            "question": "统计专利数量最多的N个企业",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.entity_type IN ['公司', '机构']
RETURN o.name AS organization, o.entity_type AS type, count(p) AS patent_count
ORDER BY patent_count DESC LIMIT {N}""",
            "pattern": "企业排名：entity_type过滤企业，LIMIT控制数量"
        },
        {
            "question": "统计各国家的专利公开数量",
            "cypher": """MATCH (p:Patent)-[:PUBLISHED_IN]->(c:Country)
RETURN c.name AS country, count(p) AS patent_count
ORDER BY patent_count DESC LIMIT 30""",
            "pattern": "国家统计"
        },
        {
            "question": "统计高校的专利数量排名",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.entity_type = '高校'
RETURN o.name AS university, count(p) AS patent_count
ORDER BY patent_count DESC LIMIT 30""",
            "pattern": "高校排名：用entity_type='高校'过滤"
        },
        {
            "question": "统计各法律状态的专利数量",
            "cypher": """MATCH (p:Patent)-[:HAS_STATUS]->(ls:LegalStatus)
RETURN ls.name AS legal_status, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "法律状态分布"
        },
    ],
    "trend": [
        {
            "question": "某技术领域在某时间段的年度专利变化",
            "cypher": """MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
WHERE p.application_date >= '{起始年}-01-01' AND p.application_date < '{结束年+1}-01-01'
RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count
ORDER BY year""",
            "pattern": "时间范围趋势：用日期字符串比较过滤"
        },
        {
            "question": "近N年某领域的专利趋势",
            "cypher": """MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
WHERE substring(p.application_date, 0, 4) IN ['{年份1}', '{年份2}', '{年份3}']
RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count
ORDER BY year""",
            "pattern": "近几年趋势：用IN列出具体年份"
        },
        {
            "question": "全部氢能专利的年度变化趋势",
            "cypher": """MATCH (p:Patent)
WHERE substring(p.application_date, 0, 4) >= '{起始年}'
RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count
ORDER BY year""",
            "pattern": "全量趋势：直接查Patent节点"
        },
        {
            "question": "储氢技术各子领域的年度变化",
            "cypher": """MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain)
WHERE td.name IN ['物理储氢', '合金储氢', '无机储氢', '有机储氢']
RETURN substring(p.application_date, 0, 4) AS year, td.name AS tech_domain, count(p) AS patent_count
ORDER BY year, tech_domain""",
            "pattern": "多领域对比趋势"
        },
        {
            "question": "某机构历年专利申请数量",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构名}'
RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count
ORDER BY year""",
            "pattern": "机构趋势"
        },
    ],
    "geographic": [
        {
            "question": "某地区氢能专利数量（通过Location节点）",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location)
WHERE loc.province = '{省份}' OR loc.city = '{城市}'
RETURN count(p) AS patent_count""",
            "pattern": "地区查询：通过Location节点精确查询"
        },
        {
            "question": "某省各城市的专利分布",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location)
WHERE loc.province = '{省份}'
RETURN loc.city AS city, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "省内城市分布：按city分组"
        },
        {
            "question": "某地区近N年的专利变化",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location)
WHERE loc.city = '{城市}'
AND substring(p.application_date, 0, 4) IN ['{年份列表}']
RETURN substring(p.application_date, 0, 4) AS year, count(p) AS patent_count
ORDER BY year""",
            "pattern": "地区+时间过滤"
        },
        {
            "question": "某地区在某技术领域有哪些机构",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location)
WHERE loc.city = '{城市}'
MATCH (p)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
RETURN o.name AS organization, count(p) AS patent_count
ORDER BY patent_count DESC LIMIT 30""",
            "pattern": "地区+领域联合查询"
        },
        {
            "question": "外国机构的专利统计（只到国家）",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)-[:LOCATED_IN]->(loc:Location)
WHERE loc.country = '{国家}' AND loc.country <> '中国'
RETURN o.name AS organization, count(p) AS patent_count
ORDER BY patent_count DESC LIMIT 20""",
            "pattern": "外国查询：只用country字段"
        },
    ],
    "comparison": [
        {
            "question": "对比两个技术领域的专利数量",
            "cypher": """MATCH (p:Patent)-[:BELONGS_TO]->(td:TechDomain)
WHERE td.name IN ['{领域A}', '{领域B}']
RETURN td.name AS tech_domain, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "领域对比：用IN匹配多个领域"
        },
        {
            "question": "对比多个机构在某领域的专利数量",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构A}' OR o.name CONTAINS '{机构B}'
MATCH (p)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
RETURN o.name AS organization, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "多机构对比：用OR连接多个CONTAINS条件"
        },
        {
            "question": "对比多个机构的氢能专利总量",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.name CONTAINS '{机构A}' OR o.name CONTAINS '{机构B}'
OPTIONAL MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
RETURN o.name AS organization, count(p) AS patent_count, collect(DISTINCT td.name) AS tech_domains
ORDER BY patent_count DESC""",
            "pattern": "多机构总量对比"
        },
        {
            "question": "对比多个国家在某领域的专利",
            "cypher": """MATCH (p:Patent)-[:PUBLISHED_IN]->(c:Country)
MATCH (p)-[:BELONGS_TO]->(td:TechDomain)
WHERE c.name IN ['{国家A}', '{国家B}']
AND td.name IN ['{领域列表}']
RETURN c.name AS country, td.name AS tech_domain, count(p) AS patent_count
ORDER BY country, patent_count DESC""",
            "pattern": "国家+领域对比"
        },
        {
            "question": "对比不同类型专利的数量",
            "cypher": """MATCH (p:Patent)
WHERE p.patent_type IS NOT NULL
RETURN p.patent_type AS patent_type, count(p) AS patent_count
ORDER BY patent_count DESC""",
            "pattern": "专利类型对比"
        },
    ],
    "multi_hop": [
        {
            "question": "专利最多的企业主要研究哪个技术领域",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
WHERE o.entity_type IN ['公司', '机构']
WITH o, count(p) AS total_patents
ORDER BY total_patents DESC LIMIT 1
MATCH (p2:Patent)-[:APPLIED_BY]->(o)
MATCH (p2)-[:BELONGS_TO]->(td:TechDomain)
RETURN o.name AS top_organization, total_patents, td.name AS tech_domain, count(p2) AS domain_patents
ORDER BY domain_patents DESC""",
            "pattern": "多跳查询：先找Top企业，再查其技术分布"
        },
        {
            "question": "某领域专利最多的企业有哪些交易",
            "cypher": """MATCH (p:Patent)-[:APPLIED_BY]->(o:Organization)
MATCH (p)-[:BELONGS_TO]->(td:TechDomain {name: '{技术领域}'})
WITH o, count(p) AS field_patents
ORDER BY field_patents DESC LIMIT 5
MATCH (p2:Patent)-[:APPLIED_BY]->(o)
WHERE p2.license_count > 0 OR p2.transfer_count > 0
RETURN o.name AS organization, field_patents,
       p2.title_cn AS patent, p2.license_count AS licenses, p2.transfer_count AS transfers""",
            "pattern": "领域Top企业的交易情况"
        },
    ],
    "fuzzy": [
        {
            "question": "搜索某技术关键词相关的专利",
            "cypher": """CALL db.index.fulltext.queryNodes('patent_fulltext', '{关键词1} {关键词2} {关键词3}')
YIELD node, score
OPTIONAL MATCH (node)-[:BELONGS_TO]->(td:TechDomain)
OPTIONAL MATCH (node)-[:APPLIED_BY]->(applicant)
RETURN node.application_no AS app_no, node.title_cn AS title, td.name AS tech_domain,
       collect(DISTINCT COALESCE(applicant.name, applicant.uid))[..3] AS applicants, score
ORDER BY score DESC LIMIT 20""",
            "pattern": "全文搜索：用空格分隔多个关键词"
        },
        {
            "question": "搜索某材料/技术相关研究",
            "cypher": """CALL db.index.fulltext.queryNodes('patent_fulltext', '{搜索词}')
YIELD node, score
WHERE score > 0.5
OPTIONAL MATCH (node)-[:BELONGS_TO]->(td:TechDomain)
RETURN node.application_no AS app_no, node.title_cn AS title, td.name AS tech_domain, score
ORDER BY score DESC LIMIT 20""",
            "pattern": "语义搜索：添加score阈值过滤"
        },
    ],
}

# ==============================================================================
# 实体消解 Prompt（基础版）
# ==============================================================================
ENTITY_MERGE_PROMPT = """你是一个专利数据实体消解专家。请分析这些名称，将表示**同一实体**的名称合并，选择其中**最规范、最完整**的名称作为标准名。

注意：
1. 中英文名称可能指同一实体（如"丰田汽车"和"Toyota Motor Corporation"）
2. 简称和全称可能指同一实体（如"中石化"和"中国石油化工集团有限公司"）
3. 有些名称虽然相似但是不同实体，不要错误合并
4. 个人姓名如果上下文不同则不要合并

待分析的名称列表：
{names}

请返回JSON格式的映射结果，key是原始名称，value是标准名称：
{{"原名1": "标准名", "原名2": "标准名", ...}}"""

CROSS_LANGUAGE_PROMPT = """你是专利数据实体匹配专家。请判断以下中文名称和英文名称中，哪些指的是同一个实体。

中文名称：
{cn_names}

英文名称：
{en_names}

对于匹配的对，选择**中文名**作为标准名（如果中文名更规范的话）。

请返回JSON格式的映射结果，key是英文名，value是匹配的中文标准名：
{{"English Name 1": "中文标准名1", ...}}

如果某个英文名没有对应的中文名，不要包含在结果中。"""

# ==============================================================================
# 增强实体消解 Prompt（向量粗筛 + LLM决策，输出中文标准名）
# ==============================================================================
ENHANCED_ENTITY_RESOLUTION_PROMPT = """你是专利数据实体消解专家。请分析以下企业/机构名称，判断哪些指的是同一实体，并选择或生成标准的**中文名称**。

## 任务要求
1. 判断这些名称是否指同一实体
2. 如果是同一实体，输出统一的**中文标准名称**
3. 中文标准名称规范：
   - 使用正式全称（如"丰田汽车公司"而非"丰田"）
   - 外国公司使用通用中文译名
   - 保留"有限公司"、"股份有限公司"等后缀
   - 高校/研究机构使用官方中文名称

## 示例
输入：["Toyota Motor Corporation", "丰田", "TOYOTA", "トヨタ自動車株式会社"]
输出：{{"is_same": true, "standard_name_cn": "丰田汽车公司", "confidence": 0.95, "reason": "日本丰田汽车的不同语言表示"}}

输入：["清华大学", "Tsinghua University", "清华"]
输出：{{"is_same": true, "standard_name_cn": "清华大学", "confidence": 0.98, "reason": "清华大学的中英文名和简称"}}

输入：["北京大学", "上海交通大学"]
输出：{{"is_same": false, "entities": [{{"name": "北京大学", "standard_name_cn": "北京大学"}}, {{"name": "上海交通大学", "standard_name_cn": "上海交通大学"}}]}}

## 待分析实体组
{entities}

## 输出格式
返回JSON：
- 如果是同一实体：{{"is_same": true, "standard_name_cn": "中文标准名", "confidence": 0.0-1.0, "reason": "简要理由"}}
- 如果不是同一实体：{{"is_same": false, "entities": [{{"name": "原名", "standard_name_cn": "中文标准名"}}, ...]}}

请分析并返回JSON："""

# ==============================================================================
# 统一实体处理 Prompt（实体消解 + 类型判断 + 地点提取 三合一）
# ==============================================================================
UNIFIED_ENTITY_RESOLUTION_PROMPT = """你是专利数据实体分析专家。请分析以下名称，完成三个任务：
1. 判断这些名称是否指同一实体（实体消解）
2. 判断实体类型是"person"(个人)还是"organization"(机构)
3. 如果是organization，提取地点信息

## 实体类型判断规则
- **person（个人）**：发明人、自然人，如"张伟"、"Dong, Xueliang"、"田中太郎"
- **organization（机构）**：公司、大学、研究院、政府机构等

## 地点提取规则（仅organization需要）
- **中国机构**：精确到省-市-区（如有），如"广东省/深圳市/南山区"
- **外国机构**：只到国家，省市区全部填null
- **特殊地区**：香港→province:"中国香港"，澳门→"中国澳门"，台湾→"中国台湾"
- **无法确定**：location整体为null
- **person类型**：location必须为null

## 待分析实体
{entities}

## 输出格式
返回JSON，有两种情况：

**情况1：同一实体（所有名称指向同一个实体）**
```json
{{
  "is_same": true,
  "standard_name_cn": "中文标准名称",
  "entity_type": "organization 或 person",
  "location": {{
    "country": "国家",
    "province": "省份或null",
    "city": "城市或null",
    "district": "区县或null"
  }}
}}
```

**情况2：不同实体（名称指向不同实体）**
```json
{{
  "is_same": false,
  "entities": [
    {{
      "name": "原始名称1",
      "standard_name_cn": "标准名1",
      "entity_type": "organization 或 person",
      "location": {{...}} 或 null
    }},
    {{
      "name": "原始名称2",
      "standard_name_cn": "标准名2",
      "entity_type": "person",
      "location": null
    }}
  ]
}}
```

## 示例

输入：["Toyota Motor Corporation", "丰田汽车", "トヨタ自動車"]
输出：
```json
{{
  "is_same": true,
  "standard_name_cn": "丰田汽车公司",
  "entity_type": "organization",
  "location": {{"country": "日本", "province": null, "city": null, "district": null}}
}}
```

输入：["张伟", "张伟科技有限公司"]
输出：
```json
{{
  "is_same": false,
  "entities": [
    {{"name": "张伟", "standard_name_cn": "张伟", "entity_type": "person", "location": null}},
    {{"name": "张伟科技有限公司", "standard_name_cn": "张伟科技有限公司", "entity_type": "organization", "location": {{"country": "中国", "province": null, "city": null, "district": null}}}}
  ]
}}
```

输入：["深圳市比亚迪股份有限公司", "BYD Company"]
输出：
```json
{{
  "is_same": true,
  "standard_name_cn": "比亚迪股份有限公司",
  "entity_type": "organization",
  "location": {{"country": "中国", "province": "广东省", "city": "深圳市", "district": null}}
}}
```

请分析并返回JSON（不要包含```json标记）："""

# ==============================================================================
# 批量实体类型和地点判断 Prompt（用于处理单独实体，无需判断是否相同）
# ==============================================================================
BATCH_ENTITY_CLASSIFICATION_PROMPT = """你是专利数据实体分析专家。请分析以下实体名称，判断每个实体的类型和地点信息。

## 实体类型
- **person**：个人/自然人，如"张伟"、"Smith, John"、"田中太郎"
- **organization**：公司、大学、研究机构等

## 地点规则
- person类型：location为null
- organization类型（中国）：精确到省-市-区
- organization类型（外国）：只到国家，省市区为null
- 特殊地区：香港/澳门/台湾的province填"中国香港"/"中国澳门"/"中国台湾"
- 无法确定地点：location为null

## 待分析实体
{entities}

## 输出格式
返回JSON数组：
```json
[
  {{
    "name": "原始名称",
    "standard_name_cn": "中文标准名（如果是外文则翻译，中文保持原样）",
    "entity_type": "person 或 organization",
    "location": {{"country": "...", "province": "...", "city": "...", "district": "..."}} 或 null
  }},
  ...
]
```

## 示例

输入：["Dong, Xueliang", "苏州氢策科技有限公司", "Honda Motor Co., Ltd."]
输出：
```json
[
  {{"name": "Dong, Xueliang", "standard_name_cn": "Dong, Xueliang", "entity_type": "person", "location": null}},
  {{"name": "苏州氢策科技有限公司", "standard_name_cn": "苏州氢策科技有限公司", "entity_type": "organization", "location": {{"country": "中国", "province": "江苏省", "city": "苏州市", "district": null}}}},
  {{"name": "Honda Motor Co., Ltd.", "standard_name_cn": "本田技研工业株式会社", "entity_type": "organization", "location": {{"country": "日本", "province": null, "city": null, "district": null}}}}
]
```

请分析并返回JSON数组（不要包含```json标记）："""

# ==============================================================================
# 地点提取 Prompt（保留，用于兼容）
# ==============================================================================
LOCATION_EXTRACTION_PROMPT = """你是地理信息提取专家。请从以下机构/组织名称中提取详细的地点信息。

## 规则
1. **中国地址**：尽量精确到省-市-区/县
   - 直辖市（北京、上海、天津、重庆）：省份填"北京市"/"上海市"等
   - 普通省份：省份填"XX省"，城市填"XX市"
   - 如果只能确定省份，城市和区县填null
2. **外国地址**：只需要国家级别，省市区全部填null
3. **特殊地区**：
   - 香港 → country:"中国", province:"中国香港"
   - 澳门 → country:"中国", province:"中国澳门"
   - 台湾 → country:"中国", province:"中国台湾"
4. 如果完全无法确定地点，所有字段填null

## 待分析机构
{organizations}

## 输出格式
返回JSON：
{{
  "locations": [
    {{
      "organization": "机构原名",
      "country": "中国/日本/美国/...",
      "province": "省份（仅中国，带'省'或'市'后缀）或null",
      "city": "城市（仅中国，带'市'后缀）或null",
      "district": "区县（仅中国）或null",
      "confidence": 0.0-1.0
    }}
  ]
}}

## 示例
输入：["深圳市比亚迪股份有限公司", "Toyota Motor Corporation", "香港科技大学"]
输出：
{{
  "locations": [
    {{"organization": "深圳市比亚迪股份有限公司", "country": "中国", "province": "广东省", "city": "深圳市", "district": null, "confidence": 0.95}},
    {{"organization": "Toyota Motor Corporation", "country": "日本", "province": null, "city": null, "district": null, "confidence": 0.90}},
    {{"organization": "香港科技大学", "country": "中国", "province": "中国香港", "city": null, "district": null, "confidence": 0.95}}
  ]
}}

请提取地点信息："""

# ==============================================================================
# Agentic RAG - 结果评估 Prompt
# ==============================================================================
RESULT_EVALUATION_PROMPT = """请评估以下检索结果是否足够回答用户问题。

用户问题：{question}

检索结果数量：{result_count}
检索结果摘要：
{result_summary}

请判断：
1. 结果数量是否充足？
2. 结果内容是否与问题相关？
3. 是否需要补充检索？

返回JSON：
{{
    "is_sufficient": true/false,
    "relevance_score": 0.0-1.0,
    "need_supplement": true/false,
    "supplement_keywords": ["补充搜索关键词1", "..."]  // 仅当need_supplement为true时填写
}}"""
