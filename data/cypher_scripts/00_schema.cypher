// 唯一性约束
CREATE CONSTRAINT patent_app_no IF NOT EXISTS FOR (p:Patent) REQUIRE p.application_no IS UNIQUE;
CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;
CREATE CONSTRAINT person_uid IF NOT EXISTS FOR (p:Person) REQUIRE p.uid IS UNIQUE;
CREATE CONSTRAINT tech_domain_name IF NOT EXISTS FOR (t:TechDomain) REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT ipc_code IF NOT EXISTS FOR (i:IPCCode) REQUIRE i.code IS UNIQUE;
CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT legal_status_name IF NOT EXISTS FOR (l:LegalStatus) REQUIRE l.name IS UNIQUE;
CREATE CONSTRAINT family_id IF NOT EXISTS FOR (f:PatentFamily) REQUIRE f.family_id IS UNIQUE;
CREATE CONSTRAINT litigation_type_name IF NOT EXISTS FOR (lt:LitigationType) REQUIRE lt.name IS UNIQUE;

// 全文索引（分离中英文以提高搜索精度）
// 混合索引（兼容旧版本）
CREATE FULLTEXT INDEX patent_fulltext IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_cn, p.title_en, p.abstract_cn, p.abstract_en, p.full_text];
// 中文专用索引
CREATE FULLTEXT INDEX patent_fulltext_cn IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_cn, p.abstract_cn];
// 英文专用索引
CREATE FULLTEXT INDEX patent_fulltext_en IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_en, p.abstract_en];
// 机构索引
CREATE FULLTEXT INDEX org_fulltext IF NOT EXISTS FOR (o:Organization) ON EACH [o.name, o.name_aliases];

// 普通索引
CREATE INDEX patent_type_idx IF NOT EXISTS FOR (p:Patent) ON (p.patent_type);
CREATE INDEX patent_date_idx IF NOT EXISTS FOR (p:Patent) ON (p.application_date);
CREATE INDEX patent_ipc_idx IF NOT EXISTS FOR (p:Patent) ON (p.ipc_main);
CREATE INDEX tech_domain_level_idx IF NOT EXISTS FOR (t:TechDomain) ON (t.level);
CREATE INDEX patent_transfer_idx IF NOT EXISTS FOR (p:Patent) ON (p.transfer_count);
CREATE INDEX patent_license_idx IF NOT EXISTS FOR (p:Patent) ON (p.license_count);
CREATE INDEX patent_pledge_idx IF NOT EXISTS FOR (p:Patent) ON (p.pledge_count);
CREATE INDEX patent_litigation_idx IF NOT EXISTS FOR (p:Patent) ON (p.litigation_count);

// Location节点约束和索引
CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE;
CREATE INDEX location_country_idx IF NOT EXISTS FOR (l:Location) ON (l.country);
CREATE INDEX location_province_idx IF NOT EXISTS FOR (l:Location) ON (l.province);
CREATE INDEX location_city_idx IF NOT EXISTS FOR (l:Location) ON (l.city);
CREATE INDEX location_level_idx IF NOT EXISTS FOR (l:Location) ON (l.level);