# -*- coding: utf-8 -*-
"""
Cypher DSL生成器
生成Neo4j导入用的Cypher语句文件
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import (
    OUTPUT_DIR, NEO4J_IMPORT, TECH_DOMAIN_HIERARCHY,
    VECTOR_SEARCH
)
from utils.logger import get_logger

logger = get_logger(__name__)


class CypherGenerator:
    """
    Cypher DSL生成器
    - 生成Schema（约束、索引）
    - 生成节点创建语句
    - 生成关系创建语句
    - 支持批量分文件
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = NEO4J_IMPORT["batch_size"]

    def generate_all(self, records: List[Dict],
                     unified_mapping: Dict[str, Dict]) -> Dict[str, int]:
        """
        生成所有Cypher文件

        Args:
            records: 专利记录（已应用实体消解，包含xxx_resolved字段）
            unified_mapping: 统一实体映射表 {原名: {standard_name, entity_type, location}}

        Returns:
            {filename: statement_count}
        """
        logger.info("开始生成Cypher DSL文件...")

        stats = {}

        # 1. Schema
        stats.update(self._generate_schema())

        # 2. 向量索引
        stats.update(self._generate_vector_index())

        # 3. 技术领域节点
        stats.update(self._generate_tech_domains())

        # 4. IPC代码节点
        ipc_codes = self._collect_ipc_codes(records)
        stats.update(self._generate_ipc_codes(ipc_codes))

        # 5. 国家节点
        countries = self._collect_countries(records)
        stats.update(self._generate_countries(countries))

        # 6. 法律状态节点
        legal_statuses = self._collect_legal_statuses(records)
        stats.update(self._generate_legal_statuses(legal_statuses))

        # 从unified_mapping中分离机构和人物
        orgs_info = {}  # {standard_name: {aliases, location}}
        persons_info = {}  # {name: [uid]}

        for original_name, info in unified_mapping.items():
            # 获取标准名，如果为None则使用原始名
            standard_name = info.get("standard_name") or original_name
            # 跳过空值
            if not standard_name or not original_name:
                continue

            entity_type = info.get("entity_type", "organization")
            location = info.get("location")

            if entity_type == "person":
                # 人物：生成uid
                if standard_name not in persons_info:
                    persons_info[standard_name] = []
                # 人物uid使用原始名@标准名格式
                uid = f"{original_name}@{standard_name}" if original_name != standard_name else standard_name
                if uid not in persons_info[standard_name]:
                    persons_info[standard_name].append(uid)
            else:
                # 机构
                if standard_name not in orgs_info:
                    orgs_info[standard_name] = {
                        "aliases": set(),
                        "location": location
                    }
                if original_name != standard_name:
                    orgs_info[standard_name]["aliases"].add(original_name)
                # 更新location（如果之前没有）
                if not orgs_info[standard_name]["location"] and location:
                    orgs_info[standard_name]["location"] = location

        # 7. 机构节点
        stats.update(self._generate_organizations_unified(orgs_info))

        # 8. 人物节点
        stats.update(self._generate_persons_unified(persons_info))

        # 9. 专利族节点
        patent_families = self._collect_patent_families(records)
        stats.update(self._generate_patent_families(patent_families))

        # 10. 诉讼类型节点
        litigation_types = self._collect_litigation_types(records)
        stats.update(self._generate_litigation_types(litigation_types))

        # 11. Location节点（从orgs_info中提取）
        stats.update(self._generate_locations_unified(orgs_info))

        # 11. 专利节点
        stats.update(self._generate_patents(records))

        # 12. 关系
        stats.update(self._generate_relationships_unified(records, unified_mapping))

        # 13. Organization-Location关系
        stats.update(self._generate_org_location_relationships(orgs_info))

        # 生成清单文件
        self._generate_manifest(stats)

        total = sum(stats.values())
        logger.info(f"Cypher生成完成: {len(stats)} 个文件, {total} 条语句")

        return stats

    def _generate_schema(self) -> Dict[str, int]:
        """生成Schema约束和索引"""
        statements = [
            "// 唯一性约束",
            "CREATE CONSTRAINT patent_app_no IF NOT EXISTS FOR (p:Patent) REQUIRE p.application_no IS UNIQUE;",
            "CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;",
            "CREATE CONSTRAINT person_uid IF NOT EXISTS FOR (p:Person) REQUIRE p.uid IS UNIQUE;",
            "CREATE CONSTRAINT tech_domain_name IF NOT EXISTS FOR (t:TechDomain) REQUIRE t.name IS UNIQUE;",
            "CREATE CONSTRAINT ipc_code IF NOT EXISTS FOR (i:IPCCode) REQUIRE i.code IS UNIQUE;",
            "CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE;",
            "CREATE CONSTRAINT legal_status_name IF NOT EXISTS FOR (l:LegalStatus) REQUIRE l.name IS UNIQUE;",
            "CREATE CONSTRAINT family_id IF NOT EXISTS FOR (f:PatentFamily) REQUIRE f.family_id IS UNIQUE;",
            "CREATE CONSTRAINT litigation_type_name IF NOT EXISTS FOR (lt:LitigationType) REQUIRE lt.name IS UNIQUE;",
            "",
            "// 全文索引（分离中英文以提高搜索精度）",
            "// 混合索引（兼容旧版本）",
            "CREATE FULLTEXT INDEX patent_fulltext IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_cn, p.title_en, p.abstract_cn, p.abstract_en, p.full_text];",
            "// 中文专用索引",
            "CREATE FULLTEXT INDEX patent_fulltext_cn IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_cn, p.abstract_cn];",
            "// 英文专用索引",
            "CREATE FULLTEXT INDEX patent_fulltext_en IF NOT EXISTS FOR (p:Patent) ON EACH [p.title_en, p.abstract_en];",
            "// 机构索引",
            "CREATE FULLTEXT INDEX org_fulltext IF NOT EXISTS FOR (o:Organization) ON EACH [o.name, o.name_aliases];",
            "",
            "// 普通索引",
            "CREATE INDEX patent_type_idx IF NOT EXISTS FOR (p:Patent) ON (p.patent_type);",
            "CREATE INDEX patent_date_idx IF NOT EXISTS FOR (p:Patent) ON (p.application_date);",
            "CREATE INDEX patent_ipc_idx IF NOT EXISTS FOR (p:Patent) ON (p.ipc_main);",
            "CREATE INDEX tech_domain_level_idx IF NOT EXISTS FOR (t:TechDomain) ON (t.level);",
            "CREATE INDEX patent_transfer_idx IF NOT EXISTS FOR (p:Patent) ON (p.transfer_count);",
            "CREATE INDEX patent_license_idx IF NOT EXISTS FOR (p:Patent) ON (p.license_count);",
            "CREATE INDEX patent_pledge_idx IF NOT EXISTS FOR (p:Patent) ON (p.pledge_count);",
            "CREATE INDEX patent_litigation_idx IF NOT EXISTS FOR (p:Patent) ON (p.litigation_count);",
            "",
            "// Location节点约束和索引",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE;",
            "CREATE INDEX location_country_idx IF NOT EXISTS FOR (l:Location) ON (l.country);",
            "CREATE INDEX location_province_idx IF NOT EXISTS FOR (l:Location) ON (l.province);",
            "CREATE INDEX location_city_idx IF NOT EXISTS FOR (l:Location) ON (l.city);",
            "CREATE INDEX location_level_idx IF NOT EXISTS FOR (l:Location) ON (l.level);",
        ]

        return self._write_file("00_schema.cypher", statements)

    def _generate_vector_index(self) -> Dict[str, int]:
        """生成向量索引"""
        dim = VECTOR_SEARCH["embedding_dim"]
        index_name = VECTOR_SEARCH["vector_index_name"]

        statements = [
            f"// 向量索引 (需要Neo4j 5.11+)",
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS FOR (p:Patent) ON (p.embedding) OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: 'cosine'}}}};",
        ]

        return self._write_file("00_vector_index.cypher", statements)

    def _generate_tech_domains(self) -> Dict[str, int]:
        """生成技术领域节点和层级关系"""
        statements = ["// 技术领域节点"]

        # 创建节点
        for name, info in TECH_DOMAIN_HIERARCHY.items():
            level = info["level"]
            parent = info.get("parent", "")
            stmt = f"MERGE (t:TechDomain {{name: '{self._escape(name)}'}}) SET t.level = {level}, t.parent_name = '{self._escape(parent)}';"
            statements.append(stmt)

        statements.append("")
        statements.append("// 技术领域层级关系")

        # 创建层级关系
        for name, info in TECH_DOMAIN_HIERARCHY.items():
            parent = info.get("parent")
            if parent:
                stmt = f"MATCH (child:TechDomain {{name: '{self._escape(name)}'}}) MATCH (parent:TechDomain {{name: '{self._escape(parent)}'}}) MERGE (child)-[:PARENT_DOMAIN]->(parent);"
                statements.append(stmt)

        return self._write_file("01_tech_domains.cypher", statements)

    def _generate_ipc_codes(self, ipc_codes: Set[str]) -> Dict[str, int]:
        """生成IPC代码节点"""
        statements = ["// IPC分类节点"]

        for code in sorted(ipc_codes):
            if not code:
                continue

            # 解析IPC
            section = code[0] if len(code) >= 1 else ""
            class_code = code[:3] if len(code) >= 3 else ""
            subclass = code[:4] if len(code) >= 4 else ""

            stmt = f"MERGE (i:IPCCode {{code: '{self._escape(code)}'}}) SET i.section = '{section}', i.class_code = '{class_code}', i.subclass = '{subclass}';"
            statements.append(stmt)

        return self._write_file("02_ipc_codes.cypher", statements)

    def _generate_countries(self, countries: Set[str]) -> Dict[str, int]:
        """生成国家节点"""
        statements = ["// 国家节点"]

        for country in sorted(countries):
            if not country:
                continue
            stmt = f"MERGE (c:Country {{name: '{self._escape(country)}'}});"
            statements.append(stmt)

        return self._write_file("03_countries.cypher", statements)

    def _generate_legal_statuses(self, statuses: Set[str]) -> Dict[str, int]:
        """生成法律状态节点"""
        statements = ["// 法律状态节点"]

        for status in sorted(statuses):
            if not status:
                continue
            stmt = f"MERGE (l:LegalStatus {{name: '{self._escape(status)}'}});"
            statements.append(stmt)

        return self._write_file("04_legal_statuses.cypher", statements)

    def _generate_organizations(self, unique_orgs: Set[str],
                                org_mapping: Dict[str, str]) -> Dict[str, int]:
        """生成机构节点（旧版兼容）"""
        statements = ["// 机构节点"]

        # 收集别名
        aliases_map = defaultdict(set)
        for original, standard in org_mapping.items():
            if original != standard:
                aliases_map[standard].add(original)

        for org_name in sorted(unique_orgs):
            if not org_name:
                continue

            # 判断实体类型
            entity_type = self._classify_org_type(org_name)

            # 收集别名
            aliases = aliases_map.get(org_name, set())
            aliases_str = "; ".join(sorted(aliases)) if aliases else ""

            stmt = f"MERGE (o:Organization {{name: '{self._escape(org_name)}'}}) SET o.entity_type = '{entity_type}', o.name_aliases = '{self._escape(aliases_str)}';"
            statements.append(stmt)

        return self._write_file("05_organizations.cypher", statements)

    def _generate_organizations_unified(self, orgs_info: Dict[str, Dict]) -> Dict[str, int]:
        """生成机构节点（新版统一处理）"""
        statements = ["// 机构节点"]

        # 过滤并排序，跳过None键
        valid_items = [(k, v) for k, v in orgs_info.items() if k]
        for org_name, info in sorted(valid_items, key=lambda x: x[0]):

            # 判断实体类型
            entity_type = self._classify_org_type(org_name)

            # 收集别名
            aliases = info.get("aliases", set())
            aliases_str = "; ".join(sorted(aliases)) if aliases else ""

            stmt = f"MERGE (o:Organization {{name: '{self._escape(org_name)}'}}) SET o.entity_type = '{entity_type}', o.name_aliases = '{self._escape(aliases_str)}';"
            statements.append(stmt)

        return self._write_file("05_organizations.cypher", statements)

    def _generate_persons(self, person_info: Dict[str, List[Dict]]) -> Dict[str, int]:
        """生成人物节点（旧版兼容）"""
        statements = ["// 人物节点"]

        for name, identities in person_info.items():
            for identity in identities:
                uid = identity["uid"]
                affiliated_org = identity.get("affiliated_org", "")

                stmt = f"MERGE (p:Person {{uid: '{self._escape(uid)}'}}) SET p.name = '{self._escape(name)}', p.affiliated_org = '{self._escape(affiliated_org)}';"
                statements.append(stmt)

        return self._write_file("06_persons.cypher", statements)

    def _generate_persons_unified(self, persons_info: Dict[str, List[str]]) -> Dict[str, int]:
        """生成人物节点（新版统一处理）"""
        statements = ["// 人物节点"]

        # 过滤None键
        valid_items = [(k, v) for k, v in persons_info.items() if k]
        for name, uids in valid_items:
            for uid in uids:
                if uid:  # 跳过空uid
                    stmt = f"MERGE (p:Person {{uid: '{self._escape(uid)}'}}) SET p.name = '{self._escape(name)}';"
                    statements.append(stmt)

        return self._write_file("06_persons.cypher", statements)

    def _generate_patent_families(self, families: Dict[str, List[str]]) -> Dict[str, int]:
        """生成专利族节点"""
        statements = ["// 专利族节点"]

        for family_id, members in families.items():
            if not family_id:
                continue

            members_str = "; ".join(members[:20])  # 限制成员数量
            stmt = f"MERGE (f:PatentFamily {{family_id: '{self._escape(family_id)}'}}) SET f.members = '{self._escape(members_str)}';"
            statements.append(stmt)

        return self._write_file("07_patent_families.cypher", statements)

    def _generate_locations(self, location_data: Dict[str, Dict]) -> Dict[str, int]:
        """
        生成Location节点（旧版兼容）

        Args:
            location_data: {org_name: {country, province, city, district, full_path, level, location_id}}
        """
        statements = ["// Location节点"]

        # 收集所有唯一的location节点
        unique_locations = {}

        for org_name, loc_info in location_data.items():
            if not loc_info.get("country"):
                continue

            # 国家级节点
            country = loc_info.get("country")
            if country:
                country_id = f"cn" if country == "中国" else country[:2].lower()
                if country_id not in unique_locations:
                    unique_locations[country_id] = {
                        "location_id": country_id,
                        "name": country,
                        "level": 1,
                        "country": country,
                        "province": "",
                        "city": "",
                        "district": "",
                        "full_path": country,
                    }

            # 省级节点（仅中国）
            province = loc_info.get("province")
            if province and country == "中国":
                loc_id = loc_info.get("location_id", "")
                # 取到省级的ID
                parts = loc_id.split("-")
                if len(parts) >= 2:
                    province_id = "-".join(parts[:2])
                    if province_id not in unique_locations:
                        unique_locations[province_id] = {
                            "location_id": province_id,
                            "name": province,
                            "level": 2,
                            "country": country,
                            "province": province,
                            "city": "",
                            "district": "",
                            "full_path": f"{country}/{province}",
                        }

            # 市级节点（仅中国）
            city = loc_info.get("city")
            if city and country == "中国":
                loc_id = loc_info.get("location_id", "")
                parts = loc_id.split("-")
                if len(parts) >= 3:
                    city_id = "-".join(parts[:3])
                    if city_id not in unique_locations:
                        unique_locations[city_id] = {
                            "location_id": city_id,
                            "name": city,
                            "level": 3,
                            "country": country,
                            "province": province or "",
                            "city": city,
                            "district": "",
                            "full_path": f"{country}/{province}/{city}" if province else f"{country}/{city}",
                        }

            # 区县级节点（仅中国）
            district = loc_info.get("district")
            if district and country == "中国":
                loc_id = loc_info.get("location_id", "")
                if loc_id and loc_id not in unique_locations:
                    unique_locations[loc_id] = {
                        "location_id": loc_id,
                        "name": district,
                        "level": 4,
                        "country": country,
                        "province": province or "",
                        "city": city or "",
                        "district": district,
                        "full_path": loc_info.get("full_path", ""),
                    }

        # 生成Cypher语句
        for loc_id, loc in sorted(unique_locations.items()):
            stmt = (
                f"MERGE (loc:Location {{location_id: '{self._escape(loc['location_id'])}'}}) "
                f"SET loc.name = '{self._escape(loc['name'])}', "
                f"loc.level = {loc['level']}, "
                f"loc.country = '{self._escape(loc['country'])}', "
                f"loc.province = '{self._escape(loc['province'])}', "
                f"loc.city = '{self._escape(loc['city'])}', "
                f"loc.district = '{self._escape(loc['district'])}', "
                f"loc.full_path = '{self._escape(loc['full_path'])}';"
            )
            statements.append(stmt)

        # 生成Location层级关系（仅中国）
        statements.append("")
        statements.append("// Location层级关系")

        for loc_id, loc in unique_locations.items():
            if loc["level"] > 1 and loc["country"] == "中国":
                # 找到父节点
                parts = loc_id.split("-")
                if len(parts) > 1:
                    parent_id = "-".join(parts[:-1])
                    if parent_id in unique_locations:
                        stmt = (
                            f"MATCH (child:Location {{location_id: '{self._escape(loc_id)}'}}) "
                            f"MATCH (parent:Location {{location_id: '{self._escape(parent_id)}'}}) "
                            f"MERGE (child)-[:PARENT_LOCATION]->(parent);"
                        )
                        statements.append(stmt)

        return self._write_file("10_locations.cypher", statements)

    def _generate_locations_unified(self, orgs_info: Dict[str, Dict]) -> Dict[str, int]:
        """
        生成Location节点（新版统一处理）

        从orgs_info中提取location信息生成节点
        """
        statements = ["// Location节点"]

        # 收集所有唯一的location节点
        unique_locations = {}

        # 过滤None键
        valid_items = [(k, v) for k, v in orgs_info.items() if k]
        for org_name, info in valid_items:
            loc_info = info.get("location")
            if not loc_info or not loc_info.get("country"):
                continue

            country = loc_info.get("country")
            province = loc_info.get("province")
            city = loc_info.get("city")
            district = loc_info.get("district")

            # 国家级节点
            country_id = self._gen_location_id(country)
            if country_id not in unique_locations:
                unique_locations[country_id] = {
                    "location_id": country_id,
                    "name": country,
                    "level": 1,
                    "country": country,
                    "province": "",
                    "city": "",
                    "district": "",
                    "full_path": country,
                }

            # 省级节点（仅中国）
            if province and country == "中国":
                province_id = f"{country_id}-{self._get_pinyin_abbr(province)}"
                if province_id not in unique_locations:
                    unique_locations[province_id] = {
                        "location_id": province_id,
                        "name": province,
                        "level": 2,
                        "country": country,
                        "province": province,
                        "city": "",
                        "district": "",
                        "full_path": f"{country}/{province}",
                    }

                # 市级节点
                if city:
                    city_id = f"{province_id}-{self._get_pinyin_abbr(city)}"
                    if city_id not in unique_locations:
                        unique_locations[city_id] = {
                            "location_id": city_id,
                            "name": city,
                            "level": 3,
                            "country": country,
                            "province": province,
                            "city": city,
                            "district": "",
                            "full_path": f"{country}/{province}/{city}",
                        }

                    # 区县级节点
                    if district:
                        district_id = f"{city_id}-{self._get_pinyin_abbr(district)}"
                        if district_id not in unique_locations:
                            unique_locations[district_id] = {
                                "location_id": district_id,
                                "name": district,
                                "level": 4,
                                "country": country,
                                "province": province,
                                "city": city,
                                "district": district,
                                "full_path": f"{country}/{province}/{city}/{district}",
                            }

        # 生成Cypher语句
        for loc_id, loc in sorted(unique_locations.items()):
            stmt = (
                f"MERGE (loc:Location {{location_id: '{self._escape(loc['location_id'])}'}}) "
                f"SET loc.name = '{self._escape(loc['name'])}', "
                f"loc.level = {loc['level']}, "
                f"loc.country = '{self._escape(loc['country'])}', "
                f"loc.province = '{self._escape(loc['province'])}', "
                f"loc.city = '{self._escape(loc['city'])}', "
                f"loc.district = '{self._escape(loc['district'])}', "
                f"loc.full_path = '{self._escape(loc['full_path'])}';"
            )
            statements.append(stmt)

        # 生成Location层级关系（仅中国）
        statements.append("")
        statements.append("// Location层级关系")

        for loc_id, loc in unique_locations.items():
            if loc["level"] > 1 and loc["country"] == "中国":
                parts = loc_id.split("-")
                if len(parts) > 1:
                    parent_id = "-".join(parts[:-1])
                    if parent_id in unique_locations:
                        stmt = (
                            f"MATCH (child:Location {{location_id: '{self._escape(loc_id)}'}}) "
                            f"MATCH (parent:Location {{location_id: '{self._escape(parent_id)}'}}) "
                            f"MERGE (child)-[:PARENT_LOCATION]->(parent);"
                        )
                        statements.append(stmt)

        return self._write_file("10_locations.cypher", statements)

    def _gen_location_id(self, country: str) -> str:
        """生成国家级location_id"""
        country_codes = {
            "中国": "cn", "日本": "jp", "韩国": "kr", "美国": "us",
            "德国": "de", "法国": "fr", "英国": "uk", "加拿大": "ca",
            "澳大利亚": "au", "意大利": "it", "西班牙": "es", "荷兰": "nl",
            "瑞典": "se", "瑞士": "ch", "俄罗斯": "ru", "印度": "in",
            "新加坡": "sg",
        }
        return country_codes.get(country, country[:2].lower() if country else "xx")

    def _get_pinyin_abbr(self, text: str, length: int = 2) -> str:
        """获取拼音首字母缩写"""
        pinyin_map = {
            "北京": "bj", "上海": "sh", "天津": "tj", "重庆": "cq",
            "广东": "gd", "江苏": "js", "浙江": "zj", "山东": "sd",
            "河南": "hn", "四川": "sc", "湖北": "hb", "湖南": "hun",
            "河北": "heb", "福建": "fj", "安徽": "ah", "辽宁": "ln",
            "陕西": "sx", "江西": "jx", "广西": "gx", "云南": "yn",
            "山西": "shx", "贵州": "gz", "黑龙江": "hlj", "吉林": "jl",
            "甘肃": "gs", "内蒙古": "nmg", "新疆": "xj", "海南": "hain",
            "宁夏": "nx", "青海": "qh", "西藏": "xz",
            "广州": "gz", "深圳": "sz", "杭州": "hz", "南京": "nj",
            "武汉": "wh", "成都": "cd", "西安": "xa", "苏州": "suz",
            "中国香港": "hk", "中国澳门": "mo", "中国台湾": "tw",
        }

        # 去除后缀
        clean_text = re.sub(r'[省市区县]$', '', text)

        if clean_text in pinyin_map:
            return pinyin_map[clean_text]

        return clean_text[:length].lower()

    def _generate_org_location_relationships(self, orgs_info: Dict[str, Dict]) -> Dict[str, int]:
        """生成Organization到Location的关系"""
        statements = ["// Organization-Location关系 (LOCATED_IN)"]

        # 过滤None键
        valid_items = [(k, v) for k, v in orgs_info.items() if k]
        for org_name, info in valid_items:
            loc_info = info.get("location")
            if not loc_info or not loc_info.get("country"):
                continue

            # 生成对应的location_id
            country = loc_info.get("country")
            province = loc_info.get("province")
            city = loc_info.get("city")
            district = loc_info.get("district")

            # 确定最详细的location_id
            location_id = self._gen_location_id(country)
            if province and country == "中国":
                location_id = f"{location_id}-{self._get_pinyin_abbr(province)}"
                if city:
                    location_id = f"{location_id}-{self._get_pinyin_abbr(city)}"
                    if district:
                        location_id = f"{location_id}-{self._get_pinyin_abbr(district)}"

            stmt = (
                f"MATCH (o:Organization {{name: '{self._escape(org_name)}'}}) "
                f"MATCH (loc:Location {{location_id: '{self._escape(location_id)}'}}) "
                f"MERGE (o)-[:LOCATED_IN]->(loc);"
            )
            statements.append(stmt)

        return self._write_file("11_org_location_rels.cypher", statements)

    def _generate_location_relationships(self, org_location_mapping: Dict[str, str]) -> Dict[str, int]:
        """
        生成Organization到Location的关系（旧版兼容）

        Args:
            org_location_mapping: {org_name: location_id}
        """
        statements = ["// Organization-Location关系 (LOCATED_IN)"]

        for org_name, location_id in org_location_mapping.items():
            if not org_name or not location_id:
                continue

            stmt = (
                f"MATCH (o:Organization {{name: '{self._escape(org_name)}'}}) "
                f"MATCH (loc:Location {{location_id: '{self._escape(location_id)}'}}) "
                f"MERGE (o)-[:LOCATED_IN]->(loc);"
            )
            statements.append(stmt)

        return self._write_file("11_org_location_rels.cypher", statements)

    def _generate_relationships(self, records: List[Dict],
                                org_mapping: Dict[str, str]) -> Dict[str, int]:
        """生成关系（旧版兼容）"""
        all_relationships = []

        for record in records:
            app_no = record.get("application_no")
            if not app_no:
                continue

            # APPLIED_BY
            for applicant in record.get("applicants", []):
                all_relationships.append(self._make_entity_rel(
                    app_no, applicant, "APPLIED_BY", org_mapping
                ))

            # OWNED_BY
            for owner in record.get("current_rights_holders", []):
                all_relationships.append(self._make_entity_rel(
                    app_no, owner, "OWNED_BY", org_mapping
                ))

            # BELONGS_TO (TechDomain)
            tech_domain = record.get("tech_domain")
            if tech_domain:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (t:TechDomain {{name: '{self._escape(tech_domain)}'}}) "
                    f"MERGE (p)-[:BELONGS_TO]->(t);"
                )

            # CLASSIFIED_AS (IPC)
            ipc_main = record.get("ipc_main")
            if ipc_main:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (i:IPCCode {{code: '{self._escape(ipc_main)}'}}) "
                    f"MERGE (p)-[:CLASSIFIED_AS]->(i);"
                )

            # PUBLISHED_IN (Country)
            country = record.get("publication_country")
            if country:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (c:Country {{name: '{self._escape(country)}'}}) "
                    f"MERGE (p)-[:PUBLISHED_IN]->(c);"
                )

            # HAS_STATUS
            legal_status = record.get("legal_status")
            if legal_status:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (l:LegalStatus {{name: '{self._escape(legal_status)}'}}) "
                    f"MERGE (p)-[:HAS_STATUS]->(l);"
                )

            # IN_FAMILY
            family = record.get("patent_family")
            if family:
                family_id = self._generate_family_id(family)
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (f:PatentFamily {{family_id: '{self._escape(family_id)}'}}) "
                    f"MERGE (p)-[:IN_FAMILY]->(f);"
                )

            # TRANSFERRED_TO
            for transferee in record.get("transferees", []):
                all_relationships.append(self._make_entity_rel(
                    app_no, transferee, "TRANSFERRED_TO", org_mapping
                ))

            # LICENSED_TO
            for licensee in record.get("licensees", []):
                all_relationships.append(self._make_entity_rel(
                    app_no, licensee, "LICENSED_TO", org_mapping
                ))

            # PLEDGED_TO
            for pledgee in record.get("pledgees", []):
                all_relationships.append(self._make_entity_rel(
                    app_no, pledgee, "PLEDGED_TO", org_mapping
                ))

            # LITIGATED_WITH (plaintiffs and defendants)
            for plaintiff in record.get("plaintiffs", []):
                rel = self._make_entity_rel(app_no, plaintiff, "LITIGATED_WITH", org_mapping)
                rel = rel.replace("MERGE (p)-[:LITIGATED_WITH]->",
                                  "MERGE (p)-[:LITIGATED_WITH {role: '原告'}]->")
                all_relationships.append(rel)

            for defendant in record.get("defendants", []):
                rel = self._make_entity_rel(app_no, defendant, "LITIGATED_WITH", org_mapping)
                rel = rel.replace("MERGE (p)-[:LITIGATED_WITH]->",
                                  "MERGE (p)-[:LITIGATED_WITH {role: '被告'}]->")
                all_relationships.append(rel)

        # 过滤空值
        all_relationships = [r for r in all_relationships if r]

        # 分批写入
        all_stats = {}
        batch_num = 1

        for i in range(0, len(all_relationships), self.batch_size):
            batch = all_relationships[i:i + self.batch_size]
            statements = [f"// 关系 批次 {batch_num}"] + batch

            filename = f"09_relationships_batch_{batch_num:03d}.cypher"
            stats = self._write_file(filename, statements)
            all_stats.update(stats)
            batch_num += 1

        return all_stats

    def _generate_relationships_unified(self, records: List[Dict],
                                        unified_mapping: Dict[str, Dict]) -> Dict[str, int]:
        """生成关系（新版统一处理）"""
        all_relationships = []

        # 定义需要处理的实体字段和关系类型
        entity_rel_fields = [
            ("applicants_resolved", "APPLIED_BY"),
            ("current_rights_holders_resolved", "OWNED_BY"),
            ("transferors_resolved", "TRANSFERRED_FROM"),       # 转让人
            ("transferees_resolved", "TRANSFERRED_TO"),         # 受让人
            ("licensors_resolved", "LICENSED_FROM"),            # 许可人
            ("licensees_resolved", "LICENSED_TO"),              # 被许可人
            ("current_licensees_resolved", "CURRENT_LICENSED_TO"),  # 当前被许可人
            ("pledgors_resolved", "PLEDGED_FROM"),              # 出质人
            ("pledgees_resolved", "PLEDGED_TO"),                # 质权人
            ("current_pledgees_resolved", "CURRENT_PLEDGED_TO"),   # 当前质权人
        ]

        for record in records:
            app_no = record.get("application_no")
            if not app_no:
                continue

            # 实体关系（使用resolved字段）
            for field, rel_type in entity_rel_fields:
                for entity_info in record.get(field, []):
                    rel = self._make_entity_rel_unified(app_no, entity_info, rel_type)
                    if rel:
                        all_relationships.append(rel)

            # 诉讼关系（带角色）
            for entity_info in record.get("plaintiffs_resolved", []):
                rel = self._make_entity_rel_unified(app_no, entity_info, "LITIGATED_WITH")
                if rel:
                    rel = rel.replace("MERGE (p)-[:LITIGATED_WITH]->",
                                     "MERGE (p)-[:LITIGATED_WITH {role: '原告'}]->")
                    all_relationships.append(rel)

            for entity_info in record.get("defendants_resolved", []):
                rel = self._make_entity_rel_unified(app_no, entity_info, "LITIGATED_WITH")
                if rel:
                    rel = rel.replace("MERGE (p)-[:LITIGATED_WITH]->",
                                     "MERGE (p)-[:LITIGATED_WITH {role: '被告'}]->")
                    all_relationships.append(rel)

            # 非实体关系（TechDomain, IPC, Country, LegalStatus, Family）
            # BELONGS_TO (TechDomain)
            tech_domain = record.get("tech_domain")
            if tech_domain:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (t:TechDomain {{name: '{self._escape(tech_domain)}'}}) "
                    f"MERGE (p)-[:BELONGS_TO]->(t);"
                )

            # CLASSIFIED_AS (IPC)
            ipc_main = record.get("ipc_main")
            if ipc_main:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (i:IPCCode {{code: '{self._escape(ipc_main)}'}}) "
                    f"MERGE (p)-[:CLASSIFIED_AS]->(i);"
                )

            # PUBLISHED_IN (Country)
            country = record.get("publication_country")
            if country:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (c:Country {{name: '{self._escape(country)}'}}) "
                    f"MERGE (p)-[:PUBLISHED_IN]->(c);"
                )

            # HAS_STATUS
            legal_status = record.get("legal_status")
            if legal_status:
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (l:LegalStatus {{name: '{self._escape(legal_status)}'}}) "
                    f"MERGE (p)-[:HAS_STATUS]->(l);"
                )

            # IN_FAMILY
            family = record.get("patent_family")
            if family:
                family_id = self._generate_family_id(family)
                all_relationships.append(
                    f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                    f"MATCH (f:PatentFamily {{family_id: '{self._escape(family_id)}'}}) "
                    f"MERGE (p)-[:IN_FAMILY]->(f);"
                )

            # HAS_LITIGATION_TYPE (诉讼类型)
            litigation_types = record.get("litigation_types", [])
            if isinstance(litigation_types, list):
                for lt in litigation_types:
                    if lt and lt.strip():
                        all_relationships.append(
                            f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                            f"MATCH (lt:LitigationType {{name: '{self._escape(lt.strip())}'}}) "
                            f"MERGE (p)-[:HAS_LITIGATION_TYPE]->(lt);"
                        )

        # 过滤空值
        all_relationships = [r for r in all_relationships if r]

        # 分批写入
        all_stats = {}
        batch_num = 1

        for i in range(0, len(all_relationships), self.batch_size):
            batch = all_relationships[i:i + self.batch_size]
            statements = [f"// 关系 批次 {batch_num}"] + batch

            filename = f"09_relationships_batch_{batch_num:03d}.cypher"
            stats = self._write_file(filename, statements)
            all_stats.update(stats)
            batch_num += 1

        return all_stats

    def _make_entity_rel_unified(self, app_no: str, entity_info: Dict,
                                 rel_type: str) -> Optional[str]:
        """生成实体关系语句（新版统一处理）"""
        if not entity_info:
            return None

        standard_name = entity_info.get("standard_name")
        entity_type = entity_info.get("entity_type", "organization")

        if not standard_name:
            return None

        if entity_type == "person":
            # 人物：使用uid
            original_name = entity_info.get("original_name", standard_name)
            uid = f"{original_name}@{standard_name}" if original_name != standard_name else standard_name
            return (
                f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                f"MATCH (e:Person {{uid: '{self._escape(uid)}'}}) "
                f"MERGE (p)-[:{rel_type}]->(e);"
            )
        else:
            # 机构
            return (
                f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                f"MATCH (e:Organization {{name: '{self._escape(standard_name)}'}}) "
                f"MERGE (p)-[:{rel_type}]->(e);"
            )

    def _generate_patents(self, records: List[Dict]) -> Dict[str, int]:
        """生成专利节点（分批）"""
        all_stats = {}
        batch_num = 1

        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            statements = [f"// 专利节点 批次 {batch_num}"]

            for record in batch:
                app_no = record.get("application_no")
                if not app_no:
                    continue

                # 构建属性
                props = {
                    "application_no": app_no,
                    "title_cn": record.get("title_cn", ""),
                    "title_en": record.get("title_en", ""),
                    "abstract_cn": record.get("abstract_cn", "")[:2000] if record.get("abstract_cn") else "",
                    "abstract_en": record.get("abstract_en", "")[:2000] if record.get("abstract_en") else "",
                    "application_date": record.get("application_date", ""),
                    "publication_date": record.get("publication_date", ""),
                    "publication_no": record.get("publication_no", ""),
                    "patent_type": record.get("patent_type", ""),
                    "ipc_main": record.get("ipc_main", ""),
                    "tech_domain": record.get("tech_domain", ""),
                    "transfer_count": self._safe_int(record.get("transfer_count", 0)),
                    "license_count": self._safe_int(record.get("license_count", 0)),
                    "pledge_count": self._safe_int(record.get("pledge_count", 0)),
                    "litigation_count": self._safe_int(record.get("litigation_count", 0)),
                }

                # 构建SET子句
                set_parts = []
                for k, v in props.items():
                    if isinstance(v, int):
                        set_parts.append(f"p.{k} = {v}")
                    else:
                        set_parts.append(f"p.{k} = '{self._escape(str(v))}'")

                set_clause = ", ".join(set_parts)
                stmt = f"MERGE (p:Patent {{application_no: '{self._escape(app_no)}'}}) SET {set_clause};"
                statements.append(stmt)

            filename = f"08_patents_batch_{batch_num:03d}.cypher"
            stats = self._write_file(filename, statements)
            all_stats.update(stats)
            batch_num += 1

        return all_stats

    def _make_entity_rel(self, app_no: str, entity_name: str,
                         rel_type: str, org_mapping: Dict[str, str]) -> Optional[str]:
        """生成实体关系语句（旧版兼容）"""
        if not entity_name:
            return None

        # 判断是机构还是人
        if "@" in entity_name:
            # 人物 (uid格式)
            return (
                f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                f"MATCH (e:Person {{uid: '{self._escape(entity_name)}'}}) "
                f"MERGE (p)-[:{rel_type}]->(e);"
            )
        else:
            # 机构
            return (
                f"MATCH (p:Patent {{application_no: '{self._escape(app_no)}'}}) "
                f"MATCH (e:Organization {{name: '{self._escape(entity_name)}'}}) "
                f"MERGE (p)-[:{rel_type}]->(e);"
            )

    def _generate_manifest(self, stats: Dict[str, int]):
        """生成导入清单文件"""
        # 收集Location相关文件
        location_files = sorted([f for f in stats.keys() if f.startswith("10_") or f.startswith("11_")])

        manifest = {
            "phases": [
                {
                    "name": "schema",
                    "files": ["00_schema.cypher"],
                    "parallel": False,
                },
                {
                    "name": "vector_index",
                    "files": ["00_vector_index.cypher"],
                    "parallel": False,
                },
                {
                    "name": "dimension_nodes",
                    "files": [
                        "01_tech_domains.cypher",
                        "02_ipc_codes.cypher",
                        "03_countries.cypher",
                        "04_legal_statuses.cypher",
                        "05_organizations.cypher",
                        "06_persons.cypher",
                        "07_patent_families.cypher",
                    ] + ([f for f in location_files if f.startswith("10_")] if location_files else []),
                    "parallel": True,
                },
                {
                    "name": "patents",
                    "files": sorted([f for f in stats.keys() if f.startswith("08_")]),
                    "parallel": True,
                },
                {
                    "name": "relationships",
                    "files": sorted([f for f in stats.keys() if f.startswith("09_")]) +
                             ([f for f in location_files if f.startswith("11_")] if location_files else []),
                    "parallel": True,
                },
            ],
            "file_stats": stats,
            "total_statements": sum(stats.values()),
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _write_file(self, filename: str, statements: List[str]) -> Dict[str, int]:
        """写入Cypher文件"""
        filepath = self.output_dir / filename

        # 过滤空语句和注释
        valid_statements = [s for s in statements if s.strip() and not s.strip().startswith("//")]

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(statements))

        logger.debug(f"写入 {filename}: {len(valid_statements)} 条语句")
        return {filename: len(valid_statements)}

    @staticmethod
    def _escape(s: str) -> str:
        """转义Cypher字符串"""
        if not s:
            return ""
        # 转义单引号和反斜杠
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", "")

    @staticmethod
    def _safe_int(val) -> int:
        """安全转换为整数"""
        try:
            return int(float(val)) if val else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _classify_org_type(name: str) -> str:
        """判断机构类型"""
        if not name:
            return "机构"

        univ_keywords = ["大学", "学院", "University", "College", "学校"]
        research_keywords = ["研究所", "研究院", "实验室", "Research", "Institute", "Laboratory"]

        for kw in univ_keywords:
            if kw.lower() in name.lower():
                return "高校"

        for kw in research_keywords:
            if kw.lower() in name.lower():
                return "研究机构"

        return "公司"

    def _collect_ipc_codes(self, records: List[Dict]) -> Set[str]:
        """收集所有IPC代码"""
        codes = set()
        for r in records:
            ipc = r.get("ipc_main")
            if ipc:
                codes.add(ipc.strip())
        return codes

    def _collect_countries(self, records: List[Dict]) -> Set[str]:
        """收集所有国家"""
        countries = set()
        for r in records:
            country = r.get("publication_country")
            if country:
                countries.add(country.strip())
        return countries

    def _collect_legal_statuses(self, records: List[Dict]) -> Set[str]:
        """收集所有法律状态"""
        statuses = set()
        for r in records:
            status = r.get("legal_status")
            if status:
                statuses.add(status.strip())
        return statuses

    def _collect_patent_families(self, records: List[Dict]) -> Dict[str, List[str]]:
        """收集专利族"""
        families = defaultdict(list)
        for r in records:
            family = r.get("patent_family")
            app_no = r.get("application_no")
            if family and app_no:
                family_id = self._generate_family_id(family)
                families[family_id].append(app_no)
        return dict(families)

    @staticmethod
    def _generate_family_id(family_str: str) -> str:
        """生成专利族ID"""
        if not family_str:
            return ""
        # 使用第一个成员作为ID
        members = family_str.split(";")
        if members:
            return members[0].strip()[:50]
        return family_str[:50]

    def _collect_litigation_types(self, records: List[Dict]) -> Set[str]:
        """收集所有诉讼类型"""
        litigation_types = set()
        for r in records:
            types = r.get("litigation_types", [])
            if isinstance(types, list):
                for t in types:
                    if t and t.strip():
                        litigation_types.add(t.strip())
            elif types and isinstance(types, str):
                litigation_types.add(types.strip())
        return litigation_types

    def _generate_litigation_types(self, litigation_types: Set[str]) -> Dict[str, int]:
        """生成诉讼类型节点"""
        statements = ["// 诉讼类型节点"]

        for lt in sorted(litigation_types):
            if not lt:
                continue
            stmt = f"MERGE (lt:LitigationType {{name: '{self._escape(lt)}'}});"
            statements.append(stmt)

        return self._write_file("04_litigation_types.cypher", statements)
