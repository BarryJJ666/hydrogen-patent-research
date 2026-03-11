# -*- coding: utf-8 -*-
"""
地点提取器
- 从机构名称中提取详细地点信息
- 中国：详细到省-市-区
- 外国：只到国家级别
- 特殊处理：中国香港、中国澳门、中国台湾
"""
import re
import json
import hashlib
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
from config.settings import LLM_CONFIG, CACHE_DIR
from config.prompts import LOCATION_EXTRACTION_PROMPT
from utils.llm_client import call_llm
from utils.logger import get_logger
from utils.cache import get_cache_manager

logger = get_logger(__name__)


class LocationExtractor:
    """
    地点信息提取器

    功能：
    1. 从机构名称中提取地点信息
    2. 使用LLM进行详细地点分析
    3. 中国地址：省-市-区/县
    4. 外国地址：只到国家级别
    5. 特殊处理：中国香港、中国澳门、中国台湾必须带"中国"前缀
    """

    # 中国省份和直辖市
    CHINA_PROVINCES = {
        # 直辖市
        "北京": ("北京市", True),
        "上海": ("上海市", True),
        "天津": ("天津市", True),
        "重庆": ("重庆市", True),
        # 省份
        "广东": ("广东省", False),
        "江苏": ("江苏省", False),
        "浙江": ("浙江省", False),
        "山东": ("山东省", False),
        "河南": ("河南省", False),
        "四川": ("四川省", False),
        "湖北": ("湖北省", False),
        "湖南": ("湖南省", False),
        "河北": ("河北省", False),
        "福建": ("福建省", False),
        "安徽": ("安徽省", False),
        "辽宁": ("辽宁省", False),
        "陕西": ("陕西省", False),
        "江西": ("江西省", False),
        "广西": ("广西壮族自治区", False),
        "云南": ("云南省", False),
        "山西": ("山西省", False),
        "贵州": ("贵州省", False),
        "黑龙江": ("黑龙江省", False),
        "吉林": ("吉林省", False),
        "甘肃": ("甘肃省", False),
        "内蒙古": ("内蒙古自治区", False),
        "新疆": ("新疆维吾尔自治区", False),
        "海南": ("海南省", False),
        "宁夏": ("宁夏回族自治区", False),
        "青海": ("青海省", False),
        "西藏": ("西藏自治区", False),
    }

    # 特殊地区（必须带"中国"前缀）
    SPECIAL_REGIONS = {
        "香港": "中国香港",
        "澳门": "中国澳门",
        "台湾": "中国台湾",
        "Hong Kong": "中国香港",
        "Macao": "中国澳门",
        "Macau": "中国澳门",
        "Taiwan": "中国台湾",
        "HK": "中国香港",
    }

    # 外国关键词模式
    FOREIGN_PATTERNS = {
        r"Japan|日本|株式会社|合同会社": "日本",
        r"Korea|韩国|주식회사": "韩国",
        r"USA|United States|美国|Inc\.|Corp\.": "美国",
        r"Germany|德国|GmbH|AG(?![a-z])": "德国",
        r"France|法国|S\.A\.|SARL": "法国",
        r"UK|United Kingdom|英国|Ltd\.(?!\s*公司)": "英国",
        r"Canada|加拿大": "加拿大",
        r"Australia|澳大利亚|澳洲": "澳大利亚",
        r"Italy|意大利": "意大利",
        r"Spain|西班牙": "西班牙",
        r"Netherlands|荷兰": "荷兰",
        r"Sweden|瑞典": "瑞典",
        r"Switzerland|瑞士": "瑞士",
        r"Russia|俄罗斯": "俄罗斯",
        r"India|印度": "印度",
        r"Singapore|新加坡": "新加坡",
    }

    def __init__(self):
        self.cache = get_cache_manager()
        self._llm_cache = {}
        self._cache_lock = Lock()
        self._load_cache()

    def _load_cache(self):
        """加载缓存"""
        cache_data = self.cache.load_json("location_extraction_cache.json", {})
        self._llm_cache = cache_data

    def _save_cache(self):
        """保存缓存"""
        self.cache.save_json("location_extraction_cache.json", self._llm_cache)

    def extract_locations_batch(self, org_names: List[str]) -> Dict[str, Dict]:
        """
        批量提取地点信息

        Args:
            org_names: 机构名称列表

        Returns:
            {org_name: {
                "country": "中国",
                "province": "广东省",      # 外国为None
                "city": "深圳市",           # 外国为None
                "district": "南山区",       # 外国为None
                "full_path": "中国/广东省/深圳市/南山区",
                "level": 4,
                "location_id": "cn-gd-sz-ns"
            }}
        """
        logger.info(f"开始提取 {len(org_names)} 个机构的地点信息...")

        results = {}

        # Step 1: 规则快速提取
        logger.info("Step 1: 规则快速提取...")
        rule_extracted = []
        need_llm = []

        for org_name in org_names:
            result = self._rule_based_extract(org_name)
            if result and result.get("confidence", 0) > 0.8:
                results[org_name] = result
                rule_extracted.append(org_name)
            else:
                need_llm.append(org_name)

        logger.info(f"  规则提取: {len(rule_extracted)} 个, 需要LLM: {len(need_llm)} 个")

        # Step 2: LLM批量提取
        if need_llm:
            logger.info("Step 2: LLM批量提取...")
            llm_results = self._llm_batch_extract(need_llm)
            results.update(llm_results)

        # Step 3: 后处理，生成location_id和full_path
        logger.info("Step 3: 后处理...")
        for org_name, loc_info in results.items():
            loc_info["full_path"] = self._build_full_path(loc_info)
            loc_info["level"] = self._determine_level(loc_info)
            loc_info["location_id"] = self._generate_location_id(loc_info)

        # 保存缓存
        self._save_cache()

        # 统计
        countries = defaultdict(int)
        for loc_info in results.values():
            country = loc_info.get("country", "未知")
            countries[country] += 1

        logger.info(f"地点提取完成: {len(results)} 个机构")
        logger.info(f"  国家分布: {dict(countries)}")

        return results

    def _rule_based_extract(self, org_name: str) -> Optional[Dict]:
        """规则快速提取地点"""
        result = {
            "country": None,
            "province": None,
            "city": None,
            "district": None,
            "confidence": 0.0
        }

        # 检查特殊地区（香港、澳门、台湾）
        for keyword, normalized in self.SPECIAL_REGIONS.items():
            if keyword in org_name:
                result["country"] = "中国"
                result["province"] = normalized
                result["confidence"] = 0.95
                return result

        # 检查中国省份
        for province_key, (province_name, is_municipality) in self.CHINA_PROVINCES.items():
            if province_key in org_name:
                result["country"] = "中国"
                result["province"] = province_name
                result["confidence"] = 0.85

                # 尝试提取城市
                if not is_municipality:
                    city_match = re.search(r'([\u4e00-\u9fa5]{2,4}市)', org_name)
                    if city_match:
                        city = city_match.group(1)
                        # 排除省名+市（如"广东市"这种不存在的）
                        if province_key not in city:
                            result["city"] = city
                            result["confidence"] = 0.90

                return result

        # 检查外国关键词
        for pattern, country in self.FOREIGN_PATTERNS.items():
            if re.search(pattern, org_name, re.IGNORECASE):
                result["country"] = country
                # 外国只到国家级别，province/city/district全部为None
                result["province"] = None
                result["city"] = None
                result["district"] = None
                result["confidence"] = 0.85
                return result

        return None

    def _llm_batch_extract(self, org_names: List[str]) -> Dict[str, Dict]:
        """使用LLM批量提取地点"""
        results = {}
        batch_size = 20

        # 先检查缓存
        uncached = []
        for org_name in org_names:
            cache_key = hashlib.md5(org_name.encode()).hexdigest()
            if cache_key in self._llm_cache:
                results[org_name] = self._llm_cache[cache_key]
            else:
                uncached.append(org_name)

        if not uncached:
            return results

        logger.info(f"  缓存命中: {len(org_names) - len(uncached)}, 需要LLM调用: {len(uncached)}")

        # 批量调用LLM
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i:i+batch_size]
            batch_results = self._llm_extract_batch(batch)
            results.update(batch_results)

            # 更新缓存
            for org_name, loc_info in batch_results.items():
                cache_key = hashlib.md5(org_name.encode()).hexdigest()
                with self._cache_lock:
                    self._llm_cache[cache_key] = loc_info

        return results

    def _llm_extract_batch(self, org_names: List[str]) -> Dict[str, Dict]:
        """单批次LLM提取"""
        prompt = LOCATION_EXTRACTION_PROMPT.format(
            organizations=json.dumps(org_names, ensure_ascii=False, indent=2)
        )

        response = call_llm(prompt, max_retries=3)
        if not response:
            # LLM失败，返回默认值
            return {name: self._default_location() for name in org_names}

        try:
            # 解析JSON响应
            cleaned = re.sub(r'```json?\s*', '', response)
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()
            parsed = json.loads(cleaned)

            results = {}
            for item in parsed.get("locations", []):
                org_name = item.get("organization")
                if org_name and org_name in org_names:
                    # 外国地址强制清空省市区
                    country = item.get("country")
                    if country and country != "中国":
                        results[org_name] = {
                            "country": country,
                            "province": None,
                            "city": None,
                            "district": None,
                            "confidence": item.get("confidence", 0.75)
                        }
                    else:
                        results[org_name] = {
                            "country": item.get("country"),
                            "province": item.get("province"),
                            "city": item.get("city"),
                            "district": item.get("district"),
                            "confidence": item.get("confidence", 0.75)
                        }

            # 确保所有机构都有结果
            for name in org_names:
                if name not in results:
                    results[name] = self._default_location()

            return results

        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"解析LLM响应失败: {e}")
            return {name: self._default_location() for name in org_names}

    def _default_location(self) -> Dict:
        """默认地点（无法识别时）"""
        return {
            "country": None,
            "province": None,
            "city": None,
            "district": None,
            "confidence": 0.0
        }

    def _build_full_path(self, loc_info: Dict) -> str:
        """构建地点完整路径"""
        parts = []

        if loc_info.get("country"):
            parts.append(loc_info["country"])
        if loc_info.get("province"):
            parts.append(loc_info["province"])
        if loc_info.get("city"):
            parts.append(loc_info["city"])
        if loc_info.get("district"):
            parts.append(loc_info["district"])

        return "/".join(parts) if parts else ""

    def _determine_level(self, loc_info: Dict) -> int:
        """确定地点级别"""
        if loc_info.get("district"):
            return 4  # 区县
        if loc_info.get("city"):
            return 3  # 市
        if loc_info.get("province"):
            return 2  # 省/直辖市
        if loc_info.get("country"):
            return 1  # 国家
        return 0  # 未知

    def _generate_location_id(self, loc_info: Dict) -> str:
        """生成地点ID"""
        parts = []

        # 国家简码
        country = loc_info.get("country", "")
        country_codes = {
            "中国": "cn", "日本": "jp", "韩国": "kr", "美国": "us",
            "德国": "de", "法国": "fr", "英国": "uk", "加拿大": "ca",
            "澳大利亚": "au", "意大利": "it", "西班牙": "es", "荷兰": "nl",
            "瑞典": "se", "瑞士": "ch", "俄罗斯": "ru", "印度": "in",
            "新加坡": "sg",
        }
        parts.append(country_codes.get(country, country[:2].lower() if country else "xx"))

        # 省份简码（仅中国）
        province = loc_info.get("province", "")
        if province:
            # 特殊地区
            if "香港" in province:
                parts.append("hk")
            elif "澳门" in province:
                parts.append("mo")
            elif "台湾" in province:
                parts.append("tw")
            else:
                # 取省份前两个汉字的拼音首字母
                parts.append(self._get_pinyin_abbr(province, 2))

        # 城市简码（仅中国）
        city = loc_info.get("city", "")
        if city:
            parts.append(self._get_pinyin_abbr(city, 2))

        # 区县简码（仅中国）
        district = loc_info.get("district", "")
        if district:
            parts.append(self._get_pinyin_abbr(district, 2))

        return "-".join(parts) if parts else "unknown"

    def _get_pinyin_abbr(self, text: str, length: int = 2) -> str:
        """获取拼音首字母缩写（简化版）"""
        # 简单的汉字到拼音首字母映射（常见地名）
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
        }

        # 去除后缀
        clean_text = re.sub(r'[省市区县]$', '', text)

        if clean_text in pinyin_map:
            return pinyin_map[clean_text]

        # 默认取前几个字符
        return clean_text[:length].lower()

    def build_location_nodes(self, location_data: Dict[str, Dict]) -> List[Dict]:
        """
        构建Location节点数据

        Returns:
            [{location_id, name, level, country, province, city, district, full_path}, ...]
        """
        unique_locations = {}

        for org_name, loc_info in location_data.items():
            if not loc_info.get("country"):
                continue

            # 按层级添加所有中间节点
            # 国家级
            if loc_info.get("country"):
                loc_id = self._generate_location_id({
                    "country": loc_info["country"]
                })
                if loc_id not in unique_locations:
                    unique_locations[loc_id] = {
                        "location_id": loc_id,
                        "name": loc_info["country"],
                        "level": 1,
                        "country": loc_info["country"],
                        "province": None,
                        "city": None,
                        "district": None,
                        "full_path": loc_info["country"],
                    }

            # 省级（仅中国）
            if loc_info.get("province"):
                loc_id = self._generate_location_id({
                    "country": loc_info["country"],
                    "province": loc_info["province"]
                })
                if loc_id not in unique_locations:
                    unique_locations[loc_id] = {
                        "location_id": loc_id,
                        "name": loc_info["province"],
                        "level": 2,
                        "country": loc_info["country"],
                        "province": loc_info["province"],
                        "city": None,
                        "district": None,
                        "full_path": f"{loc_info['country']}/{loc_info['province']}",
                    }

            # 市级（仅中国）
            if loc_info.get("city"):
                loc_id = self._generate_location_id({
                    "country": loc_info["country"],
                    "province": loc_info["province"],
                    "city": loc_info["city"]
                })
                if loc_id not in unique_locations:
                    unique_locations[loc_id] = {
                        "location_id": loc_id,
                        "name": loc_info["city"],
                        "level": 3,
                        "country": loc_info["country"],
                        "province": loc_info["province"],
                        "city": loc_info["city"],
                        "district": None,
                        "full_path": f"{loc_info['country']}/{loc_info['province']}/{loc_info['city']}",
                    }

            # 区县级（仅中国）
            if loc_info.get("district"):
                loc_id = self._generate_location_id(loc_info)
                if loc_id not in unique_locations:
                    unique_locations[loc_id] = {
                        "location_id": loc_id,
                        "name": loc_info["district"],
                        "level": 4,
                        "country": loc_info["country"],
                        "province": loc_info["province"],
                        "city": loc_info["city"],
                        "district": loc_info["district"],
                        "full_path": loc_info.get("full_path", ""),
                    }

        return list(unique_locations.values())

    def build_org_location_mapping(self, location_data: Dict[str, Dict]) -> Dict[str, str]:
        """
        构建机构到location_id的映射

        Returns:
            {org_name: location_id}
        """
        mapping = {}
        for org_name, loc_info in location_data.items():
            if loc_info.get("location_id"):
                mapping[org_name] = loc_info["location_id"]
        return mapping


def extract_locations(org_names: List[str]) -> Tuple[Dict[str, Dict], List[Dict], Dict[str, str]]:
    """
    快捷地点提取函数

    Returns:
        (location_data, location_nodes, org_location_mapping)
    """
    extractor = LocationExtractor()

    # 提取地点信息
    location_data = extractor.extract_locations_batch(org_names)

    # 构建Location节点
    location_nodes = extractor.build_location_nodes(location_data)

    # 构建机构-地点映射
    org_location_mapping = extractor.build_org_location_mapping(location_data)

    return location_data, location_nodes, org_location_mapping
