"""导入相关提示词模板"""

# 内容类型标准值（规范后用于 schema/content_type 字段）
CONTENT_TYPE_MAP = {
    "景点": "景点介绍",
    "线路": "线路推荐",
    "酒店": "酒店信息",
    "住宿": "酒店信息",
    "美食": "美食推荐",
    "交通": "交通指南",
    "文化": "文化民俗",
    "民俗": "文化民俗",
}


def normalize_content_type(raw: str) -> str:
    """把元数据里五花八门的内容类型规整为 6 类标准值"""
    if not raw:
        return ""
    for key, val in CONTENT_TYPE_MAP.items():
        if key in raw:
            return val
    return raw


# 实体识别（备用：当元数据缺失/异常时由 LLM 提取文件级主体实体）
ENTITY_NAME_SYSTEM_PROMPT = """你是一个旅游信息抽取专家。你的任务是从一份旅游 Markdown 文档中，
提取该文档描述的核心实体信息，并以 JSON 输出：
{"region": "地区/城市", "entity_name": "核心主体名", "content_type": "景点介绍|线路推荐|酒店信息|美食推荐|交通指南|文化民俗"}
如果某个字段无法判断，输出空字符串。不要输出任何多余内容。"""

ENTITY_NAME_USER_PROMPT_TEMPLATE = """请分析以下旅游文档，提取核心实体信息：

【文件标题】
{file_title}

【文档内容前几段】
{context}

JSON 输出：
"""
