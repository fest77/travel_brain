"""实体识别节点：规整 content_type/region，确定文件级主体实体并回填到 chunks"""
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.prompt.import_prompt import normalize_content_type


class EntityRecognitionNode(BaseNode):
    """基于元数据确定该文档的 地区/内容类型/主体实体。

    说明：旅游 md 的 '## 元数据' 已显式给出 内容类型/城市 等，用规则规整即可，
    无需对每份文档调用 LLM（稳健且省成本）。每个 chunk 会回填以下标量：
        content_type / region / source_file / entity_name
    entity_name 取地区名（如“三亚”），用于查询侧按地区/类型过滤。
    """

    name = "entity_recognition_node"

    def process(self, state):
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(self.name, "chunks", list)

        file_title = state.get("file_title", "")
        source_file = state.get("source_file", "")

        # 1. 规整内容类型
        content_type = normalize_content_type(state.get("content_type", ""))
        if not content_type:
            # 兜底：从文件名推断（如 "三亚景点推荐" / 目录类）
            content_type = self._infer_content_type(file_title)
        state["content_type"] = content_type

        # 2. 地区
        region = (state.get("region") or "").strip()
        if not region:
            region = self._infer_region(file_title)
        state["region"] = region

        # 3. 主体实体 = 地区（无地区则退化为文件标题）
        entity_name = region or file_title
        state["entity_names"] = [entity_name] if entity_name else []
        state["entity_name"] = entity_name

        # 4. 回填每个 chunk
        for chunk in chunks:
            chunk["content_type"] = content_type
            chunk["region"] = region
            chunk["source_file"] = source_file
            chunk["entity_name"] = entity_name
            chunk["file_title"] = chunk.get("file_title") or file_title

        self.logger.info(
            f"实体识别完成: region={region}, content_type={content_type}, "
            f"chunks={len(chunks)}"
        )
        return state

    @staticmethod
    def _infer_content_type(text: str) -> str:
        from knowledge.prompt.import_prompt import CONTENT_TYPE_MAP
        for key, val in CONTENT_TYPE_MAP.items():
            if key in text:
                return val
        return ""

    @staticmethod
    def _infer_region(text: str) -> str:
        # 文件名前缀常为地区，如“三亚景点推荐”
        for city in ["三亚", "云南", "厦门", "张家界", "成都", "杭州"]:
            if text.startswith(city):
                return city
        return ""
