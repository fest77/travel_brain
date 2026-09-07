"""入口节点：读取 md 文件并解析元数据"""
import os

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.utils.markdown_util import parse_metadata_section


class EntryNode(BaseNode):
    """读取 md 文件内容，解析 '## 元数据' 段，规整 content_type / region。

    旅游 md 结构（纯文本）:
        # 三亚景点推荐
        ## 元数据
        - 内容类型：景点介绍
        - 城市：三亚
        - 主题：...
        ## 目的地概览
        ...
    """

    name = "entry_node"

    def process(self, state):
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise ValidationError("import_file_path 为空", self.name)
        if not os.path.exists(import_file_path):
            raise ValidationError(f"文件不存在: {import_file_path}", self.name)

        # 1. 读取文件
        with open(import_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 2. 解析 H1 标题 + 元数据段 + 正文
        h1_title, metadata, body_content = parse_metadata_section(content)

        # 3. 文件标题（H1 优先，缺失用文件名 stem）
        filename = os.path.basename(import_file_path)
        file_title = h1_title or os.path.splitext(filename)[0]

        # 4. 元数据字段（值可能形如 "景点介绍"；key 可能带空格）
        content_type_raw = ""
        region_raw = ""
        for key, val in metadata.items():
            k = key.strip()
            if "内容类型" in k:
                content_type_raw = val
            elif "城市" in k or "目的地" in k:
                region_raw = val

        state["import_file_path"] = import_file_path
        state["file_dir"] = state.get("file_dir") or os.path.dirname(import_file_path)
        state["file_title"] = file_title
        state["source_file"] = filename
        state["content_type"] = content_type_raw   # 原始值，entity 节点规整
        state["region"] = region_raw
        state["md_content"] = body_content or content

        return state
