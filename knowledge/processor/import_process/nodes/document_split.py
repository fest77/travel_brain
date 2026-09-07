"""文档切分节点：按标题切分、长切短合，生成 chunks"""
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.utils.markdown_util import (
    split_by_headings,
    split_long_section,
    merge_short_sections,
)


class DocumentSplitNode(BaseNode):
    """把正文 md 按 H2/H3 切分成 chunk，处理超长/超短片段"""

    name = "document_split_node"

    def process(self, state):
        # 1. 校验输入
        md_content = state.get("md_content")
        file_title = state.get("file_title")
        if not md_content:
            raise StateFieldError(self.name, "md_content", str)
        if not file_title:
            raise StateFieldError(self.name, "file_title", str)

        max_len = self.config.max_content_length     # 单块内容上限
        min_len = self.config.min_content_length     # 合并阈值

        # 2. 按标题切分
        sections = split_by_headings(md_content, file_title)
        if not sections:
            raise ValidationError("未切分到任何内容", self.name)

        # 3. 长切短合
        final_sections = []
        for sec in sections:
            final_sections.extend(split_long_section(sec, max_len))
        final_sections = merge_short_sections(final_sections, min_len)

        # 4. 组装 chunk（content = title + 正文）
        chunks = []
        for sec in final_sections:
            body = sec["body"]
            if not body:
                continue
            chunks.append({
                "title": sec["title"],
                "parent_title": sec["parent_title"],
                "file_title": sec["file_title"],
                "content": f"{sec['title']}\n\n{body}",
            })

        self.logger.info(f"切分完成: 共 {len(chunks)} 个 chunk")
        state["chunks"] = chunks
        return state
