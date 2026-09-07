"""Markdown 处理工具（纯文本：按标题切分、长切短合，无表格/图片/代码块逻辑）

旅游数据为纯文本 md。标题原文为 Markdown（## xxx），入库前转为纯文本小节名。
"""
import re
from typing import Dict, List

# 标题行，如 "### 亚龙湾"
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")


def parse_metadata_section(content: str) -> tuple:
    """解析文首 H1 标题 + '## 元数据' 列表段。

    Args:
        content: 完整 md 内容

    Returns:
        (title, metadata_dict, body_content)
        - title: H1 去掉 # 的文本（可能为空）
        - metadata_dict: 元数据键值（去掉 key 中的空格等），如 {"内容类型":"景点介绍","城市":"三亚"}
        - body_content: 去掉 H1、元数据段后的正文
    """
    lines = content.split("\n")
    title = ""
    body = []
    metadata = {}
    in_metadata = False
    metadata_lines = []
    h1_done = False

    for line in lines:
        m = HEADING_RE.match(line)

        # 1) H1 标题 → 文件标题
        if not h1_done and m and len(m.group(1)) == 1:
            title = m.group(2).strip()
            h1_done = True
            continue

        # 2) 进入 ## 元数据 段
        if not in_metadata and m and len(m.group(1)) == 2 and "元数据" in m.group(2):
            in_metadata = True
            continue

        # 3) 元数据段内：收集 "- key：value"，遇到下一个标题则结束该段
        if in_metadata:
            if m:                      # 下一个标题(如 ## 目的地概览) → 元数据段结束，标题归入正文
                in_metadata = False
                body.append(line)
                continue
            s = line.strip()
            if s.startswith("- "):
                metadata_lines.append(s)
            # 元数据段内的空行/散文本直接忽略
            continue

        # 4) 正文：非空行原样保留
        if line.strip():
            body.append(line)

    for item in metadata_lines:
        text = item[2:].strip()  # 去掉 "- "
        if "：" in text:
            k, v = text.split("：", 1)
        elif ":" in text:
            k, v = text.split(":", 1)
        else:
            continue
        metadata[k.strip()] = v.strip()

    return title, metadata, "\n".join(body)


def split_by_headings(body: str, file_title: str) -> List[Dict[str, str]]:
    """按 H2/H3/H4 标题将正文切成 sections。

    Returns:
        [{ "title": 小节纯文本名, "parent_title": 上级 H2 名, "body": 正文,
           "file_title": 文件名 }, ...]
    顶层（正文开头无标题的段落）归入 file_title 名下。
    """
    if not body or not body.strip():
        return []

    sections = []
    current_title = ""            # 当前小节标题（纯文本）
    current_level = 0             # 当前标题级别
    hierarchy = [""] * 7          # 各级标题文本缓存
    collected = []                # 当前小节正文行

    def flush():
        nonlocal collected
        if current_title or any(l.strip() for l in collected):
            # 找父级标题（上一级非空标题）
            parent_title = ""
            for lev in range(current_level - 1, 0, -1):
                if hierarchy[lev]:
                    parent_title = hierarchy[lev]
                    break
            if not parent_title:
                parent_title = file_title
            text = "\n".join(l.strip() for l in collected).strip()
            sections.append({
                "title": current_title or file_title,
                "parent_title": parent_title,
                "body": text,
                "file_title": file_title,
            })
        collected = []

    for line in body.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            name = m.group(2).strip()
            current_level = level
            current_title = name
            hierarchy[level] = name
            for i in range(level + 1, 7):
                hierarchy[i] = ""
        else:
            collected.append(line)

    flush()
    return sections


def split_long_section(section: Dict[str, str], max_len: int = 1000,
                       overlap_chars: int = 100) -> List[Dict[str, str]]:
    """长 section 按句子切分（含标题前缀与重叠）"""
    title = section["title"]
    parent_title = section["parent_title"]
    body = section["body"]
    file_title = section["file_title"]

    if len(body) <= max_len:
        return [section]

    # 句子切分符（保留标点）
    sentences = re.split(r"(?<=[。！？.!?；;])", body)
    sentences = [s.strip() for s in sentences if s.strip()]

    sub_sections = []
    buf = ""
    part = 0
    for sent in sentences:
        if buf and len(buf) + len(sent) > max_len:
            part += 1
            sub_sections.append({
                "parent_title": parent_title,
                "title": f"{title}-{part}",
                "body": buf,
                "file_title": file_title,
            })
            # 重叠尾部
            buf = buf[-overlap_chars:] if overlap_chars else ""
        buf += sent
    if buf:
        part += 1
        sub_sections.append({
            "parent_title": parent_title,
            "title": f"{title}-{part}" if part > 1 else title,
            "body": buf,
            "file_title": file_title,
        })
    return sub_sections


def merge_short_sections(sections: List[Dict[str, str]], min_len: int = 200) -> List[Dict[str, str]]:
    """过滤空/过短小节，保留各自标题。

    旅游数据每个 H2/H3 小节都是完整语义单元（景点介绍/日程/建议等），
    若把短节并入其它标题节会造成「正文与标题错位」，故不做跨主题合并，
    仅去除纯标题空节。
    """
    if not sections:
        return []
    # 仅保留有实际正文的小节（内容可短，但避免纯标题空块）
    return [sec for sec in sections if sec.get("body", "").strip()]
