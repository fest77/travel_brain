"""导入流程状态类型定义"""
from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict, total=False):
    """导入流程图状态（total=False 表示所有字段可选）"""

    # ==================== 任务标识 ====================
    task_id: str                    # 任务 ID，用于任务追踪

    # ==================== 路径信息 ====================
    import_file_path: str           # 导入文件路径（原始输入）
    file_dir: str                   # 导入(出)文件目录

    # ==================== 文件信息 ====================
    file_title: str                 # 文件标题（不含扩展名）
    source_file: str                # 来源文件名
    content_type: str               # 内容类型：景点介绍/线路推荐/酒店信息/美食推荐/交通指南/文化民俗

    # ==================== 实体信息 ====================
    entity_name: str                # 文件级主体实体（地区优先）
    entity_names: List              # 识别出的景点/线路/目的地实体列表
    region: str                     # 地区/城市

    # ==================== 处理中间数据 ====================
    md_content: str                 # Markdown 文档内容
    chunks: List                    # 文档切片列表


# ==================== 默认状态模板 ====================
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "import_file_path": "",
    "file_dir": "",
    "file_title": "",
    "source_file": "",
    "content_type": "",
    "entity_name": "",
    "entity_names": [],
    "region": "",
    "md_content": "",
    "chunks": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态副本（避免全局污染）"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
