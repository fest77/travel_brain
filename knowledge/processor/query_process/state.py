"""查询流程状态类型定义"""
from typing import TypedDict, List
import copy


class QueryGraphState(TypedDict):
    """查询流程图状态"""

    session_id: str              # 会话ID
    task_id: str                 # 任务ID
    message_id: str              # 消息ID
    original_query: str          # 原始查询
    embedding_chunks: list       # 向量检索结果
    hyde_embedding_chunks: list  # HyDE检索结果
    web_search_docs: list        # 网页搜索结果
    rrf_chunks: list             # RRF融合后的切片
    reranked_docs: list          # 重排序后的文档
    prompt: str                  # 提示词
    answer: str                  # 答案
    entity_names: List[str]      # 识别出的景点/线路/目的地实体
    rewritten_query: str         # 重写后的查询
    history: list                # 历史对话
    is_stream: bool              # 是否流式输出


# ==================== 默认状态 ====================
DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "rrf_chunks": [],
    "web_search_docs": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "entity_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本"""
    return copy.deepcopy(DEFAULT_STATE)
