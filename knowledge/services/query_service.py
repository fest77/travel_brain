"""查询业务层：启动查询图、会话/任务/结果/历史"""
import uuid
from typing import List, Dict, Any

from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history as mongo_clear
from knowledge.utils.task_util import (
    update_task_status, set_task_result,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
    get_task_result,
)


class QueryService:
    """查询流程业务层类"""

    # ------------------------------------------------------------ #
    def run_query_graph(self, original_query, session_id, task_id, is_stream=False):
        """执行查询图（同步，供调用方在后台线程/执行器里跑）"""
        try:
            init_state = {
                "original_query": original_query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream,
            }
            update_task_status(task_id, TASK_STATUS_PROCESSING)
            query_app.invoke(init_state)
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            # 失败也把错误信息塞回 result，便于前端展示
            set_task_result(task_id, "answer", f"查询流程出错：{e}")

    # ------------------------------------------------------------ #
    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def get_task_result(task_id: str):
        return get_task_result(task_id, "answer")

    # ------------------------------------------------------------ #
    # 历史
    # ------------------------------------------------------------ #
    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        records = get_recent_messages(session_id, limit=limit)
        records.reverse()  # 升序返回便于前端展示
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "entity_names": r.get("entity_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return mongo_clear(session_id)
