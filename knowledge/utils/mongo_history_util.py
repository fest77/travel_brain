"""对话历史 MongoDB 存储"""
import logging
from datetime import datetime
from typing import List, Dict, Any

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection():
    """chat_message 集合（travel_kb 库内，与掌柜项目隔离）"""
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        entity_names: List[str] = None,
) -> str:
    """保存一条对话消息（user / assistant）"""
    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "entity_names": entity_names or [],
        "ts": datetime.now().timestamp(),
    }
    try:
        result = _get_collection().insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"保存对话消息失败: {e}")
        return ""


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """按时间倒序取最近 limit 条消息（返回给上游做指代消解/多轮上下文）"""
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", -1)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"读取历史消息失败: {e}")
        return []


def clear_history(session_id: str) -> int:
    """清空某会话历史，返回删除条数"""
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        return result.deleted_count
    except Exception as e:
        logger.error(f"清空历史失败: {e}")
        return 0
