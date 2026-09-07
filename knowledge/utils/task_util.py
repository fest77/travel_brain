"""任务状态管理（内存容器）

一个上传/问答请求 = 一个任务（task_id），用于追踪节点执行进度。
"""
from collections import defaultdict
from typing import Dict, List

_tasks_running_list: Dict[str, List[str]] = defaultdict(list)
_tasks_done_list: Dict[str, List[str]] = defaultdict(list)
_tasks_duration: Dict[str, Dict[str, float]] = defaultdict(dict)
_tasks_result: Dict[str, Dict[str, str]] = defaultdict(dict)
_tasks_status: Dict[str, str] = {}

TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_NODE_NAME_TO_CN: Dict[str, str] = {
    # --- Import 流程节点 ---
    "upload_file": "上传文件",
    "entry_node": "读取文件",
    "document_split_node": "文档切分",
    "entity_recognition_node": "实体识别",
    "bge_embedding_node": "向量生成",
    "import_milvus_node": "导入向量数据库",
    "__end__": "处理完成",
    # --- Query 流程节点 ---
    "entity_confirm": "确认游玩主题",
    "answer_output": "生成答案",
    "rerank": "重排序",
    "rrf": "倒排融合",
    "web_search_mcp": "网络搜索",
    "search_embedding": "切片搜索",
    "search_embedding_hyde": "切片搜索(假设性文档)",
}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str) -> None:
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)


def add_done_task(task_id: str, node_name: str) -> None:
    if node_name in _tasks_running_list[task_id]:
        _tasks_running_list[task_id].remove(node_name)
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)


def get_running_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def get_done_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_task_status(task_id: str) -> str:
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str) -> None:
    _tasks_status[task_id] = status_name


def set_task_result(task_id: str, key: str, value) -> None:
    """存储任务结果字段（如 answer）"""
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default=""):
    """获取任务结果字段"""
    return _tasks_result.get(task_id, {}).get(key, default)


def add_node_duration(task_id: str, node_name: str, duration: float) -> None:
    cn_name = _to_cn(node_name)
    _tasks_duration[task_id][cn_name] = round(duration, 2)


def get_node_durations(task_id: str) -> Dict[str, float]:
    return dict(_tasks_duration.get(task_id, {}))


def get_task_info(task_id: str) -> Dict[str, any]:
    return {
        "status": get_task_status(task_id),
        "running_list": get_running_task_list(task_id),
        "done_list": get_done_task_list(task_id),
        "durations": get_node_durations(task_id),
    }
