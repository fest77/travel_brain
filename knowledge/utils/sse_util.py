"""SSE 流式输出工具（队列生产 → 生成器消费）"""
import asyncio
import json
import logging
import queue
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    PROGRESS = "progress"   # 节点进度
    DELTA = "delta"         # LLM 流式增量
    FINAL = "final"         # 整轮结束


# 每个 task_id -> queue.Queue（后端节点往里塞事件，SSE 生成器取走）
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    q = queue.Queue()
    _task_stream[task_id] = q
    return q


def remove_sse_queue(task_id: str):
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event(task_id: str, event: str, data: Dict[str, Any]):
    """节点侧：向任务队列写入一条 SSE 事件（无队列则忽略，用于非流式/离线测试）"""
    stream_queue = get_sse_queue(task_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})


async def sse_generator(task_id: str, request: Request) -> AsyncGenerator:
    """SSE 消费者：把队列事件逐条 yield 给前端（FastAPI StreamingResponse 用）"""
    sse_queue = _task_stream.get(task_id)
    if sse_queue is None:
        return
    loop = asyncio.get_event_loop()
    try:
        while True:
            # 客户端断开则提前结束
            if await request.is_disconnected():
                return
            try:
                # 阻塞最多 1 秒取一条；避免空转占满事件循环
                msg = await loop.run_in_executor(None, sse_queue.get, True, 1)
                yield _sse_pack(msg.get("event"), msg.get("data"))
            except queue.Empty:
                logging.info(f"[SSE] 队列为空 task={task_id} ...继续等待")
                continue
    except (ConnectionResetError, BrokenPipeError):
        return
    except asyncio.CancelledError:
        raise
    finally:
        remove_sse_queue(task_id)
