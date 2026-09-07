"""查询流程节点基类"""
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional

from knowledge.processor.query_process.config import QueryConfig, get_config
from knowledge.processor.query_process.exceptions import QueryProcessError
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import (
    add_running_task, add_done_task,
    get_task_status, get_running_task_list, get_done_task_list,
)

T = TypeVar("T")


class BaseNode(ABC):
    """查询节点基类：统一日志、任务追踪、SSE 进度推送"""

    name: str = "base_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口：记录节点运行/完成、流式时推送进度、answer_output 结束时推送 final"""
        is_stream = state.get("is_stream")
        task_id = state.get("task_id")
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

            result = self.process(state)

            if task_id:
                add_done_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)
                    # answer_output 完成即整轮结束：推送 FINAL，通知前端关闭 SSE
                    if self.name == "answer_output":
                        push_sse_event(task_id=task_id, event=SSEEvent.FINAL,
                                       data={"answer": state.get("answer")})
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise QueryProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        raise NotImplementedError

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

    def _push_progress(self, task_id):
        """推送整轮节点进度（运行中/已完成）给前端 SSE"""
        push_sse_event(task_id=task_id, event=SSEEvent.PROGRESS, data={
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
    )
