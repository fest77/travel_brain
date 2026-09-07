"""导入流程节点基类"""
import datetime
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Optional

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")


class BaseNode(ABC):
    """导入流程节点基类：统一日志、任务追踪、异常包装"""

    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        task_id = state.get("task_id", "")
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                add_running_task(task_id, self.name)

            start = datetime.datetime.now()
            result = self.process(state)
            duration = (datetime.datetime.now() - start).total_seconds()

            self.logger.info(f"--- {self.name} 完成 ---")
            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, duration)
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(message=str(e), node_name=self.name, cause=e)

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑（子类必须实现）"""
        raise NotImplementedError

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    """配置导入流程日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
