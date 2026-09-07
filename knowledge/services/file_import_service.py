"""导入文件业务层：上传(md)→本地保存(可选 MinIO 双写)→ 异步跑导入图 → 任务状态

流程(步骤注释):
    upload_file: 生成 task_id → 按日期建目录 → add_running(upload_file)
                 → 存本地(必需) → 尝试存 MinIO(失败仅告警，不影响) → add_done(upload_file)
    run_import_graph: 更新状态 processing → 流式跑导入图打日志 → completed / failed
"""
import datetime
import logging
import os
import uuid

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.main_graph import kb_import_process_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import (
    add_running_task, add_done_task, update_task_status,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)

logger = logging.getLogger(__name__)


class ImportFileService:
    """处理上传文件业务层类"""

    # ------------------------------------------------------------ #
    def upload_file(self, file: UploadFile):
        """上传 md：双写本地 + MinIO，返回 (task_id, file_dir, import_file_path)"""
        task_id = self._generate_task_id()
        date_path = os.path.join(get_local_base_dir(),
                                 datetime.datetime.now().strftime("%Y%m%d"))
        file_dir = os.path.join(date_path, task_id)
        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id, "upload_file")

        # 1. 保存本地（必需）
        import_file_path = self._save_local(file, file_dir)

        # 2. 保存 MinIO（可选，失败降级）
        self._try_upload_minio(import_file_path, file.filename)

        add_done_task(task_id, "upload_file")
        return task_id, file_dir, import_file_path

    # ------------------------------------------------------------ #
    def run_import_graph(self, task_id, file_dir, import_file_path):
        """异步执行导入图（供 FastAPI BackgroundTasks 调用）"""
        init_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,
        }
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        try:
            for event in kb_import_process_graph.stream(init_state):
                for node_name, _state in event.items():
                    logger.info(f"[{task_id}] 运行节点: {node_name}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"[{task_id}] 导入流程执行失败: {e}")

    # ------------------------------------------------------------ #
    # 私有工具
    # ------------------------------------------------------------ #
    @staticmethod
    def _generate_task_id() -> str:
        return uuid.uuid4().hex[:8]

    def _save_local(self, file: UploadFile, file_dir: str) -> str:
        import_file_path = os.path.join(file_dir, file.filename)
        with open(import_file_path, "wb") as f:
            # 分块拷贝，避免大文件一次性读入内存
            while chunk := file.file.read(1024 * 1024):
                f.write(chunk)
        return import_file_path

    def _try_upload_minio(self, import_file_path, file_name):
        """尝试上传 MinIO；客户端/上传失败仅告警，不影响主流程"""
        try:
            from knowledge.processor.import_process.config import get_config
            client = StorageClients.get_minio_client()
            cfg = get_config()
            obj_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}"
            client.fput_object(cfg.minio_bucket, obj_name, import_file_path)
            logger.info(f"已上传 MinIO: {obj_name}")
        except Exception as e:
            logger.warning(f"MinIO 上传跳过(降级): {e}")
