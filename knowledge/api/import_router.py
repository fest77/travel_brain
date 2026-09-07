"""导入流程 API（端口 8000）

端点:
    GET  /import            导入页(前端静态页)
    GET  /                  重定向到 /import
    POST /upload            上传 md 并异步启动导入流程
    GET  /status/{task_id}  查询任务状态
"""
import os
import sys

# 允许直接运行本文件(python import_router.py)启动服务时找到 knowledge 包
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app):

    @app.get("/")
    def index():
        return RedirectResponse(url="/import")

    @app.get("/import")
    def import_page():
        # 返回上传前端页面
        return FileResponse(os.path.join(get_front_page_dir(), "import.html"))

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            service: ImportFileService = Depends(get_import_file_service),
            file: UploadFile = File(...)):
        """上传 md：保存后异步启动导入图，立即返回 task_id 供前端轮询状态"""
        # 1. 保存文件(本地+尝试MinIO)
        task_id, file_dir, import_file_path = service.upload_file(file)
        # 2. 异步跑导入图(不阻塞请求)
        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)
        return UploadResponse(task_id=task_id, message="文件上传成功，导入流程已启动")

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """按 task_id 查询导入任务状态/节点进度"""
        return TaskStatusResponse(**get_task_info(task_id))


def create_app():
    app = FastAPI(title="旅游知识库 - 导入服务", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    # 挂载前端静态资源
    front_dir = get_front_page_dir()
    if os.path.exists(front_dir):
        app.mount("/front", StaticFiles(directory=front_dir), name="front")
    register_router(app)
    return app


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
