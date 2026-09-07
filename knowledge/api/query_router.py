"""查询流程 API（端口 8001）

端点:
    GET  /chat                  对话前端页
    GET  /                      重定向到 /chat
    POST /query                 提问（支持 is_stream: false/true）
    GET  /stream/{task_id}      SSE 流式输出
    GET  /history/{session_id}  历史记录
    DELETE /history/{session_id} 清空历史
"""
import asyncio
import os
import sys
import threading

# 允许直接运行本文件(python query_router.py)启动服务时找到 knowledge 包
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import QueryRequest, QueryResponse, StreamSubmitResponse
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


def register_router(app):

    @app.get("/")
    def index():
        return RedirectResponse(url="/chat")

    @app.get("/chat")
    def chat_page():
        # 返回对话前端页面
        return FileResponse(os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(
            request: QueryRequest,
            background_tasks: BackgroundTasks,
            service: QueryService = Depends(get_query_service)):
        """处理查询请求

        - 非流式: 在默认线程池执行查询图，同步等答案返回
        - 流式:   后台执行并推送 SSE，前端连 /stream/{task_id}
        """
        session_id = request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        is_stream = request.is_stream

        if is_stream:
            # 流式：先建 SSE 队列，再用"独立守护线程"跑同步查询图
            # 关键①：不能同步跑(会阻塞事件循环 → SSE 无法推送 → 变一次性输出)
            # 关键②：用 threading.Thread 而不用 loop.run_in_executor —— 后者在本环境调度不可靠
            create_sse_queue(task_id)
            threading.Thread(
                target=service.run_query_graph,
                args=(request.query, session_id, task_id, is_stream),
                daemon=True,
            ).start()
            return StreamSubmitResponse(message="查询已启动", session_id=session_id, task_id=task_id)

        # 非流式：丢进默认线程池执行，避免阻塞事件循环
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, service.run_query_graph,
                                   request.query, session_id, task_id, is_stream)
        answer = service.get_task_result(task_id)
        return QueryResponse(message="生成完成", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """SSE 流式输出端点"""
        return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(session_id: str, limit: int = 50,
                          service: QueryService = Depends(get_query_service)):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取历史失败: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(session_id: str,
                                 service: QueryService = Depends(get_query_service)):
        count = service.clear_history(session_id)
        return {"message": "历史已清空", "deleted_count": count}


def create_app():
    app = FastAPI(title="旅游知识库 - 查询服务", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )
    front_dir = get_front_page_dir()
    if os.path.exists(front_dir):
        app.mount("/front", StaticFiles(directory=front_dir), name="front")
    register_router(app)
    return app


if __name__ == "__main__":
    setup_logging()
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
