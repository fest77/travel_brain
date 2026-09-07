"""查询相关 Schema 定义"""
from typing import Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="查询内容")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式返回")


class QueryResponse(BaseModel):
    """非流式响应"""
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    answer: str = Field("", description="生成的答案")


class StreamSubmitResponse(BaseModel):
    """流式提交响应"""
    message: str = Field(..., description="响应消息")
    session_id: str = Field(..., description="会话ID")
    task_id: str = Field(..., description="任务ID，前端用此 ID 建立 SSE 连接")


class HistoryItem(BaseModel):
    """历史记录条目"""
    id: str = Field("", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    entity_names: List[str] = Field(default_factory=list)
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    """历史记录响应"""
    session_id: str
    items: List[HistoryItem]
