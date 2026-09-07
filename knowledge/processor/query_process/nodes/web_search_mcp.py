"""MCP 联网搜索节点（备用实现，图中默认未启用）

启用条件（满足后才真正生效）：
    1. pip install openai-agents
    2. 在 DashScope 控制台开通「Web 搜索 MCP」，并在 .env 配置
       MCP_DASHSCOPE_BASE_URL / DASHSCOPE_API_KEY

说明：
    · 当前查询图采用「双路检索 + answer_output 的 enable_search 实时联网兜底」，
      本节点为可选的显式 MCP 联网实现，保留代码供日后购买/开通后接入。
    · 接入方式：在 main_graph.create_query_graph() 中加入节点并连到 multi_search→join 即可。
    · 未满足上述条件时调用会异常，本实现已做 best-effort，失败返回空列表，不影响主流程。
"""
import asyncio
import json

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError


class WebSearchMcpNode(BaseNode):
    """联网补充检索：结果结构 {title, url, snippet, source:web}"""

    name = "web_search_mcp"

    def process(self, state):
        # Step1 校验输入
        rewritten_query = state.get("rewritten_query") or state.get("original_query")
        if not rewritten_query:
            raise StateFieldError(self.name, "rewritten_query", str)

        # Step2 尝试联网；任何失败均降级返回空列表（best-effort）
        docs = []
        try:
            docs = asyncio.run(self._web_search(rewritten_query))
        except Exception as e:
            self.logger.warning(f"联网搜索不可用，降级返回空: {e}")

        self.logger.info(f"联网搜索返回 {len(docs)} 条")
        return {"web_search_docs": docs}

    # ------------------------------------------------------------ #
    async def _web_search(self, query: str):
        """通过 DashScope WebSearch MCP 调用网络搜索"""
        try:
            from agents.mcp import MCPServerStreamableHttp
        except ImportError as e:
            raise RuntimeError("未安装 openai-agents，无法联网(MCP)搜索") from e

        url = self.config.mcp_dashscope_base_url
        key = self.config.dashscope_api_key
        if not url or not key:
            raise RuntimeError("缺少 MCP_DASHSCOPE_BASE_URL / DASHSCOPE_API_KEY")

        async with MCPServerStreamableHttp(
                name="旅游网络搜索",
                params={
                    "url": url,
                    "headers": {"Authorization": f"Bearer {key}"},
                    "timeout": 60,
                    "terminate_on_close": True,
                },
                max_retry_attempts=1,
                client_session_timeout_seconds=20,
                cache_tools_list=True,
        ) as client:
            result = await client.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 3},
            )
            if not result or not result.content or not result.content[0].text:
                return []
            text_json = json.loads(result.content[0].text)
            pages = text_json.get("pages") or []
            return [
                {"title": p.get("title", ""), "url": p.get("url", ""),
                 "snippet": p.get("snippet", ""), "source": "web"}
                for p in pages if p.get("url")
            ]
