"""查询流程节点

- entity_confirm.py    实体确认节点（目的地/意图提取 + 多轮指代消解）
- vector_search.py     向量检索节点
- hyde_search.py       HyDE 检索节点
- rrf.py               RRF 融合节点
- rerank.py            重排序节点
- answer_output.py     答案生成节点（本地资料不足时自动转 enable_search 实时联网）
- web_search_mcp.py    MCP 联网搜索节点（备用保留：需购买/开通 DashScope Web 搜索 MCP 后接入图中）

注：当前库外问题由 answer_output 用 DashScope enable_search 实时联网兜底；
    web_search_mcp 实现已保留备用，日后购买 MCP 后可接入 multi_search→join。
"""
