"""查询流程主图（LangGraph）

流程结构:
    entity_confirm ─(无预置答案)→ multi_search ─┬→ search_embedding(向量)
                                                ├→ search_embedding_hyde(HyDE)
                                                └→ web_search_mcp(联网,可选)
    → join → rrf(融合) → rerank(精排) → answer_output(答案+引用+历史) → END

联网策略说明(两层)：
    1) web_search_mcp：显式 MCP 联网节点(图中可见)。best-effort——
       需安装 openai-agents 并开通 DashScope Web 搜索 MCP 才真正返回结果；
       未满足时返回空列表，不影响主流程。日后购买/开通后自动生效。
    2) answer_output 兜底：本地检索最高分低于阈值(库外目的地)时，
       自动改用 DashScope enable_search 实时联网回答(零额外依赖)。
"""
import os
import sys

# 关闭 LangSmith 遥测（避免请求 api.smith.langchain.com 被墙）
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_TRACING", "false")

# 允许直接运行本文件(python main_graph.py)打印图结构
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langgraph.graph import StateGraph, END  # noqa: E402

from knowledge.processor.query_process.state import QueryGraphState  # noqa: E402
from knowledge.processor.query_process.nodes.entity_confirm import EntityConfirmNode  # noqa: E402
from knowledge.processor.query_process.nodes.vector_search import VectorSearchNode  # noqa: E402
from knowledge.processor.query_process.nodes.hyde_search import HyDeSearchNode  # noqa: E402
from knowledge.processor.query_process.nodes.web_search_mcp import WebSearchMcpNode  # noqa: E402
from knowledge.processor.query_process.nodes.rrf import RrfNode  # noqa: E402
from knowledge.processor.query_process.nodes.rerank import RerankNode  # noqa: E402
from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode  # noqa: E402


def route_after_entity_confirm(state: QueryGraphState) -> bool:
    """实体确认后路由：若已预置 answer（需澄清），跳过检索直接生成；否则走多路检索"""
    return bool(state.get("answer"))


def create_query_graph():
    nodes = {
        "entity_confirm": EntityConfirmNode(),
        "multi_search": lambda x: x,   # 虚拟分发节点
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp": WebSearchMcpNode(),  # 联网检索(可选，best-effort)
        "join": lambda x: {},          # 虚拟汇合节点
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode(),
    }
    graph = StateGraph(QueryGraphState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.set_entry_point("entity_confirm")
    # 实体确认 → 有 answer 则直达答案；否则并行三路检索(向量 + HyDE + MCP联网)
    graph.add_conditional_edges(
        "entity_confirm",
        route_after_entity_confirm,
        {False: "multi_search", True: "answer_output"},
    )
    graph.add_edge("multi_search", "search_embedding")
    graph.add_edge("multi_search", "search_embedding_hyde")
    graph.add_edge("multi_search", "web_search_mcp")
    graph.add_edge("search_embedding", "join")
    graph.add_edge("search_embedding_hyde", "join")
    graph.add_edge("web_search_mcp", "join")
    graph.add_edge("join", "rrf")
    graph.add_edge("rrf", "rerank")
    graph.add_edge("rerank", "answer_output")
    graph.add_edge("answer_output", END)

    return graph.compile()


# 全局查询图实例
query_app = create_query_graph()


if __name__ == "__main__":
    # 直接运行本文件：打印查询流程图结构
    query_app.get_graph().print_ascii()


# 预计输出见下(仅记录)：直接运行本文件打印查询图 ASCII 结构
r"""D:\A_Py_Java\pyFile\travel_brain\.venv\Scripts\python.exe D:\A_Py_Java\pyFile\travel_brain\knowledge\processor\query_process\main_graph.py
                                            +-----------+                                    
                                            | __start__ |                                    
                                            +-----------+                                    
                                                   *                                         
                                                   *                                         
                                                   *                                         
                                          +----------------+                                 
                                          | entity_confirm |..                               
                                          +----------------+  .......                        
                                            ...                      .......                 
                                           .                                ......           
                                         ..                                       .......    
                               +--------------+                                          ....
                               | multi_search |                                             .
                           ****+--------------+*****                                        .
                       ****            *            ****                                    .
                  *****                *                *****                               .
               ***                     *                     ***                            .
+------------------+       +-----------------------+       +----------------+               .
| search_embedding |       | search_embedding_hyde |       | web_search_mcp |               .
+------------------+***    +-----------------------+    ***+----------------+               .
                       *****           *           *****                                    .
                            *****      *      *****                                         .
                                 ***   *   ***                                              .
                                   +------+                                                 .
                                   | join |                                                 .
                                   +------+                                                 .
                                       *                                                    .
                                       *                                                    .
                                       *                                                    .
                                    +-----+                                                 .
                                    | rrf |                                                 .
                                    +-----+                                                 .
                                       *                                                    .
                                       *                                                    .
                                       *                                                    .
                                  +--------+                                             ....
                                  | rerank |                                      .......    
                                  +--------+                                ......           
                                            ***                      .......                 
                                               *              .......                        
                                                **        ....                               
                                          +---------------+                                  
                                          | answer_output |                                  
                                          +---------------+                                  
                                                   *                                         
                                                   *                                         
                                                   *                                         
                                              +---------+                                    
                                              | __end__ |                                    
                                              +---------+                                    

进程已结束，退出代码为 0

"""