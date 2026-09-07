"""导入流程主图（LangGraph）

流程结构:
    entry_node -> document_split_node -> entity_recognition_node
               -> bge_embedding_node -> import_milvus_node -> END
"""
import os
import sys

# 关闭 LangSmith 遥测（避免请求 api.smith.langchain.com 被墙）
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_TRACING", "false")

# 允许直接运行本文件（python main_graph.py）也能找到 knowledge 包
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entity_recognition import EntityRecognitionNode
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode


def create_import_graph():
    """创建并编译导入流程图"""
    nodes = {
        "entry_node": EntryNode(),
        "document_split_node": DocumentSplitNode(),
        "entity_recognition_node": EntityRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    graph = StateGraph(ImportGraphState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.set_entry_point("entry_node")
    graph.add_edge("entry_node", "document_split_node")
    graph.add_edge("document_split_node", "entity_recognition_node")
    graph.add_edge("entity_recognition_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")
    graph.add_edge("import_milvus_node", END)

    return graph.compile()


# 全局图实例
kb_import_process_graph = create_import_graph()


def run_import_graph(input_state: ImportGraphState) -> ImportGraphState:
    """运行导入图并打印每个执行节点（供测试/脚本直接调用）

    input_state 示例:
        {"import_file_path": r"D:\\...\\三亚景点推荐.md"}
    """
    final_state = None
    for event in kb_import_process_graph.stream(input_state):
        for node_name, process_state in event.items():
            print(f"运行节点: {node_name}")
            final_state = process_state

    return final_state or input_state


if __name__ == "__main__":
    # 直接运行本文件：仅展示导入流程图结构（不触发真实导入/模型加载）
    kb_import_process_graph.get_graph().print_ascii()


# 预计输出（直接运行本文件时的图结构）
r"""D:\A_Py_Java\pyFile\travel_brain\.venv\Scripts\python.exe D:\A_Py_Java\pyFile\travel_brain\knowledge\processor\import_process\main_graph.py
       +-----------+         
       | __start__ |         
       +-----------+         
              *              
              *              
              *              
      +------------+         
      | entry_node |         
      +------------+         
              *              
              *              
              *              
  +---------------------+    
  | document_split_node |    
  +---------------------+    
              *              
              *              
              *              
+-------------------------+  
| entity_recognition_node |  
+-------------------------+  
              *              
              *              
              *              
  +--------------------+     
  | bge_embedding_node |     
  +--------------------+     
              *              
              *              
              *              
  +--------------------+     
  | import_milvus_node |     
  +--------------------+     
              *              
              *              
              *              
        +---------+          
        | __end__ |          
        +---------+          

进程已结束，退出代码为 0

"""