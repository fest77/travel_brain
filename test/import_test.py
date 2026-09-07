"""
导入阶段跑全程_测试代码（travel_brain）

运行前提:
1. Milvus(192.168.6.170:19530) 已启动
2. .env 中 DASHSCOPE_API_KEY / BGE_M3_PATH / BGE_RERANKER_LARGE 正确
3. 旅游数据 md 文件路径正确

作用:
    用一份真实旅游 md 跑通完整导入图（解析 → 切分 → 实体识别 → 向量化 → 写库），
    并验证最终向量是否成功入库。

用法:
    python test/import_test.py            # 默认样例(三亚线路推荐)，增量入库
    python test/import_test.py --fresh    # 先清空 travel_* 集合再导入（幂等演示）
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from knowledge.processor.import_process.base import setup_logging  # noqa: E402
from knowledge.processor.import_process.main_graph import run_import_graph  # noqa: E402
from knowledge.processor.import_process.config import get_config  # noqa: E402
from knowledge.utils.client.storage_clients import StorageClients  # noqa: E402

# 换成任意真实旅游 md（默认一份三亚线路推荐）
TRAVEL_DATA_DIR = os.getenv(
    "TRAVEL_DATA_DIR",
    r"D:\Aruanjian_coding_tools\Obsidian_vault\AI应用开发\4_LangChain&RAG架构\项目实战_掌柜智库\资料\data\旅游数据",
)
SAMPLE = os.path.join(TRAVEL_DATA_DIR, "线路推荐", "三亚线路推荐.md")


def _ensure_md(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"样例文件不存在: {path}")
    return path


def _count(client, collection: str) -> int:
    """返回集合当前真实行数（先 flush）"""
    client.flush(collection)
    stats = client.get_collection_stats(collection)
    return int(stats.get("row_count", 0))


if __name__ == "__main__":
    setup_logging()
    fresh = "--fresh" in sys.argv

    SAMPLE = _ensure_md(SAMPLE)
    config = get_config()
    client = StorageClients.get_milvus_client()

    chunks_col = config.chunks_collection
    entity_col = config.entity_name_collection

    if fresh:
        print(f"[--fresh] 清空 travel 集合: {chunks_col}, {entity_col}")
        for col in [chunks_col, entity_col]:
            if client.has_collection(col):
                client.drop_collection(col)

    # 0. 导入前行数
    old_chunks = _count(client, chunks_col) if client.has_collection(chunks_col) else 0
    print(f"导入前 travel_chunks_v1 行数: {old_chunks}")

    # 1. 跑完整导入图
    print("\n=== 运行导入图 ===")
    input_state = {"task_id": "import_test_task", "import_file_path": SAMPLE}
    final_state = run_import_graph(input_state)

    # 2. 逐项验证流程产物
    print("\n=== 逐项验证流程产物 ===")
    print("file_title   :", final_state.get("file_title"))
    print("content_type :", final_state.get("content_type"))
    print("region       :", final_state.get("region"))
    print("entity_name  :", final_state.get("entity_name"))
    chunks = final_state.get("chunks", [])
    print("chunks 数     :", len(chunks))
    if chunks:
        print("chunk 标题预览:", [c.get("title") for c in chunks[:6]], "...")
        # 检查关键字段是否齐全
        keys = ["content", "title", "parent_title", "content_type", "region",
                "source_file", "entity_name", "dense_vector", "sparse_vector"]
        missing = [k for k in keys if not chunks[0].get(k)]
        print("chunk 字段齐全 :", "是" if not missing else f"缺 {missing}")

    # 3. 去 Milvus 验证向量入库
    print("\n=== Milvus 入库校验 ===")
    new_chunks = _count(client, chunks_col)
    print(f"travel_chunks_v1  行数: 导入前={old_chunks}, 导入后={new_chunks}, 新增={new_chunks - old_chunks}")
    print(f"travel_entity_names_v1 行数: {_count(client, entity_col)}")

    # 4. 抽样验证一条切片字段
    if new_chunks > 0:
        sample = client.query(chunks_col, filter="", output_fields=[
            "title", "region", "content_type", "source_file", "entity_name",
        ], limit=2)
        print("样例记录:")
        for row in sample:
            print("  ", row)

    ok = new_chunks > old_chunks and len(chunks) > 0
    print("\n结论:", "导入流程验证通过 ✔（向量成功入库）" if ok else "导入未生效，请检查日志")
    sys.exit(0 if ok else 1)


"""预计输出:
D:\A_Py_Java\pyFile\travel_brain\.venv\Scripts\python.exe D:\A_Py_Java\pyFile\travel_brain\test\import_test.py 
2026-09-05 15:04:09 - knowledge.utils.client.storage_clients - INFO - Milvus 客户端初始化成功: http://192.168.6.170:19530
导入前 travel_chunks_v1 行数: 26

=== 运行导入图 ===
运行节点: entry_node
运行节点: document_split_node
运行节点: entity_recognition_node
2026-09-05 15:04:09 - import.entry_node - INFO - --- entry_node 开始 ---
2026-09-05 15:04:09 - import.entry_node - INFO - --- entry_node 完成 ---
2026-09-05 15:04:09 - import.document_split_node - INFO - --- document_split_node 开始 ---
2026-09-05 15:04:09 - import.document_split_node - INFO - 切分完成: 共 26 个 chunk
2026-09-05 15:04:09 - import.document_split_node - INFO - --- document_split_node 完成 ---
2026-09-05 15:04:09 - import.entity_recognition_node - INFO - --- entity_recognition_node 开始 ---
2026-09-05 15:04:09 - import.entity_recognition_node - INFO - 实体识别完成: region=三亚, content_type=线路推荐, chunks=26
2026-09-05 15:04:09 - import.entity_recognition_node - INFO - --- entity_recognition_node 完成 ---
2026-09-05 15:04:09 - import.bge_embedding_node - INFO - --- bge_embedding_node 开始 ---
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 34630.00it/s]
2026-09-05 15:04:11 - FlagEmbedding.finetune.embedder.encoder_only.m3.runner - INFO - loading existing colbert_linear and sparse_linear---------
2026-09-05 15:04:11 - knowledge.utils.client.ai_clients - INFO - BGE-M3 初始化成功 (device=cpu)
2026-09-05 15:04:14 - import.bge_embedding_node - INFO - 向量化进度 8/26
2026-09-05 15:04:15 - import.bge_embedding_node - INFO - 向量化进度 16/26
2026-09-05 15:04:17 - import.bge_embedding_node - INFO - 向量化进度 24/26
运行节点: bge_embedding_node
2026-09-05 15:04:17 - import.bge_embedding_node - INFO - 向量化进度 26/26
2026-09-05 15:04:17 - import.bge_embedding_node - INFO - --- bge_embedding_node 完成 ---
2026-09-05 15:04:17 - import.import_milvus_node - INFO - --- import_milvus_node 开始 ---
2026-09-05 15:04:17 - import.import_milvus_node - INFO - 写入切片集合 travel_chunks_v1: 26 条
2026-09-05 15:04:17 - import.import_milvus_node - INFO - 写入实体索引集合 travel_entity_names_v1: 1 条
2026-09-05 15:04:17 - import.import_milvus_node - INFO - --- import_milvus_node 完成 ---
运行节点: import_milvus_node

=== 逐项验证流程产物 ===
file_title   : 三亚线路推荐
content_type : 线路推荐
region       : 三亚
entity_name  : 三亚
chunks 数     : 26
chunk 标题预览: ['线路概览', '第一层核心体验', '第二层补充体验', '第 1 天：抵达三亚，先进入度假节奏', '第 2 天：亚龙湾和周边轻松活动', '第 3 天：蜈支洲岛一日游'] ...
chunk 字段齐全 : 是

=== Milvus 入库校验 ===
travel_chunks_v1  行数: 导入前=26, 导入后=52, 新增=26
travel_entity_names_v1 行数: 2
样例记录:
   {'chunk_id': 468867484652898628, 'title': '线路概览', 'region': '三亚', 'content_type': '线路推荐', 'source_file': '三亚线路推荐.md', 'entity_name': '三亚'}
   {'chunk_id': 468867484652898629, 'title': '第一层核心体验', 'region': '三亚', 'content_type': '线路推荐', 'source_file': '三亚线路推荐.md', 'entity_name': '三亚'}

结论: 导入流程验证通过 ✔（向量成功入库）

进程已结束，退出代码为 0

"""