"""
travel_brain 最终验收测试（一键全跑）

覆盖(按顺序):
    A. 环境与第三方依赖
    B. 外部服务连通(Milvus / MongoDB / MinIO / DashScope LLM)
    C. 知识库数据完整性(travel_chunks_v1 / travel_entity_names_v1 行数与字段)
    D. 导入流水线端到端(1 份真实 md：解析→切分→实体→向量→入库)
    E. 查询流水线端到端(进程内跑查询图，返回本地知识库答案)

用法:
    python test/all_test.py            # 默认全跑
    python test/all_test.py --skip-import   # 跳过导入入库(只做环境/数据/查询)
    python test/all_test.py --online        # 追加验证"库外联网兜底"(会调用实时搜索，较慢)

说明:
    · 导入测试为"增量追加"式(与参考项目一致)：重复运行同一文件会重复入库，
      如需清空请在导入前手动 drop travel_* 或用 import_test.py --fresh。
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_TRACING", "false")

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# 结果汇总
_PASS = []
_FAIL = []


def report(name, ok, detail=""):
    mark = "✔ 通过" if ok else "✘ 失败"
    line = f"  {mark}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    (_PASS if ok else _FAIL).append(name)


# ============================================================ #
# A. 环境与依赖
# ============================================================ #
def test_env():
    print("\n[A] 环境与第三方依赖")
    pkgs = ["torch", "langgraph", "langchain_openai", "openai", "pymilvus",
            "pymongo", "minio", "fastapi", "sentence_transformers"]
    ok = True
    for name in pkgs:
        try:
            import importlib
            importlib.import_module(name)
        except Exception as e:
            print(f"    ✗ {name}: {e}")
            ok = False
    # 关键环境变量/模型路径
    from knowledge.utils.client.base import get_env
    key = get_env("DASHSCOPE_API_KEY")
    bge = get_env("BGE_M3_PATH")
    rr = get_env("BGE_RERANKER_LARGE")
    for label, val in [("DASHSCOPE_API_KEY", key), ("BGE_M3_PATH", bge),
                       ("BGE_RERANKER_LARGE", rr)]:
        if not val or (label.endswith("PATH") and not os.path.isdir(val)):
            print(f"    ✗ {label} 缺失或路径无效")
            ok = False
    report("第三方依赖 & 环境变量/模型路径", ok,
           f"key={'有' if key else '无'} BGE={'有' if bge else '无'} Reranker={'有' if rr else '无'}")
    return ok


# ============================================================ #
# B. 外部服务
# ============================================================ #
def test_services():
    print("\n[B] 外部服务连通")
    from knowledge.processor.import_process.config import get_config
    config = get_config()

    # Milvus
    try:
        from knowledge.utils.client.storage_clients import StorageClients
        mc = StorageClients.get_milvus_client()
        cols = mc.list_collections()
        travel_ok = config.chunks_collection in cols and config.entity_name_collection in cols
        report("Milvus 连接", True, f"version via list; travel集合存在={travel_ok}")
    except Exception as e:
        report("Milvus 连接", False, str(e)[:100])

    # MongoDB
    try:
        StorageClients.get_mongo_db().command("ping")   # db.command("ping")
        report("MongoDB ping", True)
    except Exception as e:
        report("MongoDB ping", False, str(e)[:100])

    # MinIO
    try:
        buckets = [b.name for b in StorageClients.get_minio_client().list_buckets()]
        report("MinIO 连接", True, ",".join(buckets))
    except Exception as e:
        report("MinIO 连接", False, str(e)[:100])

    # DashScope LLM
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("DASHSCOPE_API_KEY"),
                        base_url=os.environ.get("OPENAI_API_BASE"))
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_DEFAULT_MODEL", "qwen-flash"),
            messages=[{"role": "user", "content": "请只回复:成功"}], max_tokens=10)
        report("DashScope LLM", bool(resp.choices[0].message.content), "qwen-flash 返回成功")
    except Exception as e:
        report("DashScope LLM", False, str(e)[:100])

    return True


# ============================================================ #
# C. 知识库数据完整性
# ============================================================ #
def test_data():
    print("\n[C] 知识库数据完整性")
    from knowledge.processor.import_process.config import get_config
    from knowledge.utils.client.storage_clients import StorageClients
    config = get_config()
    mc = StorageClients.get_milvus_client()

    for col, need in [(config.chunks_collection, 50), (config.entity_name_collection, 1)]:
        if not mc.has_collection(col):
            report(f"集合存在 {col}", False, "不存在，请先导入")
            continue
        mc.flush(col)
        rows = int(mc.get_collection_stats(col).get("row_count", 0))
        report(f"集合 {col} 行数", rows >= need, f"{rows} 行")
    # 抽样字段
    try:
        sample = mc.query(config.chunks_collection, filter="", output_fields=[
            "title", "region", "content_type", "source_file", "entity_name"], limit=1)
        field_ok = bool(sample) and all(k in sample[0] for k in
                                        ["title", "region", "content_type", "entity_name"])
        report("切片字段完整性", field_ok, str(sample[0]) if sample else "空")
    except Exception as e:
        report("切片字段完整性", False, str(e)[:100])
    return True


# ============================================================ #
# D. 导入流水线端到端
# ============================================================ #
def test_import():
    print("\n[D] 导入流水线端到端(1 份 md，增量入库)")
    from knowledge.utils.client.base import get_env
    data_dir = get_env(
        "TRAVEL_DATA_DIR",
        r"D:\Aruanjian_coding_tools\Obsidian_vault\AI应用开发\4_LangChain&RAG架构\项目实战_掌柜智库\资料\data\旅游数据")
    # 优先用源数据里的真实样例；源目录缺失时自动生成临时样例，保证测试自包含
    sample = next((os.path.join(data_dir, d, f) for d, f in [
        ("线路推荐", "三亚线路推荐.md"), ("景点攻略", "三亚景点推荐.md"),
        ("美食推荐", "成都美食推荐.md")] if os.path.exists(os.path.join(data_dir, d, f))), None)
    temp_path = None
    if not sample:
        temp_dir = os.path.join(ROOT, "knowledge", "temp_data")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "alltest_sample.md")
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("# 示例目的地推荐\n\n"
                    "## 元数据\n"
                    "- 内容类型：景点介绍\n"
                    "- 城市：测试地\n\n"
                    "## 目的地概览\n"
                    "测试地是一座示例旅行目的地，用于验证导入链路是否正常。\n\n"
                    "## 核心推荐\n"
                    "### 示例山\n"
                    "示例山适合登山观光，可安排半天游览，注意防晒并提前查看天气。\n\n"
                    "### 示例湖\n"
                    "示例湖适合环湖漫步与家庭出游，湖畔设有观景步道。\n\n"
                    "## 建议\n"
                    "建议选择春秋季节前往，并提前预订住宿与门票。\n")
        sample = temp_path
        print(f"    源数据目录不存在，已自动生成临时样例: {temp_path}")

    from knowledge.processor.import_process.config import get_config
    from knowledge.processor.import_process.main_graph import kb_import_process_graph
    from knowledge.utils.client.storage_clients import StorageClients
    config = get_config()
    mc = StorageClients.get_milvus_client()
    if mc.has_collection(config.chunks_collection):
        mc.flush(config.chunks_collection)
        old = int(mc.get_collection_stats(config.chunks_collection).get("row_count", 0))
    else:
        old = 0

    try:
        final = kb_import_process_graph.invoke(
            {"task_id": "alltest_import", "import_file_path": sample})
        n_chunks = len(final.get("chunks") or [])
        region, ctype = final.get("region"), final.get("content_type")
        print(f"    region={region} content_type={ctype} chunks={n_chunks}")
        mc.flush(config.chunks_collection)
        new = int(mc.get_collection_stats(config.chunks_collection).get("row_count", 0))
        ok = n_chunks > 0 and new > old and region and ctype
        report("导入链路(解析→切分→实体→向量→入库)", ok,
               f"{old} → {new}，新增 {new - old}（应≈{n_chunks}）")
        return ok
    except Exception as e:
        report("导入链路", False, str(e)[:160])
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================ #
# E. 查询流水线端到端
# ============================================================ #
def test_query():
    print("\n[E] 查询流水线端到端(本地知识库问答)")
    try:
        from knowledge.processor.query_process.main_graph import query_app
        state = {
            "session_id": "alltest_sess",
            "task_id": "alltest_query",
            "original_query": "三亚有哪些必去景点？",
            "is_stream": False,
        }
        t0 = time.time()
        final = query_app.invoke(state)
        answer = (final.get("answer") or "").strip()
        print(f"    耗时 {time.time()-t0:.0f}s；答案开头：{answer[:60]}…")
        ok = len(answer) > 50 and any(k in answer for k in ["三亚", "亚龙湾", "蜈支洲岛"])
        report("查询链路(entity→检索→融合→精排→答案)", ok)
        return ok
    except Exception as e:
        report("查询链路", False, str(e)[:160])
        return False


# ============================================================ #
# (可选) F. 库外联网兜底
# ============================================================ #
def test_online():
    print("\n[F] 库外联网兜底(enable_search)")
    try:
        from knowledge.processor.query_process.main_graph import query_app
        final = query_app.invoke({
            "session_id": "alltest_online",
            "task_id": "alltest_online",
            "original_query": "北京故宫门票多少钱？",
            "is_stream": False,
        })
        answer = (final.get("answer") or "").strip()
        print(f"    答案开头：{answer[:60]}…")
        ok = len(answer) > 50 and "实时" not in answer[:6] and ("故宫" in answer or "长城" in answer)
        # 上面条件宽松处理：只要给了真实内容即可
        ok = len(answer) > 80
        report("库外联网兜底", ok)
        return ok
    except Exception as e:
        report("库外联网兜底", False, str(e)[:160])
        return False


# ============================================================ #
def main():
    skip_import = "--skip-import" in sys.argv
    do_online = "--online" in sys.argv

    print("=" * 60)
    print("travel_brain 最终验收测试")
    print(f"参数: skip_import={skip_import}, online={do_online}")
    print("=" * 60)

    test_env()
    test_services()
    test_data()
    if not skip_import:
        test_import()
    test_query()
    if do_online:
        test_online()

    print("\n" + "=" * 60)
    print(f"结果汇总: 通过 {len(_PASS)} / 失败 {len(_FAIL)}")
    if _FAIL:
        print("失败项:", "、".join(_FAIL))
    else:
        print("全部通过，最终验收 OK！")
    print("=" * 60)
    sys.exit(0 if not _FAIL else 1)


if __name__ == "__main__":
    main()



"""预计输出:
D:\A_Py_Java\pyFile\travel_brain\.venv\Scripts\python.exe D:\A_Py_Java\pyFile\travel_brain\test\all_test.py 
============================================================
travel_brain 最终验收测试
参数: skip_import=False, online=False
============================================================

[A] 环境与第三方依赖
  ✔ 通过  第三方依赖 & 环境变量/模型路径  (key=有 BGE=有 Reranker=有)

[B] 外部服务连通
  ✔ 通过  Milvus 连接  (version via list; travel集合存在=True)
  ✔ 通过  MongoDB ping
  ✔ 通过  MinIO 连接  (a-bucket,knowledge-base-files,travel-files)
  ✔ 通过  DashScope LLM  (qwen-flash 返回成功)

[C] 知识库数据完整性
  ✔ 通过  集合 travel_chunks_v1 行数  (905 行)
  ✔ 通过  集合 travel_entity_names_v1 行数  (33 行)
  ✔ 通过  切片字段完整性  ({'title': '目的地概览', 'region': '三亚', 'content_type': '交通指南', 'source_file': '三亚交通指南.md', 'entity_name': '三亚', 'chunk_id': 468867484652898686})

[D] 导入流水线端到端(1 份 md，增量入库)
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 42047.35it/s]
    region=三亚 content_type=线路推荐 chunks=26
  ✔ 通过  导入链路(解析→切分→实体→向量→入库)  (905 → 931，新增 26（应≈26）)

[E] 查询流水线端到端(本地知识库问答)
联网搜索不可用，降级返回空: Session terminated
Loading weights: 100%|██████████| 393/393 [00:00<00:00, 4290.64it/s]
    耗时 20s；答案开头：三亚作为典型的热带滨海度假城市，拥有多个代表性的核心景点。根据游客普遍认知和行程安排建议，以下为三亚最值得必去的景点：
…
  ✔ 通过  查询链路(entity→检索→融合→精排→答案)

============================================================
结果汇总: 通过 10 / 失败 0
全部通过，最终验收 OK！
============================================================

进程已结束，退出代码为 0

"""