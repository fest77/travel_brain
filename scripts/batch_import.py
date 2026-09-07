"""批量导入脚本：扫描旅游数据目录，把全部 md 依次跑导入图

用法:
    python scripts/batch_import.py                 # 默认先清空 travel_* 集合再全量导入
    python scripts/batch_import.py --no-fresh      # 增量导入(不清空，可能重复)
    python scripts/batch_import.py --dir <路径>    # 指定数据目录

流程(步骤注释):
    1. 定位数据目录，递归收集所有 .md
    2. (默认) drop travel_chunks_v1 / travel_entity_names_v1，保证干净入库
    3. 逐个文件运行 kb_import_process_graph，打印进度与结果
    4. 汇总统计并 flush
"""
import os
import sys
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_TRACING", "false")

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from knowledge.processor.import_process.base import setup_logging  # noqa: E402
from knowledge.processor.import_process.main_graph import kb_import_process_graph  # noqa: E402
from knowledge.processor.import_process.config import get_config  # noqa: E402
from knowledge.utils.client.storage_clients import StorageClients  # noqa: E402


def collect_md_files(data_dir: str):
    """递归收集 md 文件，返回 [(相对分类, 绝对路径)]"""
    files = []
    for root, _dirs, names in os.walk(data_dir):
        for n in sorted(names):
            if n.lower().endswith((".md", ".markdown")):
                rel = os.path.relpath(root, data_dir)  # 形如 景点攻略
                files.append((rel, os.path.join(root, n)))
    return files


def main():
    setup_logging()
    fresh = "--no-fresh" not in sys.argv
    # 手动解析 --dir
    data_dir = os.getenv("TRAVEL_DATA_DIR", "")
    if "--dir" in sys.argv:
        data_dir = sys.argv[sys.argv.index("--dir") + 1]

    if not data_dir or not os.path.isdir(data_dir):
        print(f"数据目录无效: {data_dir}（请在 .env 配置 TRAVEL_DATA_DIR 或传 --dir）")
        sys.exit(1)

    config = get_config()
    client = StorageClients.get_milvus_client()

    # 1. 收集文件
    md_files = collect_md_files(data_dir)
    print(f"发现 {len(md_files)} 个 md 文件: {data_dir}")

    # 2. 默认清空 travel 集合（干净入库）
    if fresh:
        for col in [config.chunks_collection, config.entity_name_collection]:
            if client.has_collection(col):
                client.drop_collection(col)
        print("已清空 travel_* 集合（fresh 模式）")

    # 3. 逐个导入
    ok, fail = 0, 0
    for idx, (category, path) in enumerate(md_files, 1):
        task_id = f"batch_{uuid.uuid4().hex[:6]}"
        print(f"\n[{idx}/{len(md_files)}] 导入: {category}/{os.path.basename(path)}")
        try:
            init_state = {"task_id": task_id, "import_file_path": path}
            final = kb_import_process_graph.invoke(init_state)
            n_chunks = len(final.get("chunks") or [])
            print(f"    ✔ {final.get('region')} / {final.get('content_type')} → {n_chunks} chunks")
            ok += 1
        except Exception as e:
            print(f"    ✘ 导入失败: {e}")
            fail += 1

    # 4. flush + 统计
    client.flush(config.chunks_collection)
    client.flush(config.entity_name_collection)
    c1 = client.get_collection_stats(config.chunks_collection).get("row_count")
    c2 = client.get_collection_stats(config.entity_name_collection).get("row_count")
    print("\n" + "=" * 50)
    print(f"批量导入完成: 成功 {ok} / 失败 {fail}")
    print(f"travel_chunks_v1      行数 = {c1}")
    print(f"travel_entity_names_v1 行数 = {c2}")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
