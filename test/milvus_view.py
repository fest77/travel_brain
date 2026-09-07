"""Milvus 数据查看工具（只读）

作用：不依赖 Attu，直接用 pymilvus 查看 travel_* 集合里的数据。

用法:
    python test/milvus_view.py list                                   # 列出所有集合 + 行数
    python test/milvus_view.py show <集合名> [--limit N]              # 预览某集合前 N 条(默认5)
    python test/milvus_view.py show <集合名> --filter 'region=="三亚"' --limit 3   # 按条件过滤
    python test/milvus_view.py schema <集合名>                        # 查看字段结构
    python test/milvus_view.py all                                    # 两个 travel 集合各看 3 条
"""
import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from knowledge.processor.import_process.config import get_config  # noqa: E402
from knowledge.utils.client.storage_clients import StorageClients  # noqa: E402

# 展示时忽略的大字段（向量列）
VECTOR_FIELDS = {"dense_vector", "sparse_vector", "vector"}


def _trunc(v, n=150):
    s = str(v)
    return s if len(s) <= n else s[:n] + f"...<len={len(s)}>"


def list_collections():
    client = StorageClients.get_milvus_client()
    for col in client.list_collections():
        client.flush(col)
        stats = client.get_collection_stats(col)
        print(f"  {col:28s} 行数={stats.get('row_count')}")


def show_schema(name):
    client = StorageClients.get_milvus_client()
    if not client.has_collection(name):
        print(f"集合不存在: {name}")
        return
    info = client.describe_collection(name)
    fields = info.get("fields", [])
    print(f"集合: {name}  字段:")
    for f in fields:
        if isinstance(f, dict):
            extra = f.get("params") or {}
            dim = extra.get("dim", "")
            print(f"  - {f.get('name')}  type={f.get('type')}" + (f"  dim={dim}" if dim else ""))


def show(name, limit=5, expr="", show_vector=False):
    client = StorageClients.get_milvus_client()
    if not client.has_collection(name):
        print(f"集合不存在: {name}")
        return
    client.flush(name)

    # 只取标量字段，向量列默认省略
    info = client.describe_collection(name)
    scalar_fields = [f["name"] for f in info.get("fields", [])
                     if isinstance(f, dict) and f.get("name") not in VECTOR_FIELDS]

    rows = client.query(name, filter=expr, output_fields=scalar_fields, limit=limit)
    print(f"[{name}] 过滤={expr or '(无)'} 返回 {len(rows)} 条")
    for i, row in enumerate(rows, 1):
        print(f"\n--- {i} ---")
        for k, v in row.items():
            if isinstance(v, list) or isinstance(v, dict):
                print(f"  {k}: <{type(v).__name__} len={len(v)}>" if not show_vector else _trunc(v, 60))
            else:
                print(f"  {k}: {_trunc(v, 200)}")


def main():
    parser = argparse.ArgumentParser(description="Milvus 只读查看")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")
    sub.add_parser("all")

    p_schema = sub.add_parser("schema")
    p_schema.add_argument("collection")

    p_show = sub.add_parser("show")
    p_show.add_argument("collection")
    p_show.add_argument("--limit", type=int, default=5)
    p_show.add_argument("--filter", default="")
    p_show.add_argument("--vector", action="store_true", help="显示向量列(很长)")

    args = parser.parse_args()

    if args.cmd == "list":
        list_collections()
    elif args.cmd == "schema":
        show_schema(args.collection)
    elif args.cmd == "show":
        show(args.collection, args.limit, args.filter, args.vector)
    elif args.cmd == "all":
        config = get_config()
        for col in [config.chunks_collection, config.entity_name_collection]:
            show(col, 3)
            print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
