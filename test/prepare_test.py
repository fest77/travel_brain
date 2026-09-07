#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
travel_brain 旅游知识库 - 环境准备 / 连通性测试

用法:
    python test/prepare_test.py              # 快速环境检查（依赖 / 服务 / LLM / 数据目录）
    python test/prepare_test.py model        # 本地模型加载测试（BGE-M3 嵌入 + BGE-Reranker 精排）
    python test/prepare_test.py all          # 上述全部

覆盖项:
    1. 环境变量与模型路径存在性
    2. 关键第三方依赖及版本
    3. Milvus / MongoDB / MinIO 服务连通
    4. DashScope LLM(qwen-flash) 调用
    5. 旅游数据目录可读、md 文件数量
    6. [model] 本地 BGE-M3 / BGE-Reranker 模型加载
"""
import os
import sys
from dotenv import load_dotenv

# Windows 控制台/管道下中文输出编码兼容（Python 3.7+）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 加载 .env；已存在的系统环境变量（如 DASHSCOPE_API_KEY）不会被覆盖
load_dotenv()


def _read_registry_env(name):
    """从 Windows 系统/用户环境变量（注册表）读取，作为进程环境变量的回退。

    背景：Windows 环境变量在进程启动时快照。若 PyCharm/终端在配置系统环境变量
    (setx / 系统设置)之前已启动，其子进程读不到新变量。这里直接读注册表，
    保证旧进程也能拿到系统级配置。
    """
    if os.name != "nt":
        return None
    try:
        import winreg
    except Exception:
        return None
    for hive, key_path in [
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    ]:
        try:
            with winreg.OpenKey(hive, key_path) as key:
                val, _ = winreg.QueryValueEx(key, name)
                if val:
                    return val
        except OSError:
            continue
    return None


# 进程环境变量缺失时，回退到注册表中的系统/用户环境变量
if not os.getenv("DASHSCOPE_API_KEY"):
    _reg_key = _read_registry_env("DASHSCOPE_API_KEY")
    if _reg_key:
        os.environ["DASHSCOPE_API_KEY"] = _reg_key

TRAVEL_DATA_DIR = os.getenv(
    "TRAVEL_DATA_DIR",
    r"D:\Aruanjian_coding_tools\Obsidian_vault\AI应用开发\4_LangChain&RAG架构\项目实战_掌柜智库\资料\data\旅游数据",
)


# ============================================================ #
#                         快速环境检查                          #
# ============================================================ #

def test_env():
    """检查环境变量与本地模型路径"""
    print("检查环境变量与本地模型路径...")
    ok = True

    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        print(f"  ✓ DASHSCOPE_API_KEY 已配置 (len={len(key)})")
    else:
        print("  ✗ DASHSCOPE_API_KEY 未配置")
        ok = False

    for var in ["BGE_M3_PATH", "BGE_RERANKER_LARGE"]:
        path = os.getenv(var)
        if path and os.path.isdir(path):
            print(f"  ✓ {var} 存在\n      {path}")
        else:
            print(f"  ✗ {var} 路径无效或不存在: {path}")
            ok = False
    return ok


def test_versions():
    """检查关键第三方依赖是否可导入及版本"""
    print("检查关键第三方依赖...")
    pkgs = [
        "torch", "langgraph", "langchain_openai", "openai",
        "pymilvus", "sentence_transformers", "FlagEmbedding",
        "pymongo", "minio", "fastapi", "uvicorn", "pydantic",
    ]
    ok = True
    for name in pkgs:
        try:
            import importlib
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✓ {name}: {ver}")
        except Exception as e:
            ok = False
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
    return ok


def test_milvus():
    """测试 Milvus 连接"""
    print("测试 Milvus 连接...")
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(os.getenv("MILVUS_URL", "http://192.168.6.170:19530"))
        version = client.get_server_version()
        cols = client.list_collections()
        print(f"  ✓ Milvus 连接成功, 版本: {version}")
        print(f"      现有集合: {cols}")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ Milvus 连接失败: {e}")
        return False


def test_mongodb():
    """测试 MongoDB 连接"""
    print("测试 MongoDB 连接...")
    try:
        from pymongo import MongoClient
        client = MongoClient(
            os.getenv("MONGO_URL", "mongodb://192.168.6.170:27017"),
            serverSelectionTimeoutMS=5000,
        )
        client.admin.command("ping")
        print(f"  ✓ MongoDB 连接成功")
        client.close()
        return True
    except Exception as e:
        print(f"  ✗ MongoDB 连接失败: {e}")
        return False


def test_minio():
    """测试 MinIO 连接"""
    print("测试 MinIO 连接...")
    try:
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "192.168.6.170:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
        buckets = [b.name for b in client.list_buckets()]
        print(f"  ✓ MinIO 连接成功, 存储桶: {buckets}")
        return True
    except Exception as e:
        print(f"  ✗ MinIO 连接失败: {e}")
        return False


def test_llm():
    """测试 DashScope LLM 调用"""
    print("测试 DashScope LLM 调用...")
    try:
        from openai import OpenAI
        key = os.getenv("DASHSCOPE_API_KEY")
        base = os.getenv("OPENAI_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")
        if not key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "请只回复两个字:成功"}],
            max_tokens=10,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  ✓ LLM({model}) 回复: {reply}")
        return True
    except Exception as e:
        print(f"  ✗ LLM 调用失败: {type(e).__name__}: {e}")
        return False


def test_data_dir():
    """检查旅游数据目录与 md 文件数量"""
    print("检查旅游数据目录...")
    data_dir = TRAVEL_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"  ✗ 目录不存在: {data_dir}")
        return False
    md_files = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".md"):
                md_files.append(os.path.join(root, f))
    print(f"  ✓ 目录存在, md 文件共 {len(md_files)} 个")
    return len(md_files) > 0


# ============================================================ #
#                    本地模型加载测试（较慢）                    #
# ============================================================ #

def test_bge_m3():
    """测试 BGE-M3 嵌入模型加载与编码"""
    print("测试 BGE-M3 嵌入模型加载与编码...")
    try:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        ef = BGEM3EmbeddingFunction(
            model_name=os.getenv("BGE_M3_PATH"),
            device=os.getenv("BGE_DEVICE", "cpu"),
            use_fp16=os.getenv("BGE_FP16", "0").lower() in ("1", "true"),
        )
        out = ef.encode_documents(["三亚亚龙湾是热门的海滨度假目的地，以优质沙滩和度假酒店著称。"])
        # 注：pymilvus-model>=0.3 返回 dense 为 list[np.ndarray]，sparse 为 scipy csr_array
        dense = out["dense"]
        import numpy as np
        dense_mat = np.asarray(dense)
        sparse = out["sparse"]
        print(f"  ✓ BGE-M3 加载成功, dense 维度: {dense_mat.shape}, sparse 形状: {sparse.shape}")
        return True
    except Exception as e:
        print(f"  ✗ BGE-M3 加载失败: {type(e).__name__}: {e}")
        return False


def test_reranker():
    """测试 BGE-Reranker 模型加载与打分"""
    print("测试 BGE-Reranker 模型加载与打分...")
    try:
        from FlagEmbedding import FlagReranker
        rk = FlagReranker(
            model_name_or_path=os.getenv("BGE_RERANKER_LARGE"),
            device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
            use_fp16=os.getenv("BGE_RERANKER_FP16", "0").lower() in ("1", "true"),
        )
        pairs = [
            ("三亚亚龙湾怎么样", "亚龙湾拥有优质沙滩、海景和成熟的海滨度假酒店带。"),
            ("蜈支洲岛怎么去", "蜈支洲岛需先到码头候船，再乘船上岛，适合安排整天。"),
        ]
        scores = rk.compute_score(pairs, normalize=True)
        print(f"  ✓ Reranker 加载成功, 示例分数: {scores}")
        return True
    except Exception as e:
        print(f"  ✗ Reranker 加载失败: {type(e).__name__}: {e}")
        return False


# ============================================================ #
#                         汇总输出                             #
# ============================================================ #

QUICK_TESTS = {
    "环境变量/模型路径": test_env,
    "第三方依赖": test_versions,
    "Milvus": test_milvus,
    "MongoDB": test_mongodb,
    "MinIO": test_minio,
    "DashScope LLM": test_llm,
    "旅游数据目录": test_data_dir,
}

MODEL_TESTS = {
    "BGE-M3 嵌入": test_bge_m3,
    "BGE-Reranker": test_reranker,
}


def run(tests: dict) -> bool:
    print("\n" + "=" * 60)
    print("测试执行")
    print("=" * 60)
    results = {}
    for name, fn in tests.items():
        results[name] = fn()
    return _summary(results)


def _summary(results: dict) -> bool:
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    print("=" * 60)
    if all_passed:
        print("所有测试通过，环境正常！")
    else:
        print("存在失败的测试，请根据上方信息检查。")
    return all_passed


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

    print("=" * 60)
    print("travel_brain 旅游知识库 - 环境准备测试")
    print(f"模式: {mode}")
    print("=" * 60)

    if mode == "model":
        ok = run(MODEL_TESTS)
    elif mode == "all":
        ok = run(QUICK_TESTS)
        if ok:
            ok = run(MODEL_TESTS)
    else:
        ok = run(QUICK_TESTS)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


"""预计输出:
D:\A_Py_Java\pyFile\travel_brain\.venv\Scripts\python.exe D:\A_Py_Java\pyFile\travel_brain\test\prepare_test.py 
============================================================
travel_brain 旅游知识库 - 环境准备测试
模式: quick
============================================================

============================================================
测试执行
============================================================
检查环境变量与本地模型路径...
  ✓ DASHSCOPE_API_KEY 已配置 (len=115)
  ✓ BGE_M3_PATH 存在
      D:\ai_models\modelscope_cache\models\BAAI\bge-m3
  ✓ BGE_RERANKER_LARGE 存在
      D:\ai_models\modelscope_cache\models\BAAI\bge-reranker-large
检查关键第三方依赖...
  ✓ torch: 2.14.0+cpu
  ✓ langgraph: ?
  ✓ langchain_openai: 1.6.0
  ✓ openai: 3.8.0
  ✓ pymilvus: 3.0.1
  ✓ sentence_transformers: 6.0.1
  ✓ FlagEmbedding: ?
  ✓ pymongo: 4.18.0
  ✓ minio: 7.2.20
  ✓ fastapi: 0.141.1
  ✓ uvicorn: 0.52.4
  ✓ pydantic: 2.13.5
测试 Milvus 连接...
  ✓ Milvus 连接成功, 版本: pkg/v2.5.5
      现有集合: ['kb_item_names_v1', 'kb_chunks_v1']
测试 MongoDB 连接...
  ✓ MongoDB 连接成功
测试 MinIO 连接...
  ✓ MinIO 连接成功, 存储桶: ['a-bucket', 'knowledge-base-files']
测试 DashScope LLM 调用...
  ✓ LLM(qwen-flash) 回复: 成功
检查旅游数据目录...
  ✓ 目录存在, md 文件共 30 个

============================================================
测试结果汇总
============================================================
  环境变量/模型路径: ✓ 通过
  第三方依赖: ✓ 通过
  Milvus: ✓ 通过
  MongoDB: ✓ 通过
  MinIO: ✓ 通过
  DashScope LLM: ✓ 通过
  旅游数据目录: ✓ 通过
============================================================
所有测试通过，环境正常！

进程已结束，退出代码为 0

"""