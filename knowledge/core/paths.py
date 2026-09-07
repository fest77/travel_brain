"""路径常量定义"""
import os

# 项目根目录  D:\A_Py_Java\pyFile\travel_brain\knowledge
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录  ...\knowledge\temp_data
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")

# 前端页面静态资源目录  ...\knowledge\front
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")

# 原始 md 数据目录（批量导入用）  ...\travel_brain\data
DATA_DIR = os.path.join(os.path.abspath(os.path.join(KNOWLEDGE_ROOT, "..")), "data")


def get_local_base_dir() -> str:
    """获取本地文件存储基础目录"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录"""
    return FRONT_PAGE_DIR


def get_data_dir() -> str:
    """获取原始 md 数据目录"""
    return DATA_DIR
