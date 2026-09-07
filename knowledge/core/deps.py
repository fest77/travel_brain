"""依赖注入 + 单例缓存"""
from functools import cache, lru_cache


@cache
def get_import_file_service():
    """获取导入文件服务单例（缓存长期有效）"""
    from knowledge.services.file_import_service import ImportFileService
    return ImportFileService()


@lru_cache
def get_query_service():
    """获取查询服务单例（LRU 缓存）"""
    from knowledge.services.query_service import QueryService
    return QueryService()
