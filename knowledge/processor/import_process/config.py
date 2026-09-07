"""导入流程配置管理模块"""
from dataclasses import dataclass, field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 1000      # 切片最大长度
    min_content_length: int = 200       # 合并短内容的最小长度
    overlap_sentences: int = 1          # 句子级切分时重叠句数
    entity_chunk_k: int = 3             # 实体识别时使用的切片数量
    entity_chunk_size: int = 2500       # 实体识别时使用的切片内容长度

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    default_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", ""))
    entity_model: str = field(default_factory=lambda: os.getenv("ENTITY_MODEL", ""))

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))
    entity_name_collection: str = field(default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", ""))

    # ==================== MinIO 配置（可选） ====================
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", ""))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", ""))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", ""))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", ""))
    minio_secure: bool = False

    # ==================== 向量配置 ====================
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_batch_size: int = 8

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self) -> str:
        """获取 MinIO 基础 URL"""
        protocol = "https://" if self.minio_secure else "http://"
        return protocol + self.minio_endpoint


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
