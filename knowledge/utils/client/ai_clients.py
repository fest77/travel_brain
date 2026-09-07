"""AI 模型客户端：OpenAI LLM（文本/JSON） / BGE-M3 / BGE-Reranker"""
import logging
import threading
from typing import Optional

from openai import OpenAI
from langchain_openai import ChatOpenAI
from FlagEmbedding import FlagReranker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.base import BaseClientManager, get_env

logger = logging.getLogger(__name__)


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    # ---------- LLM：ChatOpenAI（DashScope 兼容） ----------
    _llm_text_client: Optional[ChatOpenAI] = None
    _llm_text_lock = threading.Lock()
    _llm_json_client: Optional[ChatOpenAI] = None
    _llm_json_lock = threading.Lock()

    @classmethod
    def get_llm_openai(cls, response_format: bool = True) -> ChatOpenAI:
        """获取 LLM 客户端。response_format=True 返回 json_object 模式客户端"""
        if response_format:
            return cls._get_or_create("_llm_json_client", cls._llm_json_lock,
                                      lambda: cls._create_llm_openai(response_format))
        return cls._get_or_create("_llm_text_client", cls._llm_text_lock,
                                  lambda: cls._create_llm_openai(response_format))

    @classmethod
    def _create_llm_openai(cls, response_format: bool) -> ChatOpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")
            temperature = float(get_env("LLM_DEFAULT_TEMPERATURE", "0.1"))

            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
                model_kwargs=model_kwargs,
            )
            logger.info(f"ChatOpenAI LLM 初始化成功 (model={model_name}, json={response_format})")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI LLM 初始化失败: {e}")
            raise ConnectionError(f"LLM 连接失败: {e}") from e

    # ---------- LLM(实时联网版 enable_search)：本地知识库兜底 ----------
    _llm_online_client: Optional[ChatOpenAI] = None
    _llm_online_lock = threading.Lock()

    @classmethod
    def get_llm_online(cls, response_format: bool = False) -> ChatOpenAI:
        """获取带 enable_search(实时联网) 的 LLM，用于知识库无该地资料时的兜底回答"""
        return cls._get_or_create("_llm_online_client", cls._llm_online_lock,
                                  lambda: cls._create_llm_online(response_format))

    @classmethod
    def _create_llm_online(cls, response_format: bool) -> ChatOpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")
            temperature = float(get_env("LLM_DEFAULT_TEMPERATURE", "0.1"))
            model_kwargs = {"enable_search": True}   # DashScope 实时搜索开关
            if response_format:
                model_kwargs["response_format"] = {"type": "json_object"}
            client = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
                model_kwargs=model_kwargs,
            )
            logger.info(f"ChatOpenAI LLM(联网版) 初始化成功 (model={model_name})")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"ChatOpenAI LLM(联网版) 初始化失败: {e}")
            raise ConnectionError(f"LLM 连接失败: {e}") from e

    # ---------- OpenAI 原生客户端（兼容 DashScope；用于 enable_search 实时联网） ----------
    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    @classmethod
    def get_openai(cls) -> OpenAI:
        """原生 OpenAI 客户端：用 extra_body={'enable_search': True} 才能真正触发 DashScope 实时搜索"""
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_openai)

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("DASHSCOPE_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("OpenAI(原生)客户端初始化成功")
            return client
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI(原生)客户端初始化失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ---------- BGE-M3 嵌入 ----------
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    @classmethod
    def get_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls) -> BGEM3EmbeddingFunction:
        try:
            model_path = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            fp16 = get_env("BGE_FP16", "0").lower() in ("1", "true")
            ef = BGEM3EmbeddingFunction(model_name=model_path, device=device, use_fp16=fp16)
            logger.info(f"BGE-M3 初始化成功 (device={device})")
            return ef
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-M3 初始化失败: {e}")
            raise ConnectionError(f"BGE-M3 连接失败: {e}") from e

    # ---------- BGE-Reranker ----------
    _bge_rerank_client: Optional[FlagReranker] = None
    _bge_rerank_lock = threading.Lock()

    @classmethod
    def get_bge_m3_rerank_client(cls) -> FlagReranker:
        return cls._get_or_create("_bge_rerank_client", cls._bge_rerank_lock, cls._create_bge_rerank_client)

    @classmethod
    def _create_bge_rerank_client(cls) -> FlagReranker:
        try:
            model_path = cls._require_env("BGE_RERANKER_LARGE")
            device = get_env("BGE_RERANKER_DEVICE", "cpu")
            fp16 = get_env("BGE_RERANKER_FP16", "0").lower() in ("1", "true")
            reranker = FlagReranker(model_name_or_path=model_path, device=device, use_fp16=fp16)
            logger.info(f"BGE-Reranker 初始化成功 (device={device})")
            return reranker
        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"BGE-Reranker 初始化失败: {e}")
            raise ConnectionError(f"BGE-Reranker 连接失败: {e}") from e
