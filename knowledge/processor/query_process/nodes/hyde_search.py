"""HyDE 检索节点：LLM 先生成"假想攻略"再检索，提升召回（Hypothetical Document Embeddings）"""
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.prompt.query_prompt import HYDE_USER_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

_OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title",
                  "region", "content_type", "source_file", "entity_name"]


class HyDeSearchNode(BaseNode):
    """HyDE 检索：检索信号 = 问题 + LLM 生成的假设攻略片段"""

    name = "search_embedding_hyde"

    def process(self, state):
        # Step1 校验输入
        rewritten_query = state.get("rewritten_query") or state.get("original_query")
        if not rewritten_query:
            raise StateFieldError(self.name, "rewritten_query", str)
        region = "、".join(state.get("entity_names") or [])

        # Step2 生成假设性攻略（失败则空串，仍用原问题检索）
        hy_doc = self._generate_hy_document(region, rewritten_query)

        # Step3 检索文本 = 问题 + 假设攻略 → 向量化 → 混合检索
        embedding_text = f"{rewritten_query}\n{hy_doc}".strip()
        try:
            model = AIClients.get_bge_m3_client()
            vec = generate_bge_m3_hybrid_vectors(model, [embedding_text], is_query=True)
            client = StorageClients.get_milvus_client()
            reqs = create_hybrid_search_requests(
                dense_vector=vec["dense"][0],
                sparse_vector=vec["sparse"][0],
                limit=self.config.hyde_search_limit,
            )
            reps = execute_hybrid_search_query(
                client,
                collection_name=self.config.chunks_collection,
                search_requests=reqs,
                limit=self.config.hyde_search_limit,
                output_fields=_OUTPUT_FIELDS,
            )
            hits = reps[0] if reps and reps[0] else []
            self.logger.info(f"HyDE 检索命中 {len(hits)} 条")
            return {"hyde_embedding_chunks": hits}
        except Exception as e:
            self.logger.warning(f"HyDE 检索失败: {e}")
            return {"hyde_embedding_chunks": []}

    # ------------------------------------------------------------ #
    def _generate_hy_document(self, region: str, query: str) -> str:
        """调用 LLM 生成一段假设攻略；失败返回空串"""
        try:
            llm = AIClients.get_llm_openai(response_format=False)
            user_prompt = HYDE_USER_PROMPT_TEMPLATE.format(region=region or "未知目的地",
                                                           rewritten_query=query)
            system_prompt = "你是一位资深旅游顾问，擅长撰写实用的旅游攻略。"
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            return (resp.content or "").strip()
        except Exception as e:
            self.logger.warning(f"HyDE 假设文档生成失败(降级): {e}")
            return ""
