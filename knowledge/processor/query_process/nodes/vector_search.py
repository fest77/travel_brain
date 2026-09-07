"""向量检索节点：把重写问题做混合检索(dense+sparse)召回切片"""
from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

# 检索返回时需要的切片字段（含来源与元数据，供 rrf/rerank/答案 引用）
_OUTPUT_FIELDS = ["chunk_id", "content", "title", "parent_title",
                  "region", "content_type", "source_file", "entity_name"]


class VectorSearchNode(BaseNode):
    """向量检索：全文语义检索（不做硬过滤，避免小库空召回）"""

    name = "search_embedding"

    def process(self, state):
        # Step1 校验输入
        rewritten_query = state.get("rewritten_query") or state.get("original_query")
        if not rewritten_query:
            raise StateFieldError(self.name, "rewritten_query", str)

        # Step2 问题向量化（encode_queries，问题专用指令前缀）
        try:
            model = AIClients.get_bge_m3_client()
            vec = generate_bge_m3_hybrid_vectors(model, [rewritten_query], is_query=True)
        except Exception as e:
            self.logger.warning(f"问题向量化失败: {e}")
            return {"embedding_chunks": []}

        # Step3 混合检索
        try:
            client = StorageClients.get_milvus_client()
            reqs = create_hybrid_search_requests(
                dense_vector=vec["dense"][0],
                sparse_vector=vec["sparse"][0],
                limit=self.config.embedding_search_limit,
            )
            reps = execute_hybrid_search_query(
                client,
                collection_name=self.config.chunks_collection,
                search_requests=reqs,
                limit=self.config.embedding_search_limit,
                output_fields=_OUTPUT_FIELDS,
            )
            hits = reps[0] if reps and reps[0] else []
            self.logger.info(f"向量检索命中 {len(hits)} 条")
            return {"embedding_chunks": hits}
        except Exception as e:
            self.logger.warning(f"混合检索失败: {e}")
            return {"embedding_chunks": []}
