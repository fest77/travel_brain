"""向量化节点：对 chunks 批量生成 BGE-M3 dense+sparse 向量"""
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    """分批把 chunk.content 编码为稠密+稀疏向量并回填到 chunk"""

    name = "bge_embedding_node"

    def process(self, state):
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(self.name, "chunks", list)

        # 去空内容
        valid = [c for c in chunks if c.get("content")]
        if not valid:
            raise StateFieldError(self.name, "chunks(有内容的)", list)

        try:
            model = AIClients.get_bge_m3_client()
        except Exception as e:
            self.logger.error(f"获取 BGE-M3 客户端失败: {e}")
            raise

        batch_size = getattr(self.config, "embedding_batch_size", 8)
        final_chunks = []
        total = len(valid)

        for start in range(0, total, batch_size):
            batch = valid[start:start + batch_size]
            contents = [c["content"] for c in batch]

            result = model.encode_documents(contents)
            csr = result["sparse"]

            for idx, chunk in enumerate(batch):
                # dense（统一转成 list，便于序列化/判空）
                dense = result["dense"][idx]
                chunk["dense_vector"] = list(dense) if hasattr(dense, "tolist") else list(dense)
                # sparse (scipy csr -> dict)
                p0, p1 = csr.indptr[idx], csr.indptr[idx + 1]
                token_ids = csr.indices[p0:p1].tolist()
                weights = csr.data[p0:p1].tolist()
                chunk["sparse_vector"] = dict(zip(token_ids, weights))
            final_chunks.extend(batch)

            self.logger.info(f"向量化进度 {min(start + batch_size, total)}/{total}")

        state["chunks"] = final_chunks
        return state
