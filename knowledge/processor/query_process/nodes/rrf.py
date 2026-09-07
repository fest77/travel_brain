"""RRF 融合节点：把多路检索结果（向量 + HyDE，可选联网）按 RRF 公式融合排序

RRF 公式: score(doc) = Σ weight / (k + rank_index+1)
"""
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode


class RrfNode(BaseNode):
    """多路结果倒序融合（Reciprocal Rank Fusion）"""

    name = "rrf"

    def process(self, state):
        # Step1 取各路检索原始命中（形如 [{"id":..,"distance":..,"entity":{...}}]）
        embedding = state.get("embedding_chunks") or []
        hyde = state.get("hyde_embedding_chunks") or []

        # Step2 统一抽取出 entity（标量字典），保留原排序位置用于 RRF 序号
        norm1 = self._normalize(embedding)
        norm2 = self._normalize(hyde)
        inputs = [(norm1, 1.0), (norm2, 1.0)]

        # Step3 RRF 融合
        merged = self._rrf_merge(inputs, top_k=self.config.rrf_max_results)

        # Step4 回填
        state["rrf_chunks"] = [doc for doc, _ in merged]
        self.logger.info(f"RRF 融合出 {len(merged)} 条")
        return state

    def _normalize(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 [{entity:{...}}] → [{chunk_id/content/title/...}]，去掉外层 id/distance"""
        out = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            entity = hit.get("entity")
            if isinstance(entity, dict):
                out.append(entity)
        return out

    def _rrf_merge(self, rrf_inputs, k: int = 60, top_k: int = 10):
        """对多路已归一化文档做 RRF；同一 chunk_id 去重累计分"""
        scores: Dict[Any, float] = {}
        data: Dict[Any, Dict] = {}
        for docs, weight in rrf_inputs:
            for idx, doc in enumerate(docs):
                cid = doc.get("chunk_id")
                if cid is None:
                    continue
                scores[cid] = scores.get(cid, 0.0) + weight / (k + (idx + 1))
                data.setdefault(cid, doc)  # 先到者保留其完整字段
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(data[cid], score) for cid, score in ranked[:top_k]]
