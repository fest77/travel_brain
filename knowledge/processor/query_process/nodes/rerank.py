"""重排序节点：合并本地(RRF)与联网结果 → BGE-Reranker 交叉编码精排 → 断崖截断"""
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    """对候选片段与问题做交叉编码相关性打分并重排（BGE-Reranker-Large）"""

    name = "rerank"

    def process(self, state):
        # Step1 取查询（重写优先）
        query = state.get("rewritten_query") or state.get("original_query") or ""

        # Step2 合并多源文档：本地 rrf_chunks + 联网 web_search_docs
        merge_docs = self._merge_multi_source(state)

        # Step3 Reranker 精排（失败则保持原序、score=None 降级）
        rerank_docs = self._rerank(query, merge_docs)

        # Step4 断崖截断：不固定数量，第一明显分差处截断（保底 top2，封顶 top10）
        cutoff = self._cliff_cutoff(rerank_docs)

        state["reranked_docs"] = cutoff
        self.logger.info(f"Rerank 精排后保留 {len(cutoff)} 条")
        return state

    # ------------------------------------------------------------ #
    def _merge_multi_source(self, state) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        # 本地切片（含元数据）
        for c in state.get("rrf_chunks") or []:
            docs.append({
                "chunk_id": c.get("chunk_id"),
                "content": c.get("content", ""),
                "title": c.get("title", ""),
                "file_title": c.get("file_title", c.get("source_file", "")),
                "region": c.get("region", ""),
                "content_type": c.get("content_type", ""),
                "source": "local",
            })
        # 联网结果（title/url/snippet）
        for d in state.get("web_search_docs") or []:
            docs.append({
                "content": d.get("content") or d.get("snippet", ""),
                "title": d.get("title", ""),
                "url": d.get("url", ""),
                "source": "web",
            })
        return [d for d in docs if d.get("content")]

    def _rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return []
        try:
            reranker = AIClients.get_bge_m3_rerank_client()
            pairs = [[query, d["content"]] for d in docs]
            scores = reranker.compute_score(sentence_pairs=pairs, normalize=True)
            if isinstance(scores, (float, int)):      # 单文档时 FlagEmbedding 返回标量
                scores = [scores]
            scored = [{**d, "score": s} for d, s in zip(docs, scores)]
            return sorted(scored, key=lambda d: d["score"], reverse=True)
        except Exception as e:
            self.logger.warning(f"Reranker 失败，保持原序降级: {e}")
            return [{**d, "score": None} for d in docs]

    def _cliff_cutoff(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """第一断崖截断：相邻分数差 ≥ gap_abs 处截断；范围 [min_top_k, max_top_k]"""
        upper = min(self.config.rerank_max_top_k, len(docs))
        lower = min(self.config.rerank_min_top_k, upper)
        if upper <= 1 or docs[0].get("score") is None:
            return docs[:upper]
        cut = upper
        for i in range(upper - 1):
            gap = docs[i]["score"] - docs[i + 1]["score"]
            if gap >= self.config.rerank_gap_abs:
                cut = i + 1
                break
        cut = max(cut, lower)
        return docs[:cut]
