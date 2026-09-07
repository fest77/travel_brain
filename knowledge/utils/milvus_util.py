"""Milvus 混合检索工具（dense + sparse 双路 → WeightedRanker 融合）"""
import logging
from typing import Optional, List, Tuple, Any, Dict

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

logger = logging.getLogger(__name__)


def create_hybrid_search_requests(
        dense_vector: List[float],
        sparse_vector: Dict[int, float],
        expr: Optional[str] = None,
        expr_params: Optional[Dict[str, Any]] = None,
        limit: int = 10,
) -> List[AnnSearchRequest]:
    """创建稠密 + 稀疏两个检索请求

    说明：travel 采用全文语义检索为主，expr 默认 None（可选预留，如按地区过滤）。
    """
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        expr=expr,
        expr_params=expr_params,
        limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        expr=expr,
        expr_params=expr_params,
        limit=limit,
    )
    return [dense_req, sparse_req]


def execute_hybrid_search_query(
        milvus_client: MilvusClient,
        collection_name: str,
        search_requests: List[AnnSearchRequest],
        ranker_weights: Tuple[float, float] = (0.5, 0.5),
        norm_score: bool = True,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict] = None,
):
    """执行混合检索（WeightedRanker 融合稠密/稀疏得分）

    Returns:
        [[{'id':..,'distance':..,'entity':{...output_fields}}, ...], ...]
        其中 entity 里含 chunk_id/content/title 等标量字段。
    """
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title"]
    reranker = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=search_requests,
        ranker=reranker,
        limit=limit,
        output_fields=output_fields,
        search_params=search_params,
    )
    return res
