"""BGE-M3 混合向量工具（dense + sparse）"""
from typing import List
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def generate_bge_m3_hybrid_vectors(model: BGEM3EmbeddingFunction,
                                   texts: List[str], is_query: bool = True):
    """为文本生成混合向量（稠密 + 稀疏），返回 {"dense": [...], "sparse": [...]}

    - is_query=True  使用 encode_queries（问题，带指令前缀）
    - is_query=False 使用 encode_documents（入库文档）
    """
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("texts 不能为空且必须为非空字符串列表")

    # 1. 编码
    if is_query:
        result = model.encode_queries(texts)
    else:
        result = model.encode_documents(texts)

    if "dense" not in result or "sparse" not in result:
        raise RuntimeError(f"嵌入结果缺少必要字段: {list(result.keys())}")

    # 2. 解析稀疏向量（scipy csr_array → dict {token_id: weight}）
    dense_vectors = []
    sparse_vectors = []
    csr = result["sparse"]
    for index in range(len(texts)):
        dense_vectors.append(result["dense"][index].tolist())
        start = csr.indptr[index]
        end = csr.indptr[index + 1]
        token_ids = csr.indices[start:end].tolist()
        weights = csr.data[start:end].tolist()
        sparse_vectors.append(dict(zip(token_ids, weights)))

    return {"dense": dense_vectors, "sparse": sparse_vectors}
