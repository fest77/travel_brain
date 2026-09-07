"""写入 Milvus 节点：确保 travel_chunks_v1 / travel_entity_names_v1 存在并插入"""
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


def _build_chunks_schema(client, dim: int):
    """切片集合 schema：标量(检索展示/过滤) + 双向量"""
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    for field, ml in [
        ("content", 65535), ("title", 65535), ("parent_title", 65535),
        ("file_title", 65535), ("source_file", 65535), ("content_type", 65535),
        ("region", 65535), ("entity_name", 65535),
    ]:
        schema.add_field(field, DataType.VARCHAR, max_length=ml)
    return schema


def _build_entity_schema(client, dim: int):
    """实体索引集合 schema：地区/类型 + 双向量，用于查询侧定位/过滤"""
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    for field, ml in [
        ("entity_name", 65535), ("entity_type", 255), ("region", 65535),
        ("content_type", 65535), ("file_title", 65535),
    ]:
        schema.add_field(field, DataType.VARCHAR, max_length=ml)
    return schema


def _build_index_params(client, collection_name: str):
    index = client.prepare_index_params(collection_name=collection_name)
    index.add_index("dense_vector", index_name="dense_vector_index",
                    index_type="AUTOINDEX", metric_type="COSINE")
    index.add_index("sparse_vector", index_name="sparse_vector_index",
                    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    return index


class ImportMilvusNode(BaseNode):
    """把带向量的 chunks 写入切片集合；把文档主体实体写入实体索引集合"""

    name = "import_milvus_node"

    def process(self, state):
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(self.name, "chunks", list)

        client = StorageClients.get_milvus_client()
        dim = len(chunks[0]["dense_vector"])

        # 1. 切片集合
        chunks_col = self.config.chunks_collection
        self._ensure_collection(client, chunks_col, dim, _build_chunks_schema)
        rows = [
            {k: chunk.get(k, "") for k in
             ["content", "title", "parent_title", "file_title", "source_file",
              "content_type", "region", "entity_name"]}
            | {"dense_vector": chunk["dense_vector"], "sparse_vector": chunk["sparse_vector"]}
            for chunk in chunks
        ]
        result = client.insert(chunks_col, rows)
        self.logger.info(f"写入切片集合 {chunks_col}: {result.get('insert_count')} 条")

        # 2. 实体索引集合（文件级主体实体一条）
        self._insert_entity(client, state)
        return state

    # ------------------------------------------------------------ #
    def _insert_entity(self, client, state):
        entity_name = (state.get("entity_name") or "").strip()
        region = (state.get("region") or "").strip()
        if not entity_name and not region:
            self.logger.warning("无主体实体，跳过实体索引写入")
            return

        # 实体行：region 级别一条；若含实体名再补一条具体实体
        entity_rows = [{
            "entity_name": entity_name or region,
            "entity_type": "region",
            "region": region,
            "content_type": state.get("content_type", ""),
            "file_title": state.get("file_title", ""),
        }]

        # 编码实体向量
        try:
            model = AIClients.get_bge_m3_client()
            texts = [r["entity_name"] for r in entity_rows]
            result = model.encode_documents(texts)
            csr = result["sparse"]
            for idx, row in enumerate(entity_rows):
                dense = result["dense"][idx]
                row["dense_vector"] = list(dense) if hasattr(dense, "tolist") else list(dense)
                p0, p1 = csr.indptr[idx], csr.indptr[idx + 1]
                token_ids = csr.indices[p0:p1].tolist()
                weights = csr.data[p0:p1].tolist()
                row["sparse_vector"] = dict(zip(token_ids, weights))
        except Exception as e:
            self.logger.error(f"实体向量化失败，跳过实体写入: {e}")
            return

        entity_col = self.config.entity_name_collection
        self._ensure_collection(client, entity_col, len(entity_rows[0]["dense_vector"]),
                                _build_entity_schema)
        res = client.insert(entity_col, entity_rows)
        self.logger.info(f"写入实体索引集合 {entity_col}: {res.get('insert_count')} 条")

    def _ensure_collection(self, client, name: str, dim: int, builder):
        if client.has_collection(name):
            return
        schema = builder(client, dim)
        index = _build_index_params(client, name)
        client.create_collection(name, schema=schema, index_params=index)
        self.logger.info(f"自动创建集合 {name} (dim={dim})")
