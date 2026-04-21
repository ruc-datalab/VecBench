import time
import numpy as np
import pandas as pd
from datetime import datetime

import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure, Property, DataType, VectorDistances, Reconfigure, VectorFilterStrategy
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from databases.base_vd import BaseVD


class Weaviate(BaseVD):
    """Implementation of the BaseVD interface for Weaviate vector database."""

    def connect(self):
        conn_params = self.config
        self.client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_url(
                conn_params["host"],
                grpc_port=conn_params.get("grpc_port", 50051)
            )
        )
        self.client.connect()
        self.collection_name = "my_table"

    def create_table(self, db: str, schema: dict):
        table_schema = schema.get(db)
        collection_name = table_schema.get("table_name")
        columns = table_schema.get("columns", [])

        if self.client.collections.exists(collection_name):
            self.client.collections.delete(collection_name)

        vector_field = None
        vector_dim = None
        distance_metric = "l2"
        properties = []

        for col in columns:
            name = col["name"]
            col_type = col["type"]

            if col_type == "FLOAT_VECTOR":
                vector_field = name
                vector_dim = col["dimension"]
                distance_metric = col.get("description", "l2").lower()
                continue

            if col_type in ("INT8", "INT16", "INT32", "INT64", "integer"):
                dtype = DataType.INT
            elif col_type == "FLOAT":
                dtype = DataType.NUMBER
            elif col_type in ("VARCHAR", "keyword", "text"):
                dtype = DataType.TEXT
            elif col_type == "ARRAY":
                dtype = DataType.TEXT_ARRAY
            elif col_type == "BOOL":
                dtype = DataType.BOOL
            else:
                raise ValueError(f"Unsupported field type: {col_type}")

            is_numeric = dtype in (DataType.INT, DataType.NUMBER)
            is_text = dtype == DataType.TEXT

            properties.append(
                Property(
                    name=name,
                    data_type=dtype,
                    index_filterable=True,
                    index_searchable=is_text,
                    index_range_filters=is_numeric
                )
            )

        dist_map = {
            "l2": VectorDistances.L2_SQUARED,
            "euclidean": VectorDistances.L2_SQUARED,
            "cosine": VectorDistances.COSINE,
            "dot": VectorDistances.DOT,
            "ip": VectorDistances.DOT
        }
        dist = dist_map.get(distance_metric, VectorDistances.L2_SQUARED)

        hnsw_config = table_schema.get("hnsw_config", {})
        self.client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=None,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=dist,
                max_connections=hnsw_config.get("M", 16),
                ef_construction=hnsw_config.get("efConstruction", 64)
            )
        )

        self.collection_name = collection_name
        self.vector_field = vector_field
        self.vector_dim = vector_dim

    def process_data(self, data: pd.DataFrame, db: str, schema: dict):
        table_schema = schema.get(db)
        col_types = {c["name"]: c["type"] for c in table_schema.get("columns", [])}

        for col, t in col_types.items():
            if col not in data.columns: continue
            
            if t in ("INT8", "INT16", "INT32", "INT64", "integer"):
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
            elif t == "FLOAT":
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0).astype(float)
            elif t == "VARCHAR":
                data[col] = data[col].astype(str)
            elif t == "ARRAY":
                def to_str_list(x):
                    if isinstance(x, (list, np.ndarray)):
                        return [str(i) for i in x]
                    return []
                data[col] = data[col].apply(to_str_list)
            elif t == "FLOAT_VECTOR":
                data[col] = data[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        return data

    # def insert_data(self, data: pd.DataFrame, db: str, schema: dict, batch_size: int = 10000):
    #     col = self.client.collections.get(self.collection_name)
    #     vector_col = self.vector_field
        
    #     id_col = "id" if "id" in data.columns else ("movieid" if "movieid" in data.columns else None)
        
    #     start_time = time.time()
    #     with col.batch.fixed_size(batch_size=batch_size) as batch:
    #         for _, row in data.iterrows():
    #             props = row.drop([vector_col, id_col] if id_col else [vector_col], errors="ignore").to_dict()
                
    #             obj_id = row[id_col] if id_col else None
    #             weaviate_id = generate_uuid5(obj_id) if obj_id is not None else None

    #             batch.add_object(
    #                 properties=props,
    #                 vector=row[vector_col],
    #                 uuid=weaviate_id
    #             )
        
    #     if len(col.batch.failed_objects) > 0:
    #         print(f"Failed to insert {len(col.batch.failed_objects)} objects.")
            
    #     return time.time() - start_time

    # 优化版本
    def insert_data(self, data: pd.DataFrame, db: str, schema: dict, batch_size: int = 10000):
        table_schema = schema.get(db)
        collection_name = table_schema.get("table_name")
        columns = table_schema.get("columns", [])
        
        vector_col = getattr(self, 'vector_field', None)
        if not vector_col:
            for col_def in columns:
                if col_def["type"] == "FLOAT_VECTOR":
                    vector_col = col_def["name"]
                    break
        
        col = self.client.collections.get(collection_name)

        if "id" in data.columns:
            id_col = "id"
        elif "movieid" in data.columns:
            id_col = "movieid"
        else:
            id_col = None

        scalar_cols = [
            c for c in data.columns 
            if c not in (vector_col, id_col)
        ]

        start_time = time.time()
        
        with col.batch.fixed_size(batch_size=batch_size) as batch:
            for _, row in data.iterrows():
                obj_uuid = generate_uuid5(int(row[id_col])) if id_col else None
                properties = row[scalar_cols].to_dict()
                
                batch.add_object(
                    properties=properties,
                    vector=row[vector_col] if vector_col in row else None,
                    uuid=obj_uuid
                )

        failed = len(col.batch.failed_objects)
        if failed > 0:
            print(f"[Weaviate] Failed to insert {failed} objects")

        return time.time() - start_time


    def create_index(self, index_params=None, args=None):
        # Weaviate builds index at collection creation time
        print("Weaviate builds vector index during collection creation.")

    def create_scalar_index(self, index_params=None, args=None):
        # Weaviate scalar indexes are enabled via index_filterable=True
        print("Weaviate scalar index handled automatically.")

    def query(self, query_config: dict, param: int):
        col = self.client.collections.get("my_table")
        
        col.config.update(
                vector_index_config=Reconfigure.VectorIndex.hnsw(
                    filter_strategy=VectorFilterStrategy.ACORN,  # 更新过滤策略
                    flat_search_cutoff=0,
                    ef=param,
            ),
        )   

        # current_config = col.config.get()
        # print(f"Verified ef: {current_config.vector_index_config}")

        ref_id = query_config["reference_vector_name"]
        scalar_filters = query_config.get("scalar_filters", [])
        limit = query_config.get("limit", 10)

        target_uuid = generate_uuid5(int(ref_id))
        obj = col.query.fetch_object_by_id(target_uuid, include_vector=True)
        if not obj or not obj.vector:
            raise ValueError(f"Reference vector for ID {ref_id} not found.")
        
        query_vector = obj.vector["default"] if isinstance(obj.vector, dict) else obj.vector

        weaviate_filters = None
        for cond in scalar_filters:
            field = cond["field"]
            op = cond["operator"]
            val = cond["value"]
            logic = cond.get("logic", "and")
            
            f = None
            if op == "==":
                f = Filter.by_property(field).equal(val)
            elif op == "<":
                f = Filter.by_property(field).less_than(val)
            elif op == "<=":
                f = Filter.by_property(field).less_than_equal(val)
            elif op == ">":
                f = Filter.by_property(field).greater_than(val)
            elif op == ">=":
                f = Filter.by_property(field).greater_than_equal(val)
            elif op == "contains":
                f = Filter.by_property(field).contains_any([str(val)]) 
            elif op == "like":
                f = Filter.by_property(field).like(f"*{val}*") 
            elif op == "is_empty":
                f = Filter.by_property(field).is_none(True)

            if f:
                if weaviate_filters is None:
                    weaviate_filters = f
                else:
                    if logic.lower() == "or":
                        weaviate_filters = weaviate_filters | f
                    else:
                        weaviate_filters = weaviate_filters & f

        start = time.time()
        res = col.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            filters=weaviate_filters,
            return_metadata=MetadataQuery(distance=True)
        )
        duration = time.time() - start

        return res.objects, duration

    def delete_by_ids(self, selected_ids: list[int]) -> float:
        # 单次删除批量有上限，默认10000条
        col = self.client.collections.get(self.collection_name)
        uuids_to_delete = [generate_uuid5(tid) for tid in selected_ids]
        
        chunk_size = 10000
        start_time = time.time()

        for i in range(0, len(uuids_to_delete), chunk_size):
            chunk = uuids_to_delete[i : i + chunk_size]
            try:
                col.data.delete_many(
                    where=Filter.by_id().contains_any(chunk)
                )
            except Exception as e:
                print(f"[Weaviate] Error deleting chunk starting at index {i}: {e}")
                
        return time.time() - start_time

    def update(self, batch_df, batch_size=10000):
        collection_name = "my_table" 
        col = self.client.collections.get(collection_name)
        
        ids_and_data = []
        for row in batch_df.itertuples(index=False):
            point_uuid = generate_uuid5(int(row.id), collection_name)
            ids_and_data.append((point_uuid, row))

        start_time = time.time()

        try:
            with col.batch.fixed_size(batch_size=batch_size, concurrent_requests=4) as batch:
                for point_uuid, row in ids_and_data:
                    properties = {
                        "equal": int(row.equal),
                        "range": int(row.range),
                        "tags": [str(t) for t in row.tags] 
                    }
                    
                    batch.add_object(
                        properties=properties,
                        vector=row.image_vec, 
                        uuid=point_uuid
                    )
            
            write_duration = time.time() - start_time

        except Exception as e:
            print(f"Weaviate batch update error: {e}")
            raise e

        return write_duration
    
    def overwrite_properties(self, property_name: str, update_data: pd.DataFrame) -> float:
        col = self.client.collections.get(self.collection_name)
        id_col = "id" if "id" in update_data.columns else "movieid"
        start_time = time.time()
        
        with col.batch.dynamic() as batch:
            for _, row in update_data.iterrows():
                target_uuid = generate_uuid5(int(row[id_col]))
                
                batch.add_object(
                    uuid=target_uuid,
                    properties={property_name: row[property_name]}
                )
                
        return time.time() - start_time

    def cal_recall(self, weaviate_result, ground_truth):
        if not ground_truth: return 0.0
        
        gt_uuids = set()
        for tid in ground_truth:
            gt_uuids.add(generate_uuid5(str(tid)))
        
        pred_uuids = set()
        for obj in weaviate_result:
            if hasattr(obj, 'uuid'):
                pred_uuids.add(str(obj.uuid))
            elif isinstance(obj, dict) and '_additional' in obj:
                pred_uuids.add(obj['_additional']['id'])

        if not gt_uuids: return 0.0
        return len(gt_uuids & pred_uuids) / len(gt_uuids)
 
    def close(self):
        if hasattr(self, 'client'):
            self.client.close()


    