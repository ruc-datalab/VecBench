import numpy as np
import pandas as pd
import time
from qdrant_client import QdrantClient, models
from qdrant_client.http import models as qmodels
from databases.base_vd import BaseVD


class Qdrant(BaseVD):
    """Implementation of the BaseVD interface for Qdrant vector database."""

    def connect(self) -> None:
        conn_params = self.config
        self.client = QdrantClient(
            host=conn_params["host"],
            port=conn_params["port"],
            timeout=300
        )
        self.collection_name = "my_table"

    def create_table(self, db: str, schema: dict, index: dict) -> None:
        table_schema = schema.get(db)
        collection_name = table_schema.get("table_name")
        columns = table_schema.get("columns", [])
        hnsw_config = table_schema.get("hnsw_config", {})

        vector_fields = [col for col in columns if col["type"] == "FLOAT_VECTOR"]
        if not vector_fields:
            raise ValueError("Qdrant requires at least one vector field.")

        vectors_config = {}
        for vf in vector_fields:
            name = vf["name"]
            dim = vf["dimension"]
            distance_str = (vf.get("description") or "Euclid").strip().lower()
            if distance_str == "euclid":
                distance = qmodels.Distance.EUCLID
            elif distance_str == "dot":
                distance = qmodels.Distance.DOT
            elif distance_str == "cosine":
                distance = qmodels.Distance.COSINE
            elif distance_str == "manhattan":
                distance = qmodels.Distance.MANHATTAN
            else:
                raise ValueError(f"Unsupported distance metric: {distance_str}")
            vectors_config[name] = qmodels.VectorParams(size=dim, distance=distance)

        payload_schema = {}
        for col in columns:
            col_type = col["type"]
            if col_type == "FLOAT_VECTOR":
                continue
            if col_type in ["integer", "keyword", "bool", "datetime"]:
                payload_schema[col["name"]] = getattr(qmodels.PayloadSchemaType, col_type.upper())

        hnsw_params = qmodels.HnswConfigDiff(**hnsw_config) if hnsw_config else None

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_params
        )
        self.collection_name = collection_name

        self.hnswp = False
        print("Whether construct filterable HNSW with payload index(yes/no):")
        process = input().strip().lower() == 'yes'
        if process:
            self.create_scalar_index(index[self.db_type])
            self.hnswp = True

    def process_data(self, data: pd.DataFrame, db: str, schema: dict) -> pd.DataFrame:
        schema = schema.get(db)
        columns = schema.get("columns", [])
        column_types = {col["name"]: col["type"] for col in columns}

        for name, type_ in column_types.items():
            if type_ == "integer":
                data[name] = pd.to_numeric(data[name], errors="coerce").fillna(0).astype(np.int64)
            elif type_ == "float":
                data[name] = pd.to_numeric(data[name], errors="coerce").fillna(0).astype(np.float32)
            elif type_ == "datetime":
                data[name] = pd.to_datetime(data[name], errors="coerce")
            elif type_ == "FLOAT_VECTOR":
                data[name] = data[name].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        return data

    def insert_data(self, data: pd.DataFrame, db: str, schema: dict, batch_size: int = 1000) -> None:
        schema = schema.get(db)
        columns = schema.get("columns", [])
        vector_cols = [col["name"] for col in columns if col["type"] == "FLOAT_VECTOR"]
        id_col = "id" if "id" in data.columns else "movieid"

        ids = data[id_col].astype(int).tolist()

        if len(vector_cols) == 1:
            vcol = vector_cols[0]
            vectors = [{vcol: np.array(v, dtype=np.float32).tolist()} for v in data[vcol]]
        else:
            vectors = [
                {vcol: np.array(row[vcol], dtype=np.float32).tolist() for vcol in vector_cols}
                for _, row in data.iterrows()
            ]

        payload = data.drop(columns=vector_cols + [id_col]).to_dict(orient="records")
        start_time = time.time()
        try:
            self.client.upload_collection(
                collection_name=self.collection_name,
                ids=ids,
                payload=payload,
                vectors=vectors,
                batch_size=batch_size,
                parallel=4
            )
        except Exception as e:
            print("Qdrant upload error:", e)
            raise
        return time.time() - start_time

    def create_index(self, config: dict, args: any) -> None:
        print("Qdrant uses HNSW by default; explicit index creation is not required.")

    def create_scalar_index(self, config: dict = None, args: any = None):
        if not config or "scalar_index" not in config:
            print("No scalar index configuration found.")
            return

        index_params = config["scalar_index"]
        results = []

        for index_param in index_params:
            field_name = index_param["index_column"]
            field_type = index_param.get("data_type", "keyword")

            start_time = time.time()
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            end_time = time.time()

            results.append({
                "index_column": field_name,
                "data_type": field_type,
                "create_time": end_time - start_time
            })
        print(results)


    def query(self, query_config: dict, param: int):
        self.collection_name = "my_table"
        vector_field = query_config["vector_field"]
        reference_vector_id = int(query_config["reference_vector_name"])
        scalar_filters = query_config["scalar_filters"]
        limit = query_config["limit"]

        vector_result = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[reference_vector_id],
            with_vectors=True
        )

        if not vector_result:
            raise ValueError(f"Vector with ID {reference_vector_id} not found.")
        query_vector = vector_result[0].vector[vector_field]

        must, should = [], []
        for cond in scalar_filters:
            field = cond["field"]
            op = cond["operator"]
            val = cond["value"]
            logic = cond.get("logic", "and")

            condition = None
            if pd.isna(val) or val == "nan" or val == []:
                condition = qmodels.IsEmptyCondition(is_empty=qmodels.PayloadField(key=field))
            elif op == "==":
                condition = qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=val))
            elif op == "like":
                condition = qmodels.FieldCondition(key=field, match=qmodels.MatchText(text=val))
            elif op == "contains":
                condition = qmodels.FieldCondition(key=field, match=qmodels.MatchValue(value=val))
            elif op in (">", ">=", "<", "<="):
                range_kwargs = {"gt": None, "gte": None, "lt": None, "lte": None}
                if op == ">":
                    range_kwargs["gt"] = val
                elif op == ">=":
                    range_kwargs["gte"] = val
                elif op == "<":
                    range_kwargs["lt"] = val
                elif op == "<=":
                    range_kwargs["lte"] = val
                if isinstance(val, str):
                    condition = qmodels.FieldCondition(key=field, range=qmodels.DatetimeRange(**range_kwargs))
                else:
                    condition = qmodels.FieldCondition(key=field, range=qmodels.Range(**range_kwargs))

            if condition:
                if logic.lower() == "or":
                    should.append(condition)
                else:
                    must.append(condition)

        qfilter = qmodels.Filter(must=must or None, should=should or None) if (must or should) else None

        start = time.time()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=(vector_field, query_vector),
            limit=limit,
            with_payload=True,
            search_params=qmodels.SearchParams(hnsw_ef=param, exact=False),
            query_filter=qfilter
        )
        end = time.time()
        return results, end - start

    def delete_by_ids(self, selected_ids: list[int]) -> float:
        start_time = time.time()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=selected_ids)
        )
        return time.time() - start_time

    def update(self, batch_df, batch_size=10000):
        ids = batch_df['id'].tolist()
        vectors_array = batch_df['image_vec'].tolist() 
        payloads = batch_df[['equal', 'range', 'tags']].to_dict('records')

        total_api_time = 0
        vector_name = "image_vec"

        for i in range(0, len(ids), batch_size):
            sub_ids = ids[i : i + batch_size]
            sub_vectors = vectors_array[i : i + batch_size]
            sub_payloads = payloads[i : i + batch_size]
            
            sub_start = time.time()
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=sub_ids,
                    vectors={
                        vector_name: sub_vectors
                    },
                    payloads=sub_payloads
                ),
                wait=True 
            )
            
            total_api_time += (time.time() - sub_start)

        maintenance_start = time.time()
        
        while True:
            info = self.client.get_collection(self.collection_name)
            if info.status == models.CollectionStatus.GREEN:
                break
            time.sleep(1) # 每秒轮询一次
            
        maintenance_duration = time.time() - maintenance_start
        print(f"api_time:{total_api_time} flush_time:{maintenance_duration}")
        return total_api_time + maintenance_duration

    def overwrite_payload(self, property_name: str, update_data: pd.DataFrame) -> None:
        for _, row in update_data.iterrows():
            self.client.overwrite_payload(
                collection_name=self.collection_name,
                payload={property_name: row[property_name]},
                points=[row["id"]],
            )

    def cal_recall(self, qdrant_result: list, ground_truth: list[int]) -> float:
        gt_ids = set(ground_truth)
        pred_ids = {point.id for point in qdrant_result}
        return len(gt_ids & pred_ids) / len(gt_ids) if gt_ids else 0.0

    def update_table(self, db: str, schema: dict) -> None:
        table_schema = schema.get(db)
        collection_name = table_schema.get("table_name")

        start_time = time.time()
        self.client.update_collection(collection_name=collection_name)
        end_time = time.time()
        print(f"Collection '{collection_name}' updated in {end_time - start_time:.3f} seconds.")
        self.collection_name = collection_name

    def close(self) -> None:
        self.client = None

    def state(self):
        info = self.client.get_collection(self.collection_name)
        print(info)
        if info.status == models.CollectionStatus.GREEN:
            print("ready")