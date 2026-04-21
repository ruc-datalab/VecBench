import time
from datetime import datetime
import numpy as np
import pandas as pd
import pymilvus
from pymilvus import Collection, connections, DataType, FieldSchema, CollectionSchema
from databases.base_vd import BaseVD


class Milvus(BaseVD):
    """Implementation of the BaseVD interface for Milvus vector database."""

    def connect(self):
        conn_params = self.config
        connections.connect(
            alias="default",
            host=conn_params["host"],
            port=conn_params["port"]
        )

    def create_table(self, db, schema):
        table_schema = schema.get(db)
        table_name = table_schema.get('table_name')
        if not table_name:
            raise ValueError(f"Table {table_name} not found in schema.")

        fields = []
        for field in table_schema.get('columns', []):
            field_name = field.get('name')
            field_type = field.get('type')
            field_description = field.get('description', '')

            if field_type == 'VARCHAR':
                max_length = field.get('max_length', 50)
                is_primary = field.get('primary', False)
                field_obj = FieldSchema(
                    name=field_name,
                    dtype=DataType.VARCHAR,
                    max_length=max_length,
                    description=field_description,
                    is_primary=is_primary
                )
            elif field_type == 'INT16':
                field_obj = FieldSchema(name=field_name, dtype=DataType.INT16, description=field_description)
            elif field_type == 'INT32':
                field_obj = FieldSchema(name=field_name, dtype=DataType.INT32, description=field_description)
            elif field_type == 'INT64':
                is_primary = field.get('primary', False)
                field_obj = FieldSchema(
                    name=field_name,
                    dtype=DataType.INT64,
                    description=field_description,
                    is_primary=is_primary
                )
            elif field_type == 'FLOAT':
                field_obj = FieldSchema(name=field_name, dtype=DataType.FLOAT, description=field_description)
            elif field_type == 'ARRAY':
                element_type = field.get('element_type')
                if element_type == 'VARCHAR':
                    element_type = DataType.VARCHAR
                else:
                    element_type = DataType.INT32
                max_length = field.get('max_length', 20)
                max_capacity = field.get('max_capacity', 10)
                field_obj = FieldSchema(
                    name=field_name,
                    dtype=DataType.ARRAY,
                    element_type=element_type,
                    max_length=max_length,
                    max_capacity=max_capacity,
                    description=field_description
                )
            elif field_type == 'FLOAT_VECTOR':
                dim = field.get('dimension')
                if dim is None:
                    raise ValueError(f"Dimension is not specified for float_vector field {field_name}")
                field_obj = FieldSchema(
                    name=field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim,
                    description=field_description
                )
            else:
                raise ValueError(f"Unsupported field type: {field_type} for field {field_name}")

            fields.append(field_obj)

        collection_schema = CollectionSchema(
            fields=fields,
            description=f"Schema for {table_name} collection"
        )
        try:
            self.collection = Collection(name=table_name, schema=collection_schema, consistency_level="Strong")
        except pymilvus.exceptions.MilvusException as e:
            print(f"Failed to create collection {table_name} in Milvus. Error: {str(e)}")
            raise

    def process_data(self, data, db, schema):
        schema = schema.get(db)
        columns = schema.get('columns', [])
        column_names = [col['name'] for col in columns if col['name'] != 'movieid']
        column_types = {col['name']: col['type'] for col in columns}

        for column_name in column_names:
            col_data = data[column_name]
            col_type = column_types[column_name]
            if col_type == 'INT8':
                data[column_name] = col_data.astype(np.int8)
            elif col_type == 'INT16':
                data[column_name] = col_data.astype(np.int16)
            elif col_type == 'FLOAT':
                data[column_name] = col_data.astype(np.float32)
            elif col_type == 'INT32':
                if column_name == 'release_time':
                    data[column_name] = col_data.apply(
                        lambda x: int(datetime.strptime(str(x), "%Y-%m-%d").strftime("%Y%m%d"))
                        if pd.notna(x) else 0
                    ).astype(np.int32)
                else:
                    data[column_name] = col_data.astype(np.int32)
            elif col_type == 'INT64':
                data[column_name] = col_data.astype(np.int64)
            elif col_type == 'VARCHAR':
                data[column_name] = col_data.fillna("").astype(str)
        return data

    def insert_data(self, data, db, schema, batch_size=5000):
        self.collection = Collection("my_table")
        schema = schema.get(db)
        column_names = [col['name'] for col in schema.get('columns', [])]

        if 'id' in column_names and 'id' not in data.columns:
            data = data.copy()
            data['id'] = range(1, len(data) + 1)

        records = data[column_names].to_dict('records')
        total = len(records)
        latency = []
        try:
            for i in range(0, total, batch_size):
                batch = records[i:i + batch_size]
                start_time = time.time()
                self.collection.insert(batch)
                end_time = time.time()
                latency.append(end_time - start_time)
                # print(f"Inserted rows {i} to {min(i + batch_size, total)}")
        except Exception as e:
            print(f"Insertion failed: {e}")
            raise
        return np.sum(latency)

    def create_index(self, config, args):
        self.collection = Collection('my_table')
        index_type = args.algorithm
        index_params = config[index_type]
        results = []
        for index_param in index_params:
            index_column = index_param["index_column"]
            distance = index_param["distance"]
            params = index_param["params"]
            if index_type == 'hnsw':
                param = {
                    "index_type": index_type.upper(),
                    "metric_type": distance,
                    "params": {
                        "M": params[0]["m"],
                        "efConstruction": params[0]["ef_construction"]
                    }
                }
            elif index_type == 'diskann':
                param = {
                    "index_type": "DISKANN",
                    "metric_type": distance,
                    "params": {}
                }
            elif index_type == 'ivfflat':
                param = {
                    "index_type": 'IVF_FLAT',
                    "metric_type": distance,
                    "params": {"nlist": params[0]["nlist"]}
                }
            else:
                raise ValueError(f"Unsupported index type: {index_type}")

            start_time = time.time()
            self.collection.create_index(
                field_name=index_column,
                index_params=param,
                index_name=f"my_table_{index_column}_idx"
            )
            end_time = time.time()
            execution_time = end_time - start_time
            results.append({
                "index_column": index_column,
                "index_type": index_type,
                "create_time": execution_time
            })
        print(results)
        self.collection.load()

    def create_scalar_index(self, config=None, args=None):
        if not config or "scalar_index" not in config:
            print("No scalar index configuration found.")
            return

        self.collection = Collection('my_table')
        index_params = config["scalar_index"]
        results = []

        for index_param in index_params:
            index_column = index_param["index_column"]
            start_time = time.time()
            self.collection.create_index(field_name=index_column)
            end_time = time.time()
            results.append({
                "index_column": index_column,
                "index_type": "scalar",
                "create_time": end_time - start_time
            })
        print(results)

    def query(self, query_config, param=None):
        self.collection = Collection('my_table')
        vector_field = query_config['vector_field']
        reference_vector_name = query_config['reference_vector_name']
        scalar_filters = query_config['scalar_filters']
        limit = query_config['limit']

        scalar_conditions = []
        for filter in scalar_filters:
            field = filter['field']
            operator = filter['operator']
            value = filter['value']
            logic = filter.get('logic', 'and')

            if field == 'release_time':
                value = int(datetime.strptime(str(value), "%Y-%m-%d").strftime("%Y%m%d"))
            if operator == '==':
                condition = f"{field} == {value}"
            elif operator == '<':
                condition = f"{field} < {value}"
            elif operator == '<=':
                condition = f"{field} <= {value}"
            elif operator == '>':
                condition = f"{field} > {value}"
            elif operator == '>=':
                condition = f"{field} >= {value}"
            elif operator == 'like':
                condition = f"{field} like '%{value}%'"
            elif operator == 'contains':
                condition = f"ARRAY_CONTAINS({field}, {value})"
            else:
                raise ValueError(f"Unsupported operator: {operator}")
            scalar_conditions.append((condition, logic))

        if scalar_conditions:
            expr = scalar_conditions[0][0]
            for condition, logic in scalar_conditions[1:]:
                expr += f" {logic.lower()} {condition}"
        else:
            expr = " "

        query_config_milvus = {
            'data': reference_vector_name,
            'ann_field': vector_field,
            'param': {
                'metric_type': "L2",
                'params': {'ef': param, 'nprobe': param, 'search_list': param},

            },
            'limit': limit,
            'expr': expr,
            'output_fields': ["id"],
        }
                # "hints": "iterative_filter"

        result = None
        execution_time = None
        try:
            vector = self.collection.query(
                expr=f"id == {reference_vector_name}",
                output_fields=[vector_field]
            )
            query_vector = np.array(vector[0][vector_field]).astype(np.float32)

            start_time = time.time()
            query_result = self.collection.search(
                data=[query_vector],
                anns_field=vector_field,
                param=query_config_milvus['param'],
                limit=limit,
                expr=expr,
                output_fields=query_config_milvus['output_fields'],
                consistency_level="Bounded",
            )
            result = query_result
            end_time = time.time()
            execution_time = end_time - start_time
        except Exception as e:
            print(f"Error executing query: {e}")

        return result, execution_time

    def drop_indexes(self, config, args):
        self.collection.release()
        index_type = args.algorithm
        index_params = config[index_type]
        for index_param in index_params:
            index_column = index_param["index_column"]
            print(f"Dropping index {index_type} on {index_column}")
            self.collection.drop_index(index_name=f"my_table_{index_column}_idx")

    def update(self, batch_df, batch_size=10000):
        self.collection = Collection('my_table')
        data_list = batch_df.to_dict('records')

        total_api_time = 0
        
        for i in range(0, len(data_list), batch_size):
            sub_batch = data_list[i : i + batch_size]
            
            sub_start = time.time()
            # 使用 self.collection.upsert 
            self.collection.upsert(data=sub_batch)
            total_api_time += (time.time() - sub_start)

        flush_start = time.time()
        self.collection.flush() 
        self.collection.load()
        
        flush_duration = time.time() - flush_start

        print(f"api_time:{total_api_time} flush_time:{flush_duration}")
        return total_api_time + flush_duration

    def cal_recall(self, milvus_result, ground_truth):
        gt_set = set(ground_truth)
        res_set = set()
        for entity_str in milvus_result[0]:
            id = entity_str.id
            if id:
                res_set.add(id)
        correct_count = len(gt_set & res_set)
        recall = correct_count / len(gt_set) if gt_set else 0
        return recall

    def delete_by_ids(self, selected_ids):
        self.collection = Collection('my_table')
        expr = f"id in [{', '.join(map(str, selected_ids))}]"
        start_time = time.time()
        self.collection.delete(expr)
        duration = time.time() - start_time
        return duration

    def close(self):
        connections.disconnect(alias="default")
