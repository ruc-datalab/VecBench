import pymilvus
from pymilvus import Collection, connections, DataType, FieldSchema, CollectionSchema
import time
from databases.base_vd import BaseVD
from utils import parse_data, parse_vector, parse_array, convert_dates
import numpy as np
import pandas as pd

class Milvus(BaseVD):
    def connect(self):
        """建立与 Milvus 的连接"""
        conn_params = self.config[self.db_type]
        connections.connect(
            alias="default",
            host=conn_params["host"],
            port=conn_params["port"]
        )

    def create_table(self, db, schema):
        """创建一个 Milvus 集合"""
        table_schema = schema.get(db)
        table_name = table_schema.get('table_name')
        if not table_name:
            raise ValueError(f"Table {table_name} not found in schema.")
        fields = []
        for field in table_schema.get('columns', []):
            field_name = field.get('name')
            field_type = field.get('type')
            field_description = field.get('description', '')

            # 根据不同类型准确构建FieldSchema对象
            if field_type == 'VARCHAR':
                max_length = field.get('max_length', 50)  # 设置默认最大长度为50，如果yaml中有定义则使用定义的值
                is_primary = field.get('primary', False)
                field_obj = FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=max_length,
                                        description=field_description, is_primary=is_primary)
            elif field_type == 'INT16':
                field_obj = FieldSchema(name=field_name, dtype=DataType.INT16, description=field_description)
            elif field_type == 'INT64':
                field_obj = FieldSchema(name=field_name, dtype=DataType.INT64, description=field_description)
            elif field_type == 'ARRAY':
                element_type = field.get('element_type')  # 设置默认元素类型为VARCHAR，如果yaml中有定义则使用定义的值
                if element_type == 'VARCHAR':
                    element_type = DataType.VARCHAR
                else:
                    element_type = DataType.INT32
                max_length = field.get('max_length', 20)  # 设置默认最大元素长度为20
                max_capacity = field.get('max_capacity', 10)  # 设置默认最大容量为10
                field_obj = FieldSchema(name=field_name, dtype=DataType.ARRAY, element_type=element_type,
                                        max_length=max_length, max_capacity=max_capacity,
                                        description=field_description)
            elif field_type == 'FLOAT_VECTOR':
                dim = field.get('dimension')
                if dim is None:
                    raise ValueError(f"Dimension is not specified for float_vector field {field_name}")
                field_obj = FieldSchema(name=field_name, dtype=DataType.FLOAT_VECTOR, dim=dim,
                                        description=field_description)
            else:
                raise ValueError(f"Unsupported field type: {field_type} for field {field_name}")
            # print(field_obj)
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
        """
        对数据进行预处理，比如类型转换等操作，使其更符合插入Milvus的要求
        """
        schema = schema.get(db)
        columns = schema.get('columns', [])
        column_names = [col['name'] for col in columns]
        # 确定每列的数据类型，用于后续数据整理
        column_types = {col['name']: col['type'] for col in columns}
        for column_name in column_names:
            col_data = data[column_name]
            col_type = column_types[column_name]
            if col_type == 'INT16':
                data[column_name] = col_data.astype(np.int16)
            elif col_type == 'INT64':
                data[column_name] = col_data.astype(np.int64)
            elif col_type == 'VARCHAR':
                data[column_name] = col_data.fillna("").astype(str)
            elif col_type == 'ARRAY':
                processed_col_data = []
                for element in col_data:
                    if element is None or pd.isnull(element):  # 考虑元素是None或者是pandas里的空值（比如NaN等情况）
                        processed_col_data.append([])
                    elif isinstance(element, str):
                        try:
                            element = element.strip("[]").split(',')
                            element = [x.strip().strip("'") for x in element if x.strip().strip("'")]  # 去除空字符串元素
                            processed_col_data.append(element)
                        except:
                            print(f"Error parsing element in {column_name} as list: {element}")
                            raise  # 抛出异常，方便查看具体哪个元素格式有问题
                    else:
                        if isinstance(element, (list, tuple)):  # 再次确认元素本身是否可迭代结构
                            processed_col_data.append(list(element))
                        else:
                            print(f"Element in {column_name} is not in valid iterable format: {element}")
                            raise ValueError(f"Invalid element format for {column_name} in ARRAY type.")
                data[column_name] = processed_col_data
            elif col_type == 'FLOAT_VECTOR':
                processed_col_data = []
                for element in col_data:
                    if element is None or pd.isnull(element) or element == []:  # 处理向量数据为空（None、pandas空值或空列表情况）
                        processed_col_data.append([])
                    else:
                        # 增加数据清洗步骤，去除多余字符，确保元素符合浮点型转换要求
                        clean_element = element.strip("[]").replace("'", "").split(',')
                        try:
                            float_vector = [float(x.strip()) for x in clean_element]
                            processed_col_data.append(float_vector)
                        except ValueError:
                            print(f"Element in {column_name} cannot be converted to float vector: {element}")
                            raise
                data[column_name] = processed_col_data
        return data


    def insert_data(self, data, db, schema):
        """插入数据到 Milvus"""
        schema = schema.get(db)
        columns = schema.get('columns', [])
        # column_names = [col['name'] for col in columns]
        # column_types = {col['name']: col['type'] for col in columns}
        i=-1
        data = data.to_dict('records')
        for record in data:
            i=i+1
            print(record)
            try:
                # 单条插入数据到collection
                self.collection.insert(record)
                print(f"Inserting row {i} successfully")
            except Exception as e:
                print(f"Error inserting a single record of data: {e}")
                raise

    def create_index(self, config, args):
        self.collection = Collection('my_table')
        """创建索引"""
        index_type = args.algorithm
        index_params = config[index_type]
        results = []
        for index_param in index_params:
            index_column = index_param["index_column"]
            distance = index_param["distance"]
            params = index_param["params"]
            if(index_type=='hnsw'):
                param = {
                    "index_type": index_type.upper(),
                    "metric_type": distance,
                    "params": {
                        "M": params[0]["m"],
                        "efConstruction": params[0]["ef_construction"]
                    }
                }
            elif(index_type=='ivfflat'):
                param = {
                    "index_type": 'IVF_FLAT',
                    "metric_type": distance,
                    "params": {
                        "nlist": params[0]["nlist"],
                    }
                }
            start_time = time.time()
            self.collection.create_index(
                field_name=index_column,
                index_params=param,
                index_name=f"my_table_{index_column}_idx"
            )
            end_time = time.time()
            execution_time = end_time - start_time
            results.append({"index_column": index_column,
                            "index_type": index_type,
                            "create_time": execution_time})
        print(results)
        self.collection.load()

    def query(self, query_config):
        """执行查询"""
        result = None
        execution_time = None
        # start_time = time.time()
        try:
            data = query_config.get('data', '')
            param = query_config.get('param', {})
            limit = query_config.get('limit', None)
            expr = query_config.get('expr', '')
            output_fields = query_config.get('output_fields', [])
            anns_field = query_config.get('anns_field', '')

            start_time = time.time()
            vector = self.collection.query(
                expr=f"movieid == '{data}'",
                output_fields=[anns_field]
            )
            query_vector = np.array(vector[0][anns_field]).astype(np.float32)
            query_result = self.collection.search(
                data=[query_vector],
                anns_field=anns_field,
                param=param,
                limit=limit,
                expr=expr,
                output_fields=output_fields
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
            print(f"drop index {index_type} on {index_column}")
            self.collection.drop_index(index_name=f"my_table_{index_column}_idx")

    def cal_recall(self, milvus_result, ground_truth):
        gt_movies = set()
        for item in ground_truth:
            gt_movies.add(item[0])
        # 处理milvus查询结果，提取关键信息（这里只提取id进行比较）
        milvus_movies = set()
        for entity_str in milvus_result[0]:
            movie_name = entity_str.id
            if movie_name:
                milvus_movies.add(movie_name)
        correct_count = len(gt_movies & milvus_movies)
        recall = correct_count / len(gt_movies) if gt_movies else 0
        return recall