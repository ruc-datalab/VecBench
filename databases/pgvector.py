import psycopg2
from utils import parse_data,  parse_array, convert_dates, flatten
from databases.base_vd import BaseVD
import time
import numpy as np

def cal_recall(res, gt):
    res = convert_dates(res)
    gt_set = set(tuple(item) for item in gt)
    tp = 0
    for item in res:
        if tuple(item) in gt_set:
            tp += 1
    fn = len(gt)
    recall = tp / fn if fn > 0 else 0
    return recall

class PGVector(BaseVD):
    def connect(self):
        conn_params = self.config[self.db_type]
        self.connection = psycopg2.connect(
            host=conn_params["host"],
            port=conn_params["port"],
            dbname=conn_params["dbname"],
            user=conn_params["user"],
            password=conn_params["password"]
        )
        self.cursor = self.connection.cursor()

    def create_table(self, db, schema):
        """ 创建一个table"""
        table_schema = schema.get(db)
        table_name = table_schema.get('table_name')
        if not table_schema:
            raise ValueError(f"Table {table_name} not found in schema.")
        columns = table_schema.get('columns', [])
        fields_sql = []
        for field in columns:
            field_name = field.get('name')
            field_type = field.get('type')
            field_constraints = field.get('constraints', '')
            dimension = field.get('dimension', None)
            # print(f"{field_type}")
            if field_type.lower() == 'vector':
                # 对于向量类型字段，添加维度信息
                fields_sql.append(f"{field_name} {field_type}({dimension}) {field_constraints}")
            else:
                # 其他字段类型（如 TEXT, INT, DATE, TEXT[] 等）
                fields_sql.append(f"{field_name} {field_type} {field_constraints}")
        # print(fields_sql)
        # 拼接最终的 CREATE TABLE SQL
        create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(fields_sql)}
                );
                """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def process_data(self, data, db, schema):
        columns = schema.get(db).get('columns', [])
        for field in columns:
            field_name = field.get('name')
            field_type = field.get('type')
            if field_type.lower() == 'vector' or field_type.lower() == 'text' or field_type.lower() == 'date':
                data[field_name] = data[field_name].apply(parse_data)
            elif field_type.lower() == 'text[]':
                data[field_name] = data[field_name].apply(parse_array)
        return data


    def insert_data(self, data, db, schema):
        # 提取表的列信息
        schema = schema.get(db)
        # table_schema = schema.get('my_table', {})
        columns = schema.get('columns', [])
        column_names = [col['name'] for col in columns ]

        # 构建插入语句
        column_list = ", ".join(column_names)  # 列名字符串
        value_placeholders = ", ".join(["%s"] * len(column_names))  # 对应的占位符
        insert_sql = f"""
        INSERT INTO my_table ({column_list}) 
        VALUES ({value_placeholders});
        """

        # 遍历数据并执行插入
        for i, row in data.iterrows():
            try:
                # 按列名顺序构建 values
                values = tuple(row[col['name']] for col in columns if col['name'] in column_names)
                self.cursor.execute(insert_sql, values)
                print(f"Inserting row {i} successfully")
            except Exception as e:
                print(f"Error inserting row {i}: {values}")
                print(f"Exception: {e}")
                raise  # 抛出异常以终止插入并定位问题
        self.connection.commit()

    def create_index(self, config, args):
        index_type = args.algorithm
        index_params = config[index_type]
        results = []
        for index_param in index_params:
            index_column = index_param["index_column"]
            distance = index_param["distance"]
            params = index_param["params"]
            param_str = ""
            if params:
                param_list = []
                for param in params:
                    for key, value in param.items():
                        param_list.append(f"{key} = {value}")
                param_str = f" with ({', '.join(param_list)})"
            create_index_sql = f"""
                CREATE INDEX ON my_table USING {index_type} ({index_column} {distance}) {param_str};
            """
            print(create_index_sql)
            start_time = time.time()
            self.cursor.execute(create_index_sql)
            end_time = time.time()
            execution_time = end_time - start_time
            results.append({"index_column": index_column,
                            "index_index_type": index_type,
                            "create_time": execution_time})
        print(results)
        self.connection.commit()

    def query(self, query):
        result = None
        execution_time = None
        start_time = time.time()
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            end_time = time.time()
            execution_time = end_time - start_time
        except Exception as e:
            print(f"Error executing query: {e}")
        return result, execution_time

    def cal_recall(self, res, gt):
        gt = [flatten(item) for item in gt]
        res = convert_dates(res)
        res = [flatten(item) for item in res]
        gt_set = set(tuple(item) for item in gt)
        tp = 0
        for item in res:
            if tuple(item) in gt_set:
                tp += 1
        fn = len(gt)
        recall = tp / fn if fn > 0 else 0
        return recall

    def sql(self, sql):
        self.cursor.execute(sql)
        self.connection.commit()

    def sql_res(self, sql):
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result

    def drop_indexes(self, config, args):
        index_type = args.algorithm
        index_params = config[index_type]
        for index_param in index_params:
            index_column = index_param["index_column"]
            sql = f'drop index "my_table_{index_column}_idx";'
            try:
                self.sql(sql)
            except Exception as e:
                print(f"Error executing sql: {e}")
