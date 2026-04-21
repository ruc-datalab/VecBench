import time
import ast
import io
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from databases.base_vd import BaseVD

class PGVector(BaseVD):
    """Implementation of the BaseVD interface for PG+pgvector vector database."""

    def connect(self):
        conn_params = self.config
        self.connection = psycopg2.connect(
            host=conn_params["host"],
            port=conn_params["port"],
            dbname=conn_params["dbname"],
            user=conn_params["user"],
            password=conn_params["password"],
        )
        self.cursor = self.connection.cursor()

    def create_table(self, db, schema):
        table_schema = schema.get(db)
        table_name = table_schema.get("table_name")
        if not table_schema:
            raise ValueError(f"Table {table_name} not found in schema.")

        fields_sql = []
        for field in table_schema.get('columns', []):
            field_name = field.get("name")
            field_type = field.get("type")
            field_constraints = field.get("constraints", "")
            dimension = field.get("dimension", None)

            if field_type.lower() == "vector":
                fields_sql.append(f"{field_name} {field_type}({dimension}) {field_constraints}")
            else:
                fields_sql.append(f"{field_name} {field_type} {field_constraints}")

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(fields_sql)}
            );
        """
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def process_data(self, data, db, schema):
        def _parse_data(value):
            try:
                return value if pd.notna(value) else None
            except Exception:
                return None

        def _parse_array(value):
            try:
                return ast.literal_eval(value) if pd.notna(value) else None
            except Exception:
                return None

        def _to_pg_array(val):
            if isinstance(val, list):
                return '{' + ','.join(f'"{str(v)}"' for v in val) + '}'
            return val

        columns = schema.get(db, {}).get("columns", [])
        for field in columns:
            name = field.get("name")
            ftype = field.get("type", "").lower()

            if ftype in ("text", "date"):
                data[name] = data[name].apply(_parse_data)
            elif ftype == "text[]":
                data[name] = data[name].apply(_parse_array).apply(_to_pg_array)
            elif ftype == "integer[]":
                data[name] = data[name].apply(_to_pg_array)
            elif ftype == "vector":
                data[name] = data[name].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
        return data


    def insert_data(self, data, db, schema, batch_size=100_000):
        schema = schema.get(db)
        columns = schema.get("columns", [])
        column_names = [col["name"] for col in columns]

        if "id" in column_names and "id" not in data.columns:
            data = data.copy()
            data["id"] = range(1, len(data) + 1)

        # print(column_names)

        total_rows = len(data)
        num_batches = (total_rows + batch_size - 1) // batch_size
        latency = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_rows)
            batch_data = data.iloc[start:end]

            buffer = io.StringIO()
            batch_data[column_names].to_csv(buffer, index=False, header=False)
            buffer.seek(0)

            try:
                start_time = time.time()
                self.cursor.copy_expert(
                    f"COPY my_table ({', '.join(column_names)}) FROM STDIN WITH (FORMAT CSV)",
                    buffer,
                )
                end_time = time.time()
                latency.append(end_time - start_time)
                self.connection.commit()
            except Exception as e:
                print(f"Batch {batch_idx + 1} insertion failed: {e}")
                raise

        return np.sum(latency)

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
                param_list = [
                    f"{key} = {value}" for param in params for key, value in param.items()
                ]
                param_str = f" with ({', '.join(param_list)})"

            create_index_sql = f"""
                SET maintenance_work_mem = '4GB';
                CREATE INDEX ON my_table USING {index_type} ({index_column} {distance}) {param_str};
            """
            print(create_index_sql)

            start_time = time.time()
            self.cursor.execute(create_index_sql)
            end_time = time.time()
            execution_time = end_time - start_time

            results.append(
                {
                    "index_column": index_column,
                    "index_type": index_type,
                    "create_time": execution_time,
                }
            )
        self.connection.commit()
        print(results)

    def create_scalar_index(self, config=None, args=None):
        if not config or "scalar_index" not in config:
            print("No scalar index configuration found.")
            return

        index_params = config["scalar_index"]
        results = []

        for index_param in index_params:
            index_column = index_param["index_column"]
            index_type = index_param["type"]

            index_name = f"{index_column}_{index_type.lower()}"
            sql = f"CREATE INDEX {index_name} ON my_table USING {index_type}({index_column});"
            print(sql)

            start_time = time.time()
            self.cursor.execute(sql)
            end_time = time.time()

            results.append(
                {
                    "index_column": index_column,
                    "index_type": index_type,
                    "create_time": end_time - start_time,
                }
            )

        self.connection.commit()
        print(results)

    def query(self, query_config, param=None):
        vector_field = query_config["vector_field"]
        reference_vector_name = query_config["reference_vector_name"]
        scalar_filters = query_config["scalar_filters"]
        limit = query_config["limit"]

        scalar_conditions = []
        for filter in scalar_filters:
            field = filter["field"]
            operator = filter["operator"]
            value = filter["value"]
            logic = filter.get("logic", "and")

            if operator == "==":
                condition = f"{field} = {value}"
            elif operator == "<":
                condition = f"{field} < '{value}'"
            elif operator == "<=":
                condition = f"{field} <= '{value}'"
            elif operator == ">":
                condition = f"{field} > '{value}'"
            elif operator == ">=":
                condition = f"{field} >= '{value}'"
            elif operator == "like":
                condition = f"{field} like '%{value}%'"
            elif operator == "contains":
                condition = f"'{value}' = any({field})"
            else:
                continue

            scalar_conditions.append((condition, logic))

        if scalar_conditions:
            scalar_condition_str = scalar_conditions[0][0]
            for condition, logic in scalar_conditions[1:]:
                scalar_condition_str += f" {logic} {condition}"
            scalar_condition_str = f"WHERE {scalar_condition_str}"
        else:
            scalar_condition_str = ""
        # SET ivfflat.probes = {param};
        # SET hnsw.ef_search = {param};
        # SET ivfflat.iterative_scan = relaxed_order;
        # SET hnsw.iterative_scan = strict_order;
        query = f"""
            SET hnsw.ef_search = {param};
            SELECT
                id
            FROM
                my_table
            {scalar_condition_str}
            ORDER BY
                ({vector_field} <-> (SELECT {vector_field} FROM my_table WHERE id = {reference_vector_name}))
            LIMIT {limit};
        """
        # print(query)
        result, execution_time = None, None
        start_time = time.time()
        try:
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            end_time = time.time()
            execution_time = end_time - start_time
        except Exception as e:
            print(f"Error executing query: {e}")

        return result, execution_time

    def update(self, batch_df, batch_size = 10000):
        data_list = [
            (int(row.id), str(row.image_vec), int(row.equal)) 
            for row in batch_df.itertuples(index=False)
        ]

        sql = """
        INSERT INTO my_table (id, image_vec, equal)
        VALUES %s
        ON CONFLICT (id) DO UPDATE SET 
            image_vec = EXCLUDED.image_vec,
            equal = EXCLUDED.equal;
        """

        total_update_time = 0

        with self.connection.cursor() as cursor:
            try:
                for i in range(0, len(data_list), batch_size):
                    sub_list = data_list[i : i + batch_size]
                    
                    sub_start = time.time()
                    psycopg2.extras.execute_values(
                        cursor, sql, sub_list, 
                        template=None, page_size=1000
                    )
                    self.connection.commit() 
                    total_update_time += (time.time() - sub_start)
                    
            except Exception as e:
                self.connection.rollback()
                print(f"pgvector update error: {e}")
                raise e
        
        return total_update_time

    def drop_indexes(self, config, args):
        index_type = args.algorithm
        index_params = config[index_type]
        for index_param in index_params:
            index_column = index_param["index_column"]
            sql = f'drop index "my_table_{index_column}_idx";'
            try:
                self.sql(sql)
            except Exception as e:
                print(f"Error executing SQL: {e}")

    def cal_recall(self, res, gt):
        def _flatten(lst):
            return [e for el in lst for e in (_flatten(el) if isinstance(el, list) else [el])]
        res_flat = [_flatten(item) for item in res]
        gt_set = set(gt)
        res_set = {item[0] for item in res_flat}
        correct = len(gt_set & res_set)
        return correct / len(gt_set) if gt_set else 0.0


    def delete_by_ids(self, selected_ids):
        id_list = ",".join(map(str, selected_ids))
        start_time = time.time()
        self.sql(f"DELETE FROM my_table WHERE id IN ({id_list});")
        return time.time() - start_time

    def close(self):
        try:
            if hasattr(self, "cursor") and self.cursor:
                self.cursor.close()
        except Exception as e:
            print(f"Error closing cursor: {e}")

        try:
            if hasattr(self, "connection") and self.connection:
                self.connection.close()
        except Exception as e:
            print(f"Error closing connection: {e}")

        self.cursor = None
        self.connection = None

    def sql(self, sql):
        self.cursor.execute(sql)
        self.connection.commit()

    def sql_res(self, sql):
        self.cursor.execute(sql)
        return self.cursor.fetchall()
    
    def get_index_usage(self, index_name):
        self.cursor.execute("SELECT pg_stat_clear_snapshot();")

        query = f"""
        SELECT idx_scan, idx_tup_read 
        FROM pg_stat_user_indexes 
        WHERE indexrelname = '{index_name}';
        """
        self.cursor.execute(query)
        result = self.cursor.fetchone()

        self.connection.commit()
        return result[0] if result else 0