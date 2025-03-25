# import uuid
# import numpy as np
# from pymilvus import (connections,FieldSchema, CollectionSchema, DataType,Collection)
# collection_name = "my_table"
# host = "192.168.114.130"
# port = 19530
# username = ""
# password = ""
# num_entities, dim = 1000, 128
# total_num = 3000
#
# def generate_uuids(number_of_uuids):
#     uuids = [str(uuid.uuid4()) for _ in range(number_of_uuids)]
#     return uuids
#
# print("start connecting to Milvus")
# connections.connect("default", host=host,
# port=port,user=username,password=password)
# fields = [
#         FieldSchema(name="movieid", dtype=DataType.VARCHAR, max_length=50, is_primary=True, auto_id=False),
#         FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=100),
#         FieldSchema(name="time", dtype=DataType.INT16),
#         FieldSchema(name="genre", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=20, max_capacity=10),
#         FieldSchema(name="release_time", dtype=DataType.VARCHAR, max_length=50),
#         FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=20),
#         FieldSchema(name="directors", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=20, max_capacity=3),
#         FieldSchema(name="writers", dtype=DataType.ARRAY, element_type=DataType.VARCHAR,max_length=20, max_capacity=3),
#         FieldSchema(name="stars", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_length=20, max_capacity=3),
#         FieldSchema(name="intro", dtype=DataType.VARCHAR, max_length=300),
#         FieldSchema(name="image_features", dtype=DataType.FLOAT_VECTOR, dim=512),
#         FieldSchema(name="text_features", dtype=DataType.FLOAT_VECTOR, dim=1024)
#     ]
#
# schema = CollectionSchema(fields, "vec_benchmark with milvus")
# print("Create collection `my_table`")
# coll = Collection(collection_name, schema, consistency_level="Bounded")
# print("Start inserting entities")
# rng = np.random.default_rng(seed=19530)
# entities = [
#             [i for i in range(num_entities)],
#             rng.random(num_entities).tolist(),
#             generate_uuids(num_entities),
#             rng.random((num_entities, dim)),
#         ]
# insert_result = coll.insert(entities)
# print("Start flush")
# coll.flush()
# print("done")


# # 连接数据库并进行向量查询
# from pymilvus import connections, Collection
# import numpy as np
# import ast, json, time
#
# # 连接Milvus数据库，这里假设你的Milvus服务运行在本地，端口是19530，根据实际情况修改地址和端口
# connections.connect(
#             host="192.168.114.130",
#             port=19530
#         )
# collection = Collection('my_table')
#
# # 先从数据库中查询 movieid 为t1对应的image_features向量
# try:
#     filter_expr = "movieid == 't1'"
#     result = collection.query(
#         expr="movieid == 't1'" ,
#         output_fields=["text_features"]
#     )
#     print(result)
#     if result:
#         query_vector = np.array(result[0]["text_features"]).astype(np.float32)
#     else:
#         print("未找到movieid为t1对应的记录，无法获取image_features向量。")
#         query_vector = None
# except Exception as e:
#     print(f"查询movieid为t1的image_features向量时出现错误: {e}")
#     query_vector = None
#
# execution_times = []
# execution_recalls = []
# if query_vector is not None:
#     for i in range(10):
#         try:
#             start_time = time.time()
#             res = collection.search(
#                 data=[query_vector],
#                 anns_field='text_features',
#                 param={
#                     "metric_type": "L2",
#                     "params": {"ef": 50},
#                 },
#                 limit=50,
#                 expr='ARRAY_CONTAINS(genre, "Adventure")',
#                 output_fields=["movieid"]
#             )
#             end_time = time.time()
#             execution_time = end_time - start_time
#             print(execution_time)
#             execution_times.append(execution_time)
#         except Exception as e:
#             print(f"执行搜索操作时出现错误: {e}")
#
# def calculate_recall(ground_truth, milvus_result):
#     """
#     计算召回率的函数
#     """
#     # 处理ground_truth，提取关键信息，这里假设关键信息是电影名称
#     gt_movies = set()
#     for item in ground_truth[2]["result"]:
#         gt_movies.add(item[0])
#
#     # 处理milvus查询结果，提取关键信息（这里从复杂字符串中提取电影名称）
#     milvus_movies = set()
#     for entity_str in milvus_result[0]:
#         # print(entity_str)
#         # print(type(entity_str))
#         # try:
#         #     entity_dict = ast.literal_eval(entity_str)  # 将字符串转换为字典
#         #     if isinstance(entity_dict, dict):
#         movie_name = entity_str.id
#         if movie_name:
#             milvus_movies.add(movie_name)
#         # except (SyntaxError, ValueError):
#         #     print("error")
#         #     continue
#     print(gt_movies)
#     print(milvus_movies)
#     # 计算召回率
#     correct_count = len(gt_movies & milvus_movies)
#     recall = correct_count / len(gt_movies) if gt_movies else 0
#     return recall
#
# with open('./data/ground_truth.json', 'r') as file:  # 将'ground_truth.json'替换为实际的文件名
#     ground_truth = json.load(file)
# recall = calculate_recall(ground_truth, res)
# print(recall)
# execution_recalls.append(recall)
#
# avg_execution_time = np.mean(execution_times)
# avg_execution_recall = np.mean(execution_recalls)
# print("*"*10)
# print(avg_execution_time)
# print(avg_execution_recall)

import csv
def clean_csv_data(input_file_path, output_file_path):
    """
    读取输入的CSV文件，清理数据（删除image_features或text_features为空的记录），并将清理后的数据写入新的CSV文件

    参数:
    input_file_path (str): 输入的CSV文件路径
    output_file_path (str): 输出的CSV文件路径，用于保存清理后的数据
    """
    cleaned_data = []
    i=0
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # 获取列名，后续用来判断列是否存在以及按列名访问数据
        headers = reader.fieldnames
        for row in reader:
            image_features = row.get('image_features')
            text_features = row.get('text_features')
            # 判断两个字段都不为空（这里简单地以非空字符串来判断，可根据实际情况调整判断逻辑，比如如果是列表等结构，判断元素个数等）
            if image_features and image_features != "NULL" and text_features and text_features != "NULL":
                cleaned_data.append(row)

    with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(cleaned_data)


# 使用示例，替换为实际的文件路径
input_file_path = './data/movies_100000.csv'
output_file_path = './data/clean_movies_100000.csv'
clean_csv_data(input_file_path, output_file_path)