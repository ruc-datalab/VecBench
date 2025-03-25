import argparse
from databases.pgvector import PGVector
from databases.milvus import Milvus
from utils import load_config, load_schema_from_yaml, load_query_from_yaml, save_ground_truth
from data_loader import load_data_from_csv
from envaluate import plot_query_metrics
import numpy as np

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--operation",
        help="init the database or run benchmark straightly",
        default='run',
    )
    parser.add_argument(
        "--database",
        help="the vec_database to benchmark",
        default='pgvector',
    )
    parser.add_argument(
        "--algorithm",
        help="the algorithm to be tested",
        default='ivfflat',
    )
    parser.add_argument(
        "--times",
        help="the times every single query runs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dataset",
        help="the scale of dataset ",
        default='IMDB_100000',
    )
    args = parser.parse_args()
    return args

def execute_save(db, queries):
    results = []
    for query in queries:
        print(query['query'])
        result, time = db.query(query['query'])
        results.append({'name': query['name'],
                        'result': result})
    save_ground_truth(results)
    return results

def execute(db, queries, ground_truth, args):
    results = []
    for query in queries:
        # 存储查询执行时间
        execution_times = []
        execution_recalls = []
        query_results = []
        gt = next((g for g in ground_truth if g['name'] == query['name']), None)
        for i in range(args.times):
            result, time = db.query(query['query'])
            execution_times.append(time)
            recall = db.cal_recall(result, gt['result'])
            execution_recalls.append(recall)
            query_results.append(result)
            # print(f"Query {query['name']} Execution {i + 1} Time: {time:.4f} seconds Recall: {recall:.4f}")
        avg_execution_time = np.mean(execution_times)
        avg_execution_recall = np.mean(execution_recalls)
        print(f"Average time for query {query['name']}: {avg_execution_time:.4f} seconds  recall: {avg_execution_recall:.4f}")
        results.append({'name': query['name'],
                        'avg_execution_time': avg_execution_time,
                        'avg_execution_recall': avg_execution_recall})
    return results

def main(args):
    # 加载配置
    config = load_config('config/db_config.yaml')
    index_config = load_config('config/index_config.yaml')
    schema = load_schema_from_yaml("config/schema.yaml")
    # 初始化数据库连接
    if args.database == 'pgvector':
        db = PGVector(config, args.database)
    elif args.database == 'milvus':
        db = Milvus(config, args.database)
    else:
        raise ValueError("不支持的数据库类型，目前仅支持 'pgvector' 或者 'milvus'")

    db.connect()
    print(f"------{db.db_type} connect successfully.")
    # 初始化benchmark的相关内容，完成表的创建、数据的加载准备以及表数据的插入
    if(args.operation == 'init'):
        # 创建表、集合
        # db.create_table(args.database, schema=schema)
        # print(f"----{db.db_type} create table successfully.")
        # # 加载并预处理数据
        # data = load_data_from_csv(args)
        # data = db.process_data(data, args.database, schema)
        # print(f"----{db.db_type} process_data successfully.")
        # # 插入数据
        # db.insert_data(data, args.database ,schema)
        # print(f"----{db.db_type} insert_data successfully.")
        if(args.database == 'pgvector'):
            # 进行query的查询并保存ground_truth，目前milvus好像没法进行KNN查询
            queries = load_query_from_yaml(db.db_type)
            execute_save(db, queries)
            print(f"----{db.db_type} save ground_truth successfully.")
    elif(args.operation == 'run'):
        # 加载ground truth
        ground_truth = db.load_ground_truth('data/ground_truth.json')
        # 创建索引
        # db.create_index(index_config[db.db_type], args)
        # print(f"----{db.db_type} create index successfully.")
        # # milvus调试使用
        # from pymilvus import Collection
        # db.collection = Collection('my_table')
        queries = load_query_from_yaml(db.db_type)
        print(f"----{db.db_type} load ground_truth successfully.")
        # 执行查询和验证
        results = execute(db, queries, ground_truth, args)
        print(results)
        # plot_query_metrics(results)
        # db.drop_indexes(index_config[db.db_type], args)
        # print(f"----{db.db_type} drop index successfully.")
    else:
        raise ValueError("不支持的操作，请确保传入operation的参数是 'init' 或 'run'")
    db.close_connection()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
