import sys
import time
import threading
import numpy as np
from tqdm import tqdm
import concurrent.futures
from queue import Queue, Empty

class DBConnectionFactory:
    """Thread-safe connection factory with connection pooling."""

    def __init__(self, config, db_type, max_connections=32):
        self.config = config
        self.db_type = db_type
        self.max_connections = max_connections
        self.connection_pool = Queue(maxsize=max_connections)
        self.lock = threading.Lock()
        self.active_connections = 0
        self.created_connections = 0

    def create_connection(self):
        if self.db_type == "pgvector":
            from databases.pgvector import PGVector
            return PGVector(self.config, self.db_type)
        elif self.db_type == "milvus":
            from databases.milvus import Milvus
            return Milvus(self.config, self.db_type)
        elif self.db_type == "qdrant":
            from databases.qdrant import Qdrant
            return Qdrant(self.config, self.db_type)
        elif self.db_type == "weaviate":
            from databases.weaviate import Weaviate
            return Weaviate(self.config, self.db_type)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def get_connection(self):
        try:
            return self.connection_pool.get_nowait()
        except Empty:
            with self.lock:
                if self.active_connections < self.max_connections:
                    db = self.create_connection()
                    db.connect()
                    self.active_connections += 1
                    self.created_connections += 1
                    return db
                return self.connection_pool.get()

    def release_connection(self, conn):
        self.connection_pool.put(conn)

    def close_all(self):
        while not self.connection_pool.empty():
            try:
                conn = self.connection_pool.get_nowait()
                conn.close()
            except Empty:
                break
        self.active_connections = 0


def setup_database_c(args, config):
    factory = DBConnectionFactory(
        config, args.database, max_connections=args.concurrency * 2
    )
    for _ in range(min(4, factory.max_connections)):
        conn = factory.create_connection()
        conn.connect()
        factory.release_connection(conn)
    print(f"------{args.database} connection factory initialized (max: {factory.max_connections})")
    return factory


def execute_single_query(db_factory, query, param, gt):
    db = db_factory.get_connection()
    try:
        start_time = time.time()
        result, _ = db.query(query, param)
        exec_time = time.time() - start_time
        recall = db.cal_recall(result, gt["result"])
        return result, exec_time, recall
    finally:
        db_factory.release_connection(db)

def execute_concurrent_hits(db, queries, ground_truth, search_params, args, db_config):
    results = []
    params = search_params[args.algorithm]

    for param in params:
        exec_times, recalls, names = [], [], []
        # print(param)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            start_idx_scans = db.get_index_usage("my_table_image_vec_idx")
            db_factory = setup_database_c(args, db_config)
            future_to_query = {}
            start_global = time.time()

            for query in queries:
                gt = next((g for g in ground_truth if g["name"] == query["name"]), None)
                if not gt:
                    print(f"Warning: query {query['name']} has no ground truth")
                    continue

                for _ in range(args.times):
                    future = executor.submit(execute_single_query, db_factory, query, param, gt)
                    future_to_query[future] = query["name"]

            # with tqdm(total=len(future_to_query), desc="Running queries") as pbar:
            for future in concurrent.futures.as_completed(future_to_query):
                query_name = future_to_query[future]
                try:
                    _, exec_time, recall = future.result()
                    exec_times.append(exec_time)
                    recalls.append(recall)
                    names.append(query_name)
                except Exception as e:
                    print(f"Query {query_name} failed: {e}")
                    # finally:
                    #     pbar.update(1)

            total_time = time.time() - start_global
            total_queries = len(exec_times)
            qps = total_queries / total_time if total_time > 0 else 0
            avg_recall = np.mean(recalls) if recalls else 0

            query_stats = {}
            for name, t, r in zip(names, exec_times, recalls):
                if name not in query_stats:
                    query_stats[name] = {"times": [], "recalls": []}
                query_stats[name]["times"].append(t)
                query_stats[name]["recalls"].append(r)

            for name, stats in query_stats.items():
                avg_time = np.mean(stats["times"])
                avg_r = np.mean(stats["recalls"])
                min_time = np.min(stats["times"])
                max_time = np.max(stats["times"])

                results.append(
                    {
                        "name": name,
                        "param": param,
                        "avg_time": avg_time,
                        "avg_recall": avg_r,
                        "min_time": min_time,
                        "max_time": max_time,
                        "concurrency": args.concurrency,
                        "qps": qps,
                    }
                )
            db_factory.close_all()

            end_idx_scans = db.get_index_usage("my_table_image_vec_idx")
            total_index_hits = end_idx_scans - start_idx_scans
            print(f"\nQPS: {qps:.2f}, avg recall: {avg_recall:.4f}, param: {param},"
            f"total latency: {total_time:.2f}s, total queries: {total_queries}, Index Hit Rate: {(total_index_hits / total_queries) * 100:.2f}%\n")

    return results

# def execute_concurrent(db_factory, queries, ground_truth, search_params, args):
    results = []
    params = search_params[args.algorithm]

    for param in params:
        exec_times, recalls, names = [], [], []
        # print(param)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_query = {}
            start_global = time.time()

            for query in queries:
                gt = next((g for g in ground_truth if g["name"] == query["name"]), None)
                if not gt:
                    print(f"Warning: query {query['name']} has no ground truth")
                    continue

                for _ in range(args.times):
                    future = executor.submit(execute_single_query, db_factory, query, param, gt)
                    future_to_query[future] = query["name"]

            # with tqdm(total=len(future_to_query), desc="Running queries") as pbar:
            for future in concurrent.futures.as_completed(future_to_query):
                query_name = future_to_query[future]
                try:
                    _, exec_time, recall = future.result()
                    exec_times.append(exec_time)
                    recalls.append(recall)
                    names.append(query_name)
                except Exception as e:
                    print(f"Query {query_name} failed: {e}")
                    # finally:
                    #     pbar.update(1)

            total_time = time.time() - start_global
            total_queries = len(exec_times)
            qps = total_queries / total_time if total_time > 0 else 0
            avg_recall = np.mean(recalls) if recalls else 0

            query_stats = {}
            for name, t, r in zip(names, exec_times, recalls):
                if name not in query_stats:
                    query_stats[name] = {"times": [], "recalls": []}
                query_stats[name]["times"].append(t)
                query_stats[name]["recalls"].append(r)

            for name, stats in query_stats.items():
                avg_time = np.mean(stats["times"])
                avg_r = np.mean(stats["recalls"])
                min_time = np.min(stats["times"])
                max_time = np.max(stats["times"])

                results.append(
                    {
                        "name": name,
                        "param": param,
                        "avg_time": avg_time,
                        "avg_recall": avg_r,
                        "min_time": min_time,
                        "max_time": max_time,
                        "concurrency": args.concurrency,
                        "qps": qps,
                    }
                )

            print(
                f"\nQPS: {qps:.2f}, avg recall: {avg_recall:.4f}, param: {param},"
                f"total latency: {total_time:.2f}s, total queries: {total_queries}\n"
            )

    db_factory.close_all()
    return results

def execute_concurrent(db_factory, queries, ground_truth, search_params, args):
    results = []
    params = search_params[args.algorithm]

    # ========================================================
    # 阶段 1: Warm-up (预热阶段)
    # 目的：将数据库数据加载到 OS Cache 和 DB Buffer Pool，消除冷启动 IO 干扰
    # ========================================================
    print("\n>>> [1/2] Starting Warm-up Phase...")
    
    warmup_size = max(1, len(queries) // 10)
    warmup_queries = queries[:warmup_size]
    warmup_param = params[0] 

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for q in warmup_queries:
            gt = next((g for g in ground_truth if g["name"] == q["name"]), None)
            if gt:
                futures.append(executor.submit(execute_single_query, db_factory, q, warmup_param, gt))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception:
                pass
                
    print(">>> Warm-up Phase Finished. Data should be in memory now.")
    
    # ========================================================
    # 阶段 2: Formal Benchmark (正式压力测试)
    # ========================================================
    # 重要：打印此行以便外部 bash 脚本捕获并开始执行 perf stat
    print("BENCHMARK_START_SIGNAL") 
    sys.stdout.flush() # 确保立即输出

    for param in params:
        exec_times, recalls, names = [], [], []
        
        print(f"\n>>> [2/2] Testing with param: {param} (Concurrency: {args.concurrency})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_query = {}
            start_global = time.time()

            for query in queries:
                gt = next((g for g in ground_truth if g["name"] == query["name"]), None)
                if not gt:
                    continue

                for _ in range(args.times):
                    future = executor.submit(execute_single_query, db_factory, query, param, gt)
                    future_to_query[future] = query["name"]

            for future in tqdm(concurrent.futures.as_completed(future_to_query), 
                              total=len(future_to_query), 
                              desc=f"Param {param}"):
                query_name = future_to_query[future]
                try:
                    _, exec_time, recall = future.result()
                    exec_times.append(exec_time)
                    recalls.append(recall)
                    names.append(query_name)
                except Exception as e:
                    print(f"Query {query_name} failed: {e}")

            total_time = time.time() - start_global
            total_queries = len(exec_times)
            qps = total_queries / total_time if total_time > 0 else 0
            avg_recall = np.mean(recalls) if recalls else 0

            # 统计每个查询的详细指标
            query_stats = {}
            for name, t, r in zip(names, exec_times, recalls):
                if name not in query_stats:
                    query_stats[name] = {"times": [], "recalls": []}
                query_stats[name]["times"].append(t)
                query_stats[name]["recalls"].append(r)

            for name, stats in query_stats.items():
                results.append({
                    "name": name,
                    "param": param,
                    "avg_time": np.mean(stats["times"]),
                    "avg_recall": np.mean(stats["recalls"]),
                    "min_time": np.min(stats["times"]),
                    "max_time": np.max(stats["times"]),
                    "concurrency": args.concurrency,
                    "qps": qps,
                })

            print(f"Result -> QPS: {qps:.2f}, Avg Recall: {avg_recall:.4f}")

    # 全部结束后打印结束信号
    print("BENCHMARK_END_SIGNAL")
    sys.stdout.flush()
    
    db_factory.close_all()
    return results