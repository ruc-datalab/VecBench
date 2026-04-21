import yaml
import json
import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date

def load_config(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def prepare_config(args):
    db_config = load_config('config/db_config.yaml')
    index_config = load_config(f'config/{args.dataset}/index.yaml')
    schema_config = load_config(f"config/{args.dataset}/schema.yaml")
    return db_config, index_config, schema_config

def load_query_from_yaml(query_type='YFCC', query_file='config/queries.yaml'):
    with open(query_file, 'r') as file:
        queries = yaml.load(file, Loader=yaml.FullLoader)
    return queries.get(query_type)

def load_ground_truth(ground_truth_file):
    with open(ground_truth_file, 'r') as f:
        return json.load(f)
    
def execute(db, queries, ground_truth, search_params,  args):
    detailed_results = [] 
    overall_results = []
    params = search_params[args.algorithm]

    for param in params:
        total_latency = []
        total_recall = []

        # for query in tqdm(queries, total=len(queries), desc="Processing"):
        for query in queries:
            gt = next((g for g in ground_truth if g['name'] == query['name']), None)
            if gt is None:
                raise ValueError(f"No ground truth found for query {query['name']}")

            execution_times = []
            execution_recalls = []
            query_results = []

            for i in range(args.times):
                result, time = db.query(query, param)
                recall = db.cal_recall(result, gt['result'])

                execution_times.append(time)
                execution_recalls.append(recall)
                query_results.append(result)

            avg_execution_time = np.mean(execution_times)
            min_execution_time = np.min(execution_times)
            max_execution_time = np.max(execution_times)
            delta_minus = avg_execution_time - min_execution_time
            delta_plus = max_execution_time - avg_execution_time
            avg_execution_recall = np.mean(execution_recalls)

            total_latency.append(avg_execution_time)
            total_recall.append(avg_execution_recall)

            print(f"{query['name']} - latency: {avg_execution_time:.4f} s "
                  f"(+{delta_plus:.4f}, -{delta_minus:.4f}), "
                  f"recall: {avg_execution_recall:.4f}, param: {param}")

            detailed_results.append({
                'name': query['name'],
                'param': param,
                'avg_execution_time': avg_execution_time,
                'min_execution_time': min_execution_time,
                'max_execution_time': max_execution_time,
                'delta_plus': delta_plus,
                'delta_minus': delta_minus,
                'avg_execution_recall': avg_execution_recall
            })


        avg_latency = np.mean(total_latency)
        avg_recall = np.mean(total_recall)

        p1_latency = np.percentile(total_latency, 1)
        p5_latency = np.percentile(total_latency, 5)
        p25_latency = np.percentile(total_latency, 25)
        p50_latency = np.percentile(total_latency, 50)
        p95_latency = np.percentile(total_latency, 95)
        p99_latency = np.percentile(total_latency, 99)

        p1_recall = np.percentile(total_recall, 1)
        p5_recall = np.percentile(total_recall, 5)
        p25_recall = np.percentile(total_recall, 25)
        p50_recall = np.percentile(total_recall, 50)
        p95_recall = np.percentile(total_recall, 95)
        p99_recall = np.percentile(total_recall, 99)

        overall_results.append({
            'param': param,
            'avg_latency': round(avg_latency, 4),
            'avg_recall': round(avg_recall, 4),
            'p1_latency': round(p1_latency, 4),
            'p5_latency': round(p5_latency, 4),
            'p25_latency': round(p25_latency, 4),
            'p50_latency': round(p50_latency, 4),
            'p95_latency': round(p95_latency, 4),
            'p99_latency': round(p99_latency, 4),
            'p1_recall': round(p1_recall, 4),
            'p5_recall': round(p5_recall, 4),
            'p25_recall': round(p25_recall, 4),
            'p50_recall': round(p50_recall, 4),
            'p95_recall': round(p95_recall, 4),
            'p99_recall': round(p99_recall, 4)
        })

    return detailed_results, overall_results

def apply_scalar_filters(df, scalar_filters):
    if not scalar_filters:
        return df

    conditions = []
    logic_operators = []
    for f in scalar_filters:
        field, operator, value, logic = f['field'], f['operator'], f['value'], f['logic']
        if operator == '==':
            condition = (df[field] == value)
        elif operator == '<':
            condition = (df[field] < value)
        elif operator == '<=':
            condition = (df[field] <= value)
        elif operator == '>':
            condition = (df[field] > value)
        elif operator == '>=':
            condition = (df[field] >= value)
        elif operator == 'like':
            condition = df[field].astype(str).str.contains(value, na=False)
        elif operator == 'contains':
            condition = df[field].apply(lambda x: isinstance(x, (list, str)) and value in x)
        else:
            raise ValueError(f"Unsupported operator {operator}")
        print(f"Filtering on {field} {operator} {value} → matched {condition.sum()} rows")
        conditions.append(condition)
        logic_operators.append(logic)

    if conditions:
        final_condition = conditions[0]
        for i in range(1, len(conditions)):
            if logic_operators[i] == 'and':
                final_condition &= conditions[i]
            elif logic_operators[i] == 'or':
                final_condition |= conditions[i]
        df = df[final_condition]

    return df

def execute_save(queries, data, outpath, save=False):
    results = []
    if isinstance(data, pd.DataFrame):
        for query in queries:
            vector_field = query['vector_field']
            reference_vector_name = query['reference_vector_name']
            scalar_filters = query['scalar_filters']
            limit = query['limit']

            reference_vector = data[data['id'] == reference_vector_name][vector_field].values[0]

            filtered_data = apply_scalar_filters(data, scalar_filters).copy()

            filtered_data['distance'] = filtered_data[vector_field].apply(
                lambda vec: np.linalg.norm(vec - reference_vector)
            )
            sorted_data = filtered_data.sort_values(by='distance')
            result = sorted_data.head(limit)['id'].tolist()

            results.append({'name': query['name'], 'result': result})
    else:
        loader = data
        query_map = {q['name']: q for q in queries}
        ref_vectors = {}

        for df_chunk, _, _ in loader():
            for q in queries:
                if q['name'] in ref_vectors:
                    continue
                ref_id = q['reference_vector_name']
                if ref_id in df_chunk['id'].values:
                    ref_vectors[q['name']] = df_chunk.loc[df_chunk['id'] == ref_id, q['vector_field']].values[0]

        not_found = [q['name'] for q in queries if q['name'] not in ref_vectors]
        if not_found:
            raise ValueError(f"Reference vectors not found for queries: {not_found}")

        heaps = {q['name']: [] for q in queries}
        
        for df_chunk, _, _ in loader():
            for q in queries:
                ref_vec = ref_vectors[q['name']]
                vector_field = q['vector_field']
                scalar_filters = q['scalar_filters']
                limit = q['limit']
                heap = heaps[q['name']]

                filtered_data = apply_scalar_filters(df_chunk, scalar_filters).copy()
                for _, row in filtered_data.iterrows(): 
                    vec = np.array(row[vector_field], dtype=np.float32)
                    ref_vec = np.array(ref_vec, dtype=np.float32)
                    dist = np.linalg.norm(vec - ref_vec)
                    if len(heap) < limit:
                        heapq.heappush(heap, (-dist, row['id']))
                    else:
                        if -heap[0][0] > dist:
                            heapq.heappop(heap)
                            heapq.heappush(heap, (-dist, row['id']))
        for q in queries:
            heap = heaps[q['name']]
            topk = sorted([(-d, idx) for d, idx in heap])
            result_ids = [idx for _, idx in topk]
            results.append({'name': q['name'], 'result': result_ids})

    if save:
        save_ground_truth(results, outpath)

    return results

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, date):
            return o.strftime('%Y-%m-%d')
        return super().default(o)

def save_ground_truth(data, outpath):
    json_data = []
    for item in data:
        json_item = {
            "name": item["name"],
            "result": item["result"]
        }
        json_data.append(json_item)
    with open(outpath, 'w') as f:
        json.dump(json_data, f, indent=4, cls=CustomJSONEncoder)