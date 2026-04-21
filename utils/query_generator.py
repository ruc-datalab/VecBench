import yaml
import pandas as pd

def gen_queries_random(data, args, n_queries=1000, outfile="config/YFCC/E2E_queries.yaml"):
    loader = data
    scale = getattr(args, 'scale', 10_000_000)
    sampled = []
    sampled_count = 0

    for df_chunk, _, _ in loader():
        if sampled_count >= n_queries:
            break
        chunk_ratio = len(df_chunk) / scale
        chunk_n = max(1, int(n_queries * chunk_ratio))
        chunk_n = min(chunk_n, len(df_chunk) - 1)
        sampled_chunk = df_chunk.sample(chunk_n, random_state=42)
        sampled.append(sampled_chunk)
        sampled_count += len(sampled_chunk)

    sampled = pd.concat(sampled, ignore_index=True)
    sampled = sampled.sample(n=min(n_queries, len(sampled)), random_state=42)
    
    queries = []
    qid = 1
    for _, row in sampled.iterrows():
        ref_id = row["id"]
        equal_val = row["equal"]
        range_val = row["range"]
        tags_list = row["tags"]

        # query1: equal
        queries.append({
            "name": f"query{qid}",
            "vector_field": "image_vec",
            "reference_vector_name": int(ref_id),
            "scalar_filters": [
                {
                    "field": "equal",
                    "operator": "==",
                    "value": int(equal_val),
                    "logic": "and"
                }
            ],
            "limit": 100
        })
        qid += 1

        # # query2: range
        # queries.append({
        #     "name": f"query{qid}",
        #     "vector_field": "image_vec",
        #     "reference_vector_name": int(ref_id),
        #     "scalar_filters": [
        #         {
        #             "field": "range",
        #             "operator": ">",
        #             "value": float(range_val - 0.5),
        #             "logic": "and"
        #         },
        #         {
        #             "field": "range",
        #             "operator": "<",
        #             "value": float(range_val + 0.5),
        #             "logic": "and"
        #         }
        #     ],
        #     "limit": 100
        # })
        # qid += 1

        # # query3: tags
        # if isinstance(tags_list, list) and len(tags_list) > 0:
        #     queries.append({
        #         "name": f"query{qid}",
        #         "vector_field": "image_vec",
        #         "reference_vector_name": int(ref_id),
        #         "scalar_filters": [
        #             {
        #                 "field": "tags",
        #                 "operator": "contains",
        #                 "value": int(tags_list[0]),
        #                 "logic": "and"
        #             }
        #         ],
        #         "limit": 100
        #     })
        #     qid += 1

    query_dict = {f"{args.dataset}": queries}
    with open(outfile, "w") as f:
        yaml.dump(query_dict, f, sort_keys=False)
    return outfile