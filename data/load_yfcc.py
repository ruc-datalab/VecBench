import os 
import gc
import numpy as np
import pandas as pd
import random
import json
import time
import h5py
from scipy.sparse import csr_matrix
from tqdm import trange
from math import ceil

def load_u8bin(filename, dim=None, mmap=True):
    with open(filename, "rb") as f:
        nvecs  = int.from_bytes(f.read(4), 'little')
        fdim   = int.from_bytes(f.read(4), 'little')
        dim    = fdim if dim is None else dim
        if dim != fdim:
            f.seek(4, os.SEEK_CUR)
        offset = f.tell()
        if mmap:
            vecs = np.memmap(filename, mode='r', dtype=np.uint8, 
                             offset=offset, shape=(nvecs, dim))
        else:
            vecs = np.fromfile(f, dtype=np.uint8).reshape(nvecs, dim)
    return vecs

def load_hdf5_vectors(path, subset='train'):
    f = h5py.File(path, 'r')
    return f[subset]

def load_sparse_matrix(path):
    with open(path, 'rb') as f:
        nrow, ncol, nnz = np.fromfile(f, np.int64, 3)
        indptr   = np.fromfile(f, np.int64, nrow + 1)
        indices  = np.fromfile(f, np.int32, nnz)
        data     = np.fromfile(f, np.float32, nnz)
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))

def save_scalar_bin(arr, filename):
    n = arr.shape[0]
    with open(filename, 'wb') as f:
        f.write(np.int32(n).tobytes())
        f.write(np.int32(1).tobytes())
        arr.tofile(f)

def load_scalar_bin(filename, dtype, mmap=True):
    with open(filename, 'rb') as f:
        n = int.from_bytes(f.read(4), 'little')
        dim = int.from_bytes(f.read(4), 'little')
        if dim != 1:
            raise ValueError("dim 应为 1")
        offset = f.tell()
        if mmap:
            return np.memmap(filename, mode='r', dtype=dtype,
                             offset=offset, shape=(n,))
        else:
            return np.fromfile(f, dtype=dtype, count=n)

def load_dataset_with_scalars(base_dir, args, as_list=True):
    def _loader(random_sample=True, seed=42):
        dataset_name = args.dataset.upper()
        chunk_rows = args.chunk_rows if hasattr(args, 'chunk_rows') else 1_000_000
        
        if dataset_name == 'YFCC':
            vecs = load_u8bin(os.path.join(base_dir, 'base.10M.1920d.u8bin'))
            smat = load_sparse_matrix(os.path.join(base_dir, 'base.metadata.10M.spmat'))
            total_n = vecs.shape[0]
        else:
            hdf5_map = {
                "GIST": "gist-960-euclidean.hdf5",
                "GLOVE": "glove-200-angular.hdf5",
                "SIFT": "sift-128-euclidean.hdf5"
            }
            h5_path = os.path.join(base_dir, hdf5_map[dataset_name])
            vecs = load_hdf5_vectors(h5_path)
            total_n = vecs.shape[0]
            smat = None

        scale = args.scale
        ratio = scale / total_n
        if ratio < 1.0:
            new_n = int(total_n * ratio)
            if random_sample:
                rng = np.random.default_rng(seed)
                idx = rng.choice(total_n, size=new_n, replace=False)
                idx.sort()
            else:
                idx = np.arange(new_n)
        else:
            idx = np.arange(total_n)
            new_n = total_n

        eq_file = "equal_cluster.u2bin" if dataset_name == 'YFCC' else f"{dataset_name.lower()}_scalar.bin"
        equal_vals = load_scalar_bin(os.path.join(base_dir, eq_file), np.uint16)[idx]
        
        if dataset_name == 'YFCC':
            range_vals = load_scalar_bin(os.path.join(base_dir, 'range.f4bin'), np.float32)[idx]
        else:
            range_vals = np.zeros(new_n, dtype=np.float32)

        start_id = 1
        batches = ceil(new_n / chunk_rows)
        
        for b in trange(batches, desc=f"Preparing {dataset_name} batches"):
            s, e = b * chunk_rows, min((b + 1) * chunk_rows, new_n)
            
            batch_idx = idx[s:e]
            vecs_batch = vecs[batch_idx].astype("float32")

            if as_list:
                image_vecs = vecs_batch.tolist()
            else:
                image_vecs = list(vecs_batch)

            data_dict = {
                "id": np.arange(start_id, start_id + (e - s)),
                "image_vec": image_vecs,
                "equal": equal_vals[s:e],
                "range": range_vals[s:e],
            }

            if smat is not None:
                data_dict["tags"] = [row.indices.tolist() for row in smat[batch_idx]]
            else:
                data_dict["tags"] = [[] for _ in range(e - s)]

            df = pd.DataFrame(data_dict)
            start_id += (e - s)

            vector_cols = ["image_vec"] 
            scalar_cols = ["equal", "range", "tags"]
            
            yield df, vector_cols, scalar_cols
            
    return _loader

def get_incremental_paths(incremental_dir):
    """Return all file paths related to incremental data (to avoid redundant joins)."""
    return {
        "info": os.path.join(incremental_dir, "incremental_info.json"),
        "vec": os.path.join(incremental_dir, "incremental.base.u8bin"),
        "meta": os.path.join(incremental_dir, "incremental.base.metadata.spmat"),
        "equal": os.path.join(incremental_dir, "incremental.equal.u2bin"),
        "range": os.path.join(incremental_dir, "incremental.range.f4bin")
    }

def load_incremental_meta(paths):
    """Load and validate metadata for the incremental dataset."""
    if not all(os.path.exists(p) for p in paths.values()):
        raise FileNotFoundError("Missing incremental data files. Please run the Incremental Load Phase first.")
    with open(paths["info"], "r") as f:
        return json.load(f)

def load_incremental_data(paths):
    """Load incremental dataset (vectors, scalars, and sparse metadata)."""
    from data.load_yfcc import load_u8bin, load_scalar_bin, load_sparse_matrix
    return {
        "vec": load_u8bin(paths["vec"]),
        "equal": load_scalar_bin(paths["equal"], np.uint16),
        "range": load_scalar_bin(paths["range"], np.float32),
        "tags": load_sparse_matrix(paths["meta"])
    }

# --------------------------  incremental_load  --------------------------
def perform_incremental_load(db, base_data_dir, schema):
    """
    Load and insert previously generated incremental dataset into the database.
    """
    incremental_dir = os.path.join(base_data_dir, "incremental_data")
    paths = get_incremental_paths(incremental_dir)

    print("Loading incremental data...")
    inc_info = load_incremental_meta(paths)
    inc_data = load_incremental_data(paths)

    start_id = inc_info["min_inc_id"]
    batches = ceil(inc_info["total_size"] / inc_info["chunk_rows"])
    total_latency = []

    for b in trange(batches, desc=f"Inserting {db.db_type} batches"):
        s, e = b * inc_info["chunk_rows"], min((b + 1) * inc_info["chunk_rows"], inc_info["total_size"])
        df_chunk = pd.DataFrame({
            "id": np.arange(start_id, start_id + (e - s)),
            "image_vec": inc_data["vec"][s:e].astype(np.float32).tolist(),
            "equal": inc_data["equal"][s:e],
            "range": inc_data["range"][s:e],
            "tags": [inc_data["tags"][row].indices.tolist() for row in range(s, e)]
        })
        start_id += (e - s)

        df_chunk = db.process_data(df_chunk, db.db_type, schema)
        latency = db.insert_data(df_chunk, db.db_type, schema)
        total_latency.append(latency)
        print(f"---- {db.db_type} batch {b + 1}/{batches} inserted.")

    duration = np.sum(total_latency)
    print(f"---- {db.db_type} incremental insert finished (time: {duration:.2f}s)")

# -------------------------- update --------------------------
# def perform_update_scalar(db, incremental_dir, schema, delta=1):
#     paths = get_incremental_paths(incremental_dir)
#     inc_info = load_incremental_meta(paths)
#     inc_data = load_incremental_data(paths)
#     min_id, max_id = inc_info["min_inc_id"], inc_info["max_inc_id"]
#     print(f"Updating {inc_info['total_size']} rows (ID: [{min_id}, {max_id}])...")

#     # update equal++
#     updated_equal = np.clip(inc_data["equal"] + delta, 1, np.iinfo(np.uint16).max)
#     batches = ceil(inc_info["total_size"] / inc_info["chunk_rows"])
#     total_duration = 0

#     for b in range(batches):
#         s, e = b * inc_info["chunk_rows"], min((b+1)*inc_info["chunk_rows"], inc_info["total_size"])
#         batch_ids = list(range(min_id + s, min_id + e))
#         batch_df = pd.DataFrame({
#             "id": batch_ids,
#             "image_vec": inc_data["vec"][s:e].astype(np.float32).tolist(),
#             "equal": updated_equal[s:e],
#             "range": inc_data["range"][s:e],
#             "tags": [inc_data["tags"][row].indices.tolist() for row in range(s, e)]
#         })

#         start = time.time()
#         if db.db_type == 'pgvector':
#             db.sql(f"UPDATE my_table SET equal = equal + {delta} WHERE id IN {tuple(batch_ids)};")
#         elif db.db_type == 'milvus':
#             batch_data = db.process_data(batch_df, db.db_type, schema)
#             start = time.time()
#             db.delete_by_ids(batch_ids)
#             db.insert_data(batch_data, db.db_type, schema)
#         elif db.db_type == 'qdrant':
#             db.overwrite_payload("equal", batch_df)
#         elif db.db_type == 'weaviate':
#             db.overwrite_properties("equal", batch_df)
 
#         batch_duration = time.time() - start
#         total_duration += batch_duration
#         print(f"---- Batch {b+1}/{batches} done (time: {batch_duration:.2f}s)")

#     print(f"---- {db.db_type} update finished (total time: {total_duration:.2f}s)")
#     return total_duration

def perform_update(db, incremental_dir, schema, delta=1):
    paths = get_incremental_paths(incremental_dir)
    inc_info = load_incremental_meta(paths)
    inc_data = load_incremental_data(paths)
    
    min_id = inc_info["min_inc_id"]
    total_size = inc_info["total_size"]
    chunk_rows = inc_info["chunk_rows"]
    
    print(f"Updating {total_size} rows (Vector + Scalar) for {db.db_type}...")

    updated_equal = np.clip(inc_data["equal"] + delta, 0, 65535).astype(np.uint16)
    raw_vecs = inc_data["vec"].astype(np.uint8).copy()
    
    rows = raw_vecs.shape[0]
    cols = raw_vecs.shape[1]
    random_dims = np.random.randint(0, cols, size=rows)
    row_indices = np.arange(rows)
    mask = raw_vecs[row_indices, random_dims] < 255
    raw_vecs[row_indices[mask], random_dims[mask]] += 1
    raw_vecs[row_indices[~mask], random_dims[~mask]] -= 1
    
    updated_vecs_list = raw_vecs.astype(np.float32)

    batches = ceil(total_size / chunk_rows)
    total_duration = 0

    for b in range(batches):
        s, e = b * chunk_rows, min((b + 1) * chunk_rows, total_size)
        batch_ids = list(range(min_id + s, min_id + e))
        
        batch_df = pd.DataFrame({
            "id": batch_ids,
            "image_vec": updated_vecs_list[s:e].tolist(),
            "equal": updated_equal[s:e],
            "range": inc_data["range"][s:e],
            "tags": [inc_data["tags"][row].indices.tolist() for row in range(s, e)]
        })

        batch_duration = db.update(batch_df, batch_size=10000)
        total_duration += batch_duration
        print(f"---- Batch {b+1}/{batches} done (time: {batch_duration:.2f}s)")
        
        del batch_df
        gc.collect()

    print(f"---- {db.db_type} Vector+Scalar Update Total Time: {total_duration:.2f}s")
    return total_duration

# -------------------------- delete --------------------------
def perform_delete(db, incremental_dir):
    paths = get_incremental_paths(incremental_dir)
    if not os.path.exists(paths["info"]):
        raise FileNotFoundError("Incremental data metadata is missing; please perform incremental loading first.")
    
    with open(paths["info"], "r") as f:
        inc_info = json.load(f)
    incremental_ids = list(range(inc_info["min_inc_id"], inc_info["max_inc_id"] + 1))
    print(f"Deleting {len(incremental_ids)} incremental rows...")
    
    duration = db.delete_by_ids(incremental_ids)
    print(f"---- {db.db_type} delete finished (time: {duration:.2f}s)")
    return duration
