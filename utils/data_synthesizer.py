import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import trange
from data.load_yfcc import load_u8bin, load_dataset_with_scalars, save_scalar_bin, get_incremental_paths

def adjust_dimension(base_dir, vec_file='base.10M.u8bin', new_dim=512,
                     mmap=True, batch_size=100_000, seed=42):
    """
    Adjust the dimensionality of a YFCC vector dataset and save it as a new .u8bin file.
    Performs batch processing to avoid memory overflow.
    """
    input_path = os.path.join(base_dir, vec_file)
    name, ext = os.path.splitext(vec_file)
    output_file = f"{name}.{new_dim}{ext}"
    output_path = os.path.join(base_dir, output_file)

    vecs = load_u8bin(input_path, mmap=mmap)
    n, d = vecs.shape
    print(f"Loaded {n} vectors with dim={d}")

    np.random.seed(seed)
    projection_matrix = np.random.normal(
        loc=0, scale=np.sqrt(1 / n), size=(d, new_dim)
    ).astype(np.float32)

    # Pass 1: compute global min/max
    global_min, global_max = float("inf"), float("-inf")
    for i in range(0, n, batch_size):
        batch = vecs[i:i + batch_size].astype(np.float32)
        proj = np.dot(batch, projection_matrix)
        global_min = min(global_min, proj.min())
        global_max = max(global_max, proj.max())
    print(f"Global min={global_min:.4f}, max={global_max:.4f}")

    # Pass 2: normalize, quantize, and write output in batches
    with open(output_path, "wb") as f:
        f.write(np.int32(n).tobytes())
        f.write(np.int32(new_dim).tobytes())

        num_batches = (n + batch_size - 1) // batch_size
        for i in range(0, n, batch_size):
            batch = vecs[i:i + batch_size].astype(np.float32)
            proj = np.dot(batch, projection_matrix)
            proj = (proj - global_min) / (global_max - global_min)
            proj = (proj * 255).astype(np.uint8)
            proj.tofile(f)
            print(f"Processed batch {i // batch_size + 1}/{num_batches}")

    print(f"Saved adjusted vectors to {output_path}")
    return output_path

def adjust_scale(base_data_dir, args, target_scale, noise_factor=0.05):
    """
    Generate a scaled-up dataset by replicating and adding noise to the base dataset.
    Example: expand 1M samples to 10M.
    """
    orig_loader = load_dataset_with_scalars(
        base_dir=base_data_dir,
        keep_sparse=True,
        chunk_rows=getattr(args, 'chunk_rows', 1_000_000)
    )

    chunks = []
    total_original = 0
    for df_chunk, _, _ in orig_loader:
        chunks.append({
            "vec": np.array(df_chunk["image_vec"].tolist(), np.float32),
            "equal": df_chunk["equal"].values,
            "range": df_chunk["range"].values,
            "tags": df_chunk["tags"].values,
            "size": len(df_chunk)
        })
        total_original += len(df_chunk)

    if total_original == 0:
        raise ValueError("No original data found to scale.")

    scale_factor = int(target_scale // total_original)
    remainder = int(target_scale % total_original)
    print(f"Original: {total_original:,}, Target: {target_scale:,}, Scale factor: {scale_factor}x + {remainder} remainder")

    scaled_data = {"vec": [], "equal": [], "range": [], "tags": []}

    # Replicate base dataset
    for rep in trange(scale_factor, desc="Replicating data"):
        for chunk in chunks:
            noisy_vec = chunk["vec"] + np.random.normal(0, noise_factor, chunk["vec"].shape)
            scaled_data["vec"].append(np.clip(noisy_vec, 0, 255).astype(np.uint8))
            scaled_data["equal"].append(chunk["equal"])
            scaled_data["range"].append(chunk["range"])
            scaled_data["tags"].append(chunk["tags"])

    # Handle remainder samples
    if remainder > 0:
        all_vec = np.concatenate([c["vec"] for c in chunks])
        all_equal = np.concatenate([c["equal"] for c in chunks])
        all_range = np.concatenate([c["range"] for c in chunks])
        all_tags = np.concatenate([c["tags"] for c in chunks])

        chosen = np.random.choice(total_original, remainder, replace=False)
        noisy_vec = all_vec[chosen] + np.random.normal(0, noise_factor, all_vec[chosen].shape)

        scaled_data["vec"].append(np.clip(noisy_vec, 0, 255).astype(np.uint8))
        scaled_data["equal"].append(all_equal[chosen])
        scaled_data["range"].append(all_range[chosen])
        scaled_data["tags"].append(all_tags[chosen])

    scaled_vec = np.concatenate(scaled_data["vec"])
    scaled_equal = np.concatenate(scaled_data["equal"])
    scaled_range = np.concatenate(scaled_data["range"])
    scaled_tags = np.concatenate(scaled_data["tags"])

    # Independent output directory
    scaled_dir = os.path.join(base_data_dir, f"scaled_{target_scale//1_000_000}M")
    os.makedirs(scaled_dir, exist_ok=True)
    print(f"Saving scaled dataset to: {scaled_dir}")

    # Save vector data
    vec_path = os.path.join(scaled_dir, f"base.{target_scale//1_000_000}M.u8bin")
    with open(vec_path, "wb") as f:
        f.write(np.int32(scaled_vec.shape[0]).tobytes())
        f.write(np.int32(scaled_vec.shape[1]).tobytes())
        scaled_vec.tofile(f)

    # Save scalar data using helper functions
    equal_path = os.path.join(scaled_dir, "equal.bin")
    range_path = os.path.join(scaled_dir, "range.bin")
    save_scalar_bin(scaled_equal, equal_path)
    save_scalar_bin(scaled_range, range_path)

    # Save sparse tag data (CSR format)
    data, indices, indptr = [], [], [0]
    for row in scaled_tags:
        data.append(row.data)
        indices.append(row.indices)
        indptr.append(indptr[-1] + len(row.data))

    meta_path = os.path.join(scaled_dir, "tags.bin")
    with open(meta_path, "wb") as f:
        np.array([len(scaled_tags), scaled_tags[0].shape[1], len(np.concatenate(indices))], np.int64).tofile(f)
        np.array(indptr, np.int64).tofile(f)
        np.concatenate(indices).astype(np.int32).tofile(f)
        np.concatenate(data).astype(np.float32).tofile(f)

    info_path = os.path.join(scaled_dir, "info.json")
    json.dump({
        "total_size": scaled_vec.shape[0],
        "vec_dim": scaled_vec.shape[1],
        "base_size": total_original,
        "scale_factor": scale_factor,
        "remainder": remainder,
        "noise_factor": noise_factor,
        "chunk_rows": getattr(args, 'chunk_rows', 1_000_000),
        "generated_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }, open(info_path, "w"), indent=2)

    print(f"Scaled dataset generated: {vec_path}")
    return scaled_dir

def generate_incremental_data(base_data_dir, args, noise_factor=0.05):
    """
    Generate incremental dataset with noise and save to incremental_data/ directory.
    """
    start_id = int(getattr(args, 'scale', 10_000_000)) + 1
    ratio = args.ratio
    incremental_dir = os.path.join(base_data_dir, "incremental_data")
    os.makedirs(incremental_dir, exist_ok=True)
    paths = get_incremental_paths(incremental_dir)
    has_existing_data = all(os.path.exists(p) for p in paths.values())

    if has_existing_data and not getattr(args, 'regen_incremental', False):
        print("Incremental data already exists. Skipping regeneration.")
        return incremental_dir

    print(f"Generating incremental data ({int(ratio * 100)}% of original with noise)...")

    original_loader = load_dataset_with_scalars(
        base_dir=base_data_dir, keep_sparse=True,
        chunk_rows=getattr(args, 'chunk_rows', 1_000_000)
    )

    original_chunks = []
    total_original = 0
    for df_chunk, _, _ in original_loader:
        original_chunks.append({
            "vec": np.array(df_chunk["image_vec"].tolist(), np.float32),
            "equal": df_chunk["equal"].values,
            "range": df_chunk["range"].values,
            "tags": df_chunk["tags"].values,
            "size": len(df_chunk)
        })
        total_original += len(df_chunk)

    incremental_size = int(total_original * ratio)
    selected_idx = sorted(random.sample(range(total_original), incremental_size))
    inc_data = {"vec": [], "equal": [], "range": [], "tags": []}
    current_idx = 0

    for chunk in original_chunks:
        chunk_end = current_idx + chunk["size"]
        chunk_selected = [i - current_idx for i in selected_idx if current_idx <= i < chunk_end]
        if chunk_selected:
            noisy_vec = chunk["vec"][chunk_selected] + np.random.normal(0, noise_factor, chunk["vec"][chunk_selected].shape)
            inc_data["vec"].append(np.clip(noisy_vec, 0, 255).astype(np.uint8))
            inc_data["equal"].append(chunk["equal"][chunk_selected])
            inc_data["range"].append(chunk["range"][chunk_selected])
            inc_data["tags"].append(chunk["tags"][chunk_selected])
        current_idx = chunk_end

    inc_vec = np.concatenate(inc_data["vec"])
    with open(paths["vec"], "wb") as f:
        f.write(np.int32(inc_vec.shape[0]).tobytes())
        f.write(np.int32(inc_vec.shape[1]).tobytes())
        inc_vec.tofile(f)

    save_scalar_bin(np.concatenate(inc_data["equal"]), paths["equal"])
    save_scalar_bin(np.concatenate(inc_data["range"]), paths["range"])

    inc_tags = np.concatenate(inc_data["tags"])
    data, indices, indptr = [], [], [0]
    for row in inc_tags:
        data.append(row.data)
        indices.append(row.indices)
        indptr.append(indptr[-1] + len(row.data))

    with open(paths["meta"], "wb") as f:
        np.array([len(inc_tags), inc_tags[0].shape[1], len(np.concatenate(indices))], np.int64).tofile(f)
        np.array(indptr, np.int64).tofile(f)
        np.concatenate(indices).astype(np.int32).tofile(f)
        np.concatenate(data).astype(np.float32).tofile(f)

    json.dump({
        "total_size": incremental_size,
        "vec_dim": inc_vec.shape[1],
        "chunk_rows": getattr(args, 'chunk_rows', 1_000_000),
        "generated_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "min_inc_id": start_id,
        "max_inc_id": start_id + incremental_size - 1
    }, open(paths["info"], "w"), indent=2)

    print(f"Incremental data saved to: {incremental_dir}")