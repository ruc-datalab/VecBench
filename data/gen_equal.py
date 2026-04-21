import h5py
import numpy as np
import os

# 参数设置
EQUAL_K = 500
ZIPF_S = 1.2

def gen_equal_vals(n):
    """Zipf 分布，值域 1..EQUAL_K"""
    values = np.arange(1, EQUAL_K + 1)
    weights = 1.0 / np.power(values, ZIPF_S)
    probs = weights / weights.sum()
    # 使用 uint16 存储，范围 1-50 完全足够
    return np.random.choice(values, size=n, p=probs).astype(np.uint16)

def save_scalar_bin(arr, filename):
    """
    u8bin/scalar 存储格式：
    int32 nrow, int32 dim(=1) + 纯数据 (arr.tobytes())
    """
    n = arr.shape[0]
    with open(filename, 'wb') as f:
        f.write(np.int32(n).tobytes())
        f.write(np.int32(1).tobytes())
        arr.tofile(f)
    print(f"Successfully saved {n} scalars to {filename}")

def process_datasets(dataset_configs):
    for name, path in dataset_configs.items():
        if not os.path.exists(path):
            print(f"Warning: File {path} not found, skipping...")
            continue
        
        print(f"Processing {name}...")
        
        # 1. 读取 HDF5 获取数据总量
        with h5py.File(path, 'r') as f:
            # 标准 ANN-Benchmarks 数据集包含 'train' 键
            if 'train' in f:
                n = f['train'].shape[0]
            else:
                # 兼容性处理：如果键名不同，打印出来检查
                print(f"Keys in {name}: {list(f.keys())}")
                continue
        
        # 2. 生成标量数据
        scalar_data = gen_equal_vals(n)
        
        # 3. 保存为 bin 文件 (放在原文件同目录下)
        output_filename = os.path.join(os.path.dirname(path), f"{name.lower()}_scalar.bin")
        save_scalar_bin(scalar_data, output_filename)

if __name__ == "__main__":
    # 数据集路径配置
    datasets = {
        "GIST": "/home/zhangx/vec_bench/VecBench/data/GIST/gist-960-euclidean.hdf5",
        "Glove": "/home/zhangx/vec_bench/VecBench/data/Glove/glove-100-angular.hdf5",
        "SIFT": "/home/zhangx/vec_bench/VecBench/data/SIFT/sift-128-euclidean.hdf5"
    }
    
    process_datasets(datasets)