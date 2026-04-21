# VecBench

A unified benchmark framework for evaluating vector databases under hybrid vector-scalar query workloads. VecBench supports multiple databases, datasets, index algorithms, and provides comprehensive metrics including recall, latency, QPS, and incremental operation performance.

## Features

- **Multi-database support**: Milvus, pgvector (PostgreSQL), Qdrant, Weaviate
- **Multiple datasets**: YFCC (10M, 1920-dim), SIFT (1M, 128-dim), GLOVE (1.2M, 200-dim), GIST (1M, 960-dim)
- **Hybrid queries**: Vector similarity search combined with scalar filters (equality, range, contains)
- **Multiple index algorithms**: HNSW, IVFFlat, DiskANN
- **Concurrent benchmarking**: Multi-threaded workload execution with warm-up phase and QPS measurement
- **Incremental operations**: Insert, Update, Delete with configurable ratios
- **Query analysis**: Filter rate and relevance analysis with automatic query modification
- **Ground truth generation**: Brute-force exact search for recall evaluation

## Quick Start

### Prerequisites

- Python 3.9+
- A running instance of at least one supported vector database

### Installation

```bash
conda create -n vecbench python=3.10
conda activate vecbench

pip install pymilvus psycopg2-binary qdrant-client weaviate-client
pip install numpy pandas h5py scipy pyyaml tqdm matplotlib
```

### 1. Prepare Data

Download the dataset files and place them under `data/<DATASET>/`. See [Dataset Preparation](#dataset-preparation) for details.

### 2. Configure Database Connection

Edit `config/db_config.yaml` with your database connection info:

```yaml
pgvector:
  host: localhost
  port: 5432
  dbname: postgres
  user: postgres
  password: your_password

milvus:
  host: localhost
  port: 19530

qdrant:
  host: localhost
  port: 6333
  grpc_port: 6334

weaviate:
  host: http://localhost:8080
  grpc_port: 50051
```

### 3. Initialize Database and Load Data

```bash
# Full YFCC 10M dataset with Milvus
python benchmark.py --case init --database milvus --dataset YFCC --scale 10000000

# SIFT 1M dataset with pgvector
python benchmark.py --case init --database pgvector --dataset SIFT --scale 1000000

# GLOVE with Qdrant, using smaller batch size
python benchmark.py --case init --database qdrant --dataset GLOVE --scale 1000000 --chunk_rows 500000

# GIST with Weaviate
python benchmark.py --case init --database weaviate --dataset GIST --scale 1000000
```

The init phase will:
1. Connect to the target database
2. Create the table/collection with the configured schema
3. Load and insert data in chunks
4. Generate ground truth for the configured queries

### 4. Run Benchmarks

```bash
# Concurrent benchmark: Milvus + HNSW, 50 threads
python benchmark.py --case test --database milvus --dataset YFCC --algorithm hnsw --concurrency 50

# pgvector + IVFFlat, 10 threads
python benchmark.py --case test --database pgvector --dataset SIFT --algorithm ivfflat --concurrency 10

# Run each query 5 times for more stable results
python benchmark.py --case test --database qdrant --dataset GLOVE --algorithm hnsw --times 5 --concurrency 50
```

The test phase runs a two-stage concurrent benchmark:
1. **Warm-up phase**: Preloads data into memory to eliminate cold-start I/O effects
2. **Formal benchmark**: Executes all queries under the specified concurrency, measures QPS and recall

Output includes per-query latency statistics and overall QPS/recall per search parameter.

## CLI Reference

```
python benchmark.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--case` | `init` | Execution mode: `data_pre`, `init`, `modify_queries`, `test` |
| `--database` | `milvus` | Target database: `milvus`, `pgvector`, `qdrant`, `weaviate` |
| `--dataset` | `YFCC` | Dataset: `YFCC`, `SIFT`, `GLOVE`, `GIST` |
| `--scale` | `10000000` | Number of vectors to load |
| `--chunk_rows` | `1000000` | Batch size for data loading |
| `--algorithm` | `hnsw` | Index algorithm: `hnsw`, `ivfflat`, `diskann` |
| `--times` | `1` | Number of times each query is executed |
| `--concurrency` | `50` | Number of concurrent threads |
| `--ratio` | `0.2` | Ratio of data for incremental generation |
| `--in_ratio` | `0.2` | Ratio of data to insert before index creation |
| `--up_ratio` | `0.2` | Ratio of data to update |
| `--de_ratio` | `0.2` | Ratio of data to delete |
| `--regen_incremental` | `False` | Force regenerate incremental data |

## Execution Modes

### `data_pre` — Data Preparation

Generates synthetic and scaled data for benchmarking:

```bash
# Adjust vector dimensionality (YFCC to 1920d) and scale to 100M
python benchmark.py --case data_pre --dataset YFCC --scale 100000000
```

This runs three operations:
- **Dimension adjustment**: Random projection to change vector dimensions
- **Scale adjustment**: Replicate data with noise to reach target scale
- **Incremental data generation**: Create insert/update/delete workload data

### `init` — Database Initialization

Creates the table, loads data, and generates ground truth:

```bash
python benchmark.py --case init --database milvus --dataset YFCC --scale 10000000
```

### `modify_queries` — Query Modification

Adjusts query filter selectivity based on filter rate or relevance analysis:

```bash
python benchmark.py --case modify_queries --database milvus --dataset YFCC
```

This uses the `Analyzer` to compute filter rates and relevances across the dataset, then modifies query scalar filters to achieve a target selectivity. Two modes are available:
- `filter`: Modify filters to match a target filter rate (fraction of data passing the filter)
- `relevance`: Modify filters to match a target relevance rate (fraction of top-k results passing the filter)

Edit the `default_filter_rate` / `default_relevance_rate` and `per_query_filter` / `per_query_relevance` dictionaries in `benchmark.py` to control target selectivities.

### `test` — Benchmark Execution

Runs the benchmark with index creation, query execution, concurrent testing, and incremental operations:

```bash
python benchmark.py --case test --database milvus --dataset YFCC --algorithm hnsw --concurrency 50
```

The test case supports multiple phases (uncomment the desired phases in `benchmark.py`):
- **P1 — Initialization**: Create vector and scalar indexes
- **P2 — Query Execution**: Single-threaded query execution with detailed per-query statistics
- **P3 — Concurrent Phase**: Multi-threaded benchmark with warm-up and QPS measurement
- **P4 — Incremental Load**: Insert new data into the indexed collection
- **P5 — Update Phase**: Update existing vectors and scalar fields
- **P6 — Delete Phase**: Delete previously inserted incremental data

## Project Structure

```
VecBench/
├── benchmark.py                 # Main entry point
├── config/
│   ├── db_config.yaml          # Database connection settings
│   ├── YFCC/
│   │   ├── schema.yaml         # Table schema for each database
│   │   ├── index.yaml          # Index configurations + search params
│   │   ├── E2E_queries.yaml    # End-to-end query definitions
│   │   └── Filtered_queries.yaml
│   ├── SIFT/
│   ├── GLOVE/
│   └── GIST/
├── data/                        # Dataset files (not included in repo)
│   ├── YFCC/
│   ├── SIFT/
│   ├── GLOVE/
│   └── GIST/
├── databases/                   # Database connector implementations
│   ├── base_vd.py              # Abstract base class
│   ├── milvus.py               # Milvus connector
│   ├── pgvector.py             # pgvector connector
│   ├── qdrant.py               # Qdrant connector
│   └── weaviate.py             # Weaviate connector
├── utils/
│   ├── workload_executor.py    # Query execution, ground truth generation
│   ├── concurrent.py           # Concurrent benchmarking with connection pool
│   ├── data_synthesizer.py     # Dimension/scale adjustment, incremental data
│   ├── analyzer.py             # Filter rate & relevance analysis
│   ├── query_generator.py      # Random query generation
│   └── plot.py                 # Result visualization
└── perf_result/                 # Benchmark output
```

## Configuration

### Schema Configuration (`config/<DATASET>/schema.yaml`)

Defines the table/collection structure for each database. Each dataset has a schema file specifying columns, types, and constraints. Example for YFCC on Milvus:

```yaml
milvus:
  table_name: my_table
  columns:
    - name: id
      type: INT64
      primary: True
    - name: equal
      type: INT16
    - name: range
      type: FLOAT
    - name: tags
      type: ARRAY
      element_type: INT32
      max_capacity: 1600
    - name: image_vec
      type: FLOAT_VECTOR
      dimension: 1920
```

The schema includes both vector fields and scalar fields used for hybrid filtering.

### Index Configuration (`config/<DATASET>/index.yaml`)

Defines index parameters and search parameters for each database and algorithm:

```yaml
milvus:
  hnsw:
    - index_column: "image_vec"
      distance: L2
      params:
        - m: 16
          ef_construction: 64
  scalar_index:
    - index_column: "equal"
    - index_column: "range"
    - index_column: "tags"

search_params:
  hnsw: [800]
  ivfflat: [50]
  diskann: [100, 1000]
```

The `search_params` section controls the search parameter sweep (e.g., `ef` for HNSW, `nprobe` for IVFFlat). The benchmark runs all queries at each parameter value and reports QPS/recall for each.

### Query Configuration (`config/<DATASET>/E2E_queries.yaml`)

Defines the benchmark queries with vector search + scalar filters:

```yaml
YFCC:
  - name: query1
    vector_field: image_vec
    reference_vector_name: 987232
    scalar_filters:
      - field: equal
        operator: "=="
        value: 14
        logic: and
    limit: 100
```

Supported filter operators: `==`, `<`, `<=`, `>`, `>=`, `like`, `contains`

## Dataset Preparation

### YFCC

The YFCC dataset uses a custom binary format. Place the following files in `data/YFCC/`:

| File | Description |
|------|-------------|
| `base.10M.1920d.u8bin` | 10M vectors (uint8, 1920 dimensions) |
| `base.metadata.10M.spmat` | Sparse tag matrix (CSR format) |
| `equal_cluster.u2bin` | Equality filter values (uint16) |
| `range.f4bin` | Range filter values (float32) |

YFCC provides three scalar fields:
- `equal`: Categorical/discrete value for equality filtering
- `range`: Continuous value for range filtering
- `tags`: Multi-value field for containment filtering

### SIFT / GLOVE / GIST

These datasets use the standard HDF5 format from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks#data-sets):

| Dataset | File | Vectors | Dimensions | Distance |
|---------|------|---------|------------|----------|
| SIFT | `sift-128-euclidean.hdf5` | 1M | 128 | L2 |
| GLOVE | `glove-200-angular.hdf5` | 1.2M | 200 | Cosine |
| GIST | `gist-960-euclidean.hdf5` | 1M | 960 | L2 |

Place the HDF5 files in `data/<DATASET>/` along with a `*_scalar.bin` file containing equality filter values for hybrid queries.

### Scaling Data

To generate larger datasets from existing ones:

```bash
# Scale YFCC to 100M vectors
python benchmark.py --case data_pre --dataset YFCC --scale 100000000
```

This replicates the base data with Gaussian noise and saves the scaled dataset to `data/YFCC/scaled_100M/`.

### Incremental Data

Incremental data for insert/update/delete operations is generated during `data_pre`:

```bash
python benchmark.py --case data_pre --dataset YFCC --scale 10000000 --ratio 0.2
```

The `--ratio` flag controls the size of incremental data as a fraction of the base dataset. Generated files are stored in `data/<DATASET>/incremental_data/`.

## Database Deployment

### Milvus

```bash
# Using Docker Compose (etcd + MinIO + Milvus standalone)
docker-compose up -d
```

Default port: 19530

### pgvector

```bash
# Pull and run PostgreSQL with pgvector extension
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_password \
  pgvector/pgvector:pg16
```

Then create the extension: `CREATE EXTENSION vector;`

### Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Weaviate

```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  cr.weaviate.io/semitechnologies/weaviate:latest
```

## Extending VecBench

### Adding a New Database

1. Create a new connector in `databases/` that inherits from `BaseVD`
2. Implement the required methods: `connect()`, `create_table()`, `create_index()`, `create_scalar_index()`, `process_data()`, `insert_data()`, `query()`, `close()`
3. Add the database configuration to `config/db_config.yaml`
4. Add per-dataset schema entries in `config/<DATASET>/schema.yaml`
5. Register the database in `setup_database()` in `benchmark.py`

### Adding a New Dataset

1. Place data files in `data/<DATASET>/`
2. Create `config/<DATASET>/schema.yaml` with table definitions
3. Create `config/<DATASET>/index.yaml` with index and search parameters
4. Create `config/<DATASET>/E2E_queries.yaml` with benchmark queries
5. Add the dataset loading logic in `data/load_yfcc.py` (for custom formats) or use the existing HDF5 loader

## Metrics

VecBench reports the following metrics:

- **Recall**: Fraction of ground truth results found by the approximate search
- **Latency**: Per-query execution time (avg, min, max, p1/p5/p25/p50/p95/p99)
- **QPS**: Queries per second under concurrent workload
- **Index creation time**: Time to build vector and scalar indexes
- **Incremental operation time**: Insert, update, delete throughput

## License

<!-- TODO: Add license -->
