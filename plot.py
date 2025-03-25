# import matplotlib.pyplot as plt
#
# pgvector_hnsw_ef_search = [100, 200, 400, 800, 1000]
# pgvector_hnsw_time = [0.0022, 0.0040, 0.0072, 0.0168, 0.0191]
# pgvector_hnsw_recall = [0.0400, 0.0400, 0.0600, 0.1600, 0.2000]
#
# pgvector_ivfflat_probs = [1, 2, 3, 4, 5]
# pgvector_ivfflat_time = [0.0012, 0.0021, 0.0142, 0.0728, 0.0727]
# pgvector_ivfflat_recall = [0.0100, 0.0200, 0.0600, 1.0000, 1.0000]
#
# milvus_hnsw_ef_search = [100, 200, 400, 800, 1000]
# milvus_hnsw_time = [0.4981, 0.5385, 0.4589, 0.5182, 0.6185]
# milvus_hnsw_recall = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
#
# milvus_ivfflat_probs = [1, 2, 3, 4, 5]
# milvus_ivfflat_time = [0.4183, 0.5981, 0.5584, 0.5582, 0.5783]
# milvus_ivfflat_recall = [0.0200, 0.0300, 0.0500, 0.1000, 0.1000]
#
# # 绘制time的折线图
# plt.figure(figsize=(10, 6))
# # pgvector hnsw曲线
# plt.plot(range(len(pgvector_hnsw_time)), pgvector_hnsw_time, color='tab:blue', marker='o', label='pgvector hnsw')
# # pgvector ivfflat曲线
# plt.plot(range(len(pgvector_ivfflat_time)), pgvector_ivfflat_time, color='tab:green', marker='s', label='pgvector ivfflat')
# # milvus hnsw曲线
# plt.plot(range(len(milvus_hnsw_time)), milvus_hnsw_time, color='tab:orange', marker='^', label='milvus hnsw')
# # milvus ivfflat曲线
# plt.plot(range(len(milvus_ivfflat_time)), milvus_ivfflat_time, color='tab:red', marker='*', label='milvus ivfflat')
#
# plt.xlabel('Parameters')
# plt.ylabel('time (s)', color='tab:blue')
# plt.title('Result of Query 5 - Time')
# # 设置横坐标刻度及标签
# parameter_labels = ['Param1', 'Param2', 'Param3', 'Param4', 'Param5']
# plt.xticks(range(len(parameter_labels)), parameter_labels)
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 绘制recall的折线图
# plt.figure(figsize=(10, 6))
# # pgvector hnsw曲线
# plt.plot(range(len(pgvector_hnsw_recall)), pgvector_hnsw_recall, color='tab:blue', marker='o', label='pgvector hnsw')
# # pgvector ivfflat曲线
# plt.plot(range(len(pgvector_ivfflat_recall)), pgvector_ivfflat_recall, color='tab:green', marker='s', label='pgvector ivfflat')
# # milvus hnsw曲线
# plt.plot(range(len(milvus_hnsw_recall)), milvus_hnsw_recall, color='tab:orange', marker='^', label='milvus hnsw')
# # milvus ivfflat曲线
# plt.plot(range(len(milvus_ivfflat_recall)), milvus_ivfflat_recall, color='tab:red', marker='*', label='milvus ivfflat')
#
# plt.xlabel('Parameters')
# plt.ylabel('recall', color='tab:green')
# plt.title('Result of Query 5 - Recall')
# # 设置横坐标刻度及标签
# parameter_labels = ['Param1', 'Param2', 'Param3', 'Param4', 'Param5']
# plt.xticks(range(len(parameter_labels)), parameter_labels)
# plt.legend()
# plt.grid(True)
# plt.show()


# dim_change top k
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 数据
# dimensions = [2000, 1024, 1000, 762, 256, 128, 64]
# top10_recall = [0.9000, 0.8000, 0.9000, 0.8000, 0.5000, 0.5000, 0.3000]
# top20_recall = [0.8000, 0.8000, 0.6500, 0.6500, 0.4500, 0.3500, 0.2000]
# top50_recall = [0.8000, 0.6600, 0.7400, 0.6200, 0.4800, 0.2600, 0.2400]
# top100_recall = [0.8300, 0.7500, 0.6900, 0.6300, 0.4600, 0.2900, 0.2400]
# top500_recall = [0.7940, 0.6900, 0.6980, 0.6600, 0.4040, 0.3080, 0.2500]
#
# # 创建折线图
# plt.figure(figsize=(10, 6))
#
# plt.plot(range(len(dimensions)), top10_recall, label='Top10 Recall', marker='o', linestyle='-', color='b')
# plt.plot(range(len(dimensions)), top20_recall, label='Top20 Recall', marker='o', linestyle='-', color='g')
# plt.plot(range(len(dimensions)), top50_recall, label='Top50 Recall', marker='o', linestyle='-', color='r')
# plt.plot(range(len(dimensions)), top100_recall, label='Top100 Recall', marker='o', linestyle='-', color='c')
# plt.plot(range(len(dimensions)), top500_recall, label='Top500 Recall', marker='o', linestyle='-', color='m')
#
# # 标题和标签
# plt.title('Order-preserving (1e5)', fontsize=16)
# plt.xlabel('Dimension', fontsize=14)
# plt.ylabel('Recall', fontsize=14)
# plt.xticks(range(len(dimensions)), dimensions)  # 设置x轴的刻度为维度
# plt.yticks(np.arange(0, 1.1, 0.1))  # 设置y轴的刻度为 0 到 1 之间的整数
#
# # 显示图例
# plt.legend()
#
# # 显示网格
# plt.grid(True)
#
# # 显示图像
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt

data = {
    "query1": {
        "ivfflat": {
            "time": [0.5940, 0.4768, 0.5507, 0.5124, 0.5744],
            "recall": [0.2000, 0.5800, 0.6300, 0.6400, 0.6700]
        },
        "hnsw": {
            "time": [0.6573, 0.6064, 0.4752, 0.5874, 0.6233],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query2": {
        "ivfflat": {
            "time": [0.5395, 0.6392, 0.5592, 0.4194, 0.5793],
            "recall": [0.1300, 0.1800, 0.2300, 0.2700, 0.4900]
        },
        "hnsw": {
            "time": [0.5199, 0.4602, 0.6411, 0.5826, 0.5437],
            "recall": [0.8500, 0.9000, 0.9200, 0.9500, 0.9700]
        }
    },
    "query3": {
        "ivfflat": {
            "time": [0.5801, 0.5800, 0.5801, 0.5600, 0.5601],
            "recall": [0.1200, 0.1600, 0.1800, 0.2100, 0.2400]
        },
        "hnsw": {
            "time": [0.5594, 0.5791, 0.4783, 0.6367, 0.5756],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query4": {
        "ivfflat": {
            "time": [0.6001, 0.5400, 0.6199, 0.6001, 0.5996],
            "recall": [0.0000, 0.0000, 0.0000, 0.1250, 0.1250]
        },
        "hnsw": {
            "time": [0.5200, 0.4400, 0.5799, 0.5799, 0.5999],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query5": {
        "ivfflat": {
            "time": [0.6716, 0.7301, 0.6919, 0.6715, 0.7316],
            "recall": [0.0700, 0.1400, 0.3200, 0.4000, 0.4100]
        },
        "hnsw": {
            "time": [0.6115, 0.6915, 0.7116, 0.7116, 0.6516],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query6": {
        "ivfflat": {
            "time": [0.6787, 0.6827, 0.6607, 0.6609, 0.7185],
            "recall": [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        },
        "hnsw": {
            "time": [0.7209, 0.6809, 0.6604, 0.7209, 0.6209],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query7": {
        "ivfflat": {
            "time": [0.6016, 0.6818, 0.6419, 0.7219, 0.7244],
            "recall": [0.1400, 0.8400, 0.8900, 0.9000, 0.9400]
        },
        "hnsw": {
            "time": [0.7422, 0.6823, 0.6425, 0.6822, 0.6823],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    },
    "query8": {
        "ivfflat": {
            "time": [0.6468, 0.5044, 0.6443, 0.5843, 0.4243],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        },
        "hnsw": {
            "time": [0.5638, 0.5639, 0.6039, 0.6041, 0.5041],
            "recall": [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        }
    }
}

# 创建画布与子图阵列
fig, axs = plt.subplots(8, 2, figsize=(22, 40), dpi=120)
probes = [1, 2, 3, 4, 5]
queries = [f'query{i}' for i in range(1, 9)]

# 全局样式设置
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 'medium',
    'axes.titlepad': 10
})

# 遍历每个query绘制子图
for idx, query in enumerate(queries):
    # ========== 时间子图 ==========
    ax = axs[idx, 0]
    # IVFFlat折线
    ax.plot(probes, data[query]["ivfflat"]["time"],
            'b--o', markersize=7, linewidth=1.8, label='IVFFlat')
    # HNSW折线
    ax.plot(probes, data[query]["hnsw"]["time"],
            'r-s', markersize=7, linewidth=1.8, label='HNSW')
    ax.set_title(f'{query} - Query Time(s)', pad=12)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('Time (s)', labelpad=8)

    # ========== Recall子图 ==========
    ax = axs[idx, 1]
    ax.plot(probes, data[query]["ivfflat"]["recall"],
            'b--o', markersize=7, linewidth=1.8)
    ax.plot(probes, data[query]["hnsw"]["recall"],
            'r-s', markersize=7, linewidth=1.8)
    ax.set_title(f'{query} - Recall Rate', pad=12)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.2)
    ax.set_ylabel('Recall', labelpad=8)

    # ========== 坐标轴设置 ==========
    # 仅最底层显示x轴标签
    if idx != 7:
        axs[idx, 0].set_xticklabels([])
        axs[idx, 1].set_xticklabels([])
    else:
        for col in [0, 1]:
            axs[idx, col].set_xlabel('Parameter Level (1-5)', labelpad=12)

    # ========== 图例设置 ==========
    if idx == 0:
        axs[idx, 0].legend(loc='upper left',
                           bbox_to_anchor=(0.15, 1.35),
                           ncol=2,
                           frameon=False,
                           fontsize=13)

# 调整布局
plt.subplots_adjust(hspace=0.3, wspace=0.25)
plt.suptitle('Milvus_Index Performance Comparison',
             y=0.995, fontsize=16, fontweight='bold')

# 保存为高分辨率图片
plt.savefig('Milvus.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.4)
plt.close()