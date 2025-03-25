import matplotlib.pyplot as plt

def plot_query_metrics(results):
    query_names = [result['name'] for result in results]
    execution_times = [result['avg_execution_time'] for result in results]
    execution_recalls = [result['avg_execution_recall'] for result in results]

    # 绘制执行时间的柱状图
    fig_time, ax_time = plt.subplots()
    bar_width = 0.6
    index_time = range(len(query_names))
    bar_time = ax_time.bar(index_time, execution_times, bar_width, label='Execution Time', color='tab:blue')
    ax_time.set_title('Query Execution Time')
    ax_time.set_xlabel('Query Name')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_xticks(index_time)
    ax_time.set_xticklabels(query_names)
    ax_time.legend()

    # 在执行时间的柱状图上添加数值标签
    def add_labels_time(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels_time(ax_time, bar_time)

    # 绘制执行召回率的柱状图
    fig_recall, ax_recall = plt.subplots()
    index_recall = range(len(query_names))
    bar_recall = ax_recall.bar(index_recall, execution_recalls, bar_width, label='Execution Recall', color='tab:orange')
    ax_recall.set_title('Query Execution Recall')
    ax_recall.set_xlabel('Query Name')
    ax_recall.set_ylabel('Recall')
    ax_recall.set_xticks(index_recall)
    ax_recall.set_xticklabels(query_names)
    ax_recall.legend()

    # 在执行召回率的柱状图上添加数值标签
    def add_labels_recall(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    add_labels_recall(ax_recall, bar_recall)

    plt.show()