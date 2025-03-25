import yaml
import pandas as pd
import ast
import json
from datetime import date

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_schema_from_yaml(file_path):
    with open(file_path, 'r') as file:
        schema = yaml.load(file, Loader=yaml.FullLoader)
    return schema

def load_query_from_yaml(query_type, query_file='config/queries.yaml'):
    with open(query_file, 'r') as file:
        queries = yaml.load(file, Loader=yaml.FullLoader)
    return queries.get(query_type)

def parse_data(value):
    """将text中的空值转换为None"""
    try:
        return value if pd.notna(value) else None
    except Exception:
        return None

def parse_array(value):
    """将字符串转换为列表（PostgreSQL数组格式）"""
    try:
        return ast.literal_eval(value) if pd.notna(value) else None
    except Exception:
        return None

# 目前这个有问题
def parse_vector(value):
    """将字符串转换为向量（列表形式）"""
    try:
        return list(map(float, value.split(','))) if pd.notna(value) else None
    except Exception:
        return None

# 递归函数：遍历所有元素并转换 datetime.date 为 '%Y-%m-%d' 格式的字符串
def convert_dates(obj):
    if isinstance(obj, date):  # 如果是 date 类型，进行转换
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, list):  # 如果是 list 类型，递归遍历
        return [convert_dates(item) for item in obj]
    elif isinstance(obj, tuple):  # 如果是 tuple 类型，递归遍历
        return tuple(convert_dates(item) for item in obj)
    elif isinstance(obj, dict):  # 如果是 dict 类型，递归遍历字典的键和值
        return {key: convert_dates(value) for key, value in obj.items()}
    else:
        return obj  # 其他类型不做转换，直接返回

def flatten(lst):
    """
    将可能存在嵌套的列表结构扁平化，方便后续比较元素
    """
    result = []
    for element in lst:
        if isinstance(element, list):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, date):
            return o.strftime('%Y-%m-%d')
        return super().default(o)

def save_ground_truth(data):
    json_data = []
    for item in data:
        json_item = {
            "name": item["name"],
            "result": item["result"]
        }
        json_data.append(json_item)
    # print(json_data)
    # 将数据保存为JSON文件，这里保存到当前目录下名为'results.json'的文件中，缩进设置为4，使格式更清晰
    with open('data/ground_truth.json', 'w') as f:
        json.dump(json_data, f, indent=4, cls=CustomJSONEncoder)