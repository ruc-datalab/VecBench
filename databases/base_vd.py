from abc import ABC, abstractmethod

class BaseVD(ABC):
    def __init__(self, config, db_type):
        self.config = config
        self.db_type = db_type
        self.connection = None

    @abstractmethod
    def connect(self):
        """数据库连接"""
        pass

    @abstractmethod
    def create_table(self, table_name, schema):
        """创建数据库表/集合"""
        pass

    @abstractmethod
    def create_index(self, index_params, args):
        """创建索引"""
        pass

    @abstractmethod
    def process_data(self, data, table_name, schema):
        """处理数据"""
        pass

    @abstractmethod
    def insert_data(self, data, db, schema):
        """插入数据"""
        pass

    @abstractmethod
    def query(self, query):
        """执行查询"""
        pass


    def load_ground_truth(self, ground_truth_file):
        """加载ground truth数据"""
        import json
        with open(ground_truth_file, 'r') as f:
            return json.load(f)

    def close_connection(self):
        if self.connection:
            self.connection.close()
